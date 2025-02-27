import asyncio
import json
import numpy as np
import websockets
import sounddevice as sd
import base64
import logging
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Frontend test client
async def run_websocket_client():
    """Simple WebSocket client to test the voice agent"""
    uri = "ws://localhost:8080/conversation"
    
    # Generate a unique conversation ID
    conversation_id = f"client-{uuid.uuid4()}"
    logger.info(f"Starting conversation with ID: {conversation_id}")
    
    # Configure WebSocket with a longer timeout
    async with websockets.connect(uri, ping_interval=20, ping_timeout=60) as websocket:
        print("Connected to server!")
        
        # Send start message with proper audio configuration
        # start_message = {
        #     "type": "websocket_start",
        #     "conversation_id": conversation_id,
        #     "transcriber_config": {
        #         "type": "transcriber_deepgram",
        #         "sampling_rate": 16000,
        #         "audio_encoding": "linear16",
        #         "chunk_size": 1024
        #     },
        #     "agent_config": {
        #         "type": "agent_chat_gpt",
        #         "initial_message": {"text": "Hello! How can I help you today?"},
        #         "prompt_preamble": "You are a helpful assistant."
        #     },
        #     "synthesizer_config": {
        #         "type": "synthesizer_eleven_labs",
        #         "sampling_rate": 16000,
        #         "audio_encoding": "linear16"
        #     }
        # }
        # await websocket.send(json.dumps(start_message))

        audio_config_start_message = {
            "type": "websocket_audio_config_start",
            "conversation_id": conversation_id,
            "input_audio_config": {
                "sampling_rate": 16000,
                "audio_encoding": "linear16",
                "chunk_size": 1024
            },
            "output_audio_config": {
                "sampling_rate": 16000,
                "audio_encoding": "linear16"
            }
        }
        await websocket.send(json.dumps(audio_config_start_message))
        print(f"Conversation started with ID: {conversation_id}")
        # Create a queue for sending audio data
        audio_queue = asyncio.Queue()
        # Set up audio capture from microphone
        def audio_callback(indata, frames, time, status):
            if status:
                print(f"Audio status: {status}")
                
            # Convert to int16 and encode as base64 for transmission
            audio_data = (indata.flatten() * 32767).astype(np.int16).tobytes()
            audio_message = {
                "type": "websocket_audio",
                "data": base64.b64encode(audio_data).decode('utf-8'),
                "sampling_rate": 16000,
                "audio_encoding": "linear16",
                "chunk_size": 1024
            }
            # Put the message in the queue instead of sending directly
            audio_queue.put_nowait(json.dumps(audio_message))
        
        # Start audio stream from microphone
        stream = sd.InputStream(
            callback=audio_callback,
            channels=1,
            samplerate=16000,
            blocksize=1024
        )
        
        # Task to send audio data from the queue
        async def send_audio():
            try:
                while True:
                    message = await audio_queue.get()
                    if websocket.open:
                        await websocket.send(message)
                    audio_queue.task_done()
            except websockets.exceptions.ConnectionClosed:
                logger.info("WebSocket connection closed")
            except Exception as e:
                logger.error(f"Error sending audio: {e}")
        
        # Task to receive messages from the server
        async def receive_messages():
            try:
                while True:
                    response = await websocket.recv()
                    message = json.loads(response)
                    logger.info(f"Received message type: {message.get('type')}")
                    
                    if message["type"] == "websocket_audio":
                        # Decode base64 audio and play through speakers
                        audio_bytes = base64.b64decode(message["data"])
                        audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767
                        sd.play(audio_data, 16000)
                        sd.wait()
                    elif message["type"] == "websocket_ready":
                        print(f"WebSocket connection ready")
                    elif message["type"] == "websocket_transcript":
                        print(f"You said: {message.get('text', '')}")
                    elif message["type"] == "websocket_message":
                        print(f"Agent: {message.get('text', '')}")
                    elif message["type"] == "websocket_error":
                        print(f"Error: {message.get('message', '')}")
                    else:
                        logger.info(f"Unknown message type: {message}")
            except websockets.exceptions.ConnectionClosed:
                logger.info("WebSocket connection closed by server")
            except Exception as e:
                logger.error(f"Error receiving messages: {e}")
        
        # Start the tasks
        send_task = asyncio.create_task(send_audio())
        receive_task = asyncio.create_task(receive_messages())
        
        # Listen for messages from the server
        try:
            stream.start()
            print("Microphone activated. Speak into your microphone.")
            print("Press Ctrl+C to stop the client.")
            
            # Wait for tasks to complete
            await asyncio.gather(send_task, receive_task)
                
        except KeyboardInterrupt:
            print("\nStopping client...")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            # Clean up
            stream.stop()
            print("Microphone deactivated")
            
            # Cancel tasks
            send_task.cancel()
            receive_task.cancel()
            
            # Try to send stop message if connection is still open
            try:
                if websocket.open:
                    stop_message = {
                        "type": "websocket_stop",
                        "conversation_id": conversation_id
                    }
                    await websocket.send(json.dumps(stop_message))
                    print("Conversation ended")
            except:
                print("Could not send stop message - connection already closed")

if __name__ == "__main__":
    try:
        asyncio.run(run_websocket_client())
    except KeyboardInterrupt:
        print("Client stopped by user")
    except Exception as e:
        print(f"Client error: {e}")

