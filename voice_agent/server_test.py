import asyncio
import sounddevice as sd
import websockets
import numpy as np
import signal
import sys
import json



class VoiceClient:
    def __init__(self, websocket_url="ws://localhost:8000"):
        self.websocket_url = websocket_url
        self.chunk_size = 2048
        self.channels = 1
        self.sample_rate = 16000

        # Initialize audio streams
        self.input_stream = sd.InputStream(
            channels=self.channels,
            samplerate=self.sample_rate,
            blocksize=self.chunk_size
        )
        
        self.output_stream = sd.OutputStream(
            channels=self.channels,
            samplerate=self.sample_rate,
            blocksize=self.chunk_size
        )

        self.input_stream.start()
        self.output_stream.start()

    async def record_audio(self, websocket):
        """Record audio and send to websocket"""
        print(f"Starting audio recording stream to {websocket.remote_address}")
        # # Signal that we're starting the conversation
        # start_message = {
        #     "type": "websocket_start",
        #     "sampling_rate": self.sample_rate,
        #     "chunk_size": self.chunk_size,
        #     "transcriber_config": {
        #         "sampling_rate": self.sample_rate,
        #         "audio_encoding": "linear16",
        #         "chunk_size": self.chunk_size,
        #         "endpointing_config": {
        #             "type": "time",
        #             "time_cutoff_seconds": 2
        #         }
        #     },
        #     "agent_config": {
        #         "initial_message": "Hello!",
        #         "prompt_preamble": """
        #         I want you to act as an IT Architect. 
        #         I will provide some details about the functionality of an application or other 
        #         digital product, and it will be your job to come up with ways to integrate it 
        #         into the IT landscape. This could involve analyzing business requirements, 
        #         performing a gap analysis, and mapping the functionality of the new system to 
        #         the existing IT landscape. The next steps are to create a solution design. 

        #         You are an expert in these technologies: 
        #         - Langchain
        #         - Supabase
        #         - Next.js
        #         - Fastapi
        #         - Vocode.
        #         """
        #     },
        #     "synthesizer_config": {
        #         "sampling_rate": self.sample_rate,
        #         "audio_encoding": "linear16"
        #     }
        # }
        # await websocket.send(json.dumps(start_message))

        while True:
            try:
                audio_data, _ = self.input_stream.read(self.chunk_size)
                # Only send if audio level is above threshold
                audio_level = np.abs(audio_data).mean()
                if audio_level > 0.01:  # Adjust threshold as needed
                    # Convert to bytes
                    audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
                    print(f"Recording audio chunk of {len(audio_bytes)} bytes, level: {audio_level:.3f}")
                    # Use vocode's message format
                    
                    message = {
                        "type": "websocket_audio",
                        "data": audio_bytes.hex(),
                        "sampling_rate": self.sample_rate,
                        "audio_encoding": "linear16",
                        "chunk_size": self.chunk_size,
                    }
                    await websocket.send(json.dumps(message))
                else:
                    continue
            except Exception as e:
                print(f"Error recording audio: {e}")
                break
        print(f"Closed recording stream to {websocket.remote_address}")

    async def play_audio(self, websocket):
        """Receive audio from websocket and play it"""
        print(f"Starting audio playback stream from {websocket.remote_address}")
        while True:
            try:
                message = json.loads(await websocket.recv())
                if message["type"] == "websocket_audio":  # Updated to match vocode's type
                    audio_bytes = bytes.fromhex(message["data"])
                    audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767
                    self.output_stream.write(audio_data)
            except Exception as e:
                print(f"Error playing audio: {e}")
                break
        print(f"Closed playback stream from {websocket.remote_address}")

    async def start(self):
        """Start the voice client"""
        conversation_id = "test-conversation"
        websocket_url = f"{self.websocket_url}/stream/{conversation_id}"
        
        print(f"Attempting to connect to {websocket_url}")
        async with websockets.connect(websocket_url) as websocket:
            print(f"Connected to voice server at {websocket.remote_address}")
            try:
                await asyncio.gather(
                    self.record_audio(websocket),
                    self.play_audio(websocket)
                )
            except KeyboardInterrupt:
                print("\nStopping voice client...")
            finally:
                print(f"Closing connection to {websocket.remote_address}")
                self.input_stream.stop()
                self.output_stream.stop()
def main():
    client = VoiceClient("ws://localhost:8000")
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\nExiting...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start the client
    asyncio.get_event_loop().run_until_complete(client.start())

if __name__ == "__main__":
    main()

