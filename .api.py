from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from typing import Dict
import json
import wave
import io
import numpy as np

from voice_agent.voice_agent import VoiceAgent
from vocode.streaming.models.audio import AudioEncoding
from vocode.streaming.models.message import MessageType
from vocode.streaming.utils import create_audio_queue as create_audio_stream
from vocode.streaming.utils import convert_audio as convert_wav_to_pcm

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active conversations
active_conversations: Dict[str, VoiceAgent] = {}

@app.websocket("/stream/{conversation_id}")
async def stream_audio(websocket: WebSocket, conversation_id: str):
    await websocket.accept()
    
    # Create audio streams for input/output
    input_stream = create_audio_stream(
        sampling_rate=16000,
        audio_encoding=AudioEncoding.LINEAR16,
        chunk_size=2048
    )
    output_stream = create_audio_stream(
        sampling_rate=16000,
        audio_encoding=AudioEncoding.LINEAR16,
        chunk_size=2048
    )

    # Create voice agent
    agent = VoiceAgent(input_stream, output_stream)
    active_conversations[conversation_id] = agent
    
    try:
        # Start agent in background task
        agent_task = asyncio.create_task(agent.start())
        
        # Handle incoming audio chunks
        async def receive_audio():
            while True:
                try:
                    data = await websocket.receive_bytes()
                    # Convert audio data to proper format if needed
                    audio_chunk = convert_wav_to_pcm(
                        data,
                        input_sample_rate=16000,
                        output_sample_rate=16000
                    )
                    await input_stream.put(audio_chunk)
                except Exception as e:
                    print(f"Error receiving audio: {e}")
                    break

        # Stream output audio back to client
        async def send_audio():
            while True:
                try:
                    chunk = await output_stream.get()
                    await websocket.send_bytes(chunk)
                except Exception as e:
                    print(f"Error sending audio: {e}")
                    break

        # Run both receiving and sending tasks
        await asyncio.gather(
            receive_audio(),
            send_audio(),
            agent_task
        )

    except Exception as e:
        print(f"WebSocket error: {e}")
    
    finally:
        # Cleanup
        if conversation_id in active_conversations:
            await agent.conversation.terminate()
            del active_conversations[conversation_id]
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
