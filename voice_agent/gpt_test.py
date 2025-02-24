import asyncio
import websockets
import pyaudio

# Configuration
WEBSOCKET_URI = "ws://127.0.0.1:8000"  # Replace with your server URL
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # 16kHz, common for speech processing
CHUNK = 1024  # Buffer size

def record_audio():
    """ Generator that records audio from the microphone and yields chunks. """
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    
    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            yield data
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

async def send_audio():
    """ Connects to the Vocode WebSocket server and streams audio data. """
    async with websockets.connect(WEBSOCKET_URI) as websocket:
        for chunk in record_audio():
            await websocket.send(chunk)
            response = await websocket.recv()  # Receive response if applicable
            print("Server Response:", response)

async def main():
    await send_audio()

if __name__ == "__main__":
    asyncio.run(main())
