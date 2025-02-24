import asyncio
import websockets
import pyaudio
import wave
import io
import sys
import signal

class VoiceClient:
    def __init__(self, websocket_url):
        self.websocket_url = websocket_url
        self.p = pyaudio.PyAudio()
        self.chunk_size = 2048
        self.sample_rate = 16000
        self.channels = 1
        self.format = pyaudio.paInt16
        
        # Setup audio stream for recording
        self.input_stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        # Setup audio stream for playback
        self.output_stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            output=True,
            frames_per_buffer=self.chunk_size
        )

    async def record_audio(self, websocket):
        """Record audio from microphone and send to websocket"""
        while True:
            try:
                audio_data = self.input_stream.read(self.chunk_size)
                # Create WAV format
                wav_buffer = io.BytesIO()
                with wave.open(wav_buffer, 'wb') as wav_file:
                    wav_file.setnchannels(self.channels)
                    wav_file.setsampwidth(self.p.get_sample_size(self.format))
                    wav_file.setframerate(self.sample_rate)
                    wav_file.writeframes(audio_data)
                await websocket.send(wav_buffer.getvalue())
            except Exception as e:
                print(f"Error recording audio: {e}")
                break

    async def play_audio(self, websocket):
        """Receive audio from websocket and play it"""
        while True:
            try:
                audio_data = await websocket.recv()
                self.output_stream.write(audio_data)
            except Exception as e:
                print(f"Error playing audio: {e}")
                break

    async def start(self):
        """Start the voice client"""
        conversation_id = "test-conversation"  # You might want to generate this dynamically
        websocket_url = f"{self.websocket_url}/stream/{conversation_id}"
        
        async with websockets.connect(websocket_url) as websocket:
            print("Connected to voice server")
            try:
                await asyncio.gather(
                    self.record_audio(websocket),
                    self.play_audio(websocket)
                )
            except KeyboardInterrupt:
                print("\nStopping voice client...")
            finally:
                self.input_stream.stop_stream()
                self.output_stream.stop_stream()
                self.input_stream.close()
                self.output_stream.close()
                self.p.terminate()

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
