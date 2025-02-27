import asyncio
import signal
import os
from dotenv import load_dotenv
import numpy as np
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict

from vocode.helpers import create_streaming_microphone_input_and_speaker_output
from vocode.logging import configure_pretty_logging
from vocode.streaming.agent.chat_gpt_agent import ChatGPTAgent
from vocode.streaming.models.agent import ChatGPTAgentConfig
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import ElevenLabsSynthesizerConfig
from vocode.streaming.models.transcriber import (
    DeepgramTranscriberConfig,
    PunctuationEndpointingConfig,
)
from vocode.streaming.streaming_conversation import StreamingConversation
from vocode.streaming.synthesizer.eleven_labs_synthesizer import ElevenLabsSynthesizer
from vocode.streaming.transcriber.deepgram_transcriber import DeepgramTranscriber
from voice_agent.actions.load_book import MyCustomActionFactory
from voice_agent.audio_processor import AudioProcessor
import webrtcvad

configure_pretty_logging()


class Settings(BaseSettings):
    """
    Settings for the streaming conversation quickstart.
    These parameters can be configured with environment variables.
    """
    load_dotenv()
    openai_api_key: str = os.getenv("OPENAI_API_KEY")
    deepgram_api_key: str = os.getenv("DEEPGRAM_API_KEY")
    elevenlabs_api_key: str = os.getenv("ELEVENLABS_API_KEY")


    # This means a .env file can be used to overload these settings
    # ex: "OPENAI_API_KEY=my_key" will set openai_api_key over the default above
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()


class VoiceAgent:
    def __init__(self, input_stream, output_stream):
        self.microphone_input = input_stream
        self.speaker_output = output_stream
        self.conversation = None
        self.is_speaking = False
        self.current_playback_chunk = None
        
        # Initialize audio processor for echo cancellation
        self.audio_processor = AudioProcessor(sample_rate=16000, frame_size=480)  # 30ms frames
        
        # Initialize VAD for human speech detection
        self.vad = webrtcvad.Vad(2)  # Medium aggressiveness
        self.frame_duration_ms = 30
        self.samples_per_frame = int(16000 * self.frame_duration_ms / 1000)
        
        # Speech detection parameters
        self.speech_frames_threshold = 3  # Number of consecutive speech frames to trigger interruption
        self.speech_frames_count = 0
        self.min_audio_level = 0.02  # Minimum audio level to consider for VAD

    def _on_start_speaking(self):
        self.is_speaking = True
        self.speech_frames_count = 0  # Reset speech counter when agent starts speaking
        
    def _on_stop_speaking(self):
        self.is_speaking = False
        self.current_playback_chunk = None

    def _on_playback_chunk(self, chunk):
        """Store the current playback chunk for echo cancellation"""
        self.current_playback_chunk = chunk

    def _detect_human_speech(self, audio_chunk: np.ndarray) -> bool:
        """
        Detect human speech using VAD and audio level analysis
        Returns True if human speech is detected
        """
        try:
            # Check audio level first
            audio_level = np.abs(audio_chunk).mean()
            if audio_level < self.min_audio_level:
                self.speech_frames_count = 0
                return False

            # Convert to int16 for VAD
            if audio_chunk.dtype == np.float32:
                audio_chunk = (audio_chunk * 32767).astype(np.int16)

            # Process frame by frame
            is_speech = False
            for i in range(0, len(audio_chunk) - self.samples_per_frame, self.samples_per_frame):
                frame = audio_chunk[i:i + self.samples_per_frame]
                try:
                    if self.vad.is_speech(frame.tobytes(), 16000):
                        self.speech_frames_count += 1
                        if self.speech_frames_count >= self.speech_frames_threshold:
                            is_speech = True
                            break
                    else:
                        self.speech_frames_count = max(0, self.speech_frames_count - 1)
                except Exception:
                    continue

            if not is_speech:
                self.speech_frames_count = max(0, self.speech_frames_count - 1)

            return is_speech

        except Exception as e:
            print(f"Error in speech detection: {e}")
            return False

    async def setup(self, story: str):            
        self.conversation = StreamingConversation(
            output_device=self.speaker_output,
            transcriber=DeepgramTranscriber(
                DeepgramTranscriberConfig.from_input_device(
                    self.microphone_input,
                    endpointing_config=PunctuationEndpointingConfig(),
                    api_key=settings.deepgram_api_key,
                    # Increase confidence threshold and add VAD
                    min_interrupt_confidence=0.95,
                    interrupt_words=["Hi Angela"],
                    language="en-US",
                    model="nova-2",
                    use_vad=True,
                    vad_config={
                        "min_speech_duration_ms": 500,
                        "min_silence_duration_ms": 500,
                    }
                ),
            ),
            agent=ChatGPTAgent(
                ChatGPTAgentConfig(
                    openai_api_key=settings.openai_api_key,
                    initial_message=BaseMessage(text=story),
                    prompt_preamble="""You are baby watcher. Your name is Angela. You are telling the story from the initial message. 
                    When you reeive some special words, like "Hi Angela", you should communicate with the kids, answer question. etc. 
                    """,
                ),
                action_factory=MyCustomActionFactory(),
            ),
            synthesizer=ElevenLabsSynthesizer(
                ElevenLabsSynthesizerConfig.from_output_device(
                    self.speaker_output,
                    api_key=settings.elevenlabs_api_key,
                    voice_id="21m00Tcm4TlvDq8ikWAM",
                    output_format="pcm_16000",
                    on_playback_start=self._on_start_speaking,
                    on_playback_end=self._on_stop_speaking,
                    on_audio_chunk=self._on_playback_chunk,
                ),
            ),
        )

    async def start(self, story: str):
        await self.setup(story=story)
        await self.conversation.start()
        print("Conversation started, press Ctrl+C to end")
        signal.signal(signal.SIGINT, lambda _0, _1: asyncio.create_task(self.conversation.terminate()))
        
        while self.conversation.is_active():
            chunk = await self.microphone_input.get_audio()
            
            # Process the audio with echo cancellation
            processed_chunk = self.audio_processor.process_microphone_input(
                chunk,
                self.current_playback_chunk if self.is_speaking else None
            )
            
            # Check for human speech in the processed audio
            is_human_speech = self._detect_human_speech(processed_chunk)
            
            # Allow interruption only if human speech is detected
            if not self.is_speaking or (self.is_speaking and is_human_speech):
                self.conversation.receive_audio(processed_chunk)


async def main(init_message: str):
    import sounddevice as sd
    
    # List available devices
    devices = sd.query_devices()
    print("\nAvailable audio devices:")
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            print(f"Input {i}: {dev['name']}")
        if dev['max_output_channels'] > 0:
            print(f"Output {i}: {dev['name']}")
    
    # Manual device selection
    try:
        input_device = int(input("\nSelect input device number: "))
        output_device = int(input("Select output device number: "))
        
        # Verify selected devices
        input_info = sd.query_devices(input_device)
        output_info = sd.query_devices(output_device)
        print(f"\nSelected input: {input_info['name']}")
        print(f"Selected output: {output_info['name']}")
        
        # Create audio streams with explicit device selection
        (
            microphone_input,
            speaker_output,
        ) = create_streaming_microphone_input_and_speaker_output(
            use_default_devices=False,
            mic_sampling_rate=16000,
            speaker_sampling_rate=16000
        )
        
        print("\nStarting voice agent...")
        agent = VoiceAgent(microphone_input, speaker_output)
        await agent.start(story=init_message)
        
    except ValueError as e:
        print(f"Error: Invalid device selection - {e}")
    except Exception as e:
        print(f"Error initializing audio devices: {e}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Voice Agent PDF Reader')
    parser.add_argument('--pdf-file', type=str, help='Path to the PDF file to read')
    parser.add_argument('--init-message', type=str, help='message to start with', default="Hi, I am Angela")
    
    args = parser.parse_args()
    
    asyncio.run(main(args.init_message))
