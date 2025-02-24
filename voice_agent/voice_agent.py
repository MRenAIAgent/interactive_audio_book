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
                    on_playback_start=lambda: setattr(self, 'is_speaking', True),
                    on_playback_end=lambda: setattr(self, 'is_speaking', False),
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
            # Only process microphone input when not speaking
            if not self.is_speaking:
                self.conversation.receive_audio(chunk)


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
