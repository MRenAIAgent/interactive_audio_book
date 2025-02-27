import asyncio
import json
import logging
import signal
import numpy as np
import websockets
from typing import Dict, List, Optional, Any, Callable
import webrtcvad
import os

from vocode.streaming.agent.chat_gpt_agent import ChatGPTAgent, ChatGPTAgentConfig
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.synthesizer.eleven_labs_synthesizer import ElevenLabsSynthesizer, ElevenLabsSynthesizerConfig
from vocode.streaming.transcriber.deepgram_transcriber import DeepgramTranscriber, DeepgramTranscriberConfig
from vocode.streaming.streaming_conversation import StreamingConversation
from vocode.streaming.models.websocket import (
    WebSocketMessage,
    WebSocketMessageType,
    AudioMessage,
    StartMessage,
    StopMessage,
    AudioConfigStartMessage,
)
from vocode.streaming.models.client_backend import InputAudioConfig, OutputAudioConfig
# Import the correct ConversationRouter from client_backend
from vocode.streaming.client_backend.conversation import ConversationRouter
from voice_agent.actions.load_book import LoadBookAction, LoadBookVocodeActionConfig
from voice_agent.actions.serpy_search import SerperSearchAction, SerperSearchVocodeActionConfig
# Define settings

import fastapi
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv


app = FastAPI(docs_url=None)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# Define API models
class ConversationRequest(BaseModel):
    conversation_id: str
    user_message: Optional[str] = None

class ConversationResponse(BaseModel):
    conversation_id: str
    status: str
    message: Optional[str] = None


class Settings:
    def __init__(self):
        self.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        self.elevenlabs_api_key = os.environ.get("ELEVENLABS_API_KEY", "")
        self.deepgram_api_key = os.environ.get("DEEPGRAM_API_KEY", "")
        self.serper_api_key = os.environ.get("SERPER_API_KEY", "")

load_dotenv()
settings = Settings()

# Audio processor for echo cancellation
class AudioProcessor:
    def __init__(self):
        self.echo_suppression_factor = 0.5
        
    def process_microphone_input(self, mic_input, speaker_output=None):
        """
        Process microphone input with echo cancellation if speaker output is provided
        """
        if speaker_output is not None and len(mic_input) == len(speaker_output):
            # Simple echo cancellation by subtracting speaker output from mic input
            processed = mic_input - (speaker_output * self.echo_suppression_factor)
            return processed
        return mic_input

# Custom action factory
from typing import Dict, Type

from vocode.streaming.action.abstract_factory import AbstractActionFactory
from vocode.streaming.action.base_action import BaseAction
from vocode.streaming.models.actions import ActionConfig

from voice_agent.actions.load_book import LoadBookAction

class MyCustomActionFactory(AbstractActionFactory):
    def __init__(self):
        self.action_map = {
            "action_load_book": LoadBookAction,
            "serper_search": SerperSearchAction,
        }
    
    def create_action(self, action_config: ActionConfig) -> BaseAction:
        """Create an action instance based on the action type"""
        if action_config.type in self.action_map:
            return self.action_map[action_config.type](action_config)
        else:
            raise Exception(f"Action type '{action_config.type}' not supported by Agent config.")


class VoiceAgentWebSocket:
    def __init__(self):
        self.router = ConversationRouter(
            agent_thunk=self.create_agent,
            transcriber_thunk=self.create_transcriber,
            synthesizer_thunk=self.create_synthesizer,
        )
        self.conversations: Dict[str, StreamingConversation] = {}
        self.audio_processors: Dict[str, AudioProcessor] = {}
        self.is_speaking: Dict[str, bool] = {}
        self.current_playback_chunk: Dict[str, Optional[np.ndarray]] = {}
        self.speech_frames_count: Dict[str, int] = {}
        
        # VAD setup
        self.vad = webrtcvad.Vad(3)  # Aggressiveness level 3 (highest)
        self.samples_per_frame = 480  # 30ms at 16kHz
        self.speech_frames_threshold = 3
        self.min_audio_level = 0.01
        
        logger.info("VoiceAgentWebSocket initialized")

    def create_agent(self):
        """Create a new agent for a conversation"""
        
        return ChatGPTAgent(
            ChatGPTAgentConfig(
                openai_api_key=settings.openai_api_key,
                initial_message=BaseMessage(text="Hi, I am Angela, your virtual assistant."),
                prompt_preamble="""You are a helpful assistant named Angela. You are here to assist with any questions or tasks.
                When you hear special words like "Hi Angela", you should respond promptly and helpfully.
                """,
                # The 'actions' parameter is used to define a list of action configurations
                # that the agent can use during the conversation
                actions=[
                    LoadBookVocodeActionConfig(type="action_load_book"),
                    # # Serpy API Search action configuration
                    # SerperSearchVocodeActionConfig(
                    #     type="serper_search",
                    #     api_key = settings.serper_api_key,
                    # ),
                ],
            ),
            # The 'action_factory' parameter is used to provide a factory class that creates
            # action instances from action configurations. It implements the actual action logic.
            action_factory=MyCustomActionFactory(),
        )
    
    def create_transcriber(self, input_audio_config: InputAudioConfig):
        """Create a new transcriber for a conversation"""
        sampling_rate = input_audio_config.sampling_rate
        audio_encoding = input_audio_config.audio_encoding
        chunk_size = input_audio_config.chunk_size
        return DeepgramTranscriber(
            DeepgramTranscriberConfig(
                sampling_rate=sampling_rate,
                audio_encoding=audio_encoding,
                chunk_size=chunk_size,
                api_key=settings.deepgram_api_key,
                min_interrupt_confidence=0.95,
                interrupt_words=["Hi Angela"],
                language="en-US",
                model="nova-2",
                use_vad=True,
                vad_config={
                    "min_speech_duration_ms": 500,
                    "min_silence_duration_ms": 500,
                }
            )
        )
    
    def create_synthesizer(self, output_audio_config: OutputAudioConfig):
        """Create a new synthesizer for a conversation"""
        sampling_rate = output_audio_config.sampling_rate
        audio_encoding = output_audio_config.audio_encoding
        return ElevenLabsSynthesizer(
            ElevenLabsSynthesizerConfig(
                api_key=settings.elevenlabs_api_key,
                voice_id="21m00Tcm4TlvDq8ikWAM",
                output_format="pcm_16000",
                sampling_rate=sampling_rate,
                audio_encoding=audio_encoding,
            )
        )

    def _on_start_speaking(self):
        self.is_speaking = True
        self.speech_frames_count = 0  # Reset speech counter when agent starts speaking
        logger.debug(f"Agent started speaking in conversation: {conversation_id}")
        
    def _on_stop_speaking(self):
        self.is_speaking = False
        self.current_playback_chunk = None

    def _on_playback_chunk(self, chunk: np.ndarray):
        """Store the current playback chunk for echo cancellation"""
        self.current_playback_chunk = chunk
        # Log audio level for debugging
        if chunk is not None:
            audio_level = np.abs(chunk).mean()
            logger.debug(f"Playback chunk in conversation, level: {audio_level:.4f}")

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
                            logger.debug("Human speech detected")
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
            logger.error(f"Error in speech detection: {e}")
            return False
    async def process_audio(self, conversation_id: str, audio_data: np.ndarray):
        """Process incoming audio from the client"""
        if conversation_id not in self.audio_processors:
            logger.info(f"Initializing audio processor for conversation: {conversation_id}")
            self.audio_processors = AudioProcessor()
            self.is_speaking = False
            self.current_playback_chunk = None
            self.speech_frames_count = 0
            
        # Process the audio with echo cancellation
        processed_chunk = self.audio_processors.process_microphone_input(
            audio_data,
            self.current_playback_chunk if self.is_speaking else None
        )
        
        # Check for human speech in the processed audio
        is_human_speech = self._detect_human_speech(conversation_id, processed_chunk)
        
        # Allow interruption only if human speech is detected
        if not self.is_speaking or (self.is_speaking and is_human_speech):
            logger.debug(f"Processing audio in conversation {conversation_id}")
            return processed_chunk
        
        logger.debug(f"Ignoring audio in conversation {conversation_id} (agent is speaking)")
        return None


voice_agent = VoiceAgentWebSocket()
app.include_router(voice_agent.router.get_router())

# Add a health check endpoint
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "active_conversations": list(voice_agent.conversations.keys()),
        "speaking_status": voice_agent.is_speaking
    }

# Make sure the server is running when this file is executed directly
if __name__ == "__main__":
    import uvicorn
    print("Starting voice agent websocket server on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8080)

