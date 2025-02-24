import logging

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
import os

from vocode.streaming.agent.chat_gpt_agent import ChatGPTAgent
from vocode.streaming.client_backend.conversation import ConversationRouter
from vocode.streaming.models.agent import ChatGPTAgentConfig
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.transcriber import DeepgramTranscriberConfig, TimeEndpointingConfig
from vocode.streaming.synthesizer.eleven_labs_synthesizer import ElevenLabsSynthesizer
from vocode.streaming.models.synthesizer import ElevenLabsSynthesizerConfig
from vocode.streaming.transcriber.deepgram_transcriber import DeepgramTranscriber
from vocode.streaming.streaming_conversation import StreamingConversation
from vocode.streaming.models.audio import AudioEncoding
from vocode.streaming.models.client_backend import InputAudioConfig, OutputAudioConfig
# from vocode.streaming.vector_db.factory import VectorDBFactory
# from vocode.streaming.vector_db.pinecone import PineconeConfig

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Ensure that the environment variable 'PINECONE_INDEX_NAME' is not None
# pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
# if pinecone_index_name is None:
#     raise ValueError("Environment variable 'PINECONE_INDEX_NAME' is not set.")

# vector_db_config = PineconeConfig(index=pinecone_index_name)

INITIAL_MESSAGE = "Hello!"
PROMPT_PREAMBLE = """
I want you to act as an IT Architect. 
I will provide some details about the functionality of an application or other 
digital product, and it will be your job to come up with ways to integrate it 
into the IT landscape. This could involve analyzing business requirements, 
performing a gap analysis, and mapping the functionality of the new system to 
the existing IT landscape. The next steps are to create a solution design. 

You are an expert in these technologies: 
- Langchain
- Supabase
- Next.js
- Fastapi
- Vocode.
"""

TIME_ENDPOINTING_CONFIG = TimeEndpointingConfig()
TIME_ENDPOINTING_CONFIG.time_cutoff_seconds = 2


OUTPUT_AUDIO_CONFIG = OutputAudioConfig(
    sampling_rate=16000,
    audio_encoding=AudioEncoding.LINEAR16,
)

# Update synthesizer configuration
ELEVENLABS_SYNTHESIZER_CONFIG = ElevenLabsSynthesizerConfig(
    sampling_rate=16000,
    audio_encoding=AudioEncoding.LINEAR16,
    voice_id="21m00Tcm4TlvDq8ikWAM",  # Default ElevenLabs voice ID
    stability=0.5,
    similarity_boost=0.75,
    model_id="eleven_monolingual_v1",
    optimize_streaming_latency=0,
    api_key=os.getenv("ELEVENLABS_API_KEY"),
)

# ELEVENLABS_SYNTHESIZER_THUNK = ElevenLabsSynthesizer(
#     ELEVENLABS_SYNTHESIZER_CONFIG,
#     logger=logger,
# )

ELEVENLABS_SYNTHESIZER_THUNK = ElevenLabsSynthesizer(
    ELEVENLABS_SYNTHESIZER_CONFIG,
)


DEEPGRAM_TRANSCRIBER_THUNK = lambda input_audio_config: DeepgramTranscriber(
    DeepgramTranscriberConfig.from_input_audio_config(
        input_audio_config=input_audio_config,
        endpointing_config=TIME_ENDPOINTING_CONFIG,
        min_interrupt_confidence=0.9,
    )
)


conversation_router = ConversationRouter(
    agent_thunk=lambda: ChatGPTAgent(
        ChatGPTAgentConfig(
            initial_message=BaseMessage(text=INITIAL_MESSAGE),
            prompt_preamble=PROMPT_PREAMBLE,
            # vector_db_config=vector_db_config,
        ),
        logger=logger,
    ),
    synthesizer_thunk=ELEVENLABS_SYNTHESIZER_THUNK,
    transcriber_thunk=DEEPGRAM_TRANSCRIBER_THUNK,
)

app.include_router(conversation_router.get_router())

# Store active conversations
active_conversations: Dict[str, StreamingConversation] = {}

@app.websocket("/stream/{conversation_id}")
async def stream_audio(websocket: WebSocket, conversation_id: str):
    try:
        conversation = await conversation_router.conversation(websocket)
        active_conversations[conversation_id] = conversation
        
        await conversation.start()
        
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        if conversation_id in active_conversations:
            del active_conversations[conversation_id]

# Client code moved to server_test.py

