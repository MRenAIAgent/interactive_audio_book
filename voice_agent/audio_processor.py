import numpy as np
import speexdsp
import logging
from typing import Optional

class AudioProcessor:
    def __init__(self, sample_rate: int = 16000, frame_size: int = 1024):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.logger = logging.getLogger(__name__)
        
        # Initialize echo canceller
        self.echo_state = speexdsp.EchoState(frame_size, int(0.1 * sample_rate))  # 100ms tail length
        self.echo_state.set_sampling_rate(sample_rate)
        
        # Configure echo canceller
        self.echo_state.set_echo_delay(0)  # Start with no delay
        self.echo_state.set_capture_gain(1.0)  # Normal capture gain
        self.echo_state.set_playback_gain(1.0)  # Normal playback gain
        
        self.logger.info(f"Initialized AudioProcessor with {sample_rate}Hz sampling rate and {frame_size} frame size")
        
    def process_microphone_input(self, mic_data: np.ndarray, playback_data: Optional[np.ndarray] = None) -> np.ndarray:
        """Process microphone input with echo cancellation"""
        try:
            # Convert float32 to int16 if needed
            if mic_data.dtype == np.float32:
                mic_data = (mic_data * 32767).astype(np.int16)
            
            # If we have playback data, use it for echo cancellation
            if playback_data is not None:
                # Convert playback data to int16 if needed
                if playback_data.dtype == np.float32:
                    playback_data = (playback_data * 32767).astype(np.int16)
                
                # Ensure both arrays are the same size
                min_size = min(len(mic_data), len(playback_data))
                mic_data = mic_data[:min_size]
                playback_data = playback_data[:min_size]
                
                # Apply echo cancellation
                try:
                    # Process frame by frame
                    processed_data = np.zeros_like(mic_data)
                    for i in range(0, len(mic_data), self.frame_size):
                        frame = mic_data[i:i + self.frame_size]
                        echo_frame = playback_data[i:i + self.frame_size]
                        
                        # Pad last frame if needed
                        if len(frame) < self.frame_size:
                            frame = np.pad(frame, (0, self.frame_size - len(frame)))
                            echo_frame = np.pad(echo_frame, (0, self.frame_size - len(echo_frame)))
                        
                        # Apply echo cancellation
                        processed_frame = self.echo_state.process(frame, echo_frame)
                        processed_data[i:i + len(processed_frame)] = processed_frame
                    
                    self.logger.debug("Echo cancellation applied successfully")
                    
                except Exception as e:
                    self.logger.error(f"Error in echo cancellation: {e}")
                    processed_data = mic_data
            else:
                processed_data = mic_data
            
            # Convert back to float32 for output
            return processed_data.astype(np.float32) / 32767.0
            
        except Exception as e:
            self.logger.error(f"Error in audio processing: {e}")
            # In case of any error, return the original audio
            return mic_data.astype(np.float32) / 32767.0 if mic_data.dtype == np.int16 else mic_data 