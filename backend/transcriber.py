"""
Audio transcription module using Faster Whisper.

This module provides functionality for transcribing audio from video files using
the Faster Whisper implementation of OpenAI's Whisper model. It supports GPU
acceleration when available and includes robust error handling and logging.
"""

import logging
import os
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import List, Optional, Tuple, Union

import ffmpeg
import torch
from faster_whisper import WhisperModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranscriptionError(Exception):
    """Base exception for transcription-related errors."""
    pass

class AudioExtractionError(TranscriptionError):
    """Raised when audio extraction from video fails."""
    pass

class ModelLoadError(TranscriptionError):
    """Raised when the Whisper model fails to load."""
    pass

class TranscriptionStatus(Enum):
    """Status of the transcription process."""
    INITIALIZING = auto()
    EXTRACTING_AUDIO = auto()
    TRANSCRIBING = auto()
    COMPLETED = auto()
    FAILED = auto()

@dataclass
class TranscriptionResult:
    """Container for transcription results."""
    text: str
    segments: List[dict]
    language: str
    duration: float

class AudioTranscriber:
    """Handles audio transcription using Faster Whisper."""

    SUPPORTED_AUDIO_FORMATS = {'.wav', '.mp3', '.m4a', '.flac'}
    SUPPORTED_VIDEO_FORMATS = {'.mp4', '.mkv', '.avi', '.mov', '.webm'}
    VALID_MODEL_SIZES = {'tiny', 'base', 'small', 'medium', 'large-v1', 'large-v2', 'large-v3'}
    
    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "default",
        num_workers: int = 1,
        cache_dir: Optional[str] = None,
        download_root: Optional[str] = None
    ):
        """
        Initialize the transcriber with the specified model and device.

        Args:
            model_size: Size of the Whisper model to use
            device: Device to run the model on ('auto', 'cuda', or 'cpu')
            compute_type: Model computation type ('default', 'auto', 'int8', 'int8_float16', etc.)
            num_workers: Number of worker threads for CPU operations
            cache_dir: Directory for caching model files
            download_root: Directory for downloading model files

        Raises:
            ModelLoadError: If the model fails to load
            ValueError: If invalid parameters are provided
        """
        if model_size not in self.VALID_MODEL_SIZES:
            raise ValueError(f"Invalid model size. Must be one of: {self.VALID_MODEL_SIZES}")

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            device = "cpu"

        try:
            logger.info(f"Loading Whisper model '{model_size}' on {device}")
            self.model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
                cpu_threads=num_workers,
                download_root=download_root,
                local_files_only=False  # Allow downloading if not in cache
            )
            self.device = device
            self._status = TranscriptionStatus.INITIALIZING
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {str(e)}")
            raise ModelLoadError(f"Failed to load Whisper model: {str(e)}")

    @property
    def status(self) -> TranscriptionStatus:
        """Get the current status of the transcriber."""
        return self._status

    def _extract_audio(self, input_path: Union[str, Path], output_path: Union[str, Path]) -> None:
        """
        Extract audio from a video file.

        Args:
            input_path: Path to the input video file
            output_path: Path where the extracted audio will be saved

        Raises:
            AudioExtractionError: If audio extraction fails
        """
        try:
            self._status = TranscriptionStatus.EXTRACTING_AUDIO
            logger.info(f"Extracting audio from {input_path}")
            
            stream = ffmpeg.input(str(input_path))
            stream = ffmpeg.output(stream, str(output_path), acodec='pcm_s16le', ac=1, ar='16k')
            ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
            
            logger.info("Audio extraction completed")
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg error during audio extraction: {e.stderr.decode()}")
            raise AudioExtractionError(f"Failed to extract audio: {e.stderr.decode()}")
        except Exception as e:
            logger.error(f"Unexpected error during audio extraction: {str(e)}")
            raise AudioExtractionError(f"Failed to extract audio: {str(e)}")

    def transcribe(
        self,
        media_path: Union[str, Path],
        language: Optional[str] = None,
        task: str = "transcribe",
        **kwargs
    ) -> TranscriptionResult:
        """
        Transcribe audio from a media file.

        Args:
            media_path: Path to the media file (audio or video)
            language: Language code (e.g., 'en', 'es', None for auto-detection)
            task: Task to perform ('transcribe' or 'translate')
            **kwargs: Additional arguments passed to the Whisper model

        Returns:
            TranscriptionResult containing the transcription text and metadata

        Raises:
            TranscriptionError: If transcription fails
            ValueError: If the input file format is not supported
        """
        media_path = Path(media_path)
        if not media_path.exists():
            raise ValueError(f"File not found: {media_path}")

        # Determine if we need to extract audio
        needs_extraction = media_path.suffix.lower() in self.SUPPORTED_VIDEO_FORMATS
        audio_path = media_path
        temp_audio = None
        
        try:
            if needs_extraction:
                temp_audio = media_path.with_suffix('.wav')
                self._extract_audio(media_path, temp_audio)
                audio_path = temp_audio

            self._status = TranscriptionStatus.TRANSCRIBING
            logger.info(f"Starting transcription of {audio_path}")
            
            segments, info = self.model.transcribe(
                str(audio_path),
                language=language,
                task=task,
                **kwargs
            )

            # Convert segments to list for iteration
            segments_list = list(segments)

            # Collect results
            text_parts = []
            segment_info = []
            for segment in segments_list:
                text_parts.append(segment.text)
                segment_info.append({
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text,
                    'words': [{'text': w.word, 'start': w.start, 'end': w.end, 'probability': w.probability} 
                             for w in segment.words] if segment.words else []
                })

            self._status = TranscriptionStatus.COMPLETED
            logger.info(f"Transcription completed: {len(segment_info)} segments, {info.duration:.2f} seconds")

            return TranscriptionResult(
                text=' '.join(text_parts),
                segments=segment_info,
                language=info.language,
                duration=info.duration
            )

        except Exception as e:
            self._status = TranscriptionStatus.FAILED
            logger.error(f"Transcription failed: {str(e)}")
            raise TranscriptionError(f"Failed to transcribe audio: {str(e)}")

        finally:
            # Clean up temporary audio file if we created one
            if temp_audio and temp_audio.exists():
                try:
                    temp_audio.unlink()
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary audio file: {str(e)}")

    def format_transcript(self, result: Union[TranscriptionResult, list], format_type: str = 'txt') -> str:
        """
        Format transcription segments into the specified format.
        
        Args:
            result: TranscriptionResult or list of segments from faster-whisper
            format_type: Output format ('txt', 'srt', or 'vtt')
            
        Returns:
            str: Formatted transcript
            
        Raises:
            ValueError: If format_type is invalid
        """
        if format_type not in ['txt', 'srt', 'vtt']:
            raise ValueError(f"Invalid format type: {format_type}")

        # Get segments from result if it's a TranscriptionResult
        segments = result.segments if isinstance(result, TranscriptionResult) else result
            
        if format_type == 'txt':
            if not segments:
                return ""
            return " ".join(segment['text'].strip() if isinstance(segment, dict) else segment.text.strip() 
                          for segment in segments)
            
        # Initialize output for VTT
        output = []
        if format_type == 'vtt':
            if not segments:
                return "WEBVTT\n"
            output.append("WEBVTT")
            output.append("")  # Required blank line after header
            
        def format_timestamp(seconds: float, format_type: str) -> str:
            """Convert seconds to timestamp format."""
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            msecs = int((seconds % 1) * 1000)
            
            if format_type == 'srt':
                return f"{hours:02d}:{minutes:02d}:{secs:02d},{msecs:03d}"
            else:  # vtt
                return f"{hours:02d}:{minutes:02d}:{secs:02d}.{msecs:03d}"
        
        # Add segments
        for i, segment in enumerate(segments, 1):
            # Get segment data based on type
            start = segment['start'] if isinstance(segment, dict) else segment.start
            end = segment['end'] if isinstance(segment, dict) else segment.end
            text = segment['text'] if isinstance(segment, dict) else segment.text
            
            if format_type == 'srt':
                # Add segment number
                output.append(str(i))
                
                # Add timestamps
                start_ts = format_timestamp(start, format_type)
                end_ts = format_timestamp(end, format_type)
                output.append(f"{start_ts} --> {end_ts}")
                
                # Add text and blank line
                output.append(text.strip())
                output.append("")
                
            else:  # vtt
                # Add timestamps
                start_ts = format_timestamp(start, format_type)
                end_ts = format_timestamp(end, format_type)
                output.append(f"{start_ts} --> {end_ts}")
                
                # Add text and blank line
                output.append(text.strip())
                output.append("")
        
        # Ensure there's a final newline
        output.append("")
        return "\n".join(output)

    def get_transcript_stats(self, result: Union[TranscriptionResult, list], audio_duration: float) -> dict:
        """
        Calculate statistics for the transcription.
        
        Args:
            result: TranscriptionResult or list of transcription segments
            audio_duration: Duration of the audio in seconds
            
        Returns:
            dict: Statistics including word count, segment count, etc.
        """
        # Get segments from result if it's a TranscriptionResult
        segments = result.segments if isinstance(result, TranscriptionResult) else result
        
        if not segments:
            return {
                "segment_count": 0,
                "word_count": 0,
                "duration": audio_duration,
                "words_per_minute": 0.0,
                "average_segment_duration": 0.0,
                "average_words_per_segment": 0.0
            }

        # Calculate statistics
        word_count = sum(len(segment['text'].split()) if isinstance(segment, dict) else len(segment.text.split())
                        for segment in segments)
        segment_count = len(segments)
        minutes = audio_duration / 60.0
        words_per_minute = word_count / minutes if minutes > 0 else 0.0
        avg_segment_duration = audio_duration / segment_count if segment_count > 0 else 0.0
        avg_words_per_segment = word_count / segment_count if segment_count > 0 else 0.0

        return {
            "segment_count": segment_count,
            "word_count": word_count,
            "duration": audio_duration,
            "words_per_minute": round(words_per_minute, 2),
            "average_segment_duration": round(avg_segment_duration, 2),
            "average_words_per_segment": round(avg_words_per_segment, 2)
        }

    def get_available_device(self) -> Tuple[str, Optional[str]]:
        """
        Get information about the currently available device.

        Returns:
            Tuple of (device_type, device_name or None)
        """
        if self.device == "cuda":
            return "cuda", torch.cuda.get_device_name(0)
        return "cpu", None
