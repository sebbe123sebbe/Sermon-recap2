"""Video processing module for Video Summarizer Pro."""

import logging
from pathlib import Path
from typing import Optional, Dict, List, Union
import ffmpeg
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class VideoMetadata:
    """Container for video metadata."""
    width: int
    height: int
    duration: float
    fps: float
    codec: str
    size_bytes: int
    audio_codec: Optional[str] = None
    audio_channels: Optional[int] = None
    audio_sample_rate: Optional[int] = None

class VideoProcessingError(Exception):
    """Base exception for video processing errors."""
    pass

class VideoProcessor:
    """Handles video processing operations using ffmpeg."""
    
    def __init__(self):
        """Initialize the video processor."""
        try:
            # Test ffmpeg availability using a simple command
            stream = ffmpeg.input('nullsrc', f='lavfi', t=0.1)
            stream.output('pipe:', format='null').run(capture_stdout=True, capture_stderr=True)
            logger.info("FFmpeg is available")
        except ffmpeg.Error as e:
            logger.error("FFmpeg is not available: %s", str(e))
            raise VideoProcessingError("FFmpeg is not available. Please install FFmpeg.") from e

    def get_video_metadata(self, video_path: Union[str, Path]) -> VideoMetadata:
        """
        Get metadata for a video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            VideoMetadata object containing video information
            
        Raises:
            VideoProcessingError: If metadata extraction fails
            ValueError: If the file doesn't exist
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise ValueError(f"Video file not found: {video_path}")
            
        try:
            probe = ffmpeg.probe(str(video_path))
            
            # Get video stream info
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            
            # Get audio stream info if available
            audio_info = next((s for s in probe['streams'] if s['codec_type'] == 'audio'), None)
            
            # Calculate FPS
            fps = 0.0
            if 'avg_frame_rate' in video_info:
                try:
                    num, den = map(int, video_info['avg_frame_rate'].split('/'))
                    fps = num / den if den != 0 else 0.0
                except (ValueError, ZeroDivisionError):
                    fps = 0.0
            
            # Get file size
            size_bytes = video_path.stat().st_size
            
            return VideoMetadata(
                width=int(video_info['width']),
                height=int(video_info['height']),
                duration=float(probe['format'].get('duration', 0)),
                fps=fps,
                codec=video_info['codec_name'],
                size_bytes=size_bytes,
                audio_codec=audio_info['codec_name'] if audio_info else None,
                audio_channels=int(audio_info['channels']) if audio_info else None,
                audio_sample_rate=int(audio_info['sample_rate']) if audio_info else None
            )
            
        except ffmpeg.Error as e:
            logger.error("Failed to get video metadata: %s", str(e))
            raise VideoProcessingError(f"Failed to get video metadata: {str(e)}") from e

    def get_duration(self, media_path: Union[str, Path]) -> float:
        """
        Get the duration of a media file in seconds.
        
        Args:
            media_path: Path to the media file
            
        Returns:
            float: Duration in seconds
            
        Raises:
            VideoProcessingError: If duration extraction fails
            ValueError: If the file doesn't exist
        """
        media_path = Path(media_path)
        if not media_path.exists():
            raise ValueError(f"Media file not found: {media_path}")
            
        try:
            probe = ffmpeg.probe(str(media_path))
            return float(probe['format']['duration'])
        except ffmpeg.Error as e:
            logger.error("Failed to get media duration: %s", str(e))
            raise VideoProcessingError(f"Failed to get media duration: {str(e)}") from e

    def extract_audio(
        self,
        video_path: Union[str, Path],
        output_audio_path: Union[str, Path],
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> str:
        """
        Extract audio from a video file.
        
        Args:
            video_path: Path to the input video
            output_audio_path: Path for the output audio
            start_time: Optional start time in seconds
            end_time: Optional end time in seconds
            
        Returns:
            str: Path to the extracted audio file
            
        Raises:
            VideoProcessingError: If audio extraction fails
            ValueError: If the input file doesn't exist
        """
        video_path = Path(video_path)
        output_audio_path = Path(output_audio_path)
        
        if not video_path.exists():
            raise ValueError(f"Video file not found: {video_path}")
            
        try:
            # Start with input
            stream = ffmpeg.input(str(video_path))
            
            # Add trim if specified
            if start_time is not None or end_time is not None:
                stream = stream.filter('atrim',
                                    start=start_time if start_time else 0,
                                    end=end_time if end_time else None)
            
            # Set up audio conversion
            stream = stream.audio.filter('aformat',
                                      sample_fmts='s16',
                                      sample_rates=16000,
                                      channel_layouts='mono')
            
            # Run the conversion
            output_audio_path.parent.mkdir(parents=True, exist_ok=True)
            stream.output(str(output_audio_path),
                        acodec='pcm_s16le',
                        ac=1,
                        ar=16000).overwrite_output().run(capture_stdout=True, capture_stderr=True)
            
            return str(output_audio_path)
            
        except ffmpeg.Error as e:
            logger.error("Failed to extract audio: %s", str(e))
            raise VideoProcessingError(f"Failed to extract audio: {str(e)}") from e

    def trim_video(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        start_time: float,
        end_time: float
    ) -> str:
        """
        Trim a video file.
        
        Args:
            input_path: Path to the input video
            output_path: Path for the output video
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            str: Path to the trimmed video file
            
        Raises:
            VideoProcessingError: If video trimming fails
            ValueError: If the input file doesn't exist or times are invalid
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        if not input_path.exists():
            raise ValueError(f"Input video not found: {input_path}")
            
        if start_time < 0 or end_time <= start_time:
            raise ValueError("Invalid time range")
            
        try:
            # Get input metadata to preserve settings
            probe = ffmpeg.probe(str(input_path))
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            
            stream = ffmpeg.input(str(input_path))
            
            # Trim the video (using seek for efficiency)
            stream = ffmpeg.input(str(input_path), ss=start_time, t=end_time-start_time)
            
            # Set up output with same codec settings
            output_path.parent.mkdir(parents=True, exist_ok=True)
            stream.output(
                str(output_path),
                acodec='copy',  # Copy audio codec
                vcodec='copy'   # Copy video codec
            ).overwrite_output().run(capture_stdout=True, capture_stderr=True)
            
            return str(output_path)
            
        except ffmpeg.Error as e:
            logger.error("Failed to trim video: %s", str(e))
            raise VideoProcessingError(f"Failed to trim video: {str(e)}") from e

    def concatenate_videos(
        self,
        video_segments: List[Union[str, Path]],
        output_path: Union[str, Path]
    ) -> str:
        """
        Concatenate multiple video segments into a single video.
        
        Args:
            video_segments: List of paths to video segments
            output_path: Path for the output video
            
        Returns:
            str: Path to the concatenated video file
            
        Raises:
            VideoProcessingError: If concatenation fails
            ValueError: If input files don't exist
        """
        # Convert paths to Path objects
        video_segments = [Path(p) for p in video_segments]
        output_path = Path(output_path)
        
        # Check all input files exist
        for segment in video_segments:
            if not segment.exists():
                raise ValueError(f"Video segment not found: {segment}")
        
        try:
            # Create concat file
            concat_file = output_path.parent / "concat.txt"
            with open(concat_file, "w") as f:
                for segment in video_segments:
                    f.write(f"file '{segment.absolute()}'\n")
            
            # Run concatenation
            output_path.parent.mkdir(parents=True, exist_ok=True)
            stream = ffmpeg.input(str(concat_file), format='concat', safe=0)
            stream.output(str(output_path), c='copy').overwrite_output().run(capture_stdout=True, capture_stderr=True)
            
            # Clean up concat file
            concat_file.unlink()
            
            return str(output_path)
            
        except ffmpeg.Error as e:
            logger.error("Failed to concatenate videos: %s", str(e))
            if concat_file.exists():
                concat_file.unlink()
            raise VideoProcessingError(f"Failed to concatenate videos: {str(e)}") from e

    def preview_trim(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        start_time: float,
        end_time: float,
        duration: float = 5.0
    ) -> str:
        """
        Create a quick preview of a video trim operation.
        
        Args:
            input_path: Path to the input video
            output_path: Path for the output preview video
            start_time: Start time in seconds
            end_time: End time in seconds
            duration: Target duration for the preview in seconds
            
        Returns:
            str: Path to the preview video file
            
        Raises:
            VideoProcessingError: If preview creation fails
            ValueError: If input file doesn't exist or times are invalid
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        if not input_path.exists():
            raise ValueError(f"Input video not found: {input_path}")
            
        if start_time < 0 or end_time <= start_time:
            raise ValueError("Invalid time range")
            
        try:
            # Calculate preview parameters
            total_duration = end_time - start_time
            if total_duration <= duration:
                # If requested segment is shorter than preview duration, just trim
                return self.trim_video(input_path, output_path, start_time, end_time)
            
            # Otherwise, create a shorter preview
            stream = ffmpeg.input(str(input_path), ss=start_time)
            
            # Set up fast encoding for preview
            output_path.parent.mkdir(parents=True, exist_ok=True)
            stream.output(
                str(output_path),
                t=duration,  # Limit duration
                vcodec='libx264',
                preset='veryfast',
                crf=28,  # Lower quality for preview
                acodec='aac',
                audio_bitrate='128k'
            ).overwrite_output().run(capture_stdout=True, capture_stderr=True)
            
            return str(output_path)
            
        except ffmpeg.Error as e:
            logger.error("Failed to create preview: %s", str(e))
            raise VideoProcessingError(f"Failed to create preview: {str(e)}") from e
