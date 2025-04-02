"""
Main controller for the Video Summarizer Pro application.
Orchestrates the video processing, transcription, and AI summarization pipeline.
"""

import asyncio
import json
import logging
import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from transcriber import AudioTranscriber
from video_processor import VideoProcessor
from ai_client import OpenRouterClient
from settings_manager import SettingsManager

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class PipelineState:
    """Data class to store pipeline state and results."""
    
    video_path: Optional[str] = None
    trimmed_video_path: Optional[str] = None
    audio_path: Optional[str] = None
    transcript_txt: Optional[str] = None
    transcript_srt: Optional[str] = None
    transcript_vtt: Optional[str] = None
    transcript_stats: Optional[Dict[str, Any]] = None
    summary: Optional[str] = None
    study_guide: Optional[str] = None
    recap_video_path: Optional[str] = None
    temp_files: List[str] = None
    
    def __post_init__(self):
        """Initialize empty lists."""
        if self.temp_files is None:
            self.temp_files = []
    
    def add_temp_file(self, file_path: str):
        """Add a temporary file to track for cleanup."""
        self.temp_files.append(file_path)
    
    def cleanup(self):
        """Clean up temporary files."""
        for file_path in self.temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"Cleaned up temporary file: {file_path}")
            except Exception as e:
                logger.error(f"Failed to clean up {file_path}: {str(e)}")
        self.temp_files.clear()

class MainController:
    """Main controller class for orchestrating the video processing pipeline."""
    
    def __init__(self):
        """Initialize the controller with required components."""
        self.settings_manager = SettingsManager()
        self.settings = self.settings_manager.load_settings()
        
        # Initialize components
        self.transcriber = AudioTranscriber(
            model_size=self.settings.get("transcriber", {}).get("model_size", "base"),
            device=self.settings.get("transcriber", {}).get("device", "auto"),
            compute_type=self.settings.get("transcriber", {}).get("compute_type", "float16")
        )
        
        self.video_processor = VideoProcessor()
        
        # Initialize AI client if API key is available
        api_key = self.settings.get("ai", {}).get("api_key")
        self.ai_client = OpenRouterClient(api_key) if api_key else None
        
        # Initialize state
        self.state = PipelineState()
        self._pipeline_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._current_task = None
    
    def _run_in_thread(self, func: Callable, *args, **kwargs) -> Any:
        """Run a function in a background thread."""
        if self._current_task and not self._current_task.done():
            raise RuntimeError("A task is already running")
        
        self._current_task = self._executor.submit(func, *args, **kwargs)
        return self._current_task
    
    async def _run_async_in_thread(self, func: Callable, *args, **kwargs) -> Any:
        """Run an async function in a background thread."""
        def wrapper():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(func(*args, **kwargs))
            finally:
                loop.close()
        
        return await asyncio.get_event_loop().run_in_executor(self._executor, wrapper)
    
    async def run_full_pipeline(
        self,
        config: Dict[str, Any],
        status_callback: Callable[[str], None],
        progress_callback: Callable[[float], None],
        completion_callback: Callable[[bool, Dict[str, Any]], None]
    ) -> None:
        """
        Run the full video processing pipeline in a background thread.
        
        Args:
            config: Pipeline configuration
            status_callback: Callback for status updates
            progress_callback: Callback for progress updates
            completion_callback: Callback for pipeline completion
        """
        async def pipeline_thread():
            try:
                with self._pipeline_lock:
                    # Reset state
                    self.state = PipelineState()
                    self.state.video_path = config["video_path"]
                    
                    # Create output directory
                    output_dir = config.get("output_dir", os.path.dirname(self.state.video_path))
                    os.makedirs(output_dir, exist_ok=True)
                    base_name = os.path.splitext(os.path.basename(self.state.video_path))[0]
                    
                    # Step 1: Trim video if needed
                    if config.get("trim_video", False):
                        status_callback("Trimming video...")
                        progress_callback(0.1)
                        
                        self.state.trimmed_video_path = self.video_processor.trim_video(
                            self.state.video_path,
                            config["start_time"],
                            config["end_time"]
                        )
                        self.state.add_temp_file(self.state.trimmed_video_path)
                    else:
                        self.state.trimmed_video_path = self.state.video_path
                    
                    # Step 2: Extract audio
                    status_callback("Extracting audio...")
                    progress_callback(0.2)
                    
                    # Generate audio output path
                    audio_output = os.path.join(output_dir, f"{base_name}_audio.wav")
                    
                    # Create output directory if it doesn't exist
                    os.makedirs(os.path.dirname(audio_output), exist_ok=True)
                    
                    self.state.audio_path = self.video_processor.extract_audio(
                        self.state.trimmed_video_path,
                        audio_output
                    )
                    self.state.add_temp_file(self.state.audio_path)
                    
                    # Step 3: Transcribe audio
                    status_callback("Transcribing audio...")
                    progress_callback(0.3)
                    
                    segments = self.transcriber.transcribe(self.state.audio_path)
                    audio_duration = self.video_processor.get_duration(self.state.audio_path)
                    
                    # Step 4: Format transcripts
                    status_callback("Formatting transcripts...")
                    progress_callback(0.4)
                    
                    # Save transcripts
                    self.state.transcript_txt = self.transcriber.format_transcript(segments, "txt")
                    self.state.transcript_srt = self.transcriber.format_transcript(segments, "srt")
                    self.state.transcript_vtt = self.transcriber.format_transcript(segments, "vtt")
                    
                    # Save transcript files
                    for ext, content in [
                        ("txt", self.state.transcript_txt),
                        ("srt", self.state.transcript_srt),
                        ("vtt", self.state.transcript_vtt)
                    ]:
                        output_path = os.path.join(output_dir, f"{base_name}.{ext}")
                        with open(output_path, "w", encoding="utf-8") as f:
                            f.write(content)
                    
                    # Get transcript stats
                    self.state.transcript_stats = self.transcriber.get_transcript_stats(
                        segments,
                        audio_duration
                    )
                    
                    # Step 5: Generate summary if AI client is available
                    if self.ai_client and config.get("generate_summary", True):
                        status_callback("Generating summary...")
                        progress_callback(0.6)
                        
                        self.state.summary = await self.ai_client.summarize(
                            self.state.transcript_txt,
                            config.get("ai", {}).get("model", "anthropic/claude-2"),
                            length=config.get("ai", {}).get("summary_length", "medium"),
                            include_timestamps=config.get("ai", {}).get("include_timestamps", False)
                        )
                    
                    # Step 6: Generate study guide if AI client is available
                    if self.ai_client and config.get("generate_study_guide", True):
                        status_callback("Generating study guide...")
                        progress_callback(0.8)
                        
                        self.state.study_guide = await self.ai_client.generate_study_guide(
                            self.state.transcript_txt,
                            config.get("ai", {}).get("model", "anthropic/claude-2"),
                            guide_type=config.get("ai", {}).get("guide_type", "outline"),
                            difficulty=config.get("ai", {}).get("difficulty", "intermediate")
                        )
                    
                    # Step 7: Create recap video if requested
                    if config.get("create_recap", False):
                        status_callback("Creating recap video...")
                        progress_callback(0.9)
                        
                        recap_output = os.path.join(output_dir, f"{base_name}_recap.mp4")
                        self.video_processor.create_recap_video(
                            self.state.trimmed_video_path,
                            os.path.join(output_dir, f"{base_name}.srt"),
                            recap_output,
                            burn_subtitles=config.get("burn_subtitles", False)
                        )
                        self.state.recap_video_path = recap_output
                    
                    # Step 8: Prepare results
                    results = {
                        "transcripts": {
                            "txt": self.state.transcript_txt,
                            "srt": self.state.transcript_srt,
                            "vtt": self.state.transcript_vtt
                        },
                        "stats": self.state.transcript_stats,
                        "summary": self.state.summary,
                        "study_guide": self.state.study_guide,
                        "recap_video": self.state.recap_video_path
                    }
                    
                    # Step 9: Clean up temporary files
                    status_callback("Cleaning up...")
                    progress_callback(1.0)
                    self.state.cleanup()
                    
                    completion_callback(True, results)
                    
            except Exception as e:
                logger.error(f"Pipeline error: {str(e)}")
                self.state.cleanup()
                completion_callback(False, {"error": str(e)})
        
        await pipeline_thread()
    
    async def regenerate_summary(
        self,
        model: str,
        length: str = "medium",
        include_timestamps: bool = False,
        custom_prompt: Optional[str] = None,
        status_callback: Optional[Callable[[str], None]] = None
    ) -> Optional[str]:
        """
        Regenerate the summary with different parameters.
        
        Args:
            model: Model to use for summarization
            length: Desired summary length
            include_timestamps: Whether to include timestamps
            custom_prompt: Optional custom prompt
            status_callback: Optional callback for status updates
            
        Returns:
            str: New summary or None if failed
        """
        if not self.ai_client or not self.state.transcript_txt:
            return None
        
        if status_callback:
            status_callback("Regenerating summary...")
        
        try:
            summary = await self.ai_client.summarize(
                self.state.transcript_txt,
                model,
                custom_prompt=custom_prompt,
                length=length,
                include_timestamps=include_timestamps
            )
            
            self.state.summary = summary
            return summary
        except Exception as e:
            logger.error(f"Failed to regenerate summary: {str(e)}")
            return None
    
    async def regenerate_study_guide(
        self,
        model: str,
        guide_type: str = "outline",
        difficulty: str = "intermediate",
        custom_prompt: Optional[str] = None,
        status_callback: Optional[Callable[[str], None]] = None
    ) -> Optional[str]:
        """
        Regenerate the study guide with different parameters.
        
        Args:
            model: Model to use for generation
            guide_type: Type of study guide
            difficulty: Difficulty level
            custom_prompt: Optional custom prompt
            status_callback: Optional callback for status updates
            
        Returns:
            str: New study guide or None if failed
        """
        if not self.ai_client or not self.state.transcript_txt:
            return None
        
        if status_callback:
            status_callback("Regenerating study guide...")
        
        try:
            guide = await self.ai_client.generate_study_guide(
                self.state.transcript_txt,
                model,
                custom_prompt=custom_prompt,
                guide_type=guide_type,
                difficulty=difficulty
            )
            
            self.state.study_guide = guide
            return guide
        except Exception as e:
            logger.error(f"Failed to regenerate study guide: {str(e)}")
            return None
    
    def get_settings(self) -> Dict[str, Any]:
        """Get current settings."""
        return self.settings
    
    def save_settings(self, settings: Dict[str, Any]) -> None:
        """
        Save new settings.
        
        Args:
            settings: New settings to save
        """
        self.settings = settings
        self.settings_manager.save_settings(settings)
    
    def manage_llm_models(self, action: str, model_name: Optional[str] = None) -> List[str]:
        """
        Manage LLM models (list/add/remove).
        
        Args:
            action: Action to perform ('list', 'add', 'remove')
            model_name: Name of model to add/remove
            
        Returns:
            list: List of available models
        """
        if not self.ai_client:
            return []
        
        return self.ai_client.manage_models(action, model_name)
    
    def get_transcript_data(self) -> Optional[Tuple[str, str, str]]:
        """
        Get transcript data in all formats.
        
        Returns:
            tuple: (txt, srt, vtt) transcripts or None if not available
        """
        if not self.state.transcript_txt:
            return None
        return (self.state.transcript_txt, self.state.transcript_srt, self.state.transcript_vtt)
    
    def get_summary_data(self) -> Optional[str]:
        """
        Get summary data.
        
        Returns:
            str: Summary or None if not available
        """
        return self.state.summary
    
    def get_study_guide_data(self) -> Optional[str]:
        """
        Get study guide data.
        
        Returns:
            str: Study guide or None if not available
        """
        return self.state.study_guide
    
    def get_video_metadata(self, video_path: str) -> Optional[Dict[str, Any]]:
        """
        Get video metadata.
        
        Args:
            video_path: Path to video file
            
        Returns:
            dict: Video metadata or None if failed
        """
        try:
            return self.video_processor.get_metadata(video_path)
        except Exception as e:
            logger.error(f"Failed to get video metadata: {str(e)}")
            return None
    
    async def run_preview_trim(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        status_callback: Callable[[str], None],
        completion_callback: Callable[[bool, Optional[str]], None]
    ) -> None:
        """
        Run video trim preview in background.
        
        Args:
            video_path: Path to video file
            start_time: Start time in seconds
            end_time: End time in seconds
            status_callback: Callback for status updates
            completion_callback: Callback for completion
        """
        async def preview_thread():
            try:
                with self._pipeline_lock:
                    status_callback("Generating preview...")
                    
                    # Create temporary output file
                    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
                        preview_path = temp_file.name
                    
                    # Generate preview
                    self.video_processor.trim_video(
                        video_path,
                        start_time,
                        end_time,
                        preview_path
                    )
                    
                    completion_callback(True, preview_path)
            except Exception as e:
                logger.error(f"Preview generation failed: {str(e)}")
                completion_callback(False, None)
        
        await preview_thread()
