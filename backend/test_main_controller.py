"""
Test suite for MainController.
"""

import asyncio
import os
import tempfile
from unittest.mock import ANY, AsyncMock, Mock, patch

import pytest
import pytest_asyncio

from main_controller import MainController, PipelineState

class TestMainController:
    """Test cases for MainController."""
    
    @pytest_asyncio.fixture(autouse=True)
    async def setup(self):
        """Set up test fixtures."""
        # Mock settings
        self.mock_settings = {
            "transcriber": {
                "model_size": "base",
                "device": "cpu",
                "compute_type": "float16"
            },
            "ai": {
                "api_key": "test_key",
                "models": ["test/model"]
            }
        }
        
        # Create temporary test files
        self.temp_dir = tempfile.mkdtemp()
        self.video_path = os.path.join(self.temp_dir, "test.mp4")
        self.audio_path = os.path.join(self.temp_dir, "test.wav")
        self.transcript_path = os.path.join(self.temp_dir, "test.txt")
        
        # Create empty files
        for path in [self.video_path, self.audio_path]:
            with open(path, "wb") as f:
                f.write(b"")
        
        with open(self.transcript_path, "w") as f:
            f.write("Test transcript")
        
        # Mock components
        with patch("main_controller.SettingsManager") as mock_settings_manager, \
             patch("main_controller.AudioTranscriber") as mock_transcriber, \
             patch("main_controller.VideoProcessor") as mock_video_processor, \
             patch("main_controller.OpenRouterClient") as mock_ai_client:
            
            # Setup mocks
            mock_settings_manager.return_value.load_settings.return_value = self.mock_settings
            self.mock_settings_manager = mock_settings_manager.return_value
            self.mock_transcriber = mock_transcriber.return_value
            self.mock_video_processor = mock_video_processor.return_value
            self.mock_ai_client = mock_ai_client.return_value
            
            # Create controller
            self.controller = MainController()
            
            yield
        
        # Cleanup
        for path in [self.video_path, self.audio_path, self.transcript_path]:
            try:
                os.remove(path)
            except:
                pass
        try:
            os.rmdir(self.temp_dir)
        except:
            pass
    
    def test_init(self):
        """Test initialization."""
        assert self.controller.settings == self.mock_settings
        assert self.controller.state is not None
        assert isinstance(self.controller.state, PipelineState)
    
    @pytest.mark.asyncio
    async def test_run_full_pipeline_success(self):
        """Test successful pipeline run."""
        # Mock callbacks
        status_cb = Mock()
        progress_cb = Mock()
        completion_cb = Mock()
        
        # Mock component methods
        self.mock_video_processor.trim_video.return_value = self.video_path
        self.mock_video_processor.extract_audio.return_value = self.audio_path
        self.mock_video_processor.get_duration.return_value = 60.0
        
        self.mock_transcriber.transcribe.return_value = [{"text": "Test", "start": 0, "end": 1}]
        self.mock_transcriber.format_transcript.return_value = "Test transcript"
        self.mock_transcriber.get_transcript_stats.return_value = {"duration": 60.0}
        
        self.mock_ai_client.summarize = AsyncMock(return_value="Test summary")
        self.mock_ai_client.generate_study_guide = AsyncMock(return_value="Test guide")
        
        # Run pipeline
        config = {
            "video_path": self.video_path,
            "trim_video": True,
            "start_time": 0,
            "end_time": 60,
            "output_dir": self.temp_dir,
            "generate_summary": True,
            "generate_study_guide": True,
            "create_recap": True,
            "ai": {
                "model": "test/model",
                "summary_length": "medium",
                "guide_type": "outline"
            }
        }
        
        self.controller.run_full_pipeline(config, status_cb, progress_cb, completion_cb)
        
        # Wait for pipeline to complete
        await asyncio.sleep(0.1)
        
        # Verify callbacks
        assert status_cb.called
        assert progress_cb.called
        
        # Check completion callback with any temporary file path
        assert completion_cb.call_count == 1
        args = completion_cb.call_args[0]
        assert args[0] is True  # Success flag
        assert isinstance(args[1], dict)  # Results dict
        assert args[1]["transcripts"]["txt"] == "Test transcript"
        assert args[1]["summary"] == "Test summary"
        assert args[1]["study_guide"] == "Test guide"
    
    @pytest.mark.asyncio
    async def test_run_full_pipeline_error(self):
        """Test pipeline error handling."""
        # Mock callbacks
        status_cb = Mock()
        progress_cb = Mock()
        completion_cb = Mock()
        
        # Mock error
        self.mock_video_processor.trim_video.side_effect = Exception("Test error")
        
        # Run pipeline
        config = {
            "video_path": self.video_path,
            "trim_video": True,
            "start_time": 0,
            "end_time": 60
        }
        
        self.controller.run_full_pipeline(config, status_cb, progress_cb, completion_cb)
        
        # Wait for pipeline to complete
        await asyncio.sleep(0.1)
        
        # Verify error callback
        completion_cb.assert_called_once_with(False, {"error": "Test error"})
    
    @pytest.mark.asyncio
    async def test_regenerate_summary(self):
        """Test summary regeneration."""
        # Setup state
        self.controller.state.transcript_txt = "Test transcript"
        
        # Mock AI client
        self.mock_ai_client.summarize = AsyncMock(return_value="New summary")
        
        # Regenerate summary
        summary = await self.controller.regenerate_summary(
            "test/model",
            length="short",
            include_timestamps=True
        )
        
        assert summary == "New summary"
        assert self.controller.state.summary == "New summary"
    
    @pytest.mark.asyncio
    async def test_regenerate_study_guide(self):
        """Test study guide regeneration."""
        # Setup state
        self.controller.state.transcript_txt = "Test transcript"
        
        # Mock AI client
        self.mock_ai_client.generate_study_guide = AsyncMock(return_value="New guide")
        
        # Regenerate guide
        guide = await self.controller.regenerate_study_guide(
            "test/model",
            guide_type="questions",
            difficulty="advanced"
        )
        
        assert guide == "New guide"
        assert self.controller.state.study_guide == "New guide"
    
    def test_settings_management(self):
        """Test settings management."""
        # Test get settings
        assert self.controller.get_settings() == self.mock_settings
        
        # Test save settings
        new_settings = {
            "transcriber": {
                "model_size": "large",
                "device": "cuda",
                "compute_type": "float32"
            },
            "ai": {
                "api_key": "new_key"
            }
        }
        
        self.controller.save_settings(new_settings)
        self.mock_settings_manager.save_settings.assert_called_once_with(new_settings)
    
    def test_model_management(self):
        """Test LLM model management."""
        # Test list models
        models = self.controller.manage_llm_models("list")
        assert models == ["test/model"]
        
        # Test add model
        models = self.controller.manage_llm_models("add", "new/model")
        assert "new/model" in models
        
        # Test remove model
        models = self.controller.manage_llm_models("remove", "test/model")
        assert "test/model" not in models
    
    def test_data_access(self):
        """Test data access methods."""
        # Setup state
        self.controller.state.transcript_txt = "Test txt"
        self.controller.state.transcript_srt = "Test srt"
        self.controller.state.transcript_vtt = "Test vtt"
        self.controller.state.summary = "Test summary"
        self.controller.state.study_guide = "Test guide"
        
        # Test transcript data
        transcripts = self.controller.get_transcript_data()
        assert transcripts == ("Test txt", "Test srt", "Test vtt")
        
        # Test summary data
        assert self.controller.get_summary_data() == "Test summary"
        
        # Test study guide data
        assert self.controller.get_study_guide_data() == "Test guide"
    
    @pytest.mark.asyncio
    async def test_preview_trim(self):
        """Test video trim preview."""
        # Mock callbacks
        status_cb = Mock()
        completion_cb = Mock()
        
        # Mock video processor
        self.mock_video_processor.trim_video.return_value = ANY
        
        # Run preview
        self.controller.run_preview_trim(
            self.video_path,
            0,
            60,
            status_cb,
            completion_cb
        )
        
        # Wait for preview to complete
        await asyncio.sleep(0.1)
        
        # Verify callbacks
        status_cb.assert_called_once_with("Creating trim preview...")
        assert completion_cb.call_count == 1
        assert completion_cb.call_args[0][0] is True
        assert isinstance(completion_cb.call_args[0][1], str)
        assert completion_cb.call_args[0][1].endswith(".mp4")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
