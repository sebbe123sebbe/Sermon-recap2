"""Test suite for the VideoProcessor class."""

import unittest
import logging
from pathlib import Path
import numpy as np
import ffmpeg
from video_processor import VideoProcessor, VideoProcessingError, VideoMetadata

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestVideoProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment once before all tests."""
        cls.test_dir = Path(__file__).parent / 'test_files'
        cls.test_dir.mkdir(exist_ok=True)
        cls.processor = VideoProcessor()
        
        # Create a test video file
        cls.test_video = cls.test_dir / 'test_video.mp4'
        cls._create_test_video(cls.test_video)
        
        # Create paths for output files
        cls.test_audio = cls.test_dir / 'test_audio.wav'
        cls.test_trim = cls.test_dir / 'test_trim.mp4'
        cls.test_preview = cls.test_dir / 'test_preview.mp4'
        cls.test_concat = cls.test_dir / 'test_concat.mp4'

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        # Clean up test files
        for file in [cls.test_video, cls.test_audio, cls.test_trim, 
                    cls.test_preview, cls.test_concat]:
            if file.exists():
                file.unlink()

    @classmethod
    def _create_test_video(cls, output_path: Path, duration: int = 5):
        """Create a test video file with color bars."""
        try:
            # Create a video with color bars
            stream = ffmpeg.input(f'testsrc=duration={duration}:size=640x480:rate=30', f='lavfi')
            
            # Add a test tone
            audio = ffmpeg.input(f'sine=frequency=440:duration={duration}', f='lavfi')
            
            # Combine video and audio
            output_path.parent.mkdir(exist_ok=True)
            stream = ffmpeg.output(
                stream,
                audio,
                str(output_path),
                acodec='aac',
                vcodec='libx264',
                pix_fmt='yuv420p',  # Required for compatibility
                video_bitrate='1M',
                audio_bitrate='128k'
            )
            
            stream.overwrite_output().run(capture_stdout=True, capture_stderr=True)
            logger.info(f"Created test video: {output_path}")
            
        except ffmpeg.Error as e:
            logger.error(f"Failed to create test video: {e}")
            raise

    def test_01_get_video_metadata(self):
        """Test video metadata extraction."""
        metadata = self.processor.get_video_metadata(self.test_video)
        
        self.assertIsInstance(metadata, VideoMetadata)
        self.assertEqual(metadata.width, 640)
        self.assertEqual(metadata.height, 480)
        self.assertGreater(metadata.duration, 0)
        self.assertGreater(metadata.fps, 0)
        self.assertIsNotNone(metadata.codec)
        self.assertGreater(metadata.size_bytes, 0)
        self.assertIsNotNone(metadata.audio_codec)
        self.assertEqual(metadata.audio_channels, 1)  # Mono test tone
        self.assertGreater(metadata.audio_sample_rate, 0)

    def test_02_extract_audio(self):
        """Test audio extraction."""
        # Extract full audio
        result = self.processor.extract_audio(self.test_video, self.test_audio)
        self.assertTrue(result)
        self.assertTrue(self.test_audio.exists())
        
        # Verify audio properties
        probe = ffmpeg.probe(str(self.test_audio))
        audio_info = next(s for s in probe['streams'] if s['codec_type'] == 'audio')
        self.assertEqual(int(audio_info['sample_rate']), 16000)
        self.assertEqual(int(audio_info['channels']), 1)

    def test_03_trim_video(self):
        """Test video trimming."""
        # Trim middle section
        start_time = 1.0
        end_time = 3.0
        
        result = self.processor.trim_video(self.test_video, self.test_trim, start_time, end_time)
        self.assertTrue(result)
        self.assertTrue(self.test_trim.exists())
        
        # Verify duration (allow 0.2s tolerance due to keyframe alignment)
        probe = ffmpeg.probe(str(self.test_trim))
        duration = float(probe['format']['duration'])
        expected_duration = end_time - start_time
        self.assertLess(abs(duration - expected_duration), 0.2,
                       f"Duration {duration:.1f}s differs from expected {expected_duration:.1f}s by more than 0.2s")

    def test_04_preview_trim(self):
        """Test preview generation."""
        result = self.processor.preview_trim(
            self.test_video,
            self.test_preview,
            start_time=1.0,
            end_time=4.0,
            duration=2.0
        )
        self.assertTrue(result)
        self.assertTrue(self.test_preview.exists())
        
        # Verify preview duration
        probe = ffmpeg.probe(str(self.test_preview))
        duration = float(probe['format']['duration'])
        self.assertAlmostEqual(duration, 2.0, places=1)

    def test_05_concatenate_videos(self):
        """Test video concatenation."""
        # Create a second test video
        test_video2 = self.test_dir / 'test_video2.mp4'
        self._create_test_video(test_video2, duration=3)
        
        # Concatenate videos
        result = self.processor.concatenate_videos(
            [self.test_video, test_video2],
            self.test_concat
        )
        self.assertTrue(result)
        self.assertTrue(self.test_concat.exists())
        
        # Verify duration (should be sum of input durations)
        probe = ffmpeg.probe(str(self.test_concat))
        duration = float(probe['format']['duration'])
        self.assertGreater(duration, 7.0)  # At least 5s + 3s
        
        # Clean up second test video
        test_video2.unlink()

    def test_06_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test nonexistent file
        with self.assertRaises(ValueError):
            self.processor.get_video_metadata('nonexistent.mp4')
        
        # Test invalid time range
        with self.assertRaises(ValueError):
            self.processor.trim_video(self.test_video, self.test_trim, 3.0, 1.0)
        
        # Test empty segment list
        with self.assertRaises(ValueError):
            self.processor.concatenate_videos([], self.test_concat)

def run_tests():
    """Run the test suite."""
    unittest.main(argv=[''], verbosity=2)

if __name__ == "__main__":
    run_tests()
