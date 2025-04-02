"""Comprehensive test suite for the AudioTranscriber class."""

import logging
import unittest
from pathlib import Path
from transcriber import (
    AudioTranscriber,
    TranscriptionError,
    ModelLoadError,
    AudioExtractionError,
    TranscriptionStatus
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestAudioTranscriber(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment once before all tests."""
        cls.test_dir = Path(__file__).parent / 'test_files'
        cls.test_dir.mkdir(exist_ok=True)
        cls.test_audio = cls.test_dir / 'test_tone.wav'

    def setUp(self):
        """Set up before each test."""
        self.transcriber = None

    def tearDown(self):
        """Clean up after each test."""
        if self.transcriber:
            del self.transcriber

    def test_01_init_with_cuda(self):
        """Test initialization with CUDA support."""
        try:
            self.transcriber = AudioTranscriber(
                model_size="tiny",
                device="cuda",
                compute_type="default"
            )
            device_type, device_name = self.transcriber.get_available_device()
            self.assertEqual(device_type, "cuda")
            self.assertIsNotNone(device_name)
            logger.info(f"Successfully initialized with CUDA: {device_name}")
        except ModelLoadError as e:
            if "CUDA requested but not available" in str(e):
                logger.warning("CUDA not available, skipping CUDA test")
                self.skipTest("CUDA not available")
            else:
                raise

    def test_02_init_with_cpu(self):
        """Test initialization with CPU."""
        self.transcriber = AudioTranscriber(
            model_size="tiny",
            device="cpu",
            compute_type="default"
        )
        device_type, device_name = self.transcriber.get_available_device()
        self.assertEqual(device_type, "cpu")
        logger.info("Successfully initialized with CPU")

    def test_03_invalid_model_size(self):
        """Test initialization with invalid model size."""
        with self.assertRaises(ValueError):
            AudioTranscriber(model_size="invalid_size")

    def test_04_status_tracking(self):
        """Test transcription status tracking."""
        self.transcriber = AudioTranscriber("tiny", device="cpu")
        self.assertEqual(self.transcriber.status, TranscriptionStatus.INITIALIZING)

    def test_05_supported_formats(self):
        """Test supported format validation."""
        self.transcriber = AudioTranscriber("tiny", device="cpu")
        
        # Check video formats
        for fmt in ['.mp4', '.mkv', '.avi', '.mov', '.webm']:
            self.assertIn(fmt, self.transcriber.SUPPORTED_VIDEO_FORMATS)
            
        # Check audio formats
        for fmt in ['.wav', '.mp3', '.m4a', '.flac']:
            self.assertIn(fmt, self.transcriber.SUPPORTED_AUDIO_FORMATS)

    def test_06_transcribe_audio(self):
        """Test audio transcription."""
        if not self.test_audio.exists():
            logger.warning(f"Test audio file not found: {self.test_audio}")
            self.skipTest("Test audio file not found")
            return

        self.transcriber = AudioTranscriber("tiny", device="cpu")
        try:
            result = self.transcriber.transcribe(self.test_audio)
            self.assertIsNotNone(result)
            self.assertIsInstance(result.text, str)
            self.assertIsInstance(result.segments, list)
            self.assertIsInstance(result.language, str)
            self.assertIsInstance(result.duration, float)
            logger.info(f"Successfully transcribed audio. Duration: {result.duration:.2f}s")
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise

    def test_07_nonexistent_file(self):
        """Test handling of nonexistent file."""
        self.transcriber = AudioTranscriber("tiny", device="cpu")
        with self.assertRaises(ValueError):
            self.transcriber.transcribe("nonexistent_file.mp4")

    def test_08_format_transcript_txt(self):
        """Test transcript formatting in TXT format."""
        from dataclasses import dataclass
        
        @dataclass
        class MockSegment:
            text: str
            start: float
            end: float
            avg_logprob: float
        
        segments = [
            MockSegment("Hello", 0.0, 1.0, -0.5),
            MockSegment("world", 1.0, 2.0, -0.3)
        ]
        
        self.transcriber = AudioTranscriber("tiny", device="cpu")
        result = self.transcriber.format_transcript(segments, 'txt')
        self.assertEqual(result, "Hello world")

    def test_09_format_transcript_srt(self):
        """Test transcript formatting in SRT format."""
        from dataclasses import dataclass
        
        @dataclass
        class MockSegment:
            text: str
            start: float
            end: float
            avg_logprob: float
        
        segments = [
            MockSegment("Hello", 0.0, 1.0, -0.5),
            MockSegment("world", 1.0, 2.0, -0.3)
        ]
        
        self.transcriber = AudioTranscriber("tiny", device="cpu")
        result = self.transcriber.format_transcript(segments, 'srt')
        expected = (
            "1\n"
            "00:00:00,000 --> 00:00:01,000\n"
            "Hello\n"
            "\n"
            "2\n"
            "00:00:01,000 --> 00:00:02,000\n"
            "world\n"
            "\n"
        )
        self.assertEqual(result, expected)

    def test_10_format_transcript_vtt(self):
        """Test transcript formatting in VTT format."""
        from dataclasses import dataclass
        
        @dataclass
        class MockSegment:
            text: str
            start: float
            end: float
            avg_logprob: float
        
        segments = [
            MockSegment("Hello", 0.0, 1.0, -0.5),
            MockSegment("world", 1.0, 2.0, -0.3)
        ]
        
        self.transcriber = AudioTranscriber("tiny", device="cpu")
        result = self.transcriber.format_transcript(segments, 'vtt')
        expected = (
            "WEBVTT\n"
            "\n"
            "00:00:00.000 --> 00:00:01.000\n"
            "Hello\n"
            "\n"
            "00:00:01.000 --> 00:00:02.000\n"
            "world\n"
            "\n"
        )
        self.assertEqual(result, expected)

    def test_11_get_transcript_stats(self):
        """Test transcript statistics calculation."""
        from dataclasses import dataclass
        
        @dataclass
        class MockSegment:
            text: str
            start: float
            end: float
            avg_logprob: float
        
        segments = [
            MockSegment("This is a test.", 0.0, 2.0, -0.5),
            MockSegment("Another test segment.", 2.0, 4.0, -0.3)
        ]
        
        self.transcriber = AudioTranscriber("tiny", device="cpu")
        stats = self.transcriber.get_transcript_stats(segments, 4.0)
        
        self.assertEqual(stats["word_count"], 7)  # "This is a test Another test segment"
        self.assertEqual(stats["segment_count"], 2)
        self.assertEqual(stats["duration"], 4.0)
        self.assertEqual(stats["words_per_minute"], 105.0)  # (7 words / 4 seconds) * 60
        self.assertEqual(stats["average_segment_duration"], 2.0)  # 4 seconds / 2 segments
        self.assertAlmostEqual(stats["confidence"], -0.4, places=4)  # average of -0.5 and -0.3

    def test_12_empty_transcript(self):
        """Test handling of empty transcripts."""
        self.transcriber = AudioTranscriber("tiny", device="cpu")
        
        # Test empty format_transcript
        self.assertEqual(self.transcriber.format_transcript([], 'txt'), "")
        self.assertEqual(self.transcriber.format_transcript([], 'srt'), "")
        self.assertEqual(self.transcriber.format_transcript([], 'vtt'), "WEBVTT\n")
        
        # Test empty get_transcript_stats
        stats = self.transcriber.get_transcript_stats([], 5.0)
        self.assertEqual(stats["word_count"], 0)
        self.assertEqual(stats["segment_count"], 0)
        self.assertEqual(stats["duration"], 5.0)
        self.assertEqual(stats["words_per_minute"], 0)
        self.assertEqual(stats["average_segment_duration"], 0)
        self.assertEqual(stats["confidence"], 0)

    def test_13_invalid_format(self):
        """Test handling of invalid format type."""
        self.transcriber = AudioTranscriber("tiny", device="cpu")
        with self.assertRaises(ValueError):
            self.transcriber.format_transcript([], 'invalid_format')

def run_tests():
    """Run the test suite."""
    # First generate test audio
    import generate_test_audio
    generate_test_audio.main()
    
    # Then run tests
    unittest.main(argv=[''], verbosity=2)

if __name__ == "__main__":
    run_tests()
