"""Test script for the AudioTranscriber class."""

import logging
from pathlib import Path
from transcriber import AudioTranscriber

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Initialize transcriber with CUDA support
    transcriber = AudioTranscriber(
        model_size="tiny",  # Use tiny model for quick testing
        device="cuda",
        compute_type="default"
    )

    # Log device information
    device_type, device_name = transcriber.get_available_device()
    logger.info(f"Using device: {device_type} ({device_name if device_name else 'N/A'})")

    # Test transcription with a sample video/audio file
    # You'll need to provide a test file path
    test_file = Path("path/to/your/test/file.mp4")  # Update this path
    
    if not test_file.exists():
        logger.error(f"Test file not found: {test_file}")
        return

    try:
        result = transcriber.transcribe(test_file)
        logger.info(f"Transcription successful!")
        logger.info(f"Detected language: {result.language}")
        logger.info(f"Duration: {result.duration:.2f} seconds")
        logger.info(f"Number of segments: {len(result.segments)}")
        logger.info("\nTranscription text:")
        logger.info("-" * 40)
        logger.info(result.text)
        logger.info("-" * 40)
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")

if __name__ == "__main__":
    main()
