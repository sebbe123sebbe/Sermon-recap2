"""
Generate a test tone audio file with spoken numbers for testing transcription.

This script uses gTTS (Google Text-to-Speech) to generate an audio file
containing spoken numbers with pauses, which is useful for testing the
transcription functionality.
"""

import os
import time
from pathlib import Path

from gtts import gTTS

def generate_test_audio(output_path: str, duration: int = 30):
    """
    Generate a test audio file with spoken numbers.
    
    Args:
        output_path: Path to save the output MP3 file
        duration: Approximate duration in seconds
    """
    # Create text with numbers and pauses
    numbers = list(range(1, int(duration/2) + 1))  # One number every 2 seconds
    text = ". ".join(str(n) for n in numbers)
    
    # Add some phrases for testing
    text = (
        "This is a test audio file for transcription. "
        "I will now count numbers with pauses. "
        f"{text}. "
        "This concludes the test audio file. Thank you for listening."
    )
    
    # Generate audio
    tts = gTTS(text=text, lang='en', slow=False)
    tts.save(output_path)

def main():
    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parent.parent / "test_files"
    output_dir.mkdir(exist_ok=True)
    
    # Generate test audio
    output_path = output_dir / "test_tone.mp3"
    generate_test_audio(str(output_path))
    print(f"Generated test audio: {output_path}")

if __name__ == "__main__":
    main()
