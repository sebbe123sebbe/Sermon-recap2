"""Generate a test audio file for transcription testing."""

import math
import wave
import struct
import os
from pathlib import Path

def generate_sine_wave(frequency, duration, sample_rate=16000):
    """Generate a sine wave audio signal."""
    num_samples = int(sample_rate * duration)
    samples = []
    
    for i in range(num_samples):
        sample = math.sin(2 * math.pi * frequency * i / sample_rate)
        samples.append(int(sample * 32767))  # Convert to 16-bit integer
    
    return samples

def create_wav_file(filename, samples, sample_rate=16000):
    """Create a WAV file with the given samples."""
    with wave.open(filename, 'w') as wav_file:
        # Set parameters: 1 channel, 2 bytes per sample, sample rate
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 2 bytes per sample
        wav_file.setframerate(sample_rate)
        
        # Convert samples to bytes
        sample_data = struct.pack('<' + 'h' * len(samples), *samples)
        wav_file.writeframes(sample_data)

def main():
    # Create test directory if it doesn't exist
    test_dir = Path(__file__).parent / 'test_files'
    test_dir.mkdir(exist_ok=True)
    
    # Generate a 1-second 440Hz tone (A4 note)
    samples = generate_sine_wave(440, 1.0)
    
    # Create the WAV file
    test_file = test_dir / 'test_tone.wav'
    create_wav_file(str(test_file), samples)
    
    print(f"Created test audio file: {test_file}")

if __name__ == '__main__':
    main()
