# Video Summarizer Pro

A powerful video summarization tool that uses AI to transcribe, summarize, and generate study guides from video content. Built with Python, featuring GPU acceleration and modern AI models.

## Features

- 🎥 Video Processing:
  - Supports multiple video formats (MP4, MKV, AVI, MOV, WEBM)
  - Automatic audio extraction
  - Video trimming and concatenation
  - Preview generation

- 🎯 Audio Transcription:
  - GPU-accelerated using Faster Whisper
  - Multiple model sizes (tiny to large-v3)
  - Multiple output formats (TXT, SRT, VTT)
  - Language detection and translation

- 🤖 AI-Powered Summarization:
  - Video content summarization
  - Study guide generation
  - Customizable prompts and parameters

- ⚙️ Advanced Features:
  - Background processing
  - Progress tracking
  - Comprehensive error handling
  - Resource cleanup

## Requirements

### Python Environment
- Python 3.10.9 or later
- Virtual environment recommended

### CUDA Support (Optional, but recommended)
- NVIDIA GPU with CUDA support
- CUDA Toolkit 11.8 or later
- cuDNN compatible with your CUDA version

### System Dependencies
- FFmpeg (for video processing)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd video-summarizer-pro
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Linux/macOS
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your OpenRouter API key:
   ```python
   import keyring
   keyring.set_password("video_summarizer", "openrouter_api_key", "your-api-key")
   ```

## Usage

### Command Line Interface
```python
from backend.main_controller import MainController

# Initialize controller
controller = MainController()

# Configure the pipeline
config = {
    "input_path": "path/to/video.mp4",
    "output_dir": "path/to/output",
    "model_size": "base",  # or "tiny", "small", "medium", "large-v3"
    "device": "cuda",      # or "cpu"
    "language": "en",      # or None for auto-detection
    "summary_length": "medium"  # or "short", "long"
}

# Run the pipeline
controller.run_full_pipeline(
    config,
    status_callback=lambda s: print(f"Status: {s}"),
    progress_callback=lambda p: print(f"Progress: {p}%"),
    completion_callback=lambda r: print(f"Complete: {r}")
)
```

## Development

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest backend/test_transcriber.py -v

# Run with coverage
pytest --cov=backend
```

### Project Structure
```
video-summarizer-pro/
├── backend/
│   ├── __init__.py
│   ├── main_controller.py    # Main orchestration
│   ├── video_processor.py    # Video handling
│   ├── transcriber.py        # Audio transcription
│   ├── ai_client.py         # AI API client
│   ├── settings_manager.py  # Settings management
│   └── utils.py            # Utilities
├── tests/
│   └── ...                 # Test files
├── requirements.txt        # Dependencies
└── README.md
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Faster Whisper](https://github.com/guillaumekln/faster-whisper) - For efficient audio transcription
- [FFmpeg](https://ffmpeg.org/) - For video processing
- [OpenRouter](https://openrouter.ai/) - For AI capabilities
