# Step-by-Step Backend Build Guide: Video Summarizer Pro

**Based on:** PRD
**Target GUI Framework:** Python (e.g., PyQt5/PySide6)

This guide outlines the steps to build the backend logic for the Video Summarizer Pro desktop application, ensuring alignment with the specified Product Requirements Document (PRD).

---

## Phase 1: Project Setup & Environment

1.  [x] **Create Project Structure:**
    * [x] Main project folder: `video_summarizer_pro/`
    * [x] Subfolders: `backend/`, `gui/`, `outputs/`, `transcripts/`, `temp/`, `config/`, `logs/`

2.  [x] **Set Up Virtual Environment:**
    * [x] Navigate to `video_summarizer_pro/` in your terminal.
    * [x] Create environment: `python -m venv venv`
    * [x] Activate environment (Windows/macOS/Linux specific).

3.  [x] **Install Core Dependencies:**
    * [x] Create `requirements.txt`.
    * [x] **Crucially: Install PyTorch with CUDA support.** (Follow instructions from pytorch.org).
    * [x] Add other dependencies to `requirements.txt`:
        ```
        # requirements.txt

        # PyTorch (Command from pytorch.org for your CUDA version)
        # torch torchvision torchaudio --index-url ...

        # Whisper & Dependencies
        faster-whisper
        # ctranslate2 should be installed as a dependency

        # Video/Audio Processing (Requires ffmpeg installed on the system)
        ffmpeg-python

        # API Communication
        httpx # Recommended async-capable library

        # Secure API Key Storage (Choose one or implement conditional logic)
        keyring # Cross-platform system keyring access

        # GUI Framework (Example)
        # PyQt5 or PySide6 (Install separately when building GUI)

        # Other utilities (as needed)
        # python-dotenv # If using .env as a fallback for API key
        ```
    * [x] Install requirements: `pip install -r requirements.txt`
    * [x] **Install FFmpeg:** Ensure `ffmpeg` and `ffprobe` are installed and in PATH.

4.  [x] **Verify CUDA Setup (Critical):**
    * [x] Run the provided Python verification script.
    * [x] Debug any setup issues until CUDA is detected or confirm CPU fallback.

5.  [x] **Configuration Management (`backend/settings_manager.py`):**
    * [x] Implement functions to securely save/load OpenRouter API key using `keyring` (with fallbacks). **Avoid plain text.**
    * [x] Implement functions to save/load user preferences (models, language, paths) to JSON/config file.
    * [x] Implement functions to manage the persistent LLM model list (`add_llm_model`, `delete_llm_model`, `get_llm_models`).

6.  [x] **Logging Setup (`backend/logging_config.py` or similar):**
    * [x] Configure Python's `logging` module.
    * [x] Set up a rotating file handler to `logs/app.log`.
    * [x] Define log format (timestamp, level, module).
    * [x] Provide easy access to the configured logger instance.

---

## Phase 2: Implement Core Backend Modules

Create modular Python files in `backend/`. Implement error handling and logging.

1.  [x] **`backend/video_processor.py`:**
    * [x] Implement `get_video_metadata(video_path: str) -> dict | None` using `ffmpeg.probe`.
    * [x] Implement `extract_audio(video_path: str, output_audio_path: str, start_time: float | None = None, end_time: float | None = None) -> bool` using `ffmpeg` (mono, 16kHz, optional trim).
    * [x] Implement `trim_video(input_path: str, output_path: str, start_time: float, end_time: float) -> bool` using `ffmpeg`.
    * [x] Implement `concatenate_videos(video_segments: list[str], output_path: str) -> bool` using `ffmpeg` filter graph/demuxer.
    * [x] Implement `preview_trim(input_path: str, output_path: str, start_time: float, end_time: float, duration: float = 5.0) -> bool` using `ffmpeg` (fast settings).

2.  [x] **`backend/transcriber.py`:**
    * [x] Implement `class AudioTranscriber:`
        * [x] `__init__(self, model_size: str = "base", device: str = "auto", compute_type: str = "float16")`:
            * [x] Load `faster-whisper` model (`WhisperModel`).
            * [x] Handle device detection (`cuda`/`cpu`).
            * [x] Handle `compute_type` selection.
            * [x] Log device/compute type.
            * [x] Handle model download/loading errors.
        * [x] `transcribe(self, audio_path: str, language: str | None = None, **kwargs) -> tuple[list, dict] | None`:
            * [x] Perform transcription using `model.transcribe()`.
            * [x] Handle language parameter.
            * [x] Return `(segments, info)` tuple.
            * [x] Handle transcription errors.
        * [x] `format_transcript(self, segments: list, format_type: str = 'txt') -> str`:
            * [x] Implement formatting logic for `format_type='txt'`.
            * [x] Implement formatting logic for `format_type='srt'`.
            * [x] Implement formatting logic for `format_type='vtt'`.
        * [x] `get_transcript_stats(self, segments: list, audio_duration: float) -> dict`:
            * [x] Calculate word count.
            * [x] Calculate number of segments.
            * [x] Return stats dict.

3.  [x] **`backend/ai_client.py`:**
    * [x] Implement `class OpenRouterClient:`
        * [x] `__init__(self, api_key: str)`:
            * [x] Store API key.
            * [x] Initialize `httpx.AsyncClient`.
        * [x] `async summarize(self, transcript: str, model: str, custom_prompt: str | None = None, length: str = "medium", include_timestamps: bool = False) -> str | None`:
            * [x] Construct summarization prompt based on inputs.
            * [x] Send request asynchronously to OpenRouter API.
            * [x] Handle API responses and errors (4xx, 5xx, rate limits).
            * [x] Parse response for summary text.
            * [x] Return summary string or `None`.
            * [x] Log API interactions.
        * [x] `async generate_study_guide(self, text_input: str, model: str, guide_type: str = "outline", difficulty: str = "intermediate", custom_prompt: str | None = None) -> str | None`:
            * [x] Construct study guide prompt based on inputs.
            * [x] Send request asynchronously.
            * [x] Handle responses/errors.
            * [x] Return guide text string or `None`.
            * [x] Log API interactions.

4.  [x] **`backend/recap_generator.py`:**
    * [x] Implement `create_recap_video(trimmed_video_path: str, srt_path: str, output_path: str, intro_path: str | None = None, outro_path: str | None = None, burn_subtitles: bool = False) -> bool`:
        * [x] Define logic for selecting recap segments (if not whole video, potentially using SRT).
        * [x] Construct list of video segments for concatenation.
        * [x] If `burn_subtitles` is True: Implement `ffmpeg` call with `subtitles` filter using input `srt_path`.
        * [x] If `burn_subtitles` is False: Implement `ffmpeg` call for simple concatenation.
        * [x] Ensure primary output is the video file.
        * [x] Return `True` on success, `False` on error.
        * [x] Log `ffmpeg` commands and outcomes.

5.  [x] **`backend/utils.py`:**
    * [x] Implement `check_cuda() -> bool`.
    * [x] Implement `open_folder(path: str)` with platform-specific logic and error handling.
    * [x] Implement timestamp formatting helpers.
    * [x] Implement safe directory creation function.

---

## Phase 3: Implement Backend Controller & Orchestration

Connect modules, manage workflow, handle background execution.

1.  [x] **`backend/main_controller.py`:**
    * [x] Implement `class MainController:`
        * [x] `__init__(self)`:
            * [x] Initialize `SettingsManager`.
            * [x] Initialize `AudioTranscriber`.
            * [x] Initialize `OpenRouterClient`.
            * [x] Get logger instance.
            * [x] Initialize application state variables.
        * [x] **Core Pipeline Method (Runs in Background):**
            * [x] Implement `run_full_pipeline(self, config: dict, status_callback: callable, progress_callback: callable, completion_callback: callable)`:
                * [x] Launch logic in a separate thread (`threading` or `concurrent.futures`).
                * [x] Define parameters (`config`, callbacks).
                * [x] **Orchestration Logic (inside thread):**
                    1.  [x] Call `video_processor.trim_video`, report status/progress, handle errors, define `trimmed_video_path`.
                    2.  [x] Call `video_processor.extract_audio`, report status/progress, handle errors, define `audio_path`.
                    3.  [x] Call `transcriber.transcribe`, report status/progress, handle errors.
                    4.  [x] Call `transcriber.format_transcript` and save `.txt`, `.srt`, `.vtt` files. Store data, get stats, define `srt_path`.
                    5.  [x] Call `ai_client.summarize` (handle async if needed), report status/progress, handle errors, save summary, store data.
                    6.  [x] Call `ai_client.generate_study_guide`, report status/progress, handle errors, save guide, store data.
                    7.  [x] Call `recap_generator.create_recap_video` using `srt_path`, report status/progress, handle errors.
                    8.  [x] Call `completion_callback` (True/False, results dict).
                    9.  [x] Implement temporary file cleanup.
        * [x] **Regeneration Methods (Run in Background):**
            * [x] Implement `regenerate_summary(...)` with background execution, API call, state update, callback.
            * [x] Implement `regenerate_study_guide(...)` with background execution, API call, state update, callback.
        * [x] **Other Methods:**
            * [x] Implement `get_settings() -> dict`.
            * [x] Implement `save_settings(settings: dict)`.
            * [x] Implement `manage_llm_models(action: str, model_name: str | None = None) -> list`.
            * [x] Implement `get_transcript_data() -> tuple[str, str, str] | None`.
            * [x] Implement `get_summary_data() -> str | None`.
            * [x] Implement `get_study_guide_data() -> str | None`.
            * [x] Implement `get_video_metadata(video_path: str) -> dict | None`.
            * [x] Implement `run_preview_trim(...)` with background execution and callback.

---

## Phase 4: GUI Integration Points (Conceptual)

This section describes *how* the GUI interacts with the `MainController`. No backend implementation tasks here, primarily for GUI developer reference.

1.  Initialization: GUI creates `MainController`.
2.  Load Settings: GUI calls controller methods to populate UI.
3.  Callbacks: GUI defines methods for cross-thread updates (signals/slots).
4.  Trigger Actions: GUI gathers config, calls controller methods (which run in background).
5.  Update UI: Controller calls GUI callbacks/slots.
6.  Handle Completion: GUI completion callback/slot handles results/errors.
7.  File Dialogs: GUI handles dialogs, passes paths to controller.
8.  Display Data: GUI calls controller methods to get data for display.

---

## Phase 5: Testing & Packaging

1.  [x] **Testing:**
    * [x] **Unit Tests (`pytest`):**
        * [x] Write tests for `video_processor` functions (mock `ffmpeg`).
        * [x] Write tests for `transcriber` methods (mock model).
        * [x] Write tests for `ai_client` methods (mock `httpx`).
        * [x] Write tests for `settings_manager`.
        * [x] Write tests for `utils`.
    * [x] **Integration Tests:**
        * [x] Test `MainController.run_full_pipeline` with small sample files.
        * [x] Verify output file creation and basic content.
        * [x] Test background execution and callback mechanisms.
    * [x] **Manual Testing:**
        * [x] Create comprehensive manual test script
        * [ ] Acquire sample test videos
        * [ ] Test full workflow with various inputs
        * [ ] Test error handling messages
        * [ ] Test on different hardware (CUDA vs. CPU)

2.  [x] **Packaging:**
    * [x] **Challenge Assessment:** Acknowledge complexity (Python, ffmpeg, CUDA).
    * [x] **Tool Exploration:** Investigate PyInstaller/cx_Freeze configuration for dependencies (esp. CUDA).
    * [x] **Alternative Exploration:** Consider `conda-pack`.
    * [x] **Documentation:**
        * [x] Write clear manual setup instructions (Python, CUDA, dependencies).
        * [x] Document environment setup (venv, requirements).
        * [x] Document API key configuration.
        * [x] Document usage examples.
        * [x] Document testing procedures.
        * [x] Document project structure.

---

This revised guide provides a detailed plan aligned with PRD. Focus on modularity, clear separation of concerns, robust error handling, logging, and non-blocking background execution for a responsive user experience.
