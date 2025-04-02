# Step-by-Step Backend Build Guide: Video Summarizer Pro

**Based on:** PRD
**Target GUI Framework:** Python (e.g., PyQt5/PySide6)

This guide outlines the steps to build the backend logic for the Video Summarizer Pro desktop application, ensuring alignment with the specified Product Requirements Document (PRD).

---

## Phase 1: Project Setup & Environment

1.  [ ] **Create Project Structure:**
    * [ ] Main project folder: `video_summarizer_pro/`
    * [ ] Subfolders: `backend/`, `gui/`, `outputs/`, `transcripts/`, `temp/`, `config/`, `logs/`

2.  [ ] **Set Up Virtual Environment:**
    * [ ] Navigate to `video_summarizer_pro/` in your terminal.
    * [ ] Create environment: `python -m venv venv`
    * [ ] Activate environment (Windows/macOS/Linux specific).

3.  [ ] **Install Core Dependencies:**
    * [ ] Create `requirements.txt`.
    * [ ] **Crucially: Install PyTorch with CUDA support.** (Follow instructions from pytorch.org).
    * [ ] Add other dependencies to `requirements.txt`:
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
    * [ ] Install requirements: `pip install -r requirements.txt`
    * [ ] **Install FFmpeg:** Ensure `ffmpeg` and `ffprobe` are installed and in PATH.

4.  [ ] **Verify CUDA Setup (Critical):**
    * [ ] Run the provided Python verification script.
    * [ ] Debug any setup issues until CUDA is detected or confirm CPU fallback.

5.  [ ] **Configuration Management (`backend/settings_manager.py`):**
    * [ ] Implement functions to securely save/load OpenRouter API key using `keyring` (with fallbacks). **Avoid plain text.**
    * [ ] Implement functions to save/load user preferences (models, language, paths) to JSON/config file.
    * [ ] Implement functions to manage the persistent LLM model list (`add_llm_model`, `delete_llm_model`, `get_llm_models`).

6.  [ ] **Logging Setup (`backend/logging_config.py` or similar):**
    * [ ] Configure Python's `logging` module.
    * [ ] Set up a rotating file handler to `logs/app.log`.
    * [ ] Define log format (timestamp, level, module).
    * [ ] Provide easy access to the configured logger instance.

---

## Phase 2: Implement Core Backend Modules

Create modular Python files in `backend/`. Implement error handling and logging.

1.  [ ] **`backend/video_processor.py`:**
    * [ ] Implement `get_video_metadata(video_path: str) -> dict | None` using `ffmpeg.probe`.
    * [ ] Implement `extract_audio(video_path: str, output_audio_path: str, start_time: float | None = None, end_time: float | None = None) -> bool` using `ffmpeg` (mono, 16kHz, optional trim).
    * [ ] Implement `trim_video(input_path: str, output_path: str, start_time: float, end_time: float) -> bool` using `ffmpeg`.
    * [ ] Implement `concatenate_videos(video_segments: list[str], output_path: str) -> bool` using `ffmpeg` filter graph/demuxer.
    * [ ] Implement `preview_trim(input_path: str, output_path: str, start_time: float, end_time: float, duration: float = 5.0) -> bool` using `ffmpeg` (fast settings).

2.  [ ] **`backend/transcriber.py`:**
    * [ ] Implement `class AudioTranscriber:`
        * [ ] `__init__(self, model_size: str = "base", device: str = "auto", compute_type: str = "float16")`:
            * [ ] Load `faster-whisper` model (`WhisperModel`).
            * [ ] Handle device detection (`cuda`/`cpu`).
            * [ ] Handle `compute_type` selection.
            * [ ] Log device/compute type.
            * [ ] Handle model download/loading errors.
        * [ ] `transcribe(self, audio_path: str, language: str | None = None, **kwargs) -> tuple[list, dict] | None`:
            * [ ] Perform transcription using `model.transcribe()`.
            * [ ] Handle language parameter.
            * [ ] Return `(segments, info)` tuple.
            * [ ] Handle transcription errors.
        * [ ] `format_transcript(self, segments: list, format_type: str = 'txt') -> str`:
            * [ ] Implement formatting logic for `format_type='txt'`.
            * [ ] Implement formatting logic for `format_type='srt'`.
            * [ ] Implement formatting logic for `format_type='vtt'`.
        * [ ] `get_transcript_stats(self, segments: list, audio_duration: float) -> dict`:
            * [ ] Calculate word count.
            * [ ] Calculate number of segments.
            * [ ] Return stats dict.

3.  [ ] **`backend/ai_client.py`:**
    * [ ] Implement `class OpenRouterClient:`
        * [ ] `__init__(self, api_key: str)`:
            * [ ] Store API key.
            * [ ] Initialize `httpx.AsyncClient`.
        * [ ] `async summarize(self, transcript: str, model: str, custom_prompt: str | None = None, length: str = "medium", include_timestamps: bool = False) -> str | None`:
            * [ ] Construct summarization prompt based on inputs.
            * [ ] Send request asynchronously to OpenRouter API.
            * [ ] Handle API responses and errors (4xx, 5xx, rate limits).
            * [ ] Parse response for summary text.
            * [ ] Return summary string or `None`.
            * [ ] Log API interactions.
        * [ ] `async generate_study_guide(self, text_input: str, model: str, guide_type: str = "outline", difficulty: str = "intermediate", custom_prompt: str | None = None) -> str | None`:
            * [ ] Construct study guide prompt based on inputs.
            * [ ] Send request asynchronously.
            * [ ] Handle responses/errors.
            * [ ] Return guide text string or `None`.
            * [ ] Log API interactions.

4.  [ ] **`backend/recap_generator.py`:**
    * [ ] Implement `create_recap_video(trimmed_video_path: str, srt_path: str, output_path: str, intro_path: str | None = None, outro_path: str | None = None, burn_subtitles: bool = False) -> bool`:
        * [ ] Define logic for selecting recap segments (if not whole video, potentially using SRT).
        * [ ] Construct list of video segments for concatenation.
        * [ ] If `burn_subtitles` is True: Implement `ffmpeg` call with `subtitles` filter using input `srt_path`.
        * [ ] If `burn_subtitles` is False: Implement `ffmpeg` call for simple concatenation.
        * [ ] Ensure primary output is the video file.
        * [ ] Return `True` on success, `False` on error.
        * [ ] Log `ffmpeg` commands and outcomes.

5.  [ ] **`backend/utils.py`:**
    * [ ] Implement `check_cuda() -> bool`.
    * [ ] Implement `open_folder(path: str)` with platform-specific logic and error handling.
    * [ ] Implement timestamp formatting helpers.
    * [ ] Implement safe directory creation function.

---

## Phase 3: Implement Backend Controller & Orchestration

Connect modules, manage workflow, handle background execution.

1.  [ ] **`backend/main_controller.py`:**
    * [ ] Implement `class MainController:`
        * [ ] `__init__(self)`:
            * [ ] Initialize `SettingsManager`.
            * [ ] Initialize `AudioTranscriber`.
            * [ ] Initialize `OpenRouterClient`.
            * [ ] Get logger instance.
            * [ ] Initialize application state variables.
        * [ ] **Core Pipeline Method (Runs in Background):**
            * [ ] Implement `run_full_pipeline(self, config: dict, status_callback: callable, progress_callback: callable, completion_callback: callable)`:
                * [ ] Launch logic in a separate thread (`threading` or `concurrent.futures`).
                * [ ] Define parameters (`config`, callbacks).
                * [ ] **Orchestration Logic (inside thread):**
                    1.  [ ] Call `video_processor.trim_video`, report status/progress, handle errors, define `trimmed_video_path`.
                    2.  [ ] Call `video_processor.extract_audio`, report status/progress, handle errors, define `audio_path`.
                    3.  [ ] Call `transcriber.transcribe`, report status/progress, handle errors.
                    4.  [ ] Call `transcriber.format_transcript` and save `.txt`, `.srt`, `.vtt` files. Store data, get stats, define `srt_path`.
                    5.  [ ] Call `ai_client.summarize` (handle async if needed), report status/progress, handle errors, save summary, store data.
                    6.  [ ] Call `ai_client.generate_study_guide`, report status/progress, handle errors, save guide, store data.
                    7.  [ ] Call `recap_generator.create_recap_video` using `srt_path`, report status/progress, handle errors.
                    8.  [ ] Call `completion_callback` (True/False, results dict).
                    9.  [ ] Implement temporary file cleanup.
        * [ ] **Regeneration Methods (Run in Background):**
            * [ ] Implement `regenerate_summary(...)` with background execution, API call, state update, callback.
            * [ ] Implement `regenerate_study_guide(...)` with background execution, API call, state update, callback.
        * [ ] **Other Methods:**
            * [ ] Implement `get_settings() -> dict`.
            * [ ] Implement `save_settings(settings: dict)`.
            * [ ] Implement `manage_llm_models(action: str, model_name: str | None = None) -> list`.
            * [ ] Implement `get_transcript_data() -> tuple[str, str, str] | None`.
            * [ ] Implement `get_summary_data() -> str | None`.
            * [ ] Implement `get_study_guide_data() -> str | None`.
            * [ ] Implement `get_video_metadata(video_path: str) -> dict | None`.
            * [ ] Implement `run_preview_trim(...)` with background execution and callback.

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

1.  [ ] **Testing:**
    * [ ] **Unit Tests (`pytest`):**
        * [ ] Write tests for `video_processor` functions (mock `ffmpeg`).
        * [ ] Write tests for `transcriber` methods (mock model).
        * [ ] Write tests for `ai_client` methods (mock `httpx`).
        * [ ] Write tests for `settings_manager`.
        * [ ] Write tests for `utils`.
    * [ ] **Integration Tests:**
        * [ ] Test `MainController.run_full_pipeline` with small sample files.
        * [ ] Verify output file creation and basic content.
        * [ ] Test background execution and callback mechanisms.
    * [ ] **Manual Testing:**
        * [ ] Test full workflow with various inputs.
        * [ ] Test GUI interaction (once GUI exists).
        * [ ] Test error handling messages.
        * [ ] Test on different hardware (CUDA vs. CPU).

2.  [ ] **Packaging:**
    * [ ] **Challenge Assessment:** Acknowledge complexity (Python, ffmpeg, CUDA).
    * [ ] **Tool Exploration:** Investigate PyInstaller/cx_Freeze configuration for dependencies (esp. CUDA).
    * [ ] **Alternative Exploration:** Consider `conda-pack`.
    * [ ] **Documentation:**
        * [ ] Write clear manual setup instructions (Python, CUDA, drivers, ffmpeg, pip).
        * [ ] Target one platform initially (e.g., Windows).

---

This revised guide provides a detailed plan aligned with PRD. Focus on modularity, clear separation of concerns, robust error handling, logging, and non-blocking background execution for a responsive user experience.
