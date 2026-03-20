# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Qwen3-TTS for Apple Silicon — a local text-to-speech application using MLX-optimized Qwen3-TTS models on macOS with Apple Silicon. Runs entirely offline after model download.

## Setup & Run

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install ffmpeg (required for audio conversion)
brew install ffmpeg

# Download models to models/ directory (see README.md for links)

# Run the application
python main.py
```

## Architecture

**Single-file application** (`main.py`, ~440 lines) with an interactive CLI menu loop.

### Three TTS Modes

Each mode uses a different 8-bit quantized 1.7B MLX model from `models/`:

1. **Custom Voice** (`run_custom_session`) — Pre-defined speakers (Ryan, Aiden, Ethan, etc.) with emotion/speed control. Uses `Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit`.
2. **Voice Design** (`run_design_session`) — Generate voices from text descriptions (e.g., "a warm female voice"). Uses `Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit`.
3. **Voice Cloning** (`run_clone_manager`) — Clone voices from reference audio files. Uses `Qwen3-TTS-12Hz-1.7B-Base-8bit`. Supports saved voice profiles in `voices/`.

### Key Dependencies

- **MLX stack** (`mlx`, `mlx-metal`, `mlx-lm`, `mlx-audio`): Apple Silicon GPU-accelerated inference
- **transformers** (v5.0.0rc3): Model architecture and tokenization
- **librosa/soundfile/sounddevice**: Audio I/O and processing
- **ffmpeg** (external): Audio format conversion to 24kHz mono WAV

### Data Flow

User input → MLX model loads from `models/` → `generate_audio()` from mlx-audio → WAV saved to `outputs/` → auto-played via macOS `afplay`

### Directory Conventions

- `models/` — Pre-downloaded MLX model weights (gitignored, 3-6GB)
- `voices/` — Saved voice profiles (`.wav` + `.txt` transcript pairs)
- `outputs/` — Generated audio files, organized by mode subdirectories
- `temp_*` — Temporary conversion files (cleaned up automatically)

## Important Implementation Details

- `mlx-audio` is installed from a pinned git commit, not PyPI
- `get_smart_path()` resolves both direct model folders and HuggingFace snapshot directory structures
- `get_safe_input()` supports drag-and-drop of `.txt` files as input (reads file contents)
- `convert_audio_if_needed()` shells out to `ffmpeg` for format conversion
- Audio sample rate is fixed at 24000 Hz
- `clean_memory()` calls `gc.collect()` and `mlx.core.metal.clear_cache()` between generations
