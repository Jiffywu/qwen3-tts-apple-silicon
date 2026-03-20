"""
FastAPI backend for Qwen3-TTS Web UI.
Wraps the same TTS pipeline from main.py in HTTP endpoints.
"""

import os
import sys
import re
import shutil
import asyncio
import wave
import webbrowser
from datetime import datetime

# Import shared constants and utilities from main.py
from main import (
    MODELS,
    SPEAKER_MAP,
    EMOTION_EXAMPLES,
    SAMPLE_RATE,
    BASE_OUTPUT_DIR,
    MODELS_DIR,
    VOICES_DIR,
    FILENAME_MAX_LEN,
    get_smart_path,
    convert_audio_if_needed,
    get_saved_voices,
    clean_memory,
    make_temp_dir,
)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI(title="Qwen3-TTS Web UI")

# ── State ────────────────────────────────────────────────────────────────────

loaded_models = {}
generation_lock = asyncio.Lock()

# ── Helpers ──────────────────────────────────────────────────────────────────


def get_or_load_model(model_key: str):
    """Load a model if not already cached."""
    if model_key in loaded_models:
        return loaded_models[model_key]

    from mlx_audio.tts.utils import load_model

    info = MODELS[model_key]
    model_path = get_smart_path(info["folder"])
    if not model_path:
        raise RuntimeError(f"Model folder not found: {info['folder']}")

    model = load_model(model_path)
    loaded_models[model_key] = model
    return model


def save_audio_file_web(temp_folder: str, subfolder: str, text_snippet: str) -> dict:
    """Move generated audio from temp dir to outputs/. Returns file info."""
    save_path = os.path.join(BASE_OUTPUT_DIR, subfolder)
    os.makedirs(save_path, exist_ok=True)

    timestamp = datetime.now().strftime("%H-%M-%S")
    clean_text = (
        re.sub(r"[^\w\s-]", "", text_snippet)[:FILENAME_MAX_LEN].strip().replace(" ", "_")
        or "audio"
    )
    filename = f"{timestamp}_{clean_text}.wav"
    final_path = os.path.join(save_path, filename)

    source_file = os.path.join(temp_folder, "audio_000.wav")
    if not os.path.exists(source_file):
        raise RuntimeError("Generation produced no audio file")

    shutil.move(source_file, final_path)

    if os.path.exists(temp_folder):
        shutil.rmtree(temp_folder, ignore_errors=True)

    # Get duration
    duration = 0.0
    try:
        with wave.open(final_path, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate > 0:
                duration = frames / rate
    except Exception:
        pass

    return {
        "filename": filename,
        "subfolder": subfolder,
        "path": final_path,
        "audio_url": f"/outputs/{subfolder}/{filename}",
        "duration": round(duration, 2),
    }


def get_audio_duration(filepath: str) -> float:
    try:
        with wave.open(filepath, "rb") as wf:
            return round(wf.getnframes() / wf.getframerate(), 2)
    except Exception:
        return 0.0


# ── Static files & outputs ───────────────────────────────────────────────────

# Serve the outputs directory so the browser can play audio
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=BASE_OUTPUT_DIR), name="outputs")

# Serve voices directory for reference audio playback
os.makedirs(VOICES_DIR, exist_ok=True)
app.mount("/voices", StaticFiles(directory=VOICES_DIR), name="voices")

# Serve static files (index.html etc.)
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


# ── Routes ───────────────────────────────────────────────────────────────────


@app.get("/")
async def index():
    return FileResponse(os.path.join(static_dir, "index.html"))


@app.get("/api/config")
async def get_config():
    """Return model/speaker/emotion config for the frontend."""
    return {
        "models": MODELS,
        "speakers": SPEAKER_MAP,
        "emotions": EMOTION_EXAMPLES,
    }


@app.post("/api/generate/custom")
async def generate_custom(
    text: str = Form(...),
    speaker: str = Form("Vivian"),
    emotion: str = Form("Normal tone"),
    speed: float = Form(1.0),
):
    from mlx_audio.tts.generate import generate_audio
    import time

    start = time.time()
    async with generation_lock:
        try:
            model = await asyncio.to_thread(get_or_load_model, "1")
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=str(e))

        temp_dir = make_temp_dir()
        try:
            await asyncio.to_thread(
                generate_audio,
                model=model,
                text=text,
                voice=speaker,
                instruct=emotion,
                speed=speed,
                output_path=temp_dir,
            )
            result = save_audio_file_web(temp_dir, MODELS["1"]["output_subfolder"], text)
        except Exception as e:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            clean_memory()

    result["processing_time"] = round(time.time() - start, 2)
    return result


@app.post("/api/generate/design")
async def generate_design(
    text: str = Form(...),
    voice_description: str = Form(...),
):
    from mlx_audio.tts.generate import generate_audio
    import time

    start = time.time()
    async with generation_lock:
        try:
            model = await asyncio.to_thread(get_or_load_model, "2")
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=str(e))

        temp_dir = make_temp_dir()
        try:
            await asyncio.to_thread(
                generate_audio,
                model=model,
                text=text,
                instruct=voice_description,
                output_path=temp_dir,
            )
            result = save_audio_file_web(temp_dir, MODELS["2"]["output_subfolder"], text)
        except Exception as e:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            clean_memory()

    result["processing_time"] = round(time.time() - start, 2)
    return result


@app.post("/api/generate/clone")
async def generate_clone(
    text: str = Form(...),
    voice_name: str = Form(...),
):
    """Clone using a saved voice profile."""
    from mlx_audio.tts.generate import generate_audio
    import time

    ref_audio = os.path.join(VOICES_DIR, f"{voice_name}.wav")
    if not os.path.exists(ref_audio):
        raise HTTPException(status_code=404, detail=f"Voice '{voice_name}' not found")

    ref_text = "."
    txt_path = os.path.join(VOICES_DIR, f"{voice_name}.txt")
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            ref_text = f.read().strip() or "."

    start = time.time()
    async with generation_lock:
        try:
            model = await asyncio.to_thread(get_or_load_model, "3")
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=str(e))

        temp_dir = make_temp_dir()
        try:
            await asyncio.to_thread(
                generate_audio,
                model=model,
                text=text,
                ref_audio=ref_audio,
                ref_text=ref_text,
                output_path=temp_dir,
            )
            result = save_audio_file_web(temp_dir, MODELS["3"]["output_subfolder"], text)
        except Exception as e:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            clean_memory()

    result["processing_time"] = round(time.time() - start, 2)
    return result


@app.post("/api/generate/clone/quick")
async def generate_clone_quick(
    text: str = Form(...),
    transcript: str = Form("."),
    audio: UploadFile = File(...),
):
    """Quick clone from an uploaded audio file."""
    from mlx_audio.tts.generate import generate_audio
    import time
    import tempfile

    # Save uploaded audio to a temp file
    suffix = os.path.splitext(audio.filename or "audio.wav")[1] or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=os.getcwd())
    try:
        content = await audio.read()
        tmp.write(content)
        tmp.close()

        # Convert if needed
        ref_audio = convert_audio_if_needed(tmp.name)
        if not ref_audio:
            raise HTTPException(status_code=400, detail="Could not process audio file")

        start = time.time()
        async with generation_lock:
            try:
                model = await asyncio.to_thread(get_or_load_model, "3")
            except RuntimeError as e:
                raise HTTPException(status_code=500, detail=str(e))

            temp_dir = make_temp_dir()
            try:
                await asyncio.to_thread(
                    generate_audio,
                    model=model,
                    text=text,
                    ref_audio=ref_audio,
                    ref_text=transcript or ".",
                    output_path=temp_dir,
                )
                result = save_audio_file_web(
                    temp_dir, MODELS["3"]["output_subfolder"], text
                )
            except Exception as e:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
                raise HTTPException(status_code=500, detail=str(e))
            finally:
                clean_memory()

        result["processing_time"] = round(time.time() - start, 2)
        return result
    finally:
        # Clean up temp files
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
        converted = tmp.name.replace(suffix, "") + "_converted.wav"
        if ref_audio and ref_audio != tmp.name and os.path.exists(ref_audio):
            os.unlink(ref_audio)


# ── Voice Library ────────────────────────────────────────────────────────────


@app.get("/api/voices")
async def list_voices():
    """List saved voices with metadata."""
    voices = get_saved_voices()
    result = []
    for name in voices:
        wav_path = os.path.join(VOICES_DIR, f"{name}.wav")
        txt_path = os.path.join(VOICES_DIR, f"{name}.txt")

        transcript = ""
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                transcript = f.read().strip()

        duration = get_audio_duration(wav_path)

        result.append(
            {
                "name": name,
                "transcript": transcript,
                "duration": duration,
                "audio_url": f"/voices/{name}.wav",
            }
        )
    return result


@app.post("/api/voices/enroll")
async def enroll_voice(
    name: str = Form(...),
    transcript: str = Form(""),
    audio: UploadFile = File(...),
):
    """Enroll a new voice from uploaded audio."""
    import tempfile

    safe_name = re.sub(r"[^\w\s-]", "", name).strip().replace(" ", "_")
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid voice name")

    # Check if already exists
    target_wav = os.path.join(VOICES_DIR, f"{safe_name}.wav")
    if os.path.exists(target_wav):
        raise HTTPException(status_code=409, detail=f"Voice '{safe_name}' already exists")

    suffix = os.path.splitext(audio.filename or "audio.wav")[1] or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=os.getcwd())
    try:
        content = await audio.read()
        tmp.write(content)
        tmp.close()

        clean_wav_path = convert_audio_if_needed(tmp.name)
        if not clean_wav_path:
            raise HTTPException(status_code=400, detail="Could not process audio file")

        os.makedirs(VOICES_DIR, exist_ok=True)

        target_txt = os.path.join(VOICES_DIR, f"{safe_name}.txt")

        shutil.copy(clean_wav_path, target_wav)
        with open(target_txt, "w", encoding="utf-8") as f:
            f.write(transcript)

        if clean_wav_path != tmp.name and os.path.exists(clean_wav_path):
            os.remove(clean_wav_path)

        return {"name": safe_name, "message": f"Voice '{safe_name}' enrolled successfully"}
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


@app.delete("/api/voices/{name}")
async def delete_voice(name: str):
    """Delete a saved voice."""
    wav_path = os.path.join(VOICES_DIR, f"{name}.wav")
    txt_path = os.path.join(VOICES_DIR, f"{name}.txt")

    if not os.path.exists(wav_path):
        raise HTTPException(status_code=404, detail=f"Voice '{name}' not found")

    os.remove(wav_path)
    if os.path.exists(txt_path):
        os.remove(txt_path)

    return {"message": f"Voice '{name}' deleted"}


# ── Output History ───────────────────────────────────────────────────────────


@app.get("/api/outputs")
async def list_outputs():
    """List generated audio files grouped by mode."""
    result = []
    if not os.path.exists(BASE_OUTPUT_DIR):
        return result

    for model_key, info in MODELS.items():
        subfolder = info["output_subfolder"]
        folder_path = os.path.join(BASE_OUTPUT_DIR, subfolder)
        if not os.path.exists(folder_path):
            continue

        for fname in sorted(os.listdir(folder_path), reverse=True):
            if not fname.endswith(".wav"):
                continue
            filepath = os.path.join(folder_path, fname)
            duration = get_audio_duration(filepath)
            stat = os.stat(filepath)
            result.append(
                {
                    "filename": fname,
                    "mode": info["name"],
                    "subfolder": subfolder,
                    "audio_url": f"/outputs/{subfolder}/{fname}",
                    "duration": duration,
                    "size": stat.st_size,
                    "created": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                }
            )

    # Sort by creation time, newest first
    result.sort(key=lambda x: x["created"], reverse=True)
    return result


@app.delete("/api/outputs/{subfolder}/{filename}")
async def delete_output(subfolder: str, filename: str):
    """Delete a generated audio file."""
    # Validate subfolder is one of the known output subfolders
    valid_subfolders = {info["output_subfolder"] for info in MODELS.values()}
    if subfolder not in valid_subfolders:
        raise HTTPException(status_code=400, detail="Invalid subfolder")

    filepath = os.path.join(BASE_OUTPUT_DIR, subfolder, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")

    os.remove(filepath)
    return {"message": f"Deleted {filename}"}


# ── Startup ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    os.makedirs(VOICES_DIR, exist_ok=True)

    port = 7860
    print(f"\n  Qwen3-TTS Web UI starting on http://localhost:{port}")
    print(f"  Press Ctrl+C to stop\n")

    webbrowser.open(f"http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
