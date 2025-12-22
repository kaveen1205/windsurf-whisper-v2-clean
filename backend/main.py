import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
import re
import uuid
import threading
import time

from typing import Optional, Dict, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import whisper


logger = logging.getLogger("panuval_maatram_backend")
TA_PROMPT_APPEND = os.getenv("TA_PROMPT_APPEND", "").strip()
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
)

app = FastAPI(title="Panuval Maatram Backend", version="1.0.0")

# Allow local development and frontend integration (e.g., Cloudflare Pages frontend calling this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this later to specific frontend origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".mp4", ".m4a", ".mov", ".mkv"}


def _get_file_extension(filename: str) -> str:
    return Path(filename).suffix.lower()


WHISPER_MODEL = None
CURRENT_DEVICE = "cpu"
# Allow override via environment variable; default to a more accurate model
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "large-v3")

DEFAULT_TA_PROMPT = (
    "இந்த ஆடியோ முழுவதும் தமிழில் உள்ளது. தமிழ் உச்சரிப்பு, வழக்கு, முறைப்பாடு எல்லாவற்றையும் துல்லியமாகப் பதிவு செய். "
    "வாக்கிய எல்லைகளை தெளிவாகப் பிரித்து, சரியான நிறுத்தக்குறிகள் (., ?, !) மற்றும் கமா (,) பயன்படுத்தவும். "
    "தமிழ் எழுத்துருவில் எழுதவும்; ஆங்கிலச் சொற்கள், பெயர்கள், தொழில்நுட்பச் சொற்கள், சுருக்கங்கள் வந்தால் அவற்றை அப்படியே ஆங்கிலத்தில் வைத்திரு (transliteration செய்யாதே). "
    "தொழில்முறைத் தரத்தில் தெளிவாகவும் சரியான இலக்கணத்துடனும் எழுதவும். பொதுவான பேசுபண்புச் சொற்கள்: "
    "நாங்க, நீங்க, உங்க, பண்ண, இருக்கே, அப்புறம், இல்ல, இதுல, அதான், ஏன், எப்படி, என்ன, இன்னும், ரொம்ப, கொஞ்சம், சும்மா."
)

# Terms that indicate GPU or memory-related failures that should trigger CPU fallback
GPU_ERROR_TERMS = (
    "cuda",
    "cudnn",
    "cublas",
    "rocm",
    "hip",
    "device-side",
    "illegal memory access",
)
MEMORY_ERROR_TERMS = (
    "out of memory",
    "oom",
    "alloc",
    "std::bad_alloc",
    "cannot allocate memory",
    "killed",
)


# In-memory job tracking for progress/result reporting
PROGRESS: Dict[str, Dict[str, Any]] = {}
RESULTS: Dict[str, str] = {}
ERRORS: Dict[str, str] = {}
JOB_TMPDIRS: Dict[str, str] = {}
PROGRESS_LOCK = threading.Lock()


@app.on_event("startup")
def load_resources() -> None:
    """Load Whisper model and verify FFmpeg availability at startup."""
    global WHISPER_MODEL, WHISPER_MODEL_NAME, CURRENT_DEVICE  # noqa: PLW0603

    # Check FFmpeg
    try:
        logger.info("Checking FFmpeg availability...")
        result = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        if result.returncode != 0:
            logger.warning(
                "FFmpeg does not seem to be available or returned a non-zero status.\nStderr: %s",
                result.stderr.strip(),
            )
        else:
            logger.info("FFmpeg is available: %s", result.stdout.splitlines()[0])
    except FileNotFoundError:
        logger.error(
            "FFmpeg executable not found. Whisper may fail on some audio/video formats. "
            "Please install FFmpeg and ensure it is on the PATH."
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error while checking FFmpeg: %s", exc)

    # Determine device honoring environment overrides
    device = _select_device()
    logger.info("PyTorch device selected: %s", device)

    # Load Whisper model once, with fallbacks on failure
    try:
        logger.info("Whisper model loading... (model=%s, device=%s)", WHISPER_MODEL_NAME, device)
        WHISPER_MODEL = whisper.load_model(WHISPER_MODEL_NAME, device=device)
        logger.info("Whisper model loaded successfully (model=%s)", WHISPER_MODEL_NAME)
        CURRENT_DEVICE = device
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to load Whisper model '%s': %s", WHISPER_MODEL_NAME, exc)
        # Attempt fallbacks to smaller models that require less memory
        for fallback in ["small", "base"]:
            if fallback == WHISPER_MODEL_NAME:
                continue
            try:
                logger.info("Attempting fallback Whisper model: %s (device=%s)", fallback, device)
                WHISPER_MODEL = whisper.load_model(fallback, device=device)
                WHISPER_MODEL_NAME = fallback
                logger.info("Fallback Whisper model loaded successfully (model=%s)", WHISPER_MODEL_NAME)
                break
            except Exception as inner_exc:  # noqa: BLE001
                logger.exception("Fallback model '%s' failed to load: %s", fallback, inner_exc)
        else:
            WHISPER_MODEL = None


@app.get("/whisper-status")
def whisper_status() -> JSONResponse:
    """Debug endpoint to report Whisper model load status."""
    loaded = WHISPER_MODEL is not None
    return JSONResponse(
        {
            "whisper_loaded": loaded,
            "model_name": WHISPER_MODEL_NAME if loaded else None,
            "device": CURRENT_DEVICE,
        }
    )


def _select_device() -> str:
    """Select best available device for Whisper (cuda if available)."""
    # Allow override via environment variable
    device_env = os.getenv("WHISPER_DEVICE")
    device = "cpu"
    try:
        import torch  # type: ignore
        if device_env:
            if device_env.lower() == "cuda" and torch.cuda.is_available():
                device = "cuda"
            elif device_env.lower() == "cpu":
                device = "cpu"
            else:
                # Unknown or unavailable request, fall back to auto
                device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        pass
    return device


def ensure_model_loaded() -> None:
    """Attempt to (re)load Whisper model with fallbacks if not already loaded."""
    global WHISPER_MODEL, WHISPER_MODEL_NAME, CURRENT_DEVICE  # noqa: PLW0603
    if WHISPER_MODEL is not None:
        return

    device = _select_device()
    candidates = [WHISPER_MODEL_NAME] + [m for m in ["small", "base"] if m != WHISPER_MODEL_NAME]
    logger.warning("Whisper model unavailable; attempting lazy load. Candidates: %s", candidates)
    for m in candidates:
        try:
            logger.info("Lazy-loading Whisper model: %s (device=%s)", m, device)
            WHISPER_MODEL = whisper.load_model(m, device=device)
            WHISPER_MODEL_NAME = m
            logger.info("Lazy-load succeeded (model=%s)", WHISPER_MODEL_NAME)
            CURRENT_DEVICE = device
            return
        except Exception as exc:  # noqa: BLE001
            logger.exception("Lazy-load failed for model '%s': %s", m, exc)
    logger.error("All lazy-load attempts failed; Whisper model remains unavailable")


def _reload_model_on_device(target_device: str) -> bool:
    """Reload the Whisper model on a specific device with fallbacks.
    Returns True on success, False on failure.
    """
    global WHISPER_MODEL, WHISPER_MODEL_NAME, CURRENT_DEVICE  # noqa: PLW0603
    candidates = [WHISPER_MODEL_NAME] + [m for m in ["small", "base"] if m != WHISPER_MODEL_NAME]
    for m in candidates:
        try:
            logger.info("Reloading Whisper model on %s: %s", target_device, m)
            WHISPER_MODEL = whisper.load_model(m, device=target_device)
            WHISPER_MODEL_NAME = m
            logger.info("Reload success (model=%s on %s)", WHISPER_MODEL_NAME, target_device)
            CURRENT_DEVICE = target_device
            return True
        except Exception as exc:  # noqa: BLE001
            logger.exception("Reload failed for model '%s' on %s: %s", m, target_device, exc)
    WHISPER_MODEL = None
    return False


def _maybe_convert_to_wav(src_path: str, ext: str) -> str:
    """If the input is a video (e.g., mp4), convert to 16kHz mono WAV for stable decoding.
    Returns the path to use for transcription (original or converted).
    """
    video_exts = {".mp4", ".mov", ".mkv", ".m4a"}
    if ext.lower() not in video_exts:
        return src_path

    dst_path = os.path.splitext(src_path)[0] + ".wav"
    try:
        logger.info("Converting input to WAV: %s -> %s", src_path, dst_path)
        proc = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                src_path,
                "-ac",
                "1",
                "-ar",
                "16000",
                "-vn",
                dst_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )
        logger.info("ffmpeg conversion success. First line: %s", proc.stdout.splitlines()[:1] or proc.stderr.splitlines()[:1])
        return dst_path
    except FileNotFoundError:
        logger.error("FFmpeg not found during conversion; proceeding with original file")
        return src_path
    except Exception as exc:  # noqa: BLE001
        logger.exception("FFmpeg conversion failed, proceeding with original file: %s", exc)
        return src_path


def _normalize_audio_to_wav(src_path: str) -> str:
    dst_path = os.path.splitext(src_path)[0] + ".norm.wav"
    try:
        logger.info("Normalizing audio to WAV (16kHz mono, loudnorm): %s -> %s", src_path, dst_path)
        denoise = os.getenv("ENABLE_DENOISE", "0").lower() in {"1", "true", "yes"}
        filters = []
        if denoise:
            filters.append("highpass=f=100")
            filters.append("afftdn=nf=-25")
            filters.append("lowpass=f=8000")
        filters.append("loudnorm=I=-16:TP=-1.5:LRA=11")
        audio_filters = ",".join(filters)
        proc = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                src_path,
                "-ac",
                "1",
                "-ar",
                "16000",
                "-af",
                audio_filters,
                dst_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )
        logger.info("ffmpeg normalize success. First line: %s", proc.stdout.splitlines()[:1] or proc.stderr.splitlines()[:1])
        return dst_path
    except FileNotFoundError:
        logger.error("FFmpeg not found during normalization; proceeding with original file")
        return src_path
    except Exception as exc:  # noqa: BLE001
        logger.exception("FFmpeg normalization failed, proceeding with original file: %s", exc)
        return src_path


def _postprocess_tamil_text(text: str) -> str:
    s = text.replace("\u200c", "").replace("\u200d", "")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" *([.,!?;:]) *", r"\1 ", s)
    s = re.sub(r"([.!?]){2,}", r"\1", s)
    s = re.sub(r"([.!?])(\S)", r"\1 \2", s)
    s = re.sub(r"\s+\n", "\n", s)
    s = re.sub(r"\n\s+", "\n", s)
    return s.strip()


def _ffprobe_duration_seconds(path: str) -> Optional[float]:
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )
        val = proc.stdout.strip()
        return float(val) if val else None
    except Exception:  # noqa: BLE001
        return None


def _set_progress(job_id: str, percent: Optional[int] = None, *, status: Optional[str] = None, message: Optional[str] = None) -> None:
    with PROGRESS_LOCK:
        state = PROGRESS.get(job_id) or {"percent": 0, "status": "processing", "message": None}
        if percent is not None:
            state["percent"] = max(0, min(100, int(percent)))
        if status is not None:
            state["status"] = status
        if message is not None:
            state["message"] = message
        PROGRESS[job_id] = state


def _progress_timer(job_id: str, start_percent: int, duration_s: Optional[float]) -> None:
    if duration_s is None or duration_s <= 0:
        duration_s = 60.0
    t0 = time.time()
    while True:
        with PROGRESS_LOCK:
            state = PROGRESS.get(job_id)
            if not state or state.get("status") != "processing":
                return
            current = int(state.get("percent", 0))
        elapsed = time.time() - t0
        frac = max(0.0, min(1.0, elapsed / duration_s))
        target = start_percent + int(frac * (95 - start_percent))
        if target > current:
            _set_progress(job_id, target)
        if target >= 95:
            time.sleep(0.5)
            continue
        time.sleep(0.5)


def _cleanup_job(job_id: str) -> None:
    tmp = JOB_TMPDIRS.pop(job_id, None)
    if tmp and os.path.isdir(tmp):
        try:
            shutil.rmtree(tmp)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to clean job tmp dir: %s", tmp)


def _run_transcription_job(job_id: str, tmp_dir: str, tmp_path: str, ext: str, language: Optional[str], initial_prompt: Optional[str], audio_duration: Optional[float], mode: Optional[str]) -> None:
    try:
        _set_progress(job_id, 5, status="processing")
        audio_path = _maybe_convert_to_wav(tmp_path, ext)
        _set_progress(job_id, 15)
        fast = (mode or "").strip().lower() in {"fast", "1", "true", "quick", "speed"}
        if not fast:
            audio_path = _normalize_audio_to_wav(audio_path)
        _set_progress(job_id, 20)

        threading.Thread(target=_progress_timer, args=(job_id, 20, audio_duration), daemon=True).start()

        if WHISPER_MODEL is None:
            ensure_model_loaded()
        if WHISPER_MODEL is None:
            _set_progress(job_id, status="error", message="Transcription model not available")
            return

        lang_norm = (language or "").strip().lower() if language else ""
        effective_language = None if lang_norm in {"", "auto", "autodetect", "detect"} else language
        effective_prompt = initial_prompt
        if not effective_prompt and effective_language == "ta":
            effective_prompt = DEFAULT_TA_PROMPT
        if TA_PROMPT_APPEND and effective_language == "ta":
            effective_prompt = (effective_prompt + " " + TA_PROMPT_APPEND) if effective_prompt else TA_PROMPT_APPEND

        logger.info("[job %s] Whisper transcribe starting (lang=%s, mode=%s)", job_id, effective_language or "auto", ("fast" if fast else "normal"))
        if fast:
            result = WHISPER_MODEL.transcribe(
                audio_path,
                fp16=(CURRENT_DEVICE == "cuda"),
                language=effective_language,
                task="transcribe",
                beam_size=1,
                temperature=0.0,
                condition_on_previous_text=True,
                initial_prompt=effective_prompt,
            )
        else:
            result = WHISPER_MODEL.transcribe(
                audio_path,
                fp16=(CURRENT_DEVICE == "cuda"),
                language=effective_language,
                task="transcribe",
                beam_size=10,
                temperature=0.0,
                patience=1,
                condition_on_previous_text=True,
                no_speech_threshold=0.3,
                logprob_threshold=-1.0,
                compression_ratio_threshold=2.6,
                initial_prompt=effective_prompt,
            )
        text = (result.get("text", "") or "").strip()
        detected_lang = result.get("language")
        if (not fast) and ((detected_lang == "ta") or (effective_language == "ta")):
            text = _postprocess_tamil_text(text)

        RESULTS[job_id] = text
        _set_progress(job_id, 100, status="done")
        logger.info("[job %s] Transcription done (lang=%s, chars=%d)", job_id, detected_lang or effective_language or "auto", len(text))
    except Exception as e:  # noqa: BLE001
        logger.exception("[job %s] Transcription failed: %s", job_id, e)
        _set_progress(job_id, status="error", message=f"{type(e).__name__}: {e}")
    finally:
        _cleanup_job(job_id)



@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),  # optional override; otherwise auto-detect
    initial_prompt: Optional[str] = Form(None),  # optional: bias vocabulary/context
) -> JSONResponse:
    """Transcribe an uploaded audio/video file using local Whisper.

    Returns a JSON object: {"text": "<full transcription>"}
    """
    logger.info("/transcribe called")

    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file uploaded.",
        )

    logger.info("Incoming file: filename=%s, content_type=%s", file.filename, file.content_type)

    # Ensure model is available (lazy-load fallback if needed)
    if WHISPER_MODEL is None:
        ensure_model_loaded()

    ext = _get_file_extension(file.filename)
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file type. Supported types: mp3, wav, mp4.",
        )

    if WHISPER_MODEL is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Transcription model is not available on the server. "
                "Try reducing model size (set env WHISPER_MODEL_NAME=base or small) and restart."
            ),
        )

    tmp_dir = tempfile.mkdtemp(prefix="panuval_maatram_")
    tmp_path = os.path.join(tmp_dir, f"upload{ext}")
    logger.info("Saving upload to temporary path: %s", tmp_path)

    total_bytes = 0

    try:
        # Save uploaded file to a temporary path
        with open(tmp_path, "wb") as buffer:
            while True:
                chunk = await file.read(1024 * 1024)  # 1 MB chunks
                if not chunk:
                    break
                total_bytes += len(chunk)
                buffer.write(chunk)

        logger.info("Finished writing file. Total bytes written: %d", total_bytes)

        if total_bytes == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )

        audio_path = _maybe_convert_to_wav(tmp_path, ext)
        audio_path = _normalize_audio_to_wav(audio_path)
        try:
            logger.info("Starting transcription for file: %s (path=%s)", file.filename, audio_path)
            lang_norm = (language or "").strip().lower() if language else ""
            effective_language = None if lang_norm in {"", "auto", "autodetect", "detect"} else language
            if effective_language:
                logger.info("Language override provided by client: %s", effective_language)
            else:
                logger.info("No language provided or 'auto' selected; Whisper will auto-detect language")
            effective_prompt = initial_prompt
            if not effective_prompt and effective_language == "ta":
                effective_prompt = DEFAULT_TA_PROMPT
                logger.info("Applied default Tamil biasing prompt (%d chars)", len(effective_prompt))
            if TA_PROMPT_APPEND and effective_language == "ta":
                effective_prompt = (effective_prompt + " " + TA_PROMPT_APPEND) if effective_prompt else TA_PROMPT_APPEND
                logger.info("Appended TA_PROMPT_APPEND to prompt (%d chars total)", len(effective_prompt))

            # Use beam search for better accuracy with tuned thresholds for Tamil audio.
            logger.info("Calling Whisper model.transcribe(...), task='transcribe', beam_size=10, patience=1")
            result = WHISPER_MODEL.transcribe(
                audio_path,
                fp16=(CURRENT_DEVICE == "cuda"),
                language=effective_language,
                task="transcribe",
                beam_size=10,
                temperature=0.0,
                patience=1,
                condition_on_previous_text=True,
                no_speech_threshold=0.3,
                logprob_threshold=-1.0,
                compression_ratio_threshold=2.6,
                initial_prompt=effective_prompt,
            )
            text = result.get("text", "").strip()
            detected_lang = result.get("language")
            if (detected_lang == "ta") or (effective_language == "ta"):
                text = _postprocess_tamil_text(text)
            if detected_lang:
                logger.info("Detected language: %s", detected_lang)
        except Exception as e:  # noqa: BLE001
            logger.exception("Transcription failed: %s", e)
            # If it's a CUDA/cuDNN issue, try CPU fallback once
            message = str(e)
            lower = message.lower()
            if any(term in lower for term in GPU_ERROR_TERMS) or any(term in lower for term in MEMORY_ERROR_TERMS):
                logger.warning("GPU/memory error detected; attempting CPU fallback reload and retry")
                if _reload_model_on_device("cpu"):
                    try:
                        result = WHISPER_MODEL.transcribe(
                            audio_path,
                            fp16=(CURRENT_DEVICE == "cuda"),
                            language=effective_language,
                            task="transcribe",
                            beam_size=10,
                            temperature=0.0,
                            patience=1,
                            condition_on_previous_text=True,
                            no_speech_threshold=0.3,
                            logprob_threshold=-1.0,
                            compression_ratio_threshold=2.6,
                            initial_prompt=effective_prompt,
                        )
                        text = result.get("text", "").strip()
                        detected_lang = result.get("language")
                        if (detected_lang == "ta") or (effective_language == "ta"):
                            text = _postprocess_tamil_text(text)
                        if detected_lang:
                            logger.info("Detected language (CPU retry): %s", detected_lang)
                    except Exception as e2:  # noqa: BLE001
                        logger.exception("CPU retry failed: %s", e2)
                        raise HTTPException(
                            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Transcription failed after CPU retry: {type(e2).__name__}: {e2}",
                        ) from e2
                else:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=(
                            "GPU/memory error and CPU fallback failed to load model. "
                            "Set WHISPER_DEVICE=cpu and/or WHISPER_MODEL_NAME=small or base, then restart."
                        ),
                    ) from e
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Transcription failed: {type(e).__name__}: {e}",
                ) from e

        if not text:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No transcription text produced.",
            )

        logger.info("Transcription completed successfully. Text length: %d characters", len(text))

        return JSONResponse({"text": text})
    except HTTPException:
        # propagate structured errors as-is so the client can display details
        raise
    except Exception as e:  # noqa: BLE001
        logger.exception("Unhandled error during /transcribe: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unhandled server error: {type(e).__name__}: {e}",
        ) from e
    finally:
        # Clean up temporary directory and file
        try:
            shutil.rmtree(tmp_dir)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to remove temporary directory: %s", tmp_dir)


@app.post("/transcribe-start")
async def transcribe_start(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    initial_prompt: Optional[str] = Form(None),
    mode: Optional[str] = Form(None),  # e.g., "fast" for speed-focused decoding
) -> JSONResponse:
    logger.info("/transcribe-start called")

    if not file or not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file uploaded.")

    ext = _get_file_extension(file.filename)
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file type. Supported types: mp3, wav, mp4.",
        )

    tmp_dir = tempfile.mkdtemp(prefix="panuval_maatram_job_")
    tmp_path = os.path.join(tmp_dir, f"upload{ext}")

    total_bytes = 0
    with open(tmp_path, "wb") as buffer:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            total_bytes += len(chunk)
            buffer.write(chunk)
    if total_bytes == 0:
        shutil.rmtree(tmp_dir)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    dur = _ffprobe_duration_seconds(tmp_path)
    job_id = uuid.uuid4().hex
    JOB_TMPDIRS[job_id] = tmp_dir
    PROGRESS[job_id] = {"percent": 0, "status": "processing", "message": None}

    threading.Thread(
        target=_run_transcription_job,
        args=(job_id, tmp_dir, tmp_path, ext, language, initial_prompt, dur, mode),
        daemon=True,
    ).start()

    logger.info("/transcribe-start created job: %s (duration=%.2fs)", job_id, dur or -1)
    return JSONResponse({"job_id": job_id})


@app.get("/transcribe-progress")
def transcribe_progress(job_id: str) -> JSONResponse:
    state = PROGRESS.get(job_id)
    if not state:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return JSONResponse(
        {
            "job_id": job_id,
            "percent": int(state.get("percent", 0)),
            "status": state.get("status", "processing"),
            "message": state.get("message"),
        }
    )


@app.get("/transcribe-result")
def transcribe_result(job_id: str) -> JSONResponse:
    state = PROGRESS.get(job_id)
    if not state:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    status_str = state.get("status")
    if status_str == "done":
        text = RESULTS.get(job_id, "")
        return JSONResponse({"text": text})
    if status_str == "error":
        msg = state.get("message") or "Transcription failed"
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=msg)
    return JSONResponse({"status": "processing"}, status_code=status.HTTP_202_ACCEPTED)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
