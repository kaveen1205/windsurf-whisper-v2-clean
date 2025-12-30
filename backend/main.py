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
import warnings

from typing import Optional, Dict, Any, List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import whisper
from youtube_transcript_api import (  # type: ignore
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    CouldNotRetrieveTranscript,
)


logger = logging.getLogger("panuval_maatram_backend")
TA_PROMPT_APPEND = os.getenv("TA_PROMPT_APPEND", "").strip()
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
)

# Suppress noisy warning from whisper when model is on CPU but CUDA is present
warnings.filterwarnings("ignore", message="Performing inference on CPU when CUDA is available")

app = FastAPI(title="Panuval Maatram Backend", version="1.0.0")

# Allow local development and frontend integration (e.g., Cloudflare Pages frontend calling this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this later to specific frontend origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".mp4", ".m4a", ".mov", ".mkv", ".webm", ".ogg"}

# Gating thresholds for live chunking (bytes-based fallback when duration probing isn't available)
MIN_SESSION_BYTES = int(os.getenv("MIN_SESSION_BYTES", "65536"))  # ~64KB before first transcription
MIN_DELTA_BYTES = int(os.getenv("MIN_DELTA_BYTES", "32768"))      # ~32KB new data between transcriptions
WORD_LOW_PROB_THRESHOLD = float(os.getenv("WORD_LOW_PROB_THRESHOLD", "0.35"))


def _get_file_extension(filename: str) -> str:
    return Path(filename).suffix.lower()


WHISPER_MODEL = None
CURRENT_DEVICE = "cpu"
# Allow override via environment variable; default to a more accurate model
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "small")

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

# Per-session state for incremental chunk processing
SID_STATE: Dict[str, Dict[str, Any]] = {}
SID_LOCK = threading.Lock()


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

    # If GPU is available at startup and not explicitly disabled, reload on CUDA
    try:
        _maybe_switch_to_cuda()
    except Exception:  # noqa: BLE001
        pass


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


@app.get("/captions")
def get_captions(video_id: str, lang: Optional[str] = None) -> JSONResponse:
    """Return available YouTube captions for a given video_id.

    Response schema:
    {
      "video_id": str,
      "language": str | null,
      "is_generated": bool | null,
      "segments": [{"start": float, "end": float, "text": str}]  # sorted by start
    }
    """
    if not video_id or not isinstance(video_id, str):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing or invalid video_id")
    try:
        langs: List[str] = []
        if lang and lang.strip().lower() not in {"auto", ""}:
            langs.append(lang.strip())
        # Common fallbacks
        langs += ["en", "en-US", "en-GB"]

        # Prefer manually created transcripts if possible
        chosen = None
        chosen_lang = None
        is_generated = None
        try:
            tlist = YouTubeTranscriptApi.list_transcripts(video_id)
            # Try to find a non-generated transcript in preferred languages
            for code in langs:
                try:
                    tr = tlist.find_transcript([code])
                    if tr and not getattr(tr, "is_generated", False):
                        chosen = tr
                        break
                except Exception:
                    continue
            # Fallback to any non-generated transcript
            if chosen is None:
                for tr in tlist:
                    if not getattr(tr, "is_generated", False):
                        chosen = tr
                        break
            # Fallback to generated transcript in preferred languages
            if chosen is None:
                for code in langs:
                    try:
                        tr = tlist.find_transcript([code])
                        if tr:
                            chosen = tr
                            break
                    except Exception:
                        continue
            # As a last resort, pick the first available
            if chosen is None:
                for tr in tlist:
                    chosen = tr
                    break
            if chosen is None:
                raise NoTranscriptFound(video_id)

            data = chosen.fetch() or []
            chosen_lang = getattr(chosen, "language_code", None)
            is_generated = bool(getattr(chosen, "is_generated", False))
        except (TranscriptsDisabled, NoTranscriptFound):
            # Try direct fetch with YouTubeTranscriptApi.get_transcript as a last attempt
            data = YouTubeTranscriptApi.get_transcript(video_id, languages=langs)
            chosen_lang = lang or None
            is_generated = None

        segments = []
        for item in data:
            try:
                s = float(item.get("start", 0.0))
                d = float(item.get("duration", 0.0))
                txt = (item.get("text") or "").strip()
                segments.append({"start": round(s, 2), "end": round(s + max(0.0, d), 2), "text": txt})
            except Exception:
                continue
        segments.sort(key=lambda x: x.get("start", 0.0))
        return JSONResponse({
            "video_id": video_id,
            "language": chosen_lang,
            "is_generated": is_generated,
            "segments": segments,
        })
    except (VideoUnavailable, CouldNotRetrieveTranscript) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Captions unavailable: {type(exc).__name__}") from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("/captions failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to fetch captions: {type(exc).__name__}: {exc}") from exc


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


def _maybe_switch_to_cuda() -> None:
    """If CUDA is available and not explicitly disabled, reload model on CUDA."""
    global CURRENT_DEVICE  # noqa: PLW0603
    try:
        device_env = os.getenv("WHISPER_DEVICE", "").strip().lower()
        if device_env == "cpu":
            logger.info("WHISPER_DEVICE=cpu set; staying on CPU")
            return  # user explicitly wants CPU
        import torch  # type: ignore
        available = torch.cuda.is_available()
        logger.info("CUDA availability check: %s (current_device=%s)", available, CURRENT_DEVICE)
        if CURRENT_DEVICE != "cuda" and available:
            logger.info("CUDA detected; reloading Whisper on cuda for faster inference")
            _reload_model_on_device("cuda")
    except Exception:  # noqa: BLE001
        # Ignore any failure and continue on current device
        pass


def _maybe_convert_to_wav(src_path: str, ext: str) -> str:
    """If the input is a video (e.g., mp4), convert to 16kHz mono WAV for stable decoding.
    Returns the path to use for transcription (original or converted).
    """
    video_exts = {".mp4", ".mov", ".mkv", ".m4a", ".webm"}
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


def _ffprobe_has_audio_stream(path: str) -> bool:
    """Return True if ffprobe detects at least one audio stream in the file."""
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "csv=p=0",
                path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )
        return bool(proc.stdout.strip())
    except Exception:  # noqa: BLE001
        return False


def _sniff_container_kind(path: str) -> str:
    """Best-effort container sniffing by magic bytes. Returns 'ogg', 'webm', or 'unknown'."""
    try:
        with open(path, "rb") as f:
            head = f.read(8)
        if len(head) >= 4 and head[:4] == b"OggS":
            return "ogg"
        if len(head) >= 4 and head[0] == 0x1A and head[1] == 0x45 and head[2] == 0xDF and head[3] == 0xA3:
            return "webm"  # EBML
    except Exception:  # noqa: BLE001
        pass
    return "unknown"


def _try_decode_short_wav(src_path: str) -> bool:
    """Attempt to decode a short segment to WAV to confirm presence of decodable audio."""
    tmp_dir = os.path.dirname(src_path)
    probe_wav = os.path.join(tmp_dir, "probe.wav")
    try:
        proc = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                src_path,
                "-t",
                "1.2",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-vn",
                probe_wav,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )
        if os.path.exists(probe_wav) and os.path.getsize(probe_wav) > 0:
            try:
                os.remove(probe_wav)
            except Exception:  # noqa: BLE001
                pass
            return True
    except FileNotFoundError:
        # ffmpeg missing; fall back to original probe only
        return False
    except Exception:  # noqa: BLE001
        return False
    finally:
        try:
            if os.path.exists(probe_wav):
                os.remove(probe_wav)
        except Exception:  # noqa: BLE001
            pass
    return False


def _has_audio_stream_lenient(path: str) -> bool:
    """Lenient probe used for small real-time chunks.

    - If ffprobe sees an audio stream, return True.
    - Otherwise, if container looks like ogg/webm and a short 0.5s decode to WAV succeeds, return True.
    - Else, return False.
    """
    if _ffprobe_has_audio_stream(path):
        return True
    kind = _sniff_container_kind(path)
    if kind in {"ogg", "webm"}:
        return _try_decode_short_wav(path)
    return False


def _is_likely_partial_container(path: str) -> bool:
    """Heuristics for MediaRecorder timeslice chunks that are not self-contained.

    Returns True for very small files or those without recognizable container magic.
    """
    try:
        size = os.path.getsize(path)
    except Exception:  # noqa: BLE001
        size = 0
    # Many valid short ogg/webm segments are >32KB; treat smaller as partial to be lenient.
    if size < 32 * 1024:
        return True
    kind = _sniff_container_kind(path)
    return kind == "unknown"


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
        # Keep prompt compact; in fast mode avoid rolling long context to reduce repetition
        use_prompt = effective_prompt
        if use_prompt and len(use_prompt) > 200:
            use_prompt = use_prompt[-200:]
        if fast:
            use_prompt = DEFAULT_TA_PROMPT if effective_language == "ta" else None

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
        segs = result.get("segments") or []
        segments_resp = []
        for s in segs:
            try:
                s_start = float(s.get("start") or 0.0) + float(start_offset or 0.0)
                s_end = float(s.get("end") or 0.0) + float(start_offset or 0.0)
            except Exception:
                s_start, s_end = 0.0, 0.0
            words_out = []
            wlist = s.get("words") or []
            for w in wlist:
                try:
                    ws = float(w.get("start") or 0.0) + float(start_offset or 0.0)
                    we = float(w.get("end") or 0.0) + float(start_offset or 0.0)
                except Exception:
                    ws, we = s_start, s_start
                prob = w.get("probability") if isinstance(w, dict) else None
                if prob is None:
                    prob = w.get("prob") if isinstance(w, dict) else None
                words_out.append({
                    "start": ws,
                    "end": we,
                    "word": (w.get("word") if isinstance(w, dict) else None) or "",
                    "prob": prob,
                    "low": (prob is not None and float(prob) < WORD_LOW_PROB_THRESHOLD),
                })
            segments_resp.append({
                "start": round(s_start, 2),
                "end": round(s_end, 2),
                "text": (s.get("text") or "").strip(),
                "avg_logprob": s.get("avg_logprob"),
                "no_speech_prob": s.get("no_speech_prob"),
                "words": words_out,
            })
        lines_marked = []
        for seg in segments_resp:
            mins = int(seg["start"] // 60)
            secs = int(seg["start"] % 60)
            ts = f"{mins:02d}:{secs:02d}"
            if seg.get("words"):
                parts = []
                for w in seg["words"]:
                    token = w.get("word") or ""
                    parts.append(f"[?{token}?]" if w.get("low") else token)
                marked = ("".join(parts)).strip()
            else:
                marked = seg.get("text") or ""
            lines_marked.append(f"[{ts}] {marked}")
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
    # If GPU is available at runtime and not explicitly disabled, use it
    _maybe_switch_to_cuda()

    ext = _get_file_extension(file.filename)
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file type. Supported types: mp3, wav, mp4, webm, ogg.",
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


@app.post("/transcribe-chunk")
async def transcribe_chunk(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    initial_prompt: Optional[str] = Form(None),
    mode: Optional[str] = Form("fast"),
    sid: Optional[str] = Form(None),  # optional session id to enable server-side append/remux
) -> JSONResponse:
    logger.info("/transcribe-chunk called")

    if not file or not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file uploaded.")

    ext = _get_file_extension(file.filename)
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file type. Supported types: mp3, wav, mp4, webm, ogg.",
        )

    if WHISPER_MODEL is None:
        ensure_model_loaded()
    if WHISPER_MODEL is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Transcription model is not available on the server.",
        )
    # If GPU is available at runtime and not explicitly disabled, use it
    _maybe_switch_to_cuda()

    tmp_dir = tempfile.mkdtemp(prefix="chunk_")
    tmp_path = os.path.join(tmp_dir, f"upload{ext}")

    total_bytes = 0
    try:
        with open(tmp_path, "wb") as buffer:
            while True:
                chunk = await file.read(512 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                buffer.write(chunk)

        if total_bytes == 0:
            # Benign: very small/silent timeslice from MediaRecorder; skip without error
            return JSONResponse({"text": ""})

        # If we have a session id, accumulate bytes into a per-session combined file to ensure a stable container.
        audio_candidate_path = tmp_path
        candidate_ext = ext
        start_offset = 0.0
        if sid:
            session_dir = os.path.join(tempfile.gettempdir(), f"ytlive_{sid}")
            try:
                os.makedirs(session_dir, exist_ok=True)
                combined_path = os.path.join(session_dir, f"combined{ext}")
                # If the combined file exists but lacks a proper container header, and this new
                # chunk DOES have a header, reset the combined file starting from this chunk.
                current_kind = _sniff_container_kind(tmp_path)
                existing_kind = _sniff_container_kind(combined_path) if os.path.exists(combined_path) else "unknown"
                reset = (existing_kind == "unknown" and current_kind in {"ogg", "webm"})
                open_mode = "wb" if reset or (not os.path.exists(combined_path)) else "ab"
                with open(tmp_path, "rb") as rf, open(combined_path, open_mode) as wf:
                    shutil.copyfileobj(rf, wf)
                audio_candidate_path = combined_path
                candidate_ext = ext
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to append to session file for sid=%s: %s", sid, exc)
                # fallback to single tmp chunk
                audio_candidate_path = tmp_path
                candidate_ext = ext

        # If we have a session, only process the newly appended tail to reduce work
        transcribe_source_path = audio_candidate_path
        new_dur = 0.0
        small_tail = False
        if sid:
            try:
                d = _ffprobe_duration_seconds(audio_candidate_path)
                new_dur = float(d) if d is not None else 0.0
            except Exception:  # noqa: BLE001
                new_dur = 0.0
            prev_dur = 0.0
            curr_bytes = 0
            try:
                curr_bytes = os.path.getsize(audio_candidate_path)
            except Exception:  # noqa: BLE001
                curr_bytes = 0
            with SID_LOCK:
                st = SID_STATE.get(sid)
                if st is None:
                    st = {"last_duration": 0.0, "last_bytes": 0, "last_ts": time.time()}
                    SID_STATE[sid] = st
                prev_dur = float(st.get("last_duration") or 0.0)
                prev_bytes = int(st.get("last_bytes") or 0)
            if new_dur > 0.0 and new_dur < 2.5:
                return JSONResponse({"text": ""})
            # Track if this is a very small new tail to be lenient on failures
            if new_dur > 0.0 and (new_dur - prev_dur) <= 2.0:
                small_tail = True
            # If less than 1s of new audio arrived, skip heavy processing
            if new_dur > 0.0 and (new_dur - prev_dur) < 1.0:
                return JSONResponse({"text": ""})
            # Bytes-based gating as fallback when duration probing is unavailable
            if curr_bytes > 0 and curr_bytes < max(40960, MIN_SESSION_BYTES):  # require at least ~40-64KB total
                return JSONResponse({"text": ""})
            if curr_bytes > 0 and (curr_bytes - prev_bytes) < max(16384, MIN_DELTA_BYTES):  # require ~16-32KB new data
                return JSONResponse({"text": ""})
            # Clip to the recent tail to avoid re-processing whole file
            try:
                session_dir2 = os.path.join(tempfile.gettempdir(), f"ytlive_{sid}")
                os.makedirs(session_dir2, exist_ok=True)
                seg_path = os.path.join(session_dir2, "tail.wav")
                start = max(0.0, prev_dur - 0.5) if prev_dur > 0.0 else 0.0
                subprocess.run(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-y",
                        "-ss",
                        f"{start:.2f}",
                        "-i",
                        audio_candidate_path,
                        "-ac",
                        "1",
                        "-ar",
                        "16000",
                        "-vn",
                        seg_path,
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True,
                    text=True,
                )
                transcribe_source_path = seg_path
                start_offset = start
            except FileNotFoundError:
                # ffmpeg not available; fall back to full candidate
                transcribe_source_path = audio_candidate_path
            except Exception:
                # clipping failed; fall back to full candidate
                transcribe_source_path = audio_candidate_path

        # Validate candidate; if not clearly audio but looks like a partial container, continue to attempt decode.
        # Only fail early when it doesn't look like partial and no audio stream is present.
        if not _has_audio_stream_lenient(audio_candidate_path):
            if not _is_likely_partial_container(audio_candidate_path):
                logger.warning("/transcribe-chunk: no audio stream detected in upload; likely tab audio not shared")
                return JSONResponse(
                    {
                        "error": "no_audio",
                        "message": "Tab audio not enabled. Please select 'This Tab' and enable 'Share tab audio'.",
                    },
                    status_code=status.HTTP_400_BAD_REQUEST,
                )

        fast = (mode or "").strip().lower() in {"fast", "1", "true", "quick", "speed"}
        # In fast mode, avoid expensive full-file conversion; use the original or the clipped tail
        if fast:
            audio_path = transcribe_source_path
        else:
            audio_path = _maybe_convert_to_wav(transcribe_source_path, candidate_ext)

        lang_norm = (language or "").strip().lower() if language else ""
        effective_language = None if lang_norm in {"", "auto", "autodetect", "detect"} else language
        effective_prompt = initial_prompt
        if not effective_prompt and effective_language == "ta":
            effective_prompt = DEFAULT_TA_PROMPT
        if TA_PROMPT_APPEND and effective_language == "ta":
            effective_prompt = (effective_prompt + " " + TA_PROMPT_APPEND) if effective_prompt else TA_PROMPT_APPEND

        # Prepare compact prompt and avoid rolling context in fast mode
        use_prompt = effective_prompt
        if use_prompt and len(use_prompt) > 200:
            use_prompt = use_prompt[-200:]
        if fast:
            # Keep prompt minimal in fast mode to reduce repetition/latency
            use_prompt = DEFAULT_TA_PROMPT if effective_language == "ta" else None

        logger.info("Calling Whisper for chunk (mode=%s, lang=%s)", ("fast" if fast else "normal"), effective_language or "auto")
        try:
            if fast:
                result = WHISPER_MODEL.transcribe(
                    audio_path,
                    fp16=(CURRENT_DEVICE == "cuda"),
                    language=effective_language,
                    task="transcribe",
                    beam_size=1,
                    temperature=0.0,
                    condition_on_previous_text=False,
                    no_speech_threshold=0.3,
                    logprob_threshold=-1.0,
                    compression_ratio_threshold=2.6,
                    initial_prompt=use_prompt,
                    word_timestamps=True,
                    verbose=False,
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
                    initial_prompt=use_prompt,
                    word_timestamps=True,
                    verbose=False,
                )
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            lower = msg.lower()
            decode_terms = (
                # Common ffmpeg/avformat/decoder errors for partial containers
                "failed to load audio",
                "invalid data found when processing input",
                "ebml header parsing failed",
                "error opening input",
                "moov atom not found",
                "header missing",
                "invalid argument",
                "end of file",
                "unexpectedly",
                "demux",
                "packet",
                "bitstream",
                "vorbis",
                "opus",
                "codec not found",
                "could not find codec",
            )
            if any(term in lower for term in decode_terms):
                if _is_likely_partial_container(audio_candidate_path):
                    logger.info("/transcribe-chunk: decode failure on likely partial chunk; returning empty text")
                    return JSONResponse({"text": ""})
                logger.warning("/transcribe-chunk: decode failure treated as no-audio: %s", msg.splitlines()[0] if msg else msg)
                return JSONResponse(
                    {
                        "error": "no_audio",
                        "message": "Tab audio not enabled or undecodable chunk.",
                    },
                    status_code=status.HTTP_400_BAD_REQUEST,
                )
            # GPU/memory error fallback: reload on CPU and retry once
            if any(term in lower for term in GPU_ERROR_TERMS) or any(term in lower for term in MEMORY_ERROR_TERMS):
                logger.warning("/transcribe-chunk: GPU/memory error detected; attempting CPU fallback reload and retry")
                if _reload_model_on_device("cpu"):
                    try:
                        if fast:
                            result = WHISPER_MODEL.transcribe(
                                audio_path,
                                fp16=(CURRENT_DEVICE == "cuda"),
                                language=effective_language,
                                task="transcribe",
                                beam_size=1,
                                temperature=0.0,
                                condition_on_previous_text=False,
                                no_speech_threshold=0.3,
                                logprob_threshold=-1.0,
                                compression_ratio_threshold=2.6,
                                initial_prompt=use_prompt,
                                word_timestamps=True,
                                verbose=False,
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
                                initial_prompt=use_prompt,
                                word_timestamps=True,
                                verbose=False,
                            )
                    except Exception as e2:  # noqa: BLE001
                        logger.exception("/transcribe-chunk: CPU retry failed: %s", e2)
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
                # Unknown error path: if this looks like a tiny/partial tail, be lenient and return empty
                try:
                    if small_tail or _is_likely_partial_container(audio_candidate_path):
                        logger.info("/transcribe-chunk: unknown error on tiny/partial tail; returning empty text: %s", msg.splitlines()[0] if msg else msg)
                        return JSONResponse({"text": ""})
                except Exception:
                    pass
                # Otherwise, re-raise to outer handler
                raise

        text = (result.get("text", "") or "").strip()
        detected_lang = result.get("language")
        if (not fast) and ((detected_lang == "ta") or (effective_language == "ta")):
            text = _postprocess_tamil_text(text)

        # Build segments with session-aligned timestamps and low-confidence marking
        segments_resp = []
        try:
            raw_segments = result.get("segments") or []
            for s in raw_segments:
                try:
                    s_start = float(s.get("start") or 0.0) + float(start_offset or 0.0)
                    s_end = float(s.get("end") or 0.0) + float(start_offset or 0.0)
                except Exception:
                    s_start, s_end = 0.0, 0.0
                words_out = []
                wlist = s.get("words") or []
                for w in wlist:
                    try:
                        ws = float(w.get("start") or 0.0) + float(start_offset or 0.0)
                        we = float(w.get("end") or 0.0) + float(start_offset or 0.0)
                    except Exception:
                        ws, we = s_start, s_start
                    prob = w.get("probability") if isinstance(w, dict) else None
                    if prob is None:
                        prob = w.get("prob") if isinstance(w, dict) else None
                    words_out.append({
                        "start": ws,
                        "end": we,
                        "word": (w.get("word") if isinstance(w, dict) else None) or "",
                        "prob": prob,
                        "low": (prob is not None and float(prob) < WORD_LOW_PROB_THRESHOLD),
                    })
                segments_resp.append({
                    "start": round(s_start, 2),
                    "end": round(s_end, 2),
                    "text": (s.get("text") or "").strip(),
                    "avg_logprob": s.get("avg_logprob"),
                    "no_speech_prob": s.get("no_speech_prob"),
                    "words": words_out,
                })
        except Exception:
            segments_resp = []
        lines_marked = []
        for seg in segments_resp:
            mins = int(seg["start"] // 60)
            secs = int(seg["start"] % 60)
            ts = f"{mins:02d}:{secs:02d}"
            if seg.get("words"):
                parts = []
                for w in seg["words"]:
                    token = w.get("word") or ""
                    parts.append(f"[?{token}?]" if w.get("low") else token)
                marked = ("".join(parts)).strip()
            else:
                marked = seg.get("text") or ""
            lines_marked.append(f"[{ts}] {marked}")

        # Update session's last processed duration/bytes
        if sid:
            with SID_LOCK:
                st = SID_STATE.get(sid) or {"last_duration": 0.0, "last_bytes": 0, "last_ts": time.time()}
                if new_dur > 0.0:
                    st["last_duration"] = new_dur
                try:
                    st["last_bytes"] = os.path.getsize(audio_candidate_path)
                except Exception:
                    pass
                st["last_ts"] = time.time()
                SID_STATE[sid] = st

        return JSONResponse({
            "text": text,
            "text_marked": "\n".join(lines_marked),
            "language": detected_lang or effective_language,
            "segments": segments_resp,
        })
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        logger.exception("/transcribe-chunk failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription failed: {type(e).__name__}: {e}",
        ) from e
    finally:
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

    # Avoid running ffmpeg/whisper if there is no audio stream at all (e.g., no tab audio shared)
    if not _ffprobe_has_audio_stream(tmp_path):
        shutil.rmtree(tmp_dir)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No audio detected in the uploaded file. Please select 'This Tab' and enable 'Share tab audio'.",
        )

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
