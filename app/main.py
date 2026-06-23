import io
import itertools
import os
import queue as queue_module
import struct
import subprocess
import threading
from collections.abc import Generator
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from kokoro import KPipeline

DEFAULT_LANG = os.getenv("KOKORO_LANG", "a")  # a=en-us, b=en-gb, e=es, f=fr, h=hi, i=it, j=ja, p=pt-br, z=zh
DEFAULT_VOICE = os.getenv("KOKORO_VOICE", "af_heart")
KOKORO_DEVICE = os.getenv("KOKORO_DEVICE") or None
API_KEY = os.getenv("API_KEY")
SAMPLE_RATE = 24000

# OpenAI voice name -> Kokoro voice
VOICE_MAP = {
    "alloy": "af_heart",
    "echo": "am_michael",
    "fable": "bm_george",
    "onyx": "am_adam",
    "nova": "af_bella",
    "shimmer": "af_sarah",
    "ash": "am_eric",
    "ballad": "bm_lewis",
    "coral": "af_nicole",
    "sage": "bf_emma",
    "verse": "bf_isabella",
}

app = FastAPI(title="kokoro OpenAI-compatible TTS API")

_pipelines: dict[str, KPipeline] = {}


def get_pipeline(lang: str) -> KPipeline:
    if lang not in _pipelines:
        _pipelines[lang] = KPipeline(lang_code=lang, device=KOKORO_DEVICE)
    return _pipelines[lang]


def verify_api_key(authorization: Optional[str] = Header(None)):
    if not API_KEY:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    if authorization.split(" ", 1)[1].strip() != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


class SpeechRequest(BaseModel):
    model: str = "tts-1"
    input: str = Field(..., max_length=4096)
    voice: str = "alloy"
    response_format: str = "mp3"
    speed: float = Field(1.0, ge=0.25, le=4.0)
    language: Optional[str] = None  # extension: pass through Kokoro lang_code
    stream: bool = False  # opt-in HTTP streaming (StreamingResponse, per segment)


# response_format values the streaming path supports. flac is not wired for
# streaming and falls back to the buffered Response.
STREAMABLE_FORMATS = {"pcm", "wav", "mp3", "opus", "aac"}

MEDIA_TYPES = {
    "pcm": "audio/pcm",
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "aac": "audio/aac",
}

# ffmpeg per-format args; identical to the buffered _encode() below so the
# streamed container matches what clients already get today.
FFMPEG_CODEC_ARGS = {
    "mp3":  ["-f", "mp3",  "-codec:a", "libmp3lame", "-q:a", "2"],
    "opus": ["-f", "ogg",  "-codec:a", "libopus",    "-b:a", "64k"],
    "aac":  ["-f", "adts", "-codec:a", "aac",        "-b:a", "128k"],
}


def _resolve_voice(voice: str) -> str:
    if voice in VOICE_MAP:
        return VOICE_MAP[voice]
    return voice  # allow passing native Kokoro voice ids


def _synthesize_stream(text: str, voice: str, speed: float, lang: str) -> "Generator[np.ndarray, None, None]":
    """Yield one float32 mono segment per KPipeline output, as it is produced.

    KPipeline is already a lazy per-segment generator; this just stops buffering.
    """
    pipeline = get_pipeline(lang)
    for _, _, audio in pipeline(text, voice=voice, speed=speed):
        if audio is None:
            continue
        arr = audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)
        yield arr.astype(np.float32)


def _synthesize(text: str, voice: str, speed: float, lang: str) -> np.ndarray:
    """Buffered path, preserved byte-for-byte for the non-streaming Response branch."""
    chunks = list(_synthesize_stream(text, voice, speed, lang))
    if not chunks:
        raise HTTPException(status_code=500, detail="No audio generated")
    return np.concatenate(chunks)


def _seg_to_pcm_bytes(seg: np.ndarray) -> bytes:
    return (np.clip(seg, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def _wav_streaming_header(sample_rate: int, channels: int = 1, bits: int = 16) -> bytes:
    """44-byte RIFF/WAVE header with unknown (streaming) sizes set to 0xFFFFFFFF."""
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    return (
        b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, channels, sample_rate, byte_rate, block_align, bits)
        + b"data" + struct.pack("<I", 0xFFFFFFFF)
    )


def _pcm_wav_byte_stream(seg_gen: "Generator[np.ndarray, None, None]", fmt: str) -> "Generator[bytes, None, None]":
    """Zero-buffer passthrough. The first segment was already validated by the
    caller, so emitting the WAV header up front is safe."""
    if fmt == "wav":
        yield _wav_streaming_header(SAMPLE_RATE)
    for seg in seg_gen:
        yield _seg_to_pcm_bytes(seg)


def _ffmpeg_byte_stream(seg_gen: "Generator[np.ndarray, None, None]", fmt: str) -> "Generator[bytes, None, None]":
    """One persistent ffmpeg encoder for the whole request (independently-encoded
    per-segment fragments would NOT form a valid mp3/ogg/adts stream). A writer
    thread pumps Kokoro segments into stdin; a reader thread drains encoded bytes
    into a bounded queue for backpressure. ffmpeg is killed on any failure or
    client disconnect (Starlette delivers .close()/GeneratorExit to this sync gen)."""
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-i", "pipe:0",
        *FFMPEG_CODEC_ARGS[fmt], "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    output_q: "queue_module.Queue[Optional[bytes]]" = queue_module.Queue(maxsize=10)
    error_q: "queue_module.Queue[Exception]" = queue_module.Queue()

    def write_to_stdin() -> None:
        try:
            for seg in seg_gen:  # pulling here pumps Kokoro inference, off the event loop
                proc.stdin.write(_seg_to_pcm_bytes(seg))
            proc.stdin.close()
        except Exception as e:  # noqa: BLE001
            error_q.put(e)

    def read_from_stdout() -> None:
        try:
            while True:
                chunk = proc.stdout.read(4096)
                if not chunk:
                    break
                output_q.put(chunk)
            output_q.put(None)
        except Exception as e:  # noqa: BLE001
            error_q.put(e)

    wt = threading.Thread(target=write_to_stdin, daemon=True)
    rt = threading.Thread(target=read_from_stdout, daemon=True)
    wt.start()
    rt.start()

    try:
        while True:
            if not error_q.empty():
                raise error_q.get()
            try:
                chunk = output_q.get(timeout=0.1)
            except queue_module.Empty:
                continue
            if chunk is None:
                break
            yield chunk
        wt.join(timeout=5.0)
        rt.join(timeout=5.0)
        rc = proc.wait(timeout=5.0)
        if rc != 0:
            err = proc.stderr.read().decode(errors="ignore") if proc.stderr else ""
            raise RuntimeError(f"ffmpeg failed (rc={rc}): {err}")
    except BaseException:  # includes GeneratorExit on client disconnect
        proc.kill()
        proc.wait()
        raise


def _encode(audio: np.ndarray, fmt: str) -> tuple[bytes, str]:
    if fmt == "wav":
        buf = io.BytesIO()
        sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        return buf.getvalue(), "audio/wav"
    if fmt == "flac":
        buf = io.BytesIO()
        sf.write(buf, audio, SAMPLE_RATE, format="FLAC")
        return buf.getvalue(), "audio/flac"
    if fmt == "pcm":
        pcm = np.clip(audio, -1.0, 1.0)
        return (pcm * 32767.0).astype("<i2").tobytes(), "audio/pcm"

    # ffmpeg-encoded formats: mp3, opus, aac
    codec_args = {
        "mp3":  ["-f", "mp3",  "-codec:a", "libmp3lame", "-q:a", "2"],
        "opus": ["-f", "ogg",  "-codec:a", "libopus",    "-b:a", "64k"],
        "aac":  ["-f", "adts", "-codec:a", "aac",        "-b:a", "128k"],
    }
    media = {"mp3": "audio/mpeg", "opus": "audio/ogg", "aac": "audio/aac"}
    if fmt not in codec_args:
        raise HTTPException(status_code=400, detail=f"Unsupported response_format: {fmt}")

    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-i", "pipe:0",
        *codec_args[fmt], "pipe:1",
    ]
    proc = subprocess.run(cmd, input=pcm, capture_output=True)
    if proc.returncode != 0:
        raise HTTPException(status_code=500, detail=f"ffmpeg failed: {proc.stderr.decode(errors='ignore')}")
    return proc.stdout, media[fmt]


@app.get("/health")
def health():
    return {"status": "ok", "default_voice": DEFAULT_VOICE, "lang": DEFAULT_LANG, "device": KOKORO_DEVICE or "auto", "sample_rate": SAMPLE_RATE}


@app.get("/v1/models")
def list_models(_: None = Depends(verify_api_key)):
    return {
        "object": "list",
        "data": [
            {"id": "tts-1", "object": "model", "owned_by": "openai-compat"},
            {"id": "tts-1-hd", "object": "model", "owned_by": "openai-compat"},
            {"id": "kokoro", "object": "model", "owned_by": "hexgrad"},
        ],
    }


@app.get("/v1/audio/voices")
def list_voices(_: None = Depends(verify_api_key)):
    return {"openai_aliases": VOICE_MAP, "default": DEFAULT_VOICE}


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest, _: None = Depends(verify_api_key)):
    if not req.input.strip():
        raise HTTPException(status_code=400, detail="input is empty")
    voice = _resolve_voice(req.voice)
    lang = req.language or voice[:1] if voice and voice[0] in "abefhijpz" else DEFAULT_LANG
    fmt = req.response_format

    # Opt-in streaming path: emit audio as Kokoro produces each segment.
    # (flac and any other format fall through to the buffered Response.)
    if req.stream and fmt in STREAMABLE_FORMATS:
        # Pre-pull the first segment so a synthesis-start failure (e.g. a bad
        # voice id) returns a clean 4xx/5xx BEFORE the 200 + streamed body starts.
        # `speech` is a plain def, so this runs in Starlette's threadpool — the
        # event loop is never blocked, and Starlette delivers .close() to the
        # byte generator on client disconnect so ffmpeg/threads are cleaned up.
        seg_gen = _synthesize_stream(req.input, voice, req.speed, lang)
        first = next(seg_gen, None)
        if first is None:
            raise HTTPException(status_code=500, detail="No audio generated")
        full_gen = itertools.chain((first,), seg_gen)
        if fmt in ("pcm", "wav"):
            byte_gen = _pcm_wav_byte_stream(full_gen, fmt)
        else:  # mp3 / opus / aac
            byte_gen = _ffmpeg_byte_stream(full_gen, fmt)
        return StreamingResponse(byte_gen, media_type=MEDIA_TYPES[fmt])

    # ---- unchanged buffered path (default) ----
    audio = _synthesize(req.input, voice, req.speed, lang)
    body, media = _encode(audio, fmt)
    return Response(content=body, media_type=media)
