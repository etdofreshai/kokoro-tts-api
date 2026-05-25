import io
import os
import subprocess
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import Response
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


def _resolve_voice(voice: str) -> str:
    if voice in VOICE_MAP:
        return VOICE_MAP[voice]
    return voice  # allow passing native Kokoro voice ids


def _synthesize(text: str, voice: str, speed: float, lang: str) -> np.ndarray:
    pipeline = get_pipeline(lang)
    chunks = []
    for _, _, audio in pipeline(text, voice=voice, speed=speed):
        if audio is None:
            continue
        arr = audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)
        chunks.append(arr.astype(np.float32))
    if not chunks:
        raise HTTPException(status_code=500, detail="No audio generated")
    return np.concatenate(chunks)


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
    audio = _synthesize(req.input, voice, req.speed, lang)
    body, media = _encode(audio, req.response_format)
    return Response(content=body, media_type=media)
