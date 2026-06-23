# kokoro-tts-api

OpenAI TTS API-compatible server backed by [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (Apache 2.0).

## Endpoints

- `POST /v1/audio/speech` — mirrors OpenAI's TTS endpoint
- `GET  /v1/models`
- `GET  /v1/audio/voices` — list OpenAI voice aliases
- `GET  /health`
- `GET  /docs`, `/openapi.json`

### `POST /v1/audio/speech` body

```json
{
  "model": "tts-1",
  "input": "Hello world",
  "voice": "alloy",
  "response_format": "mp3",
  "speed": 1.0
}
```

`response_format` supports `mp3`, `opus`, `aac`, `flac`, `wav`, `pcm` (24kHz s16le mono). `speed` in `[0.25, 4.0]`.

### Live streaming (`stream=true`)

Add `"stream": true` to the body to stream audio out **as Kokoro synthesizes
each segment**, instead of buffering the whole clip first — much lower
time-to-first-audio for an interactive/voice UI. Opt-in; omit it (or `false`) and
you get the original fully-buffered response, byte-for-byte.

- `pcm` / `wav` stream as raw 24kHz s16le mono (a streaming WAV header is sent first).
- `mp3` / `opus` / `aac` stream through a single persistent `ffmpeg` encoder.
- `flac` is not wired for streaming and falls back to the buffered response.

```bash
# lowest latency — pipe straight into a player as it arrives
curl -sN http://HOST:8000/v1/audio/speech \
  -H 'Authorization: Bearer $API_KEY' -H 'Content-Type: application/json' \
  -d '{"input":"Streaming hello from Kokoro, segment by segment.","voice":"alloy","response_format":"pcm","stream":true}' \
  | ffplay -f s16le -ar 24000 -ac 1 -nodisp -autoexit -

# streaming mp3
curl -sN http://HOST:8000/v1/audio/speech \
  -d '{"input":"...","response_format":"mp3","stream":true}' | mpv -
```

```python
# OpenAI SDK: pass stream via extra_body (the TTS endpoint has no native stream field)
with client.audio.speech.with_streaming_response.create(
    model="tts-1", voice="alloy", input="...", response_format="pcm",
    extra_body={"stream": True},
) as resp:
    resp.stream_to_file("out.pcm")
```

### Voices

OpenAI aliases mapped to Kokoro voices:

| OpenAI | Kokoro |
| --- | --- |
| alloy | af_heart |
| echo | am_michael |
| fable | bm_george |
| onyx | am_adam |
| nova | af_bella |
| shimmer | af_sarah |
| ash | am_eric |
| ballad | bm_lewis |
| coral | af_nicole |
| sage | bf_emma |
| verse | bf_isabella |

You can also pass any native Kokoro voice id (`af_*`, `am_*`, `bf_*`, `bm_*`, etc.) as `voice`.

## Environment

| Var | Default | Notes |
| --- | --- | --- |
| `KOKORO_LANG` | `a` | `a`=en-us, `b`=en-gb, `e`=es, `f`=fr, `h`=hi, `i`=it, `j`=ja, `p`=pt-br, `z`=zh |
| `KOKORO_VOICE` | `af_heart` | Default voice |
| `API_KEY` | _(unset)_ | If set, requests need `Authorization: Bearer <key>` |
| `HF_HOME` | `/models` | Cached model weights |

## Run locally

```bash
docker compose up --build
```

## Use with the OpenAI SDK

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="anything")
resp = client.audio.speech.create(model="tts-1", voice="alloy", input="Hello there.")
resp.write_to_file("out.mp3")
```

## Dokploy

Deploy as Docker Compose or Dockerfile. Mount a persistent volume at `/models` (`HF_HOME`) so weights survive redeploys. CPU is fine for Kokoro-82M.
