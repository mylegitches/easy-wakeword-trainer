# easy-wakeword-trainer

A web app that turns a text phrase into a custom openWakeWord model — no coding, no notebooks, no cloud AI.

Type a phrase, click Train, download `.onnx` + `.tflite`.

---

## How it works

The entire pipeline runs locally, programmatically, with no LLM or cloud-AI involvement:

```
phrase → YAML config → Piper TTS clips → augment → openWakeWord DNN → ONNX + TFLite
```

- **Piper TTS** (local, offline) generates synthetic positive clips for your phrase.
- **openWakeWord** `train.py` augments them with background music and room impulse responses, then trains a tiny DNN classifier.
- **onnx2tf** converts the ONNX output to TFLite.
- **FastAPI + SSE** streams live logs to the browser as each step runs.

No OpenAI, no Anthropic, no Gemini, no API keys. Everything runs in Docker on your machine.

---

## Requirements

- Docker Desktop (Windows/Mac/Linux)
- ~20 GB free disk space (training data)
- An NVIDIA GPU is helpful but not required — CPU works (~7–10 min per model)

---

## Quick start

### 1. Clone and enter the repo

```bash
git clone https://github.com/mylegitches/easy-wakeword-trainer.git
cd easy-wakeword-trainer
```

### 2. Download training data (one-time, ~5–20 GB)

```bash
docker compose run --rm prepare
```

This runs `scripts/prepare_data.py` inside the container and downloads:

| Asset | Size | Purpose |
|---|---|---|
| Piper TTS checkpoint (`en_US-libritts_r-medium.pt`) | ~1.8 GB | Generates positive training clips |
| MIT room impulse responses (`mit_rirs/`) | small | Acoustic augmentation |
| FMA background music (`fma/`) | ~400 MB | Background noise augmentation |
| ACAV negative features (`features_neg.npy`) | ~4–17 GB | Negative training examples |
| Validation features (`validation_set_features.npy`) | ~180 MB | Early stopping validation |
| Truncated validation set (`validation_set_features_small.npy`) | ~50 MB | 50k-row OOM-safe version (auto-created) |

All assets are saved to `./data/` (gitignored, bind-mounted into the container).

**Already have the data?** Point `./data` at your existing directory or symlink it — the download is skipped for files that already exist.

```bash
# Skip the large ACAV file if you already have it elsewhere
docker compose run --rm prepare -- --skip-acav
```

### 3. Start the app

```bash
docker compose up --build
```

Open **http://localhost:8000** in your browser.

### 4. Train a wake word

1. Type a phrase (e.g. `hey computer`)
2. Click **Train**
3. Watch the live log: Generate → Augment → Train → Export
4. Download `hey_computer.onnx` and/or `hey_computer.tflite`

---

## Outputs

Finished models are saved to `./outputs/<model_name>/`:

```
outputs/
  hey_computer/
    hey_computer.onnx        # use with openWakeWord Python API
    hey_computer.tflite      # use on edge/mobile
    hey_computer.zip         # both files in one download
    hey_computer.yaml        # config used (for reference / retrain)
```

---

## Phrase rules

- Letters, numbers, and spaces only
- Maximum 6 words
- `model_name` is automatically derived: `"hey computer"` → `hey_computer`

---

## Architecture

```
Browser
  └─ GET  /                      Single-page UI (static HTML/CSS/JS)
  └─ GET  /api/health            Data asset presence check
  └─ POST /api/train             Start training job
  └─ GET  /api/train/{id}/events Server-Sent Events (live log stream)
  └─ GET  /api/train/{id}/download  Download zip / onnx / tflite
```

One job at a time. A second Submit while a job is running returns HTTP 409.

---

## Configuration knobs

Trained with the settings proven in the source pipeline:

| Param | Value | Notes |
|---|---|---|
| `n_samples` | 1000 | TTS positives per train |
| `n_samples_val` | 500 | TTS positives for validation |
| `steps` | 10000 | DNN training steps |
| `layer_size` | 32 | DNN width |
| `tts_batch_size` | 50 | Piper generation batch |
| `augmentation_rounds` | 1 | Background/RIR augmentation passes |
| `target_accuracy` | 0.5 | Early-stop target |
| `target_recall` | 0.25 | Early-stop target |
| `target_false_positives_per_hour` | 0.2 | Early-stop target |

Edit `app/config_template.yaml` to change these permanently.

---

## Known limitations and gotchas

### Validation OOM (silent process death at step 7500)
The full `validation_set_features.npy` (~481k rows) creates a ~3 GB tensor during training validation, which the kernel silently kills with no Python traceback. `prepare_data.py` automatically creates a 50k-row truncated version. All configs point to the small file.

### onnx_tf is unavailable / expected error
After `train.py` writes the ONNX it tries to import `onnx_tf` for TFLite conversion. That module is not installed and the process exits non-zero. This is **expected and harmless** — the ONNX is already saved. `onnx2tf` is used instead for the TFLite step.

### CPU training speed
Page-cache warmup matters. On the first run of the day, training is ~4 it/s because `features_neg.npy` is not in the OS page cache. After the first epoch it warms to ~80–100 it/s. Expect ~7–10 min per model on a CPU-only machine.

---

## Electron (future)

The UI uses relative API URLs (`/api/...`) and binds to `127.0.0.1:8000`. An Electron shell can load `http://127.0.0.1:8000` in a `BrowserWindow` without any frontend changes. The Docker container becomes the backend process managed by Electron's `child_process`.

---

## Source

This project is based on the `train-open-wake-word` pipeline:
- Training approach: [dscripka/openWakeWord](https://github.com/dscripka/openWakeWord)
- TTS: [rhasspy/piper-sample-generator](https://github.com/rhasspy/piper-sample-generator)
- TFLite export: [onnx2tf](https://github.com/PINTO0309/onnx2tf)
