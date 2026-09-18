#!/usr/bin/env python3
"""
scripts/prepare_data.py
=======================
Idempotent download of all data assets required for training.
Run once before the first train (inside Docker or on Linux):

    docker compose run --rm prepare

Or:
    python scripts/prepare_data.py [--data-dir ./data] [--skip-acav]

Downloads
---------
1. piper-sample-generator checkpoint (en_US-libritts_r-medium.pt)
2. MIT room impulse responses  (davidscripka/MIT_environmental_impulse_responses)
3. FMA background music        (rudraml/fma, small, 1 hour)
4. ACAV negative features      (openwakeword_features_ACAV100M_2000_hrs_16bit.npy, ~4–17 GB)
5. Validation features         (validation_set_features.npy, ~180 MB)
6. Truncated validation set    (validation_set_features_small.npy, 50k rows — OOM fix)

All downloads are skipped if the file/directory already exists (idempotent).
No LLM or cloud-AI APIs are used — these are plain wget/HuggingFace dataset pulls.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ── Helpers ───────────────────────────────────────────────────────────────


def log(msg: str) -> None:
    print(f"[prepare] {msg}", flush=True)


def run(cmd: list[str], **kwargs) -> None:
    log(f"$ {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def wget(url: str, dest: Path) -> None:
    """Download url to dest using wget (or curl fallback)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("wget"):
        run(["wget", "-q", "--show-progress", "-O", str(dest), url])
    else:
        run(["curl", "-L", "--progress-bar", "-o", str(dest), url])


# ── HuggingFace base URLs ─────────────────────────────────────────────────

HF_OWW = "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main"
PIPER_TAG = "v2.0.0"
PIPER_CKPT_URL = (
    f"https://github.com/rhasspy/piper-sample-generator/releases/download/"
    f"{PIPER_TAG}/en_US-libritts_r-medium.pt"
)


# ── Step functions (all idempotent) ───────────────────────────────────────


def step_piper_checkpoint(data: Path) -> None:
    dest = data / "piper-sample-generator" / "models" / "en_US-libritts_r-medium.pt"
    if dest.exists():
        log(f"Piper checkpoint already present: {dest}")
        return
    log("Downloading Piper TTS checkpoint (~1.8 GB)…")
    wget(PIPER_CKPT_URL, dest)
    log(f"Saved: {dest}")


def step_mit_rirs(data: Path) -> None:
    out = data / "mit_rirs"
    if out.exists() and any(out.iterdir()):
        log(f"MIT RIRs already present: {out}")
        return
    out.mkdir(parents=True, exist_ok=True)
    log("Downloading MIT room impulse responses…")
    try:
        import datasets as hf_datasets
        import scipy.io.wavfile
        import numpy as np
    except ImportError:
        log("ERROR: 'datasets', 'scipy', 'numpy' must be installed. Run inside Docker.")
        sys.exit(1)

    rir_ds = hf_datasets.load_dataset(
        "davidscripka/MIT_environmental_impulse_responses",
        split="train",
        streaming=True,
    )
    count = 0
    for row in rir_ds:
        name = row["audio"]["path"].split("/")[-1]
        scipy.io.wavfile.write(
            str(out / name),
            16000,
            (row["audio"]["array"] * 32767).astype(np.int16),
        )
        count += 1
    log(f"MIT RIRs: {count} files saved to {out}")


def step_fma(data: Path) -> None:
    out = data / "fma"
    if out.exists() and any(out.iterdir()):
        log(f"FMA already present: {out}")
        return
    out.mkdir(parents=True, exist_ok=True)
    log("Downloading FMA background music (1 hour of 30-second clips)…")
    try:
        import datasets as hf_datasets
        import scipy.io.wavfile
        import numpy as np
    except ImportError:
        log("ERROR: required packages not installed. Run inside Docker.")
        sys.exit(1)

    fma_ds = hf_datasets.load_dataset(
        "rudraml/fma", name="small", split="train", streaming=True
    )
    fma_ds = iter(fma_ds.cast_column("audio", hf_datasets.Audio(sampling_rate=16000)))

    n_hours = 1
    target = n_hours * 3600 // 30  # 30-second clips
    for i in range(target):
        row = next(fma_ds)
        name = row["audio"]["path"].split("/")[-1].replace(".mp3", ".wav")
        scipy.io.wavfile.write(
            str(out / name),
            16000,
            (row["audio"]["array"] * 32767).astype(np.int16),
        )
    log(f"FMA: {target} clips saved to {out}")


def step_acav_features(data: Path) -> None:
    dest = data / "features_neg.npy"
    # Check for the canonical name as well as the long HuggingFace filename
    hf_name = data / "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
    if dest.exists():
        log(f"Negative features already present: {dest}")
        return
    if hf_name.exists():
        log(f"Symlinking HuggingFace ACAV file → features_neg.npy")
        dest.symlink_to(hf_name)
        return

    log("Downloading ACAV negative features (~4–17 GB). This may take a while…")
    wget(f"{HF_OWW}/openwakeword_features_ACAV100M_2000_hrs_16bit.npy", hf_name)
    dest.symlink_to(hf_name)
    log(f"Saved: {hf_name} → {dest}")


def step_validation_features(data: Path) -> None:
    dest = data / "validation_set_features.npy"
    if dest.exists():
        log(f"Validation features already present: {dest}")
    else:
        log("Downloading validation features (~180 MB)…")
        wget(f"{HF_OWW}/validation_set_features.npy", dest)
        log(f"Saved: {dest}")

    # Always ensure the truncated 50k-row version exists (OOM fix for step 7500)
    small = data / "validation_set_features_small.npy"
    if small.exists():
        log(f"Truncated validation set already present: {small}")
        return
    log("Truncating validation set to 50,000 rows (prevents silent OOM at training step 7500)…")
    try:
        import numpy as np
    except ImportError:
        log("ERROR: numpy not installed. Run inside Docker.")
        sys.exit(1)
    arr = np.load(str(dest), mmap_mode="r")
    log(f"Full shape: {arr.shape}  dtype: {arr.dtype}")
    np.save(str(small), arr[:50000])
    log(f"Saved: {small}  shape: {arr[:50000].shape}")


# ── Main ──────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("DATA_DIR", "./data"),
        help="Directory to store downloaded assets (default: ./data or $DATA_DIR)",
    )
    parser.add_argument(
        "--skip-acav",
        action="store_true",
        help="Skip the large ACAV feature file (training will not work without it)",
    )
    args = parser.parse_args()

    data = Path(args.data_dir).resolve()
    data.mkdir(parents=True, exist_ok=True)
    log(f"Data directory: {data}")
    log("")

    step_piper_checkpoint(data)
    log("")
    step_mit_rirs(data)
    log("")
    step_fma(data)
    log("")
    if not args.skip_acav:
        step_acav_features(data)
        log("")
    else:
        log("Skipping ACAV features (--skip-acav).")
        log("")
    step_validation_features(data)
    log("")

    log("All assets ready.")
    log(f"Data dir contents: {[p.name for p in sorted(data.iterdir())]}")


if __name__ == "__main__":
    main()
