#!/usr/bin/env python3
"""
Truncate the full validation set to avoid OOM during training.
The default validation_set_features.npy (~481k rows) causes a ~3GB tensor
at step 7500 which silently OOM-kills the training process.
Run once before training.
"""
import numpy as np, sys, os

src = sys.argv[1] if len(sys.argv) > 1 else "validation_set_features.npy"
dst = sys.argv[2] if len(sys.argv) > 2 else "validation_set_features_small.npy"
n   = int(sys.argv[3]) if len(sys.argv) > 3 else 50000

if not os.path.exists(src):
    print(f"ERROR: {src} not found")
    sys.exit(1)

data = np.load(src, mmap_mode="r")
print(f"Source shape: {data.shape}  dtype: {data.dtype}")
small = data[:n]
np.save(dst, small)
print(f"Saved {dst}  shape: {small.shape}")
