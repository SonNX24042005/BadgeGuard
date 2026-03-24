#!/usr/bin/env python3
"""
Split cropped dataset (chest/ badge/) into train/val/test
Ratio: 70% train, 20% val, 10% test

Output structure:
  split/
    train/chest/, train/badge/
    val/chest/,   val/badge/
    test/chest/,  test/badge/
"""

import random
import shutil
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
CROPPED_DIR = Path(
    "/home/samer/Desktop/PJ/BadgeGuard/data/extracted_frames/processed/classifition/cropped"
)
OUTPUT_DIR = CROPPED_DIR.parent / "split"

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.20
TEST_RATIO  = 0.10

SEED = 42
CLASSES = ["chest", "badge"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# ─────────────────────────────────────────────────────────────────────────────

random.seed(SEED)

total_stats = {}

for cls in CLASSES:
    src_dir = CROPPED_DIR / cls
    if not src_dir.exists():
        print(f"[WARN] {src_dir} does not exist, skipping.")
        continue

    # Collect all image files
    files = sorted([f for f in src_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS])
    n = len(files)
    if n == 0:
        print(f"[WARN] No images found in {src_dir}")
        continue

    # Shuffle
    random.shuffle(files)

    # Split indices
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    n_test  = n - n_train - n_val  # remainder goes to test

    splits = {
        "train": files[:n_train],
        "val":   files[n_train : n_train + n_val],
        "test":  files[n_train + n_val :],
    }

    print(f"\nClass: {cls}  (total: {n})")
    for split_name, split_files in splits.items():
        dst_dir = OUTPUT_DIR / split_name / cls
        dst_dir.mkdir(parents=True, exist_ok=True)

        for src_file in split_files:
            dst_file = dst_dir / src_file.name
            shutil.copy2(src_file, dst_file)

        print(f"  {split_name:5s}: {len(split_files):5d} images  →  {dst_dir}")

    total_stats[cls] = {"train": len(splits["train"]), "val": len(splits["val"]), "test": len(splits["test"]), "total": n}

print("\n=== Summary ===")
for cls, s in total_stats.items():
    print(f"  {cls}: train={s['train']}  val={s['val']}  test={s['test']}  (total={s['total']})")
print(f"\nOutput saved to: {OUTPUT_DIR}")
