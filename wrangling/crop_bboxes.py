#!/usr/bin/env python3
"""
Crop YOLO bounding boxes from images and save to:
  - cropped/chest/ for label 0
  - cropped/badge/ for label 1
"""

from pathlib import Path
from PIL import Image

# Paths
BASE_DIR = Path("/home/samer/Desktop/PJ/BadgeGuard/data/extracted_frames/processed/classifition")
IMAGES_DIR = BASE_DIR / "images" / "train"
LABELS_DIR = BASE_DIR / "labels" / "train"
OUTPUT_DIR = BASE_DIR / "cropped"

CHEST_DIR = OUTPUT_DIR / "chest"
BADGE_DIR = OUTPUT_DIR / "badge"

# Create output dirs
CHEST_DIR.mkdir(parents=True, exist_ok=True)
BADGE_DIR.mkdir(parents=True, exist_ok=True)

class_map = {0: CHEST_DIR, 1: BADGE_DIR}

total_chest = 0
total_badge = 0
skipped = 0

label_files = sorted(LABELS_DIR.glob("*.txt"))
print(f"Found {len(label_files)} label files")

for label_path in label_files:
    stem = label_path.stem

    # Find matching image (jpg, jpeg, png)
    img_path = None
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        candidate = IMAGES_DIR / (stem + ext)
        if candidate.exists():
            img_path = candidate
            break

    if img_path is None:
        print(f"  [SKIP] No image found for: {stem}")
        skipped += 1
        continue

    # Read label file
    lines = label_path.read_text().strip().splitlines()
    if not lines:
        continue

    # Open image
    try:
        img = Image.open(img_path)
        W, H = img.size
    except Exception as e:
        print(f"  [ERROR] Cannot open image {img_path}: {e}")
        skipped += 1
        continue

    # Count boxes per class per image to handle multiple boxes from same image
    class_counters = {}

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        try:
            cls = int(parts[0])
            cx = float(parts[1])
            cy = float(parts[2])
            bw = float(parts[3])
            bh = float(parts[4])
        except ValueError:
            continue

        if cls not in class_map:
            continue

        # Convert YOLO normalized coords to pixel coords
        x1 = int((cx - bw / 2) * W)
        y1 = int((cy - bh / 2) * H)
        x2 = int((cx + bw / 2) * W)
        y2 = int((cy + bh / 2) * H)

        # Clamp to image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(W, x2)
        y2 = min(H, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        # Handle multiple boxes of same class in same image
        class_counters[cls] = class_counters.get(cls, 0) + 1
        count = class_counters[cls]

        # Build output filename
        if count == 1:
            out_name = f"{stem}.jpg"
        else:
            out_name = f"{stem}_{count}.jpg"

        out_path = class_map[cls] / out_name

        cropped = img.crop((x1, y1, x2, y2))
        cropped.save(out_path, "JPEG", quality=95)

        if cls == 0:
            total_chest += 1
        else:
            total_badge += 1

print(f"\n=== Done ===")
print(f"  chest (class 0): {total_chest} crops  ->  {CHEST_DIR}")
print(f"  badge (class 1): {total_badge} crops  ->  {BADGE_DIR}")
if skipped:
    print(f"  Skipped: {skipped} label files (no matching image)")
