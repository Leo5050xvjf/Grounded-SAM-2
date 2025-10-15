#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filter a YOLO(seg) dataset to keep ONLY samples whose label contains ALL required class IDs.
Copies qualifying images/labels into a new dataset root (no overwrite/move of original).

Default structure assumed:
  <src_root>/
    images/<split>/*.jpg|png|...
    labels/<split>/*.txt

Output structure:
  <dst_root>/
    images/<split>/
    labels/<split>/

Usage:
  python filter_all4_yolo.py \
      --src yolo_dataset \
      --dst yolo_dataset_all4 \
      --splits train \
      --required-classes 0,1,2,3
"""

import argparse
from pathlib import Path
import shutil

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".JPG"]

def parse_required_classes(s: str):
    try:
        return set(int(x.strip()) for x in s.split(",") if x.strip() != "")
    except Exception:
        raise ValueError("required-classes must be a comma-separated list of integers, e.g. '0,1,2,3'")

def find_image_for_stem(images_dir: Path, stem: str):
    for ext in IMG_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None

def label_has_all_required(lbl_path: Path, required: set[int]) -> bool:
    """
    YOLO label format per line: class cx cy w h [poly...]
    We only need the first integer token (class id) on each line.
    """
    if not lbl_path.exists():
        return False
    present = set()
    try:
        with lbl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # first token should be class id
                tok0 = line.split()[0]
                cid = int(float(tok0))  # robust to "0.0"
                present.add(cid)
    except Exception:
        return False
    return required.issubset(present)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, default="yolo_dataset", help="source YOLO dataset root")
    ap.add_argument("--dst", type=str, default="yolo_dataset_all4", help="destination YOLO dataset root")
    ap.add_argument("--splits", type=str, default="train", help="comma-separated splits, e.g. 'train,val'")
    ap.add_argument("--required-classes", type=str, default="0,1,2,3", help="comma-separated class ids to require")
    ap.add_argument("--dry-run", action="store_true", help="only print stats; do not copy files")
    args = ap.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    required = parse_required_classes(args.required_classes)

    print(f"[INFO] Source: {src_root}")
    print(f"[INFO] Dest  : {dst_root}")
    print(f"[INFO] Splits: {splits}")
    print(f"[INFO] Require ALL classes: {sorted(required)}")
    if args.dry_run:
        print("[INFO] DRY-RUN mode (no copying)")

    total_labels = 0
    kept = 0
    per_split_stats = {}

    for split in splits:
        src_img_dir = src_root / "images" / split
        src_lbl_dir = src_root / "labels" / split
        dst_img_dir = dst_root / "images" / split
        dst_lbl_dir = dst_root / "labels" / split
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        if not src_lbl_dir.exists():
            print(f"[WARN] Missing labels dir: {src_lbl_dir} (skip split '{split}')")
            continue

        kept_split = 0
        total_split = 0

        for lbl_path in sorted(src_lbl_dir.glob("*.txt")):
            total_labels += 1
            total_split += 1
            stem = lbl_path.stem

            if not label_has_all_required(lbl_path, required):
                continue

            # find corresponding image
            img_path = find_image_for_stem(src_img_dir, stem)
            if img_path is None:
                print(f"[WARN] Image not found for label: {lbl_path} (stem={stem})")
                continue

            kept += 1
            kept_split += 1
            if not args.dry_run:
                shutil.copy2(img_path, dst_img_dir / img_path.name)
                shutil.copy2(lbl_path, dst_lbl_dir / lbl_path.name)

        per_split_stats[split] = (kept_split, total_split)

    print("\n[RESULT]")
    for split, (k, t) in per_split_stats.items():
        print(f"  {split:>6}: kept {k} / {t} ({(k/t*100.0 if t>0 else 0):.1f}%)")
    print(f"  TOTAL : kept {kept} / {total_labels} ({(kept/total_labels*100.0 if total_labels>0 else 0):.1f}%)")
    print(f"[OUT]   images -> {dst_root/'images'}")
    print(f"[OUT]   labels -> {dst_root/'labels'}")

if __name__ == "__main__":
    main()
