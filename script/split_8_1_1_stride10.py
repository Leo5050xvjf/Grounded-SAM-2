#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Split a YOLO dataset into 8:1:1 (train:val:test) with a stride-10 pattern on consecutive items.

Rule:
  - Sort samples by numeric stem (e.g., frame_000000 → 0). If no digits, fallback to lexicographic.
  - For i, item in enumerate(sorted_samples):
        b = i % 10
        0..7 -> train
        8    -> val
        9    -> test

Source structure (typical after your filtering step):
  <src_root>/
    images/<src_split>/*.jpg|png...
    labels/<src_split>/*.txt

Output:
  <dst_root>/
    images/{train,val,test}/
    labels/{train,val,test}/

Usage:
  python split_8_1_1_stride10.py \
      --src yolo_dataset_all4 \
      --src-split train \
      --dst yolo_dataset_all4_split
"""

import argparse
from pathlib import Path
import re
import shutil

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".JPG"]

def find_image_for_stem(images_dir: Path, stem: str):
    for ext in IMG_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None

_num_re = re.compile(r"(\d+)")

def numeric_key(stem: str):
    # prefer the last number in stem; if none, return None
    m = list(_num_re.finditer(stem))
    if not m:
        return None
    return int(m[-1].group(1))

def split_bucket(idx: int) -> str:
    r = idx % 10
    if r <= 7:
        return "train"
    elif r == 8:
        return "val"
    else:
        return "test"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, default="yolo_dataset_all4", help="source YOLO dataset root")
    ap.add_argument("--src-split", type=str, default="train", help="split name to read from source (usually 'train')")
    ap.add_argument("--dst", type=str, default="yolo_dataset_all4_split", help="destination dataset root")
    ap.add_argument("--dry-run", action="store_true", help="print plan only; do not copy files")
    args = ap.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)
    src_img_dir = src_root / "images" / args.src_split
    src_lbl_dir = src_root / "labels" / args.src_split

    if not src_lbl_dir.exists():
        raise SystemExit(f"[ERROR] Labels dir not found: {src_lbl_dir}")
    if not src_img_dir.exists():
        raise SystemExit(f"[ERROR] Images dir not found: {src_img_dir}")

    # Collect stems from labels (ensure paired image exists)
    pairs = []
    for lbl_path in sorted(src_lbl_dir.glob("*.txt")):
        stem = lbl_path.stem
        img_path = find_image_for_stem(src_img_dir, stem)
        if img_path is None:
            print(f"[WARN] Missing image for {lbl_path.name}, skip.")
            continue
        pairs.append((stem, img_path, lbl_path))

    if not pairs:
        raise SystemExit("[ERROR] No (image,label) pairs found.")

    # Sort by numeric key if possible; fallback to lexicographic for ties or missing numbers
    def sort_key(item):
        stem = item[0]
        nk = numeric_key(stem)
        return (0, nk) if nk is not None else (1, stem)
    pairs.sort(key=sort_key)

    # Prepare dest dirs
    for split in ("train", "val", "test"):
        (dst_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Assign and copy
    stats = {"train": 0, "val": 0, "test": 0}
    for i, (stem, img_path, lbl_path) in enumerate(pairs):
        bucket = split_bucket(i)

        dst_img = dst_root / "images" / bucket / img_path.name
        dst_lbl = dst_root / "labels" / bucket / lbl_path.name

        if args.dry_run:
            print(f"[PLAN] {stem} -> {bucket}  ({img_path.name}, {lbl_path.name})")
        else:
            shutil.copy2(img_path, dst_img)
            shutil.copy2(lbl_path, dst_lbl)

        stats[bucket] += 1

    total = sum(stats.values())
    print("\n[RESULT]")
    for k in ("train", "val", "test"):
        n = stats[k]
        pct = (n / total * 100.0) if total else 0.0
        print(f"  {k:>5}: {n:6d}  ({pct:4.1f}%)")
    print(f"  TOTAL: {total}")
    print(f"[OUT] images -> {dst_root/'images'}")
    print(f"[OUT] labels -> {dst_root/'labels'}")

if __name__ == "__main__":
    main()
