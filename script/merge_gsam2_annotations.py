#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge GDSAM2 per-class annotations into a single ALL_perframe_res.

Input structure (examples):
outputs/
├── bowl_perframe_res/{frames_json, frames_vis, ...}
├── eggs_perframe_res/{frames_json, frames_vis, ...}
├── wok_ladle_perframe_res/{frames_json, frames_vis, ...}
└── wok_perframe_res/{frames_json, frames_vis, ...}

Base RGB frames for visualization:
assets/rgb/000000.png, 000001.png, ...

Output:
outputs/ALL_perframe_res/
├── frames_json/frame_XXXXXX.json     (merged annotations)
└── frames_vis/frame_XXXXXX.png       (merged vis rendered on assets/rgb)

Usage:
  python merge_gsam2_annotations.py \
      --outputs_dir outputs \
      --rgb_dir assets/rgb \
      --classes bowl_perframe_res eggs_perframe_res wok_ladle_perframe_res wok_perframe_res
"""

import argparse
from pathlib import Path
import json
import cv2
import numpy as np
import hashlib
import sys

def require_pycoco():
    try:
        from pycocotools import mask as maskUtils
        return maskUtils
    except Exception as e:
        print("[ERROR] pycocotools is required to decode COCO RLE masks.")
        print("        Please install it first:  pip install pycocotools")
        raise

def sorted_json_paths(dir_path: Path):
    if not dir_path.exists():
        return []
    files = sorted(dir_path.glob("frame_*.json"))
    return files

def frame_id_from_fname(p: Path):
    # frame_000123.json -> 000123
    stem = p.stem  # frame_000123
    return stem.split("_")[-1]

def rgb_path_for_frame(rgb_dir: Path, fid: str):
    # expects 6 digits like '000123'
    return rgb_dir / f"{fid}.png"

def color_for_class(class_name: str):
    # deterministic color from class_name
    import hashlib as _hashlib
    h = _hashlib.md5(class_name.encode("utf-8")).hexdigest()
    b = int(h[0:2], 16)
    g = int(h[2:4], 16)
    r = int(h[4:6], 16)
    return (b, g, r)

def draw_mask(img, binary_mask, color, alpha=0.45):
    # img: HxWx3 (BGR), mask: HxW in {0,1}
    if binary_mask.dtype != np.uint8:
        m = (binary_mask > 0).astype(np.uint8)
    else:
        m = binary_mask
    colored = np.zeros_like(img, dtype=np.uint8)
    colored[:] = color
    mask3 = np.repeat(m[:, :, None], 3, axis=2)
    img[:] = np.where(mask3 == 1, (img * (1 - alpha) + colored * alpha).astype(img.dtype), img)
    return img

def draw_bbox_label(img, bbox_xyxy, label_text, color, thickness=2):
    x1, y1, x2, y2 = bbox_xyxy
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.5
    (tw, th), _ = cv2.getTextSize(label_text, font, fs, 1)
    cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color, -1)
    cv2.putText(img, label_text, (x1 + 3, y1 - 4), font, fs, (255, 255, 255), 1, cv2.LINE_AA)

def _score_of(ann):
    s = ann.get("score", None)
    try:
        return float(s)
    except Exception:
        return float("-inf")

def merge_single_frame(json_paths):
    """
    json_paths: list of Paths from different class folders but the same frame name.
    Returns merged json dict or None if all missing.

    Rule 1: For each input JSON, if it contains multiple annotations, keep ONLY the one with the highest score.
    Rule 2: After collecting from all sources, cap to at most 4 annotations by highest score.
    """
    merged = None
    picked_anns = []
    base_meta = None

    for p in json_paths:
        if not p or not p.exists():
            continue
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if base_meta is None:
            base_meta = {
                "frame_index": data.get("frame_index"),
                "box_format": data.get("box_format", "xyxy"),
                "img_width": data.get("img_width"),
                "img_height": data.get("img_height"),
                "image_path": data.get("image_path"),
            }

        anns = data.get("annotations", [])
        # 清洗無效標註
        clean_anns = []
        for a in anns:
            if not isinstance(a, dict):
                continue
            if "class_name" in a and "bbox" in a:
                clean_anns.append(a)

        if not clean_anns:
            continue

        # --- 規則 1：單一來源 JSON 若有多個，保留 score 最高者 ---
        if len(clean_anns) > 1:
            best = max(clean_anns, key=_score_of)
            picked_anns.append(best)
        else:
            picked_anns.append(clean_anns[0])

    if base_meta is None:
        return None

    # --- 規則 2：合併後最多 4 個物件，以 score 由高到低挑前 4 ---
    if len(picked_anns) > 4:
        picked_anns = sorted(picked_anns, key=_score_of, reverse=True)[:4]

    merged = dict(base_meta)
    merged["annotations"] = picked_anns
    return merged

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_dir", type=str, default="outputs")
    parser.add_argument("--rgb_dir", type=str, default="assets/rgb")
    parser.add_argument("--classes", nargs="+", default=[
        "bowl_perframe_res",
        "eggs_perframe_res",
        "wok_ladle_perframe_res",
        "wok_perframe_res",
    ])
    parser.add_argument("--all_name", type=str, default="ALL_perframe_res")
    parser.add_argument("--vis_ext", type=str, default=".png")  # output vis image extension
    parser.add_argument("--skip_vis_if_missing_rgb", action="store_true")
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    rgb_dir = Path(args.rgb_dir)
    all_dir = outputs_dir / args.all_name
    out_json_dir = all_dir / "frames_json"
    out_vis_dir = all_dir / "frames_vis"
    out_json_dir.mkdir(parents=True, exist_ok=True)
    out_vis_dir.mkdir(parents=True, exist_ok=True)

    class_json_dirs = []
    for cname in args.classes:
        cj = outputs_dir / cname / "frames_json"
        if not cj.exists():
            print(f"[WARN] Missing: {cj}")
        class_json_dirs.append(cj)

    ref_files = []
    for cj in class_json_dirs:
        ref_files = sorted_json_paths(cj)
        if ref_files:
            break
    if not ref_files:
        print("[ERROR] No reference frames_json found. Check your inputs.")
        sys.exit(1)

    maskUtils = require_pycoco()

    total = 0
    merged_cnt = 0
    vis_cnt = 0

    for ref in ref_files:
        fid = frame_id_from_fname(ref)  # e.g., "000000"
        same_name_paths = []
        for cj in class_json_dirs:
            p = cj / f"frame_{fid}.json"
            same_name_paths.append(p if p.exists() else None)

        merged = merge_single_frame(same_name_paths)
        total += 1
        if merged is None:
            print(f"[WARN] Skip frame_{fid}: no sources found.")
            continue

        rgb_path = rgb_path_for_frame(rgb_dir, fid)
        merged["source_image"] = str(rgb_path)
        merged["image_path"] = str((out_vis_dir / f"frame_{fid}").with_suffix(args.vis_ext))

        out_json_path = out_json_dir / f"frame_{fid}.json"
        with out_json_path.open("w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        merged_cnt += 1

        # render
        if not rgb_path.exists():
            msg = f"[WARN] RGB missing for {fid}: {rgb_path}"
            if args.skip_vis_if_missing_rgb:
                print(msg + " (skip vis)")
                continue
            else:
                print(msg + " (create blank canvas)")
                H = merged.get("img_height") or 320
                W = merged.get("img_width") or 320
                img = np.zeros((H, W, 3), dtype=np.uint8)
        else:
            img = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            if img is None:
                print(f"[WARN] Failed to read RGB: {rgb_path}. Using blank canvas.")
                H = merged.get("img_height") or 320
                W = merged.get("img_width") or 320
                img = np.zeros((H, W, 3), dtype=np.uint8)

        H, W = img.shape[:2]

        for ann in merged.get("annotations", []):
            cls = ann.get("class_name", "object")
            color = color_for_class(cls)
            score = ann.get("score", None)
            try:
                label = f"{cls}" + (f" {float(score):.2f}" if score is not None else "")
            except Exception:
                label = f"{cls}"

            seg = ann.get("segmentation")
            if isinstance(seg, dict) and "counts" in seg and "size" in seg:
                try:
                    rle = {
                        "counts": seg["counts"],
                        "size": [int(seg["size"][0]), int(seg["size"][1])]
                    }
                    m = maskUtils.decode(rle)
                    if m.shape[0] != H or m.shape[1] != W:
                        m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
                    img = draw_mask(img, m, color, alpha=0.45)
                except Exception as e:
                    print(f"[WARN] mask decode failed on frame_{fid}: {e}")

            bbox = ann.get("bbox")
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                draw_bbox_label(img, bbox, label, color, thickness=2)

        vis_out = (out_vis_dir / f"frame_{fid}").with_suffix(args.vis_ext)
        cv2.imwrite(str(vis_out), img)
        vis_cnt += 1

    print(f"[DONE] frames scanned: {total}, merged json: {merged_cnt}, vis written: {vis_cnt}")
    print(f"       JSON out: {out_json_dir}")
    print(f"       VIS  out: {out_vis_dir}")

if __name__ == "__main__":
    main()
