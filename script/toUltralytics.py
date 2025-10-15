# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# Convert merged GDSAM2 JSONs to Ultralytics YOLOv8/11 segmentation format with robust H/W inference.

# Input JSONs: outputs/ALL_perframe_res/frames_json/frame_*.json
# Each JSON example (merged):
# {
#   "frame_index": 0,
#   "source_image": "assets/rgb/000000.png",
#   "img_width": 320,
#   "img_height": 320,
#   "annotations": [
#     {"class_name":"metal bowl","bbox":[x1,y1,x2,y2],"segmentation":{"size":[320,320],"counts":"..."}}
#   ]
# }

# Output tree:
# yolo_dataset/
# ├── images/train/*.jpg
# └── labels/train/*.txt
# """

# from pathlib import Path
# import json, shutil
# import cv2
# import numpy as np
# from typing import Optional, Tuple

# # pip install pycocotools
# try:
#     from pycocotools import mask as maskUtils
# except Exception as e:
#     raise SystemExit("ERROR: pycocotools is required. Install with: pip install pycocotools") from e

# # ---- Customize your class mapping here ----
# CLASS_MAP = {
#     "mixing bowl": 0,
#     "fried egg": 1,
#     "metal wok": 2,
#     "metal spatula": 3,
# }

# INPUT_JSON_DIR = Path("outputs/ALL_perframe_res/frames_json")
# YOLO_ROOT = Path("yolo_dataset")
# IMG_DIR = YOLO_ROOT / "images/train"
# LBL_DIR = YOLO_ROOT / "labels/train"
# IMG_DIR.mkdir(parents=True, exist_ok=True)
# LBL_DIR.mkdir(parents=True, exist_ok=True)

# def infer_hw(data: dict, ann: Optional[dict], src_img_path: Optional[Path]) -> Optional[Tuple[int,int]]:
#     """Return (H,W) by trying multiple fallbacks."""
#     h = data.get("img_height", None)
#     w = data.get("img_width", None)
#     if isinstance(h, (int, float)) and isinstance(w, (int, float)):
#         return int(h), int(w)

#     # Try segmentation size
#     if ann is not None:
#         seg = ann.get("segmentation", None)
#         if isinstance(seg, dict) and "size" in seg and isinstance(seg["size"], (list, tuple)) and len(seg["size"]) == 2:
#             # COCO RLE size order is [height, width]
#             sh, sw = int(seg["size"][0]), int(seg["size"][1])
#             if sh > 0 and sw > 0:
#                 return sh, sw

#     # Try reading real image
#     if src_img_path is not None and src_img_path.exists():
#         img = cv2.imread(str(src_img_path), cv2.IMREAD_COLOR)
#         if img is not None:
#             H, W = img.shape[:2]
#             return H, W

#     return None

# def xyxy_to_norm_xywh(bbox, W, H):
#     x1, y1, x2, y2 = map(float, bbox)
#     bw, bh = x2 - x1, y2 - y1
#     xc, yc = x1 + bw / 2.0, y1 + bh / 2.0
#     return xc / W, yc / H, bw / W, bh / H

# def decode_mask_to_largest_contour(seg, H_expected=None, W_expected=None):
#     if not isinstance(seg, dict) or "counts" not in seg or "size" not in seg:
#         return None
#     try:
#         rle = {"counts": seg["counts"], "size": [int(seg["size"][0]), int(seg["size"][1])]}
#         m = maskUtils.decode(rle)  # (H_rle, W_rle)
#         if m.ndim == 3:
#             m = m.squeeze(-1)

#         # 🩹 修補：若 RLE size 與實際影像不一致，進行 resize 校正
#         if H_expected and W_expected and (m.shape[0] != H_expected or m.shape[1] != W_expected):
#             m = cv2.resize(m.astype(np.uint8), (W_expected, H_expected), interpolation=cv2.INTER_NEAREST)

#         cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         if not cnts:
#             return None
#         return max(cnts, key=cv2.contourArea).squeeze(1)  # (N,2)
#     except Exception:
#         return None


# def ensure_image_to_yolo_folder(src_img: Path, out_img_path: Path):
#     """Copy/convert image to images/train, enforce .jpg extension for consistency."""
#     out_img_path = out_img_path.with_suffix(".jpg")
#     if out_img_path.exists():
#         return out_img_path
#     # If source is png, we can read & re-encode as jpg
#     im = cv2.imread(str(src_img), cv2.IMREAD_COLOR)
#     if im is None:
#         raise FileNotFoundError(f"Cannot read source image: {src_img}")
#     cv2.imwrite(str(out_img_path), im, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
#     return out_img_path

# def stem_from_any_image_path(p: Path) -> str:
#     """Return filename stem without extension, e.g., frame_000000 from frame_000000.png"""
#     return p.stem

# bad_samples = 0
# ok_samples = 0

# for jp in sorted(INPUT_JSON_DIR.glob("frame_*.json")):
#     with jp.open("r", encoding="utf-8") as f:
#         data = json.load(f)

#     # Prefer source_image (原始 RGB)；若沒有就用 image_path
#     src_path_str = data.get("source_image", data.get("image_path", ""))
#     src_img_path = Path(src_path_str) if src_path_str else None
#     if src_img_path is None or not src_img_path.exists():
#         print(f"[WARN] {jp.name}: source image missing -> {src_img_path}")
#         src_img_path = None  # keep None; we may still infer size from seg

#     # 先用第一個 annotation 試著推斷 H/W（若 JSON 沒寫）
#     any_ann = data["annotations"][0] if data.get("annotations") else None
#     hw = infer_hw(data, any_ann, src_img_path)
#     if hw is None:
#         print(f"[SKIP] {jp.name}: cannot infer (H,W). JSON missing `img_width/height`, seg.size unusable, and image unreadable.")
#         bad_samples += 1
#         continue
#     H, W = hw

#     # 準備影像輸出路徑與 label 路徑（同名）
#     if src_img_path is not None:
#         stem = stem_from_any_image_path(src_img_path)
#     else:
#         # 回退：就用 json 檔名
#         stem = jp.stem  # frame_000000
#     out_img_path = IMG_DIR / f"{stem}.jpg"
#     out_txt_path = LBL_DIR / f"{stem}.txt"

#     # 確保影像在 images/train
#     if src_img_path is not None:
#         try:
#             out_img_path = ensure_image_to_yolo_folder(src_img_path, out_img_path)
#         except Exception as e:
#             print(f"[SKIP] {jp.name}: cannot export image -> {e}")
#             bad_samples += 1
#             continue
#     else:
#         # 沒影像也可先產 label（不建議，但保留可能性）
#         pass

#     yolo_lines = []
#     anns = data.get("annotations", [])
#     for ann in anns:
#         cls = ann.get("class_name", None)
#         if cls not in CLASS_MAP:
#             # 未在 CLASS_MAP 的類別直接跳過
#             continue
#         cid = CLASS_MAP[cls]

#         # bbox 轉 YOLO normalized xywh
#         bbox = ann.get("bbox", None)
#         if not bbox or len(bbox) != 4:
#             # 沒 bbox 仍可只用 polygon，但 Ultralytics 的 seg 標準首 5 欄是 class+xywh
#             # 這裡保守跳過此物件
#             continue
#         try:
#             xc, yc, bw, bh = xyxy_to_norm_xywh(bbox, W, H)
#         except Exception as e:
#             print(f"[WARN] {jp.name}: bbox -> xywh failed ({e}); skip this object.")
#             continue

#         # RLE → mask → 最大輪廓 → 多邊形 (規範化)
#         poly = []
#         seg = ann.get("segmentation", None)
#         if seg:
#             cnt = decode_mask_to_largest_contour(seg, H, W)
#             if cnt is not None and cnt.ndim == 2 and cnt.shape[1] == 2 and cnt.size >= 6:
#                 cnt = cnt.astype(np.float32)
#                 cnt[:, 0] /= W
#                 cnt[:, 1] /= H
#                 poly = cnt.flatten().tolist()

#         if not poly:
#             # 若沒有可用 polygon，Ultralytics 仍允許只有 bbox 的 seg 標註嗎？
#             # 對於 segmentation 任務，不建議；這裡保守跳過此物件。
#             # 你也可以改成用 bbox 四角當作一個簡單 polygon。
#             # -- bbox to poly (optional fallback) --
#             # x1, y1, x2, y2 = map(float, bbox)
#             # poly = [x1/W, y1/H, x2/W, y1/H, x2/W, y2/H, x1/W, y2/H]
#             print(f"[WARN] {jp.name}: no valid polygon for object `{cls}`; skip this object.")
#             continue

#         line = " ".join(
#             [str(cid), f"{xc:.6f}", f"{yc:.6f}", f"{bw:.6f}", f"{bh:.6f}"] +
#             [f"{v:.6f}" for v in poly]
#         )
#         yolo_lines.append(line)

#     if not yolo_lines:
#         print(f"[SKIP] {jp.name}: no usable objects after filtering.")
#         bad_samples += 1
#         # 也可以選擇刪掉對應影像，避免空 label
#         # if out_img_path.exists():
#         #     out_img_path.unlink()
#         continue

#     with out_txt_path.open("w", encoding="utf-8") as f:
#         f.write("\n".join(yolo_lines))

#     ok_samples += 1

# print(f"✅ Done. OK samples: {ok_samples}, Skipped: {bad_samples}")
# print(f"Images: {IMG_DIR}")
# print(f"Labels: {LBL_DIR}")


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert merged GDSAM2 JSONs to Ultralytics YOLOv8/11 segmentation format (polygon-only),
and visualize labels to verify correctness.

Input JSONs: outputs/ALL_perframe_res/frames_json/frame_*.json
Each JSON example (merged):
{
  "frame_index": 0,
  "source_image": "assets/rgb/000000.png",
  "img_width": 320,
  "img_height": 320,
  "annotations": [
    {"class_name":"metal bowl","bbox":[x1,y1,x2,y2],"segmentation":{"size":[320,320],"counts":"..."}}
  ]
}

Output tree:
yolo_dataset/
├── images/train/*.jpg
├── labels/train/*.txt
└── viz/train/*.png           # ← 讀 .txt 反繪的可視化結果
"""

from pathlib import Path
import json, cv2, numpy as np
from typing import Optional, Tuple, List

# pip install pycocotools
try:
    from pycocotools import mask as maskUtils
except Exception as e:
    raise SystemExit("ERROR: pycocotools is required. Install with: pip install pycocotools") from e

# ---- Customize your class mapping here ----
CLASS_MAP = {
    "mixing bowl": 0,
    "fried egg": 1,
    "metal wok": 2,
    "metal spatula": 3,
}

# ---- I/O ----
INPUT_JSON_DIR = Path("outputs/ALL_perframe_res/frames_json")
YOLO_ROOT = Path("yolo_dataset")
IMG_DIR = YOLO_ROOT / "images/train"
LBL_DIR = YOLO_ROOT / "labels/train"
VIZ_DIR = YOLO_ROOT / "viz/train"
for d in (IMG_DIR, LBL_DIR, VIZ_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---- Params ----
MIN_AREA = 20.0           # ignore tiny contours (px^2)
APPROX_FRAC = 0.002       # polygon simplification ratio wrt perimeter; 0 disables
VERTEX_STEP = 12          # mark every n-th vertex in viz
JPEG_QUALITY = 95

def infer_hw(data: dict, ann: Optional[dict], src_img_path: Optional[Path]) -> Optional[Tuple[int,int]]:
    """Return (H,W) by trying JSON fields -> RLE size -> reading the image."""
    h = data.get("img_height", None)
    w = data.get("img_width", None)
    if isinstance(h, (int, float)) and isinstance(w, (int, float)):
        return int(h), int(w)

    if ann is not None:
        seg = ann.get("segmentation", None)
        if isinstance(seg, dict) and "size" in seg and isinstance(seg["size"], (list, tuple)) and len(seg["size"]) == 2:
            sh, sw = int(seg["size"][0]), int(seg["size"][1])  # [H, W]
            if sh > 0 and sw > 0:
                return sh, sw

    if src_img_path is not None and src_img_path.exists():
        img = cv2.imread(str(src_img_path), cv2.IMREAD_COLOR)
        if img is not None:
            H, W = img.shape[:2]
            return H, W
    return None

def rle_to_mask(seg: dict, H_expected=None, W_expected=None) -> Optional[np.ndarray]:
    if not isinstance(seg, dict) or "counts" not in seg or "size" not in seg:
        return None
    rle = {"counts": seg["counts"], "size": [int(seg["size"][0]), int(seg["size"][1])]}
    m = maskUtils.decode(rle)  # (H, W) or (H, W, 1)
    if m.ndim == 3:
        m = m.squeeze(-1)
    m = (m > 0).astype(np.uint8)
    if H_expected and W_expected and (m.shape[0] != H_expected or m.shape[1] != W_expected):
        m = cv2.resize(m, (W_expected, H_expected), interpolation=cv2.INTER_NEAREST)
    return m

def mask_to_polys(binmask: np.ndarray, min_area: float, approx_frac: float) -> List[np.ndarray]:
    cnts, _ = cv2.findContours(binmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        if approx_frac > 0:
            peri = cv2.arcLength(c, True)
            c = cv2.approxPolyDP(c, approx_frac * peri, True)
        c = c.reshape(-1, 2)
        if c.shape[0] >= 3:
            polys.append(c)
    return polys

def ensure_image_jpg(src_img: Path, out_img_path: Path) -> Path:
    out_img_path = out_img_path.with_suffix(".jpg")
    if out_img_path.exists():  # don't re-encode
        return out_img_path
    im = cv2.imread(str(src_img), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(f"Cannot read source image: {src_img}")
    cv2.imwrite(str(out_img_path), im, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    return out_img_path

def draw_polys(img: np.ndarray, polys: List[np.ndarray], color, step=VERTEX_STEP, label=None):
    cv2.polylines(img, [p.astype(np.int32) for p in polys], True, color, 2)
    if label:
        cv2.putText(img, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    for p in polys:
        for i, (x,y) in enumerate(p):
            if i % step == 0:
                cv2.circle(img, (int(x), int(y)), 2, color, -1)

def seed_color(i: int):
    rng = np.random.default_rng(1000 + i)
    return (int(rng.integers(64,255)), int(rng.integers(64,255)), int(rng.integers(64,255)))

def write_yolo_seg_txt(txt_path: Path, polys: List[np.ndarray], cls_id: int, W: int, H: int):
    lines = []
    for poly in polys:
        xs = (poly[:,0] / W).clip(0,1)
        ys = (poly[:,1] / H).clip(0,1)
        pts = np.stack([xs, ys], axis=1).reshape(-1)
        if len(pts) < 6:  # at least 3 points
            continue
        lines.append(str(cls_id) + " " + " ".join(f"{p:.6f}" for p in pts.tolist()))
    if lines:
        with txt_path.open("a", encoding="utf-8") as f:  # append per instance
            f.write("\n".join(lines) + "\n")

def read_yolo_seg_txt(txt_path: Path, W: int, H: int) -> List[np.ndarray]:
    out = []
    if not txt_path.exists():
        return out
    for line in txt_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        nums = list(map(float, parts[1:]))
        if len(nums) % 2 != 0:
            continue
        xs = (np.array(nums[0::2]) * W).astype(np.float32)
        ys = (np.array(nums[1::2]) * H).astype(np.float32)
        out.append(np.stack([xs, ys], axis=1))
    return out

def main():
    bad, ok = 0, 0
    files = sorted(INPUT_JSON_DIR.glob("frame_*.json"))
    if not files:
        raise SystemExit(f"No JSON found in {INPUT_JSON_DIR}")

    for i, jp in enumerate(files):
        data = json.loads(jp.read_text())
        src_path_str = data.get("source_image", data.get("image_path", ""))
        src_img_path = Path(src_path_str) if src_path_str else None
        if not (src_img_path and src_img_path.exists()):
            print(f"[WARN] {jp.name}: source image missing -> {src_img_path}")
            src_img_path = None

        any_ann = data["annotations"][0] if data.get("annotations") else None
        hw = infer_hw(data, any_ann, src_img_path)
        if hw is None:
            print(f"[SKIP] {jp.name}: cannot infer (H,W).")
            bad += 1
            continue
        H, W = hw

        # decide stem
        if src_img_path is not None:
            stem = Path(src_img_path).stem
        else:
            stem = jp.stem

        out_img = IMG_DIR / f"{stem}.jpg"
        out_txt = LBL_DIR / f"{stem}.txt"
        if out_txt.exists():
            out_txt.unlink()  # clean previous

        # export/copy image
        if src_img_path is not None:
            try:
                out_img = ensure_image_jpg(src_img_path, out_img)
            except Exception as e:
                print(f"[SKIP] {jp.name}: cannot export image -> {e}")
                bad += 1
                continue

        # per-object
        wrote_any = False
        for k, ann in enumerate(data.get("annotations", [])):
            cname = ann.get("class_name", "")
            if cname not in CLASS_MAP:
                continue
            cid = CLASS_MAP[cname]

            seg = ann.get("segmentation", None)
            if not seg:
                continue
            m = rle_to_mask(seg, H, W)
            if m is None:
                continue
            polys = mask_to_polys(m, MIN_AREA, APPROX_FRAC)
            if not polys:
                continue

            write_yolo_seg_txt(out_txt, polys, cid, W, H)
            wrote_any = True

        if not wrote_any:
            # optional: delete image to avoid orphan
            # if out_img.exists(): out_img.unlink()
            print(f"[SKIP] {jp.name}: no usable objects.")
            bad += 1
            continue

        # --- visualization from .txt ---
        base = cv2.imread(str(out_img), cv2.IMREAD_COLOR) if out_img.exists() else np.full((H, W, 3), 255, np.uint8)
        polys_from_txt = read_yolo_seg_txt(out_txt, W, H)
        overlay = base.copy()
        if polys_from_txt:
            draw_polys(overlay, polys_from_txt, (0, 0, 255), step=VERTEX_STEP, label=f"{stem} (from .txt)")
        cv2.imwrite(str(VIZ_DIR / f"{stem}.png"), overlay)

        ok += 1
        if (i+1) % 50 == 0:
            print(f"[{i+1}/{len(files)}] ok={ok} skip={bad}")

    print(f"✅ Done. OK samples: {ok}, Skipped: {bad}")
    print(f"Images: {IMG_DIR}")
    print(f"Labels: {LBL_DIR}")
    print(f"Viz:    {VIZ_DIR}")

if __name__ == "__main__":
    main()
