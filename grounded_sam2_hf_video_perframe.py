# grounded_sam2_hf_video_perframe.py
# 本地：Grounding DINO (HF) + SAM2，逐幀處理 MP4 並保存結果
# - 每幀輸出：可視化 JPG、標註 JSON（RLE mask）
# - 另輸出：全片彙總 JSON；可選合成一支可視化影片
# - 僅做 bbox 尺寸過濾：任一邊 > --max-bbox-side 即剔除

import argparse
import os
import cv2
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import supervision as sv
import pycocotools.mask as mask_util

# SAM2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Grounding DINO (HF)
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# --------- 可選的自訂色盤（有就用，沒有就用預設） ----------
try:
    from supervision.draw.color import ColorPalette
    from utils.supervision_utils import CUSTOM_COLOR_MAP
    CUSTOM_PALETTE = ColorPalette.from_hex(CUSTOM_COLOR_MAP)
except Exception:
    CUSTOM_PALETTE = None
# -------------------------------------------------------------

def single_mask_to_rle(mask_bool: np.ndarray):
    """bool(H,W) -> COCO RLE dict (counts 為 utf-8 字串)"""
    rle = mask_util.encode(np.asfortranarray(mask_bool.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
# "egg white. wok with wooden handle. wok ladle. metal bowl."
def main():
    parser = argparse.ArgumentParser()
    # 你的原參數 + 影片版所需
    parser.add_argument('--grounding-model', default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--text-prompt", default="egg white.")
    parser.add_argument("--video-path", required=True, help="輸入 .mp4 檔")
    parser.add_argument("--sam2-checkpoint", default="./checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2-model-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--output-dir", default="outputs_large/grounded_sam2_hf_video")
    parser.add_argument("--box-threshold", type=float, default=0.5)
    parser.add_argument("--text-threshold", type=float, default=0.3)
    parser.add_argument("--max-bbox-side", type=float, default=300.0, help="bbox 任一邊長 > 此值(像素) 就濾除")
    parser.add_argument("--min-bbox-side", type=float, default=100.0, help="bbox 任一邊長 < 此值(像素) 就濾除")

    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=-1, help="-1 表示到最後")
    parser.add_argument("--stride", type=int, default=1, help=">1 代表抽幀")
    parser.add_argument("--write-video", action="store_true", help="同時輸出可視化影片")
    parser.add_argument("--video-fps", type=float, default=0.0, help="0 表示沿用來源 fps")
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--no-dump-json", action="store_true")
    args = parser.parse_args()

    GROUNDING_MODEL = args.grounding_model
    TEXT_PROMPT = args.text_prompt.strip()
    VIDEO_PATH = args.video_path
    SAM2_CHECKPOINT = args.sam2_checkpoint
    SAM2_MODEL_CONFIG = args.sam2_model_config
    OUTPUT_DIR = Path(args.output_dir)
    DUMP_JSON_RESULTS = not args.no_dump_json
    MAX_SIDE = float(args.max_bbox_side)
    MIN_SIDE = float(args.min_bbox_side)

    DEVICE = "cpu" if args.force_cpu or not torch.cuda.is_available() else "cuda"

    # ---- 準備輸出資料夾 ----
    VIS_DIR = OUTPUT_DIR / "frames_vis"
    JSON_DIR = OUTPUT_DIR / "frames_json"
    ensure_dir(OUTPUT_DIR)
    ensure_dir(VIS_DIR)
    ensure_dir(JSON_DIR)

    # ---- 自動混合精度、TF32 ----
    if DEVICE == "cuda":
        autocast_ctx = torch.autocast(device_type=DEVICE, dtype=torch.bfloat16)
        autocast_ctx.__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    # ---- 載入模型 ----
    print("[Init] Loading Grounding DINO HF:", GROUNDING_MODEL)
    processor = AutoProcessor.from_pretrained(GROUNDING_MODEL)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_MODEL).to(DEVICE).eval()

    print("[Init] Loading SAM2...")
    sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    # ---- 打開影片 ----
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fps_out = args.video_fps if args.video_fps > 0 else src_fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    start_f = max(args.start_frame, 0)
    end_f = total_frames if args.end_frame < 0 else min(args.end_frame, total_frames)
    stride = max(1, args.stride)

    print(f"[Video] {VIDEO_PATH} | {W}x{H} | {src_fps:.2f} fps | frames={total_frames}")
    print(f"[Plan] process frames [{start_f}..{end_f-1}] with stride={stride}")

    # ---- 可選：影片輸出 writer ----
    writer = None
    if args.write_video:
        out_path = str(OUTPUT_DIR / "annotated_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps_out, (W, H))
        print(f"[VideoOut] {out_path}")

    # ---- 全片彙總 JSON ----
    merged_ann = {
        "video_path": VIDEO_PATH,
        "width": W, "height": H, "fps": src_fps,
        "text_prompt": TEXT_PROMPT,
        "box_format": "xyxy",
        "frames": []   # list of {frame_index, image_path, annotations:[...]}
    }

    # ---- 逐幀處理 ----
    saved = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    while True:
        pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if pos >= end_f:
            break

        ret, frame_bgr = cap.read()
        if not ret:
            break

        # 抽幀
        if ((pos - start_f) % stride) != 0:
            continue

        # 命名
        base = f"frame_{pos:06d}"
        vis_path = str(VIS_DIR / f"{base}.jpg")
        json_path = str(JSON_DIR / f"{base}.json")

        # DINO 偵測（逐幀）
        pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        inputs = processor(images=pil_img, text=TEXT_PROMPT, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = grounding_model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            target_sizes=[pil_img.size[::-1]],
        )

        # 若無偵測，直接存一張原圖 & 空 JSON
        if not results or len(results[0]["boxes"]) == 0:
            cv2.imwrite(vis_path, frame_bgr)
            if DUMP_JSON_RESULTS:
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "frame_index": pos,
                        "image_path": vis_path,
                        "annotations": []
                    }, f, indent=2)
                merged_ann["frames"].append({
                    "frame_index": pos,
                    "image_path": vis_path,
                    "annotations": []
                })
            if writer is not None:
                writer.write(frame_bgr)
            saved += 1
            if saved % 50 == 0:
                print(f"[{saved}] processed")
            continue

        # === 取得 boxes/labels/scores ===
        det = results[0]
        input_boxes = det["boxes"].detach().cpu().numpy().astype(np.float32)   # (N,4)
        scores_tensor = det.get("scores", None)
        conf = np.ones((input_boxes.shape[0],), dtype=np.float32) if scores_tensor is None \
               else scores_tensor.detach().cpu().numpy().astype(np.float32)
        class_names_all = [c.strip() for c in det["labels"]]

        # === 只做 bbox 尺寸過濾：任一邊 > MAX_SIDE → 剔除 ===
        w = input_boxes[:, 2] - input_boxes[:, 0]
        h = input_boxes[:, 3] - input_boxes[:, 1]
        keep = ((MIN_SIDE <= w) & (w <= MAX_SIDE) &
        (MIN_SIDE <= h) & (h <= MAX_SIDE))


        if not np.any(keep):
            cv2.imwrite(vis_path, frame_bgr)
            if DUMP_JSON_RESULTS:
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump({"frame_index": pos, "image_path": vis_path, "annotations": []}, f, indent=2)
                merged_ann["frames"].append({"frame_index": pos, "image_path": vis_path, "annotations": []})
            if writer is not None:
                writer.write(frame_bgr)
            saved += 1
            if saved % 50 == 0:
                print(f"[{saved}] processed")
            continue

        # 同步切片（確保各欄位長度一致）
        input_boxes = input_boxes[keep]
        conf = conf[keep]
        class_names = [class_names_all[i] for i in range(len(class_names_all)) if keep[i]]

        # === SAM2 以 box 產生 mask ===
        sam2_predictor.set_image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        masks, mask_scores, logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        # 形狀歸一化：確保 masks = (N, H, W)
        if masks.ndim == 4:          # (N,1,H,W)
            masks = masks.squeeze(1)
        elif masks.ndim == 2:        # (H,W) → (1,H,W)
            masks = masks[None, ...]

        # 再次對齊 N（極少數情況 SAM2 會回少於/多於 boxes 的數量）
        N = input_boxes.shape[0]
        if masks.shape[0] != N:
            N = min(N, masks.shape[0])
            input_boxes = input_boxes[:N]
            conf = conf[:N]
            class_names = class_names[:N]
            masks = masks[:N]
            mask_scores = mask_scores[:N]

        # === 建 class_id：把 class_name 映到連續整數，單類別則全 0 ===
        name_to_id = {}
        ids = []
        for n in class_names:
            if n not in name_to_id:
                name_to_id[n] = len(name_to_id)
            ids.append(name_to_id[n])
        class_ids = np.array(ids, dtype=np.int32)

        # === 可視化（box + label + mask） ===
        detections = sv.Detections(
            xyxy=input_boxes,                 # (N,4)
            mask=masks.astype(bool),          # (N,H,W)
            class_id=class_ids                # (N,)
        )
        if CUSTOM_PALETTE is None:
            box_annot = sv.BoxAnnotator()
            label_annot = sv.LabelAnnotator(text_scale=0.5)   # 字體縮小為 1/2
            mask_annot = sv.MaskAnnotator()
        else:
            box_annot = sv.BoxAnnotator(color=CUSTOM_PALETTE)
            label_annot = sv.LabelAnnotator(color=CUSTOM_PALETTE, text_scale=0.5)
            mask_annot = sv.MaskAnnotator(color=CUSTOM_PALETTE)

        labels = [f"{n} {s:.2f}" for n, s in zip(class_names, conf)]
        vis = frame_bgr.copy()
        vis = box_annot.annotate(scene=vis, detections=detections)
        vis = label_annot.annotate(scene=vis, detections=detections, labels=labels)
        vis = mask_annot.annotate(scene=vis, detections=detections)
        cv2.imwrite(vis_path, vis)
        if writer is not None:
            writer.write(vis)

        # === 單幀 JSON（含 RLE mask） ===
        if DUMP_JSON_RESULTS:
            rles = [single_mask_to_rle(m.astype(bool)) for m in masks]
            frame_json = {
                "frame_index": pos,
                "image_path": vis_path,
                "annotations": [
                    {
                        "class_name": cn,
                        "bbox": box.tolist(),
                        "segmentation": rle,
                        "score": float(ms)  # SAM2 mask score
                    }
                    for cn, box, rle, ms in zip(class_names, input_boxes, rles, mask_scores)
                ],
                "box_format": "xyxy",
                "img_width": W,
                "img_height": H,
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(frame_json, f, indent=2)

            merged_ann["frames"].append({
                "frame_index": pos,
                "image_path": vis_path,
                "annotations": frame_json["annotations"]
            })

        saved += 1
        if saved % 50 == 0:
            print(f"[{saved}] processed")

    cap.release()
    if writer is not None:
        writer.release()

    # 寫全片彙總 JSON
    if DUMP_JSON_RESULTS:
        merged_path = OUTPUT_DIR / "video_annotations.json"
        with open(merged_path, "w", encoding="utf-8") as f:
            json.dump(merged_ann, f, indent=2)
        print(f"[OK] Merged JSON: {merged_path}")

    print(f"[DONE] Frames saved to: {VIS_DIR}")
    if writer is not None:
        print(f"[DONE] Annotated video: {OUTPUT_DIR / 'annotated_video.mp4'}")

if __name__ == "__main__":
    main()
