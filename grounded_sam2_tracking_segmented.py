# ==============================================================
# export_gsam2_to_yolo11seg_egg_robust.py
# 以 Grounding DINO + SAM2 產生「egg」分割，匯出為 YOLOv11-seg 資料集
# 重點：抗遮擋、防錯誤持續（低信心即視為無目標；LOST/FOUND；週期 Re-Detect）
# 均勻分割比例：每 10 幀 → 8 train / 1 val / 1 test
# ==============================================================

import os, cv2, torch, numpy as np
from pathlib import Path
from PIL import Image
from contextlib import nullcontext
import supervision as sv
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# === SAM2 ===
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from utils.video_utils import create_video_from_images

# ======================== 使用者可調參 =========================
MODEL_ID = "IDEA-Research/grounding-dino-base"   # Grounding DINO
VIDEO_PATH = "./assets/kitchen.mp4"

# 單一目標類別：egg
TEXT_PROMPT = "egg white. fried egg."
TARGET_YOLO_NAME = "egg white. fried egg."
GDINO_TEXT = "egg white. fried egg."

SEGMENT_SIZE = 200
INIT_SEARCH_FRAMES = 10
THRESHOLDS = [
    (0.40, 0.30),
    (0.30, 0.22),
    (0.25, 0.18),
    (0.20, 0.15),
]

# ---- 抗遮擋&防錯誤持續核心參數 ----
REQUIRE_ALWAYS = False      # 關閉強制補洞；低信心就當作「此幀無目標」
MASK_CONF_MIN = 0.55        # mask 內部平均機率下限（越高越嚴）
AREA_FRAC_MIN = 0.001       # 面積占比下限（依解析度&鏡頭距離調整）
AREA_FRAC_MAX = 0.15        # 面積占比上限（避免吃到大塊背景或整鍋）
RECHECK_EVERY = 5           # 每隔 N 幀做一次 DINO 重新驗證/重定位
LOST_PATIENCE = 3           # 連續低信心幀數達標→宣告遺失（清掉追蹤）
FOUND_PATIENCE = 2          # 連續高信心幀數→才判定恢復（抗抖動）
IOU_MIN = 0.30              # 重定位候選與歷史穩定 mask 的最低 IoU（相似度檻）
DET_CONF_MIN = 0.35         # DINO 的最低接受分數（若版本無 scores，當存在旗標）

# 可視化輸出（選配）
OUTPUT_DIR = Path("./seg_tracking_results_egg_robust")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_VIDEO_PATH = "./outputs/kitchen_segmented_tracking_egg_robust.mp4"

# SAM2 權重與設定
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"

# ===============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
amp_dtype = torch.bfloat16 if device == "cuda" else torch.float32
if device == "cuda" and torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# 建 SAM2 模型
video_predictor = build_sam2_video_predictor(SAM2_CFG, SAM2_CHECKPOINT)
sam2_image_model = build_sam2(SAM2_CFG, SAM2_CHECKPOINT)
image_predictor = SAM2ImagePredictor(sam2_image_model)

# 建 Grounding DINO
processor = AutoProcessor.from_pretrained(MODEL_ID)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device)

# 影片讀取
video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)
print(video_info)
frame_gen = sv.get_video_frames_generator(VIDEO_PATH, stride=1, start=1000, end=None)
all_frames = [f for f in frame_gen]
total_frames = len(all_frames)
print(f"Total frames: {total_frames}")

tmp_dir = Path("./tmp_frames")
tmp_dir.mkdir(exist_ok=True)
segment_count = (total_frames + SEGMENT_SIZE - 1) // SEGMENT_SIZE
print(f"Processing in {segment_count} segments...")

# ===============================================================
# Dataset folder setup (8:1:1)
# ===============================================================
DATASET_ROOT = Path("./dataset_egg_robust")
IMG_DIR_TRAIN = DATASET_ROOT / "images/train"
IMG_DIR_VAL   = DATASET_ROOT / "images/val"
IMG_DIR_TEST  = DATASET_ROOT / "images/test"
LBL_DIR_TRAIN = DATASET_ROOT / "labels/train"
LBL_DIR_VAL   = DATASET_ROOT / "labels/val"
LBL_DIR_TEST  = DATASET_ROOT / "labels/test"
for p in [IMG_DIR_TRAIN, IMG_DIR_VAL, IMG_DIR_TEST,
          LBL_DIR_TRAIN, LBL_DIR_VAL, LBL_DIR_TEST]:
    p.mkdir(parents=True, exist_ok=True)

def choose_split(global_idx: int) -> str:
    """每 10 幀：前 8 幀 train，第 9 幀 val，第 10 幀 test（均勻分布全片）"""
    pos_in_block = global_idx % 10
    if pos_in_block < 8:
        return "train"
    elif pos_in_block == 8:
        return "val"
    else:
        return "test"

DATASET_YAML = DATASET_ROOT / "dataset.yaml"

# ===============================================================
# 工具函式
# ===============================================================
def normalize_mask(arr: np.ndarray) -> np.ndarray:
    m = np.asarray(arr)
    m = np.squeeze(m)
    if m.ndim == 3:
        if m.shape[0] in (1,) or (m.shape[0] <= m.shape[-1]):
            m = m[0]
        else:
            m = m[..., 0]
    return (m > 0)

def try_gdino_on_image(image_pil: Image.Image, text: str, box_th: float, text_th: float):
    """回傳 boxes(float32 Nx4 xyxy) 與 scores(float32 N)；若無 scores 則為 1."""
    inputs = processor(images=image_pil, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids,
        box_threshold=box_th, text_threshold=text_th,
        target_sizes=[image_pil.size[::-1]],
    )
    if not results:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    r = results[0]
    boxes = r["boxes"].detach().cpu().numpy()
    scores = r.get("scores", None)
    if scores is None:
        scores = np.ones((boxes.shape[0],), dtype=np.float32)
    else:
        scores = scores.detach().cpu().numpy().astype(np.float32)
    return boxes, scores

def find_init_by_searching_first_k_frames(frame_paths, k=10):
    """在前 k 幀內找第一個可用偵測；取分數最高的 box。"""
    best = None
    for local_idx in range(min(k, len(frame_paths))):
        pil_img = Image.open(frame_paths[local_idx]).convert("RGB")
        for (bth, tth) in THRESHOLDS:
            boxes, scores = try_gdino_on_image(pil_img, GDINO_TEXT, bth, tth)
            if boxes.shape[0] > 0:
                j = int(np.argmax(scores))
                if scores[j] >= DET_CONF_MIN:
                    best = (local_idx, boxes[j:j+1], float(scores[j]))
                    break
        if best is not None:
            break
    if best is None:
        return None, None
    return best[0], best[1]

def mask_to_normalized_polygon(msk: np.ndarray):
    """mask(bool HxW) -> YOLO normalized polygon（只取最大外輪廓）"""
    m = (msk.astype(np.uint8) * 255)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []
    cnt = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon=0.002 * peri, closed=True)
    approx = approx.reshape(-1, 2)
    H, W = msk.shape[:2]
    poly = []
    for (x, y) in approx:
        poly.append(x / W)
        poly.append(y / H)
    return poly if len(poly) >= 6 else []

def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union + 1e-9)

def logits_to_conf_and_bin(logits_np: np.ndarray):
    """從 logits 計算：機率、二值、內部平均機率、面積占比"""
    proba = 1.0 / (1.0 + np.exp(-logits_np))
    m_bin = (proba > 0.5)
    p_inside = proba[m_bin].mean() if m_bin.any() else 0.0
    area_frac = float(m_bin.mean())
    return proba, m_bin, p_inside, area_frac

# ===============================================================
# 主流程
# ===============================================================
frame_idx_global = 0
prev_segment_mask = None
TARGET_CLASS_ID = 0

# 追蹤健壯化狀態
lost_cnt = 0
found_cnt = 0
prev_stable_mask = None   # 最近一次可信的 mask（做 IoU 比對）

autocast_ctx = torch.autocast(device_type=device, dtype=amp_dtype) if device == "cuda" else nullcontext()
with autocast_ctx:
    for seg_id in range(segment_count):
        start = seg_id * SEGMENT_SIZE
        end = min((seg_id + 1) * SEGMENT_SIZE, total_frames)
        print(f"\n--- Segment {seg_id+1}/{segment_count} | Frames {start}-{end-1} ---")

        # 暫存段 frame
        for f in tmp_dir.glob("*.jpg"):
            f.unlink()
        for i in range(start, end):
            cv2.imwrite(str(tmp_dir / f"{i-start:05d}.jpg"), all_frames[i])
        frame_paths = sorted(tmp_dir.glob("*.jpg"), key=lambda p: int(p.stem))
        inference_state = video_predictor.init_state(video_path=str(tmp_dir))

        # 初始化：若上一段遺失，就重新搜尋；若上一段穩定就用 mask 接力
        if prev_segment_mask is None:
            init_idx, boxes = find_init_by_searching_first_k_frames(frame_paths, k=INIT_SEARCH_FRAMES)
            if init_idx is not None:
                pil_img = Image.open(frame_paths[init_idx]).convert("RGB")
                image_predictor.set_image(np.array(pil_img))
                masks_img, _, _ = image_predictor.predict(
                    point_coords=None, point_labels=None,
                    box=boxes, multimask_output=False
                )
                if masks_img is not None and masks_img.ndim == 4:
                    masks_img = masks_img.squeeze(1)
                m0 = normalize_mask(masks_img[0])
                video_predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=int(init_idx),
                    obj_id=1,
                    mask=m0
                )
                prev_segment_mask = m0
                prev_stable_mask = m0
                lost_cnt = 0
                found_cnt = FOUND_PATIENCE
            else:
                print("[warn] No detections found in initial search for this segment.")
                prev_segment_mask = None
                prev_stable_mask = None
                lost_cnt = LOST_PATIENCE  # 視為遺失狀態開始
                found_cnt = 0
        else:
            # 只有在「非遺失狀態」才 carry mask
            video_predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=0, obj_id=1,
                mask=normalize_mask(prev_segment_mask)
            )

        # 推進
        video_segments = {}
        for fidx, obj_ids, mask_logits in video_predictor.propagate_in_video(inference_state):
            segs = {}
            global_fidx = start + int(fidx)

            # 當前預設為「無物件」，僅在通過檢核時才交付 segs[1]
            current_mask_bin = None
            current_is_confident = False

            for i, obj_id in enumerate(obj_ids):
                if obj_id != 1:
                    continue
                logits_np = mask_logits[i].detach().cpu().numpy()
                _, m_bin, p_inside, area_frac = logits_to_conf_and_bin(logits_np)
                # 信心＋面積雙閥門
                current_is_confident = (p_inside >= MASK_CONF_MIN) and (AREA_FRAC_MIN <= area_frac <= AREA_FRAC_MAX)
                if current_is_confident:
                    current_mask_bin = normalize_mask(m_bin)
                break  # 單一物件

            # 低信心就先當作「無」；高信心累計 found_cnt
            if current_is_confident:
                found_cnt += 1
                lost_cnt = 0
                if found_cnt >= FOUND_PATIENCE:
                    segs[1] = current_mask_bin
                    prev_stable_mask = current_mask_bin
                    prev_segment_mask = current_mask_bin
            else:
                found_cnt = 0
                lost_cnt += 1
                # 不直接補洞，先視為無物件；根據策略判斷是否要 recheck

            # 週期性或低信心才 re-detect
            need_recheck = (not current_is_confident) or (global_fidx % RECHECK_EVERY == 0)
            if need_recheck:
                pil_img = Image.open(frame_paths[int(fidx)]).convert("RGB")
                best_m = None
                best_score = -1.0
                for (bth, tth) in THRESHOLDS:
                    boxes, scores = try_gdino_on_image(pil_img, GDINO_TEXT, bth, tth)
                    if boxes.shape[0] == 0:
                        continue
                    j = int(np.argmax(scores))
                    if scores[j] < DET_CONF_MIN:
                        continue
                    image_predictor.set_image(np.array(pil_img))
                    mimg, _, _ = image_predictor.predict(
                        point_coords=None, point_labels=None,
                        box=boxes[j:j+1], multimask_output=False
                    )
                    if mimg is not None and mimg.ndim == 4:
                        mimg = mimg.squeeze(1)
                    cand = normalize_mask(mimg[0])
                    # 合理面積檢核
                    area2 = cand.mean()
                    if not (AREA_FRAC_MIN <= area2 <= AREA_FRAC_MAX):
                        continue
                    # 與歷史穩定 mask 做 IoU 檢核（若沒有穩定 mask，就略過 IoU）
                    iou_ok = True if prev_stable_mask is None else (mask_iou(prev_stable_mask, cand) >= IOU_MIN)
                    if iou_ok and scores[j] > best_score:
                        best_m = cand
                        best_score = float(scores[j])

                if best_m is not None:
                    segs[1] = best_m
                    prev_stable_mask = best_m
                    prev_segment_mask = best_m
                    lost_cnt = 0
                    found_cnt = FOUND_PATIENCE  # 直接視為穩定

            # 若連續 LOST_PATIENCE 幀低信心→宣告遺失，清掉 carry
            if lost_cnt >= LOST_PATIENCE:
                prev_segment_mask = None   # 不再 carry
                prev_stable_mask = None

            video_segments[int(fidx)] = segs

        if device == "cuda":
            torch.cuda.empty_cache()

        # 下一段初始化：只有在「非遺失且有穩定 mask」才接力
        if len(video_segments) > 0:
            last_fidx = max(video_segments.keys())
            last_segs = video_segments[last_fidx]
            if 1 in last_segs:
                prev_segment_mask = normalize_mask(last_segs[1])
                prev_stable_mask = prev_segment_mask
                lost_cnt = 0
                found_cnt = FOUND_PATIENCE
            else:
                prev_segment_mask = None

        # ===============================================================
        # 寫出 YOLOv11-seg dataset（單類 egg）
        # ===============================================================
        for fidx, segs in sorted(video_segments.items()):
            img_np = cv2.imread(str(frame_paths[fidx]))
            if img_np is None:
                continue

            global_fidx = start + int(fidx)
            split = choose_split(global_fidx)
            if split == "train":
                IMG_DIR, LBL_DIR = IMG_DIR_TRAIN, LBL_DIR_TRAIN
            elif split == "val":
                IMG_DIR, LBL_DIR = IMG_DIR_VAL, LBL_DIR_VAL
            else:
                IMG_DIR, LBL_DIR = IMG_DIR_TEST, LBL_DIR_TEST

            base = f"frame_{global_fidx:06d}"
            img_out_path = IMG_DIR / f"{base}.jpg"
            lbl_out_path = LBL_DIR / f"{base}.txt"

            # 原始幀存入 images（不疊可視化）
            cv2.imwrite(str(img_out_path), all_frames[global_fidx])

            # label：若此幀通過檢核才輸出 polygon；否則留空檔（或不寫檔亦可）
            lines = []
            if 1 in segs:
                poly = mask_to_normalized_polygon(segs[1])
                if poly:
                    line = " ".join([str(TARGET_CLASS_ID)] + [f"{v:.6f}" for v in poly])
                    lines.append(line)
            with open(lbl_out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

            # 可視化（選配）
            if 1 in segs:
                masks_np = np.stack([normalize_mask(segs[1])], axis=0).astype(bool)
                dets = sv.Detections(
                    xyxy=sv.mask_to_xyxy(masks_np),
                    mask=masks_np,
                    class_id=np.array([1], dtype=np.int32)
                )
                mask_ann = sv.MaskAnnotator(opacity=0.45)
                out = mask_ann.annotate(scene=img_np.copy(), detections=dets)
                box_ann = sv.BoxAnnotator()
                out = box_ann.annotate(out, dets)
                lbl_ann = sv.LabelAnnotator()
                out = lbl_ann.annotate(out, dets, labels=[TARGET_YOLO_NAME])
                cv2.imwrite(str(OUTPUT_DIR / f"frame_{frame_idx_global:05d}.jpg"), out)
            else:
                cv2.imwrite(str(OUTPUT_DIR / f"frame_{frame_idx_global:05d}.jpg"), img_np)

            frame_idx_global += 1

        print(f"Segment {seg_id+1} done, exported {len(video_segments)} label frames.")
        for f in tmp_dir.glob("*.jpg"):
            f.unlink()

# ===============================================================
# 結尾：合併影片 + dataset.yaml
# ===============================================================
print("\nCombining all annotated frames into final video...")
create_video_from_images(OUTPUT_DIR, OUTPUT_VIDEO_PATH)
print(f"✅ Done! Video saved to {OUTPUT_VIDEO_PATH}")

DATASET_YAML.write_text(
    "path: dataset\n"
    "train: images/train\n"
    "val: images/val\n"
    "test: images/test\n"
    "names:\n"
    f"  0: {TARGET_YOLO_NAME}\n",
    encoding="utf-8"
)

print("✅ YOLOv11-seg dataset exported to:", DATASET_ROOT.resolve())
print("👉 訓練示例：yolo task=segment mode=train model=yolo11n-seg.pt data=dataset_egg_robust/dataset.yaml imgsz=640 epochs=50 batch=16")
