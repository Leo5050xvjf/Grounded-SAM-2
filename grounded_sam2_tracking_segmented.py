# import os
# import cv2
# import torch
# import numpy as np
# import supervision as sv
# from pathlib import Path
# from tqdm import tqdm
# from PIL import Image

# from sam2.build_sam import build_sam2_video_predictor, build_sam2
# from sam2.sam2_image_predictor import SAM2ImagePredictor 
# from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
# from utils.video_utils import create_video_from_images

# # ======================= 可調參 ==========================
# MODEL_ID = "IDEA-Research/grounding-dino-base"  # 比 tiny 穩
# VIDEO_PATH = "./assets/hippopotamus.mp4"
# TEXT_PROMPT = "hippopotamus"
# OUTPUT_VIDEO_PATH = "./hippopotamus_tracking_demo.mp4"
# SOURCE_VIDEO_FRAME_DIR = "./custom_video_frames"
# SAVE_TRACKING_RESULTS_DIR = "./tracking_results"

# # 初始化搜尋：若第 0 幀偵測不到，往前找前 K 幀
# INIT_SEARCH_FRAMES = 12
# # 閾值回退策略（由嚴到鬆）
# THRESHOLDS = [(0.40, 0.30), (0.30, 0.22), (0.25, 0.18), (0.20, 0.15)]
# # ========================================================

# device = "cuda" if torch.cuda.is_available() else "cpu"

# # 混合精度
# if device == "cuda":
#     torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
#     if torch.cuda.get_device_properties(0).major >= 8:
#         torch.backends.cuda.matmul.allow_tf32 = True
#         torch.backends.cudnn.allow_tf32 = True

# # SAM2
# sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
# model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
# video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
# sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
# image_predictor = SAM2ImagePredictor(sam2_image_model)

# # GroundingDINO
# processor = AutoProcessor.from_pretrained(MODEL_ID)
# grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device)

# # ---------- 讀影片 → 存成影格 ----------
# video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)
# print(video_info)
# frame_generator = sv.get_video_frames_generator(VIDEO_PATH, stride=1, start=0, end=None)

# source_frames = Path(SOURCE_VIDEO_FRAME_DIR)
# source_frames.mkdir(parents=True, exist_ok=True)

# with sv.ImageSink(
#     target_dir_path=source_frames, 
#     overwrite=True, 
#     image_name_pattern="{:05d}.jpg"
# ) as sink:
#     for frame in tqdm(frame_generator, desc="Saving Video Frames"):
#         sink.save_image(frame)

# # 掃 JPEG
# frame_names = sorted(
#     [p for p in os.listdir(SOURCE_VIDEO_FRAME_DIR) if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]],
#     key=lambda p: int(os.path.splitext(p)[0])
# )

# # ---------- 幫手函式 ----------
# def normalize_mask(arr: np.ndarray) -> np.ndarray:
#     """統一 mask → 2D bool"""
#     m = np.asarray(arr)
#     m = np.squeeze(m)
#     if m.ndim == 3:
#         if m.shape[0] in (1,) or (m.shape[0] <= m.shape[-1]):
#             m = m[0]
#         else:
#             m = m[..., 0]
#     if m.ndim != 2:
#         raise ValueError(f"Mask not 2D: {arr.shape} -> {m.shape}")
#     return (m > 0)

# @torch.no_grad()
# def run_gdino(image_pil: Image.Image, text: str, box_th: float, text_th: float):
#     inputs = processor(images=image_pil, text=text, return_tensors="pt").to(device)
#     outputs = grounding_model(**inputs)
#     results = processor.post_process_grounded_object_detection(
#         outputs, inputs.input_ids,
#         box_threshold=box_th, text_threshold=text_th,
#         target_sizes=[image_pil.size[::-1]]
#     )
#     if len(results) == 0:
#         return np.zeros((0,4), dtype=np.float32), []
#     return results[0]["boxes"].detach().cpu().numpy(), results[0]["labels"]

# def find_init(frame_paths, k, thresholds):
#     """在前 k 幀、依 thresholds 搜尋第一個可用的偵測"""
#     for local_idx in range(min(k, len(frame_paths))):
#         pil = Image.open(os.path.join(SOURCE_VIDEO_FRAME_DIR, frame_paths[local_idx])).convert("RGB")
#         for (bth, tth) in thresholds:
#             boxes, labels = run_gdino(pil, TEXT_PROMPT, bth, tth)
#             if boxes.shape[0] > 0:
#                 return local_idx, boxes, labels
#     return None, None, None

# # ---------- 初始化 video predictor ----------
# inference_state = video_predictor.init_state(video_path=SOURCE_VIDEO_FRAME_DIR)

# # 在前 K 幀找第一個可用偵測
# init_idx, input_boxes, class_names = find_init(frame_names, INIT_SEARCH_FRAMES, THRESHOLDS)

# if init_idx is None:
#     print("[WARN] No detections found in the first {} frames under all thresholds.".format(INIT_SEARCH_FRAMES))
#     print("[INFO] Will skip SAM2 init and only dump original frames (no masks).")
#     video_segments = {}
# else:
#     print(f"[INFO] Initialize at local frame {init_idx} with {len(input_boxes)} boxes.")
#     # 以「mask 提示」初始化
#     img_path = os.path.join(SOURCE_VIDEO_FRAME_DIR, frame_names[init_idx])
#     image = Image.open(img_path).convert("RGB")
#     image_predictor.set_image(np.array(image))

#     masks_img, scores, logits = image_predictor.predict(
#         point_coords=None, point_labels=None,
#         box=input_boxes, multimask_output=False
#     )
#     if masks_img.ndim == 4:
#         masks_img = masks_img.squeeze(1)  # (N,H,W)

#     OBJECTS = class_names

#     for object_id, (label, mask) in enumerate(zip(OBJECTS, masks_img), start=1):
#         mask_bool = normalize_mask(mask)
#         _, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
#             inference_state=inference_state,
#             frame_idx=int(init_idx),
#             obj_id=object_id,
#             mask=mask_bool,
#         )

#     # ---------- Propagate ----------
#     video_segments = {}
#     for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
#         # logits → bool mask
#         segs = {}
#         for i, oid in enumerate(out_obj_ids):
#             m = (out_mask_logits[i] > 0.0).detach().cpu().numpy()
#             segs[int(oid)] = normalize_mask(m)
#         video_segments[int(out_frame_idx)] = segs

# # ---------- 視覺化與儲存 ----------
# os.makedirs(SAVE_TRACKING_RESULTS_DIR, exist_ok=True)
# ID_TO_OBJECTS = {i: obj for i, obj in enumerate(class_names if init_idx is not None else [], start=1)}

# for frame_idx in range(len(frame_names)):
#     img = cv2.imread(os.path.join(SOURCE_VIDEO_FRAME_DIR, frame_names[frame_idx]))

#     if init_idx is None or frame_idx not in video_segments or len(video_segments[frame_idx]) == 0:
#         # 無結果就輸出原圖，保持幀序連續
#         cv2.imwrite(os.path.join(SAVE_TRACKING_RESULTS_DIR, f"annotated_frame_{frame_idx:05d}.jpg"), img)
#         continue

#     segments = video_segments[frame_idx]
#     object_ids = list(segments.keys())
#     masks = np.stack([normalize_mask(segments[i]) for i in object_ids], axis=0).astype(bool)

#     detections = sv.Detections(
#         xyxy=sv.mask_to_xyxy(masks),
#         mask=masks,
#         class_id=np.array(object_ids, dtype=np.int32),
#     )

#     mask_annotator = sv.MaskAnnotator(opacity=0.45)
#     annotated = mask_annotator.annotate(scene=img.copy(), detections=detections)

#     box_annotator = sv.BoxAnnotator()
#     annotated = box_annotator.annotate(scene=annotated, detections=detections)

#     label_annotator = sv.LabelAnnotator()
#     labels = [ID_TO_OBJECTS.get(i, str(i)) for i in object_ids]
#     annotated = label_annotator.annotate(annotated, detections=detections, labels=labels)

#     cv2.imwrite(os.path.join(SAVE_TRACKING_RESULTS_DIR, f"annotated_frame_{frame_idx:05d}.jpg"), annotated)

# # ---------- 轉回影片 ----------
# create_video_from_images(SAVE_TRACKING_RESULTS_DIR, OUTPUT_VIDEO_PATH)
# print(f"✅ Done! Video saved to {OUTPUT_VIDEO_PATH}")


"""
Segmented Grounded-SAM2 video tracking (v4.0, MASK carry-over + robust init)
- 分段處理避免爆記憶體
- 首段支援「多幀搜尋 + 閾值回退」找到第一個可用偵測
- 段與段之間用上一段最後一幀的「mask 提示」接力（不重跑 Grounding DINO）
- 全域 obj_id 與 class label 穩定
- 確保 Detections.mask 為 bool；先畫 Mask 後畫 Box/Label
"""
import os, cv2, torch, numpy as np
from pathlib import Path
from PIL import Image
from contextlib import nullcontext
import supervision as sv
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from utils.video_utils import create_video_from_images

# ======================== 使用者可調參 =========================
MODEL_ID = "IDEA-Research/grounding-dino-base"
VIDEO_PATH = "./assets/kitchen.mp4"
TEXT_PROMPT = "egg white. wok with wooden handle. wok ladle. metal bowl."

SEGMENT_SIZE = 200                                # 每次處理的影格數
INIT_SEARCH_FRAMES = 10                           # 首段最多往前 K 幀嘗試找到可用偵測
THRESHOLDS = [                                    # 偵測不到時的閾值回退策略（由嚴到鬆）
    (0.40, 0.30),
    (0.30, 0.22),
    (0.25, 0.18),
    (0.20, 0.15),
]

OUTPUT_DIR = Path("./seg_tracking_results")       # 分段輸出影格
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_VIDEO_PATH = "./outputs/kitchen_segmented_tracking_maskcarry.mp4"

SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
# ===============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

# 混合精度設定
amp_dtype = torch.bfloat16 if device == "cuda" else torch.float32
if device == "cuda" and torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# 建 SAM2 影片/影像 predictor
video_predictor = build_sam2_video_predictor(SAM2_CFG, SAM2_CHECKPOINT)
sam2_image_model = build_sam2(SAM2_CFG, SAM2_CHECKPOINT)
image_predictor = SAM2ImagePredictor(sam2_image_model)

# 建 Grounding DINO
processor = AutoProcessor.from_pretrained(MODEL_ID)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device)

# 影片讀取
video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)
print(video_info)
frame_gen = sv.get_video_frames_generator(VIDEO_PATH, stride=1, start=0, end=None)
all_frames = [f for f in frame_gen]
total_frames = len(all_frames)
print(f"Total frames: {total_frames}")

# 暫存影像資料夾（每段用）
tmp_dir = Path("./tmp_frames")
tmp_dir.mkdir(exist_ok=True)

segment_count = (total_frames + SEGMENT_SIZE - 1) // SEGMENT_SIZE
print(f"Processing in {segment_count} segments...")

# --------------- 跨段狀態：用 mask 接力 ----------------
# 以 mask 做接力：{obj_id: {"mask": np.ndarray(H,W,bool)}}
prev_segment_prompts = None
global_id_to_cls = {}    # 穩定的 obj_id -> class label
next_obj_id = 1          # 全域遞增 id
# -------------------------------------------------------

# ---------------- 工具：標準化/偵測 --------------------
def normalize_mask(arr: np.ndarray) -> np.ndarray:
    """
    將 mask 統一成 2D (H, W) bool
    支援輸入：(H,W)、(H,W,1)、(1,H,W)、(K,H,W)、(H,W,K)；多通道取第一通道
    """
    m = np.asarray(arr)
    m = np.squeeze(m)
    if m.ndim == 3:
        if m.shape[0] in (1,) or (m.shape[0] <= m.shape[-1]):
            m = m[0]
        else:
            m = m[..., 0]
    if m.ndim != 2:
        raise ValueError(f"Mask shape not reducible to 2D: got {arr.shape} -> {m.shape}")
    return (m > 0)

def try_gdino_on_image(image_pil: Image.Image, texts: str, box_th: float, text_th: float):
    inputs = processor(images=image_pil, text=texts, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids,
        box_threshold=box_th, text_threshold=text_th,
        target_sizes=[image_pil.size[::-1]],
    )
    if len(results) == 0:
        return np.zeros((0, 4), dtype=np.float32), []
    boxes = results[0]["boxes"].detach().cpu().numpy()
    labels = results[0]["labels"]
    return boxes, labels

def find_init_by_searching_first_k_frames(frame_paths, k=10):
    """
    在前 k 幀內嘗試找到第一個有偵測的影格與對應 boxes/labels。
    各幀內套用 THRESHOLDS 從嚴到鬆回退。
    回傳: (init_idx, boxes, labels)；若失敗，回傳 (None, None, None)
    """
    for local_idx in range(min(k, len(frame_paths))):
        pil_img = Image.open(frame_paths[local_idx]).convert("RGB")
        for (bth, tth) in THRESHOLDS:
            boxes, labels = try_gdino_on_image(pil_img, TEXT_PROMPT, bth, tth)
            if boxes.shape[0] > 0:
                return local_idx, boxes, labels
    return None, None, None
# -------------------------------------------------------

frame_idx_global = 0

autocast_ctx = torch.autocast(device_type=device, dtype=amp_dtype) if device == "cuda" else nullcontext()
with autocast_ctx:
    for seg_id in range(segment_count):
        start = seg_id * SEGMENT_SIZE
        end = min((seg_id + 1) * SEGMENT_SIZE, total_frames)
        print(f"\n--- Segment {seg_id+1}/{segment_count} | Frames {start}-{end-1} ---")

        # 清空暫存夾
        for f in tmp_dir.glob("*.jpg"):
            f.unlink()

        # 將本段 frame 存到暫存夾
        for i in range(start, end):
            cv2.imwrite(str(tmp_dir / f"{i-start:05d}.jpg"), all_frames[i])
        frame_paths = sorted(tmp_dir.glob("*.jpg"), key=lambda p: int(p.stem))

        # 初始化本段的 video predictor 狀態
        inference_state = video_predictor.init_state(video_path=str(tmp_dir))

        # ====== 初始化本段：首段用 DINO+mask，後續用上一段最後一幀的 mask ======
        if prev_segment_prompts is None:
            # 在前 INIT_SEARCH_FRAMES 幀內尋找第一個可用偵測
            init_idx, boxes, labels = find_init_by_searching_first_k_frames(frame_paths, k=INIT_SEARCH_FRAMES)
            if init_idx is None:
                print("[warn] No detections in the first segment after multi-frame search; this segment will be empty.")
            else:
                # 將該幀的影像塞進 SAM2 image predictor 取 mask，並以 mask 初始化
                pil_img = Image.open(frame_paths[init_idx]).convert("RGB")
                image_predictor.set_image(np.array(pil_img))
                masks_img, _, _ = image_predictor.predict(
                    point_coords=None, point_labels=None,
                    box=boxes, multimask_output=False
                )
                if masks_img is not None and masks_img.ndim == 4:
                    masks_img = masks_img.squeeze(1)  # (N,H,W)

                # 逐物件用 mask 初始化；注意 frame_idx=init_idx（不是 0）
                for i, cls in enumerate(labels):
                    obj_id = next_obj_id
                    next_obj_id += 1
                    global_id_to_cls[obj_id] = cls

                    m0 = normalize_mask(masks_img[i])
                    video_predictor.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=int(init_idx),
                        obj_id=obj_id,
                        mask=m0.astype(bool)
                    )
        else:
            # 後續各段：沿用上一段最後一幀的 mask，放到本段第 0 幀
            for obj_id, prom in prev_segment_prompts.items():
                m = prom.get("mask", None)
                if m is None:
                    continue
                m = normalize_mask(m).astype(bool)
                video_predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=obj_id,
                    mask=m
                )

        # Propagate (本段推理)
        video_segments = {}
        for fidx, obj_ids, mask_logits in video_predictor.propagate_in_video(inference_state):
            segs = {}
            for i, obj_id in enumerate(obj_ids):
                # logits → 二值 mask；保持 bool
                m = (mask_logits[i] > 0.0).detach().cpu().numpy()
                segs[obj_id] = normalize_mask(m)
            video_segments[fidx] = segs

        # 釋放顯示卡記憶體
        if device == "cuda":
            torch.cuda.empty_cache()

        # 取本段最後一幀，保存 mask 給下一段
        if len(video_segments) > 0:
            last_fidx = max(video_segments.keys())
            last_segs = video_segments[last_fidx]   # {obj_id: mask(H,W)}
            prev_segment_prompts = {
                obj_id: {"mask": normalize_mask(m)}
                for obj_id, m in last_segs.items()
            }
        else:
            prev_segment_prompts = None

        # ===== 儲存結果 (所有段共用同一輸出資料夾) =====
        for fidx, segs in video_segments.items():
            img_np = cv2.imread(str(frame_paths[fidx]))
            if img_np is None:
                continue

            obj_ids = list(segs.keys())
            if len(obj_ids) == 0:
                out_path = OUTPUT_DIR / f"frame_{frame_idx_global:05d}.jpg"
                cv2.imwrite(str(out_path), img_np)
                frame_idx_global += 1
                continue

            masks_list = [normalize_mask(segs[i]) for i in obj_ids]
            masks_np = np.stack(masks_list, axis=0).astype(bool)  # <-- 關鍵：bool

            dets = sv.Detections(
                xyxy=sv.mask_to_xyxy(masks_np),
                mask=masks_np,
                class_id=np.array(obj_ids, dtype=np.int32)
            )

            # 先畫 Mask，再畫 Box/Label
            mask_ann = sv.MaskAnnotator(opacity=0.45)
            out = mask_ann.annotate(scene=img_np.copy(), detections=dets)

            box_ann = sv.BoxAnnotator()
            out = box_ann.annotate(out, dets)

            lbl_ann = sv.LabelAnnotator()
            labels = [str(global_id_to_cls.get(i, i)) for i in obj_ids]
            out = lbl_ann.annotate(out, dets, labels=labels)

            out_path = OUTPUT_DIR / f"frame_{frame_idx_global:05d}.jpg"
            cv2.imwrite(str(out_path), out)
            frame_idx_global += 1

        print(f"Segment {seg_id+1} finished, saved {len(video_segments)} frames.")

        # 清空暫存 frame
        for f in tmp_dir.glob("*.jpg"):
            f.unlink()

# 合併所有輸出影像成最終影片
print("\nCombining all annotated frames into final video...")
create_video_from_images(OUTPUT_DIR, OUTPUT_VIDEO_PATH)
print(f"✅ Done! Video saved to {OUTPUT_VIDEO_PATH}")
