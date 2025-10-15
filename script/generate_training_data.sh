#!/bin/bash
# ===== command.sh =====
# 啟用 conda 環境並執行 Grounded-SAM2 指令

# 確保 conda 指令可用
source ~/miniconda3/etc/profile.d/conda.sh

# 啟用 gsam2 環境
conda activate gsam2

# # fried egg.
python grounded_sam2_hf_video_perframe.py   \
--video-path ./assets/kitchen.mp4 \
--text-prompt "fried egg."   \
--box-threshold 0.35 \
--text-threshold 0.3   \
--output-dir \
outputs/eggs_perframe_res   \
--write-video \
--start-frame 0

# mixing bowl.
python grounded_sam2_hf_video_perframe.py   \
--video-path ./assets/kitchen.mp4   \
--text-prompt "mixing bowl."   \
--box-threshold 0.25 \
--text-threshold 0.25   \
--max-bbox-side 160 \
--min-bbox-side 80 \
--output-dir outputs/bowl_perframe_res   \
--write-video --start-frame 0

#wok.
python grounded_sam2_hf_video_perframe.py   \
--video-path ./assets/kitchen.mp4   \
--text-prompt "metal wok."   \
--box-threshold 0.3 \
--text-threshold 0.25   \
--min-bbox-side 150 \
--output-dir outputs/wok_perframe_res   \
--write-video --start-frame 0

#metal spatula.
python grounded_sam2_hf_video_perframe.py   \
--video-path ./assets/kitchen.mp4   \
--text-prompt "metal spatula."   \
--box-threshold 0.27 \
--text-threshold 0.25   \
--max-bbox-side 220 \
--min-bbox-side 80 \
--output-dir outputs/wok_ladle_perframe_res   \
--write-video --start-frame 0

# 生成合併後的訓練數據
python script/merge_gsam2_annotations.py 
#轉換成 Ultralytics 所需的數據格式
python script/toUltralytics.py
# 僅保留4個目標物均存在的case(optional)
python script/filter_all4_yolo.py

#split data 8:1:1
python script/split_8_1_1_stride10.py \
  --src yolo_dataset_all4 \
  --src-split train \
  --dst yolo_dataset_all4_split


mv yolo_dataset_all4_split/ ../../yolov11-seg_ros/