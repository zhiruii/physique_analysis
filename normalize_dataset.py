"""
normalize_dataset.py

Reads dataset/{beginner,intermediate,advanced}/
Runs MediaPipe Pose on each image to detect shoulder + hip landmarks.
Crops to the upper body region (shoulders to hips) with padding.
Saves normalized crops to dataset_normalized/{beginner,intermediate,advanced}/

Images where MediaPipe fails to detect landmarks are skipped and reported.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from pathlib import Path

INPUT_DIR  = 'dataset'
OUTPUT_DIR = 'dataset_normalized'
TASK_PATH  = 'backend/pose_landmarker_lite.task'
CLASSES    = ['beginner', 'intermediate', 'advanced']

# Padding as a fraction of the crop height/width added on each side
PAD_TOP    = 0.15
PAD_BOTTOM = 0.05
PAD_SIDES  = 0.15

# Upper body landmark indices — includes raised arms for double bicep poses
UPPER_BODY_LANDMARKS = [11, 12, 13, 14, 15, 16, 23, 24]  # shoulders, elbows, wrists, hips
LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12
LEFT_HIP       = 23
RIGHT_HIP      = 24

base_options = mp_python.BaseOptions(model_asset_path=TASK_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_poses=1,
    min_pose_detection_confidence=0.3,
    min_pose_presence_confidence=0.3,
    min_tracking_confidence=0.3,
)
landmarker = vision.PoseLandmarker.create_from_options(options)

passed = 0
failed = 0
failed_files = []

for cls in CLASSES:
    in_dir  = Path(INPUT_DIR) / cls
    out_dir = Path(OUTPUT_DIR) / cls
    out_dir.mkdir(parents=True, exist_ok=True)

    image_files = list(in_dir.glob('*.png')) + list(in_dir.glob('*.jpg')) + list(in_dir.glob('*.jpeg'))

    for img_path in image_files:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f'  [SKIP] Could not read: {img_path.name}')
            failed += 1
            failed_files.append(str(img_path))
            continue

        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = landmarker.detect(mp_image)

        if not result.pose_landmarks:
            print(f'  [FAIL] No landmarks: {cls}/{img_path.name}')
            failed += 1
            failed_files.append(str(img_path))
            continue

        lm = result.pose_landmarks[0]

        # Require shoulders and hips to be visible (core anchors)
        core = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]
        if any(lm[i].visibility < 0.3 for i in core):
            print(f'  [FAIL] Low visibility: {cls}/{img_path.name}')
            failed += 1
            failed_files.append(str(img_path))
            continue

        # Bounding box from all upper body landmarks with reasonable visibility
        visible = [i for i in UPPER_BODY_LANDMARKS if lm[i].visibility >= 0.3]
        xs = [lm[i].x for i in visible]
        ys = [lm[i].y for i in visible]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        crop_w = x_max - x_min
        crop_h = y_max - y_min

        # Add padding
        x_min = x_min - crop_w * PAD_SIDES
        x_max = x_max + crop_w * PAD_SIDES
        y_min = y_min - crop_h * PAD_TOP
        y_max = y_max + crop_h * PAD_BOTTOM

        # Force square crop so side poses don't get clipped
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        half = max(x_max - x_min, y_max - y_min) / 2
        x_min = cx - half
        x_max = cx + half
        y_min = cy - half
        y_max = cy + half

        # Clamp to image bounds
        x_min = max(0.0, x_min)
        x_max = min(1.0, x_max)
        y_min = max(0.0, y_min)
        y_max = min(1.0, y_max)

        px1 = int(x_min * w)
        px2 = int(x_max * w)
        py1 = int(y_min * h)
        py2 = int(y_max * h)

        crop = img_bgr[py1:py2, px1:px2]
        if crop.size == 0:
            print(f'  [FAIL] Empty crop: {cls}/{img_path.name}')
            failed += 1
            failed_files.append(str(img_path))
            continue

        out_path = out_dir / img_path.name
        cv2.imwrite(str(out_path), crop)
        print(f'  [OK]   {cls}/{img_path.name}')
        passed += 1

landmarker.close()

print(f'\n--- Summary ---')
print(f'Passed: {passed}')
print(f'Failed: {failed}')
if failed_files:
    print(f'\nFailed files:')
    for f in failed_files:
        print(f'  {f}')
print(f'\nNormalized dataset saved to: {OUTPUT_DIR}/')
