import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from pathlib import Path

HERE        = Path(__file__).parent
MODEL_PATH  = str(HERE / 'models' / 'physique_classifier_3class.h5')
TASK_PATH   = str(HERE / 'pose_landmarker_lite.task')
IMG_SIZE    = (224, 224)

# Keras assigns class indices alphabetically
CLASS_LABELS    = ['advanced', 'beginner', 'intermediate']
CLASS_MIDPOINTS = {'advanced': 95, 'beginner': 24, 'intermediate': 71}

model = tf.keras.models.load_model(MODEL_PATH)

base_options = mp_python.BaseOptions(model_asset_path=TASK_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = vision.PoseLandmarker.create_from_options(options)

LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12

# Crop geometry — must stay identical to normalize_dataset.py and App.jsx
UPPER_BODY_LANDMARKS = [11, 12, 13, 14, 15, 16, 23, 24]
PAD_TOP    = 0.15
PAD_BOTTOM = 0.05
PAD_SIDES  = 0.15

def landmark_crop(rgb, pose_landmarks):
    """Crop upper body using landmark bounding box geometry (matches normalize_dataset.py)."""
    h, w = rgb.shape[:2]
    lm = pose_landmarks[0]
    visible = [i for i in UPPER_BODY_LANDMARKS if lm[i].visibility >= 0.3]
    if not visible:
        return None
    xs = [lm[i].x for i in visible]
    ys = [lm[i].y for i in visible]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    crop_w = x_max - x_min
    crop_h = y_max - y_min
    x_min -= crop_w * PAD_SIDES
    x_max += crop_w * PAD_SIDES
    y_min -= crop_h * PAD_TOP
    y_max += crop_h * PAD_BOTTOM
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    half = max(x_max - x_min, y_max - y_min) / 2
    x_min, x_max = max(0.0, cx - half), min(1.0, cx + half)
    y_min, y_max = max(0.0, cy - half), min(1.0, cy + half)
    px1, px2 = int(x_min * w), int(x_max * w)
    py1, py2 = int(y_min * h), int(y_max * h)
    crop = rgb[py1:py2, px1:px2]
    return crop if crop.size > 0 else None

# Body-only connections — no face (0-10) or finger detail (17-22)
BODY_CONNECTIONS = [
    (11,12),                          # shoulders
    (11,13),(13,15),                  # left arm
    (12,14),(14,16),                  # right arm
    (11,23),(12,24),(23,24),          # torso
    (23,25),(25,27),(27,29),(27,31),(29,31),  # left leg
    (24,26),(26,28),(28,30),(28,32),(30,32),  # right leg
]
BODY_LANDMARKS = set(range(11, 17)) | set(range(23, 33))  # shoulders→wrists, hips→feet

def draw_pose(frame, pose_landmarks):
    """Overlay body skeleton (no face or fingers) on frame in-place."""
    if not pose_landmarks:
        return
    h, w = frame.shape[:2]
    lms = pose_landmarks[0]
    for a, b in BODY_CONNECTIONS:
        if lms[a].visibility > 0.5 and lms[b].visibility > 0.5:
            p1 = (int(lms[a].x * w), int(lms[a].y * h))
            p2 = (int(lms[b].x * w), int(lms[b].y * h))
            cv2.line(frame, p1, p2, (255, 255, 0), 2)
    for i in BODY_LANDMARKS:
        if lms[i].visibility > 0.5:
            cv2.circle(frame, (int(lms[i].x * w), int(lms[i].y * h)), 5, (0, 255, 255), -1)

def compute_score(probs):
    return round(sum(probs[i] * CLASS_MIDPOINTS[CLASS_LABELS[i]] for i in range(3)))

def upper_body_visible(pose_landmarks):
    if not pose_landmarks:
        return False
    lm = pose_landmarks[0]
    return lm[LEFT_SHOULDER].visibility > 0.6 and lm[RIGHT_SHOULDER].visibility > 0.6

cap = cv2.VideoCapture(0)
print("Physique Analyser — press Q to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result   = landmarker.detect(mp_image)

    draw_pose(frame, result.pose_landmarks)

    if upper_body_visible(result.pose_landmarks):
        crop = landmark_crop(rgb, result.pose_landmarks)
        if crop is None:
            crop = rgb
        crop_resized = cv2.resize(crop, IMG_SIZE)
        img   = preprocess_input(crop_resized.astype('float32'))
        probs = model.predict(np.expand_dims(img, axis=0), verbose=0)[0]

        label = CLASS_LABELS[np.argmax(probs)]
        score = compute_score(probs)

        cv2.putText(frame, f'{label.upper()}  {score}/100',
                    (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 230, 0), 2)

        for i, cls in enumerate(CLASS_LABELS):
            cv2.putText(frame, f'{cls}: {probs[i]:.2f}',
                        (30, 110 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
    else:
        cv2.putText(frame, 'Position upper body in frame',
                    (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)

    cv2.imshow('Physique Analyser', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()
