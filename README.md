# Physique analyser

A web app that analyses physique development in real-time using your device camera. MediaPipe detects upper body landmarks, a fine-tuned MobileNetV2 classifies physique into beginner / intermediate / advanced, and a weighted score from 0–100 is returned live.

Live site → https://pheno-physique.vercel.app

---

## Client-side inference

All inference runs client-side via ONNX Runtime Web. No images are ever transmitted, stored, or logged. The server serves static files only. Privacy compliance is architectural.

---

## ML Pipeline

Training data consists of AI-generated images supplemented with real photos collected from friends and public sources, across 5 bodybuilding poses labeled into 3 classes — 281 images total.

MediaPipe Pose extracts upper body landmarks from each image. A square crop is derived from the landmark bounding box, centred on the upper body. This ensures the model sees the same framing regardless of how far the user stands from the camera. The same crop logic runs in the browser during live inference, ensuring consistent input between training and inference.

MobileNetV2 pretrained on ImageNet is fine-tuned via transfer learning in PyTorch. Feature layers are frozen; only the classifier head is trained. The trained model is exported to ONNX and loaded in a Web Worker in the browser.

---

## Scoring

The model outputs a softmax probability across 3 classes, a weighted score is computed:

```
score = (P_beginner × 24) + (P_intermediate × 71) + (P_advanced × 95)
```

This produces a continuous 0–100 scale that reflects confidence distribution rather than snapping to a single class boundary.

---

## Known Limitations

The majority of training data is AI-generated. Real webcam footage has different lighting and texture characteristics from synthetic images, which is the primary source of real-world inaccuracy. Reliable classification is limited to the five trained poses.

87.5% on the validation set, though this overstates true generalisation as the same split was used for early stopping to determine best epoch. The next meaningful improvement requires a larger real-photo dataset.

---

## Tech Stack

- Training: PyTorch, torchvision, MediaPipe (Python)
- Browser inference: ONNX Runtime Web, MediaPipe Pose (JS)
- Frontend: React, Vite
- Deployment: Vercel
