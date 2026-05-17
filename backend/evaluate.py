import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

MODEL_PATH  = 'models/physique_classifier_3class.h5'
DATASET_DIR = '../dataset'
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 16

# Alphabetical — must match training order
CLASS_LABELS = ['advanced', 'beginner', 'intermediate']

model = tf.keras.models.load_model(MODEL_PATH)

datagen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.2)

val_data = datagen.flow_from_directory(
    DATASET_DIR, target_size=IMG_SIZE,
    batch_size=BATCH_SIZE, class_mode='categorical',
    subset='validation', shuffle=False
)

print(f"\nEvaluating on {val_data.samples} validation images...\n")

probs     = model.predict(val_data, verbose=1)
y_pred    = np.argmax(probs, axis=1)
y_true    = val_data.classes

print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred, target_names=CLASS_LABELS))

print("--- Confusion Matrix ---")
print(f"{'':>14}", "  ".join(f"{c:>12}" for c in CLASS_LABELS))
cm = confusion_matrix(y_true, y_pred)
for i, row in enumerate(cm):
    print(f"{CLASS_LABELS[i]:>14}", "  ".join(f"{v:>12}" for v in row))
print()
print("Rows = actual class, Columns = predicted class")
