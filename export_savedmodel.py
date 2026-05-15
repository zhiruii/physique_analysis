"""Export Keras .h5 to TF SavedModel format (needed for Node.js TF.js conversion)."""
import tensorflow as tf
from pathlib import Path

HERE       = Path(__file__).parent
MODEL_PATH = HERE / "backend" / "models" / "physique_classifier_3class.h5"
SAVE_PATH  = HERE / "backend" / "models" / "saved_model"

print(f"Loading {MODEL_PATH} ...")
model = tf.keras.models.load_model(MODEL_PATH)

print(f"Exporting to {SAVE_PATH} ...")
model.export(str(SAVE_PATH))
print("Done.")
