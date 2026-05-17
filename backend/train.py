import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

DATASET_DIR = '../dataset_normalized'
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 16
EPOCHS      = 30

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    zoom_range=0.15,
    rotation_range=10,
    shear_range=0.1
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    DATASET_DIR, target_size=IMG_SIZE,
    batch_size=BATCH_SIZE, class_mode='categorical', subset='training'
)

val_data = val_datagen.flow_from_directory(
    DATASET_DIR, target_size=IMG_SIZE,
    batch_size=BATCH_SIZE, class_mode='categorical', subset='validation'
)

base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu')(x)
output = layers.Dense(train_data.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

os.makedirs('models', exist_ok=True)

callbacks = [
    ModelCheckpoint('models/physique_classifier_3class.h5', save_best_only=True, monitor='val_accuracy', verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True, verbose=1)
]

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks
)

print("\nClass indices:", train_data.class_indices)
print("Training complete. Model saved to models/physique_classifier_3class.h5")
