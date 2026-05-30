import tensorflow as tf
import numpy as np
import json
import matplotlib.pyplot as plt

from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# SETTINGS

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10

dataset_path = "dataset"

# DATA PREPROCESSING

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

# TRAINING DATA

train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

# VALIDATION DATA

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# SAVE CLASS NAMES

class_names = list(train_data.class_indices.keys())

with open("models/class_names.json", "w") as f:
    json.dump(class_names, f)

print("\nClasses:")
print(class_names)

# COMPUTE CLASS WEIGHTS

weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)

class_weights = tf.constant(
    weights,
    dtype=tf.float32
)

print("\nClass Weights:")
print(weights)

# CUSTOM LOSS FUNCTION

def weighted_cce(class_weights):

    def loss(y_true, y_pred):

        cce = tf.keras.losses.categorical_crossentropy(
            y_true,
            y_pred
        )

        sample_weights = tf.reduce_sum(
            y_true * class_weights,
            axis=1
        )

        return cce * sample_weights

    return loss

# LOAD VGG16

base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze pretrained layers

for layer in base_model.layers:
    layer.trainable = False

# BUILD MODEL

model = Sequential([
    base_model,

    GlobalAveragePooling2D(),

    Dense(
        256,
        activation='relu'
    ),

    Dropout(0.5),

    Dense(
        train_data.num_classes,
        activation='softmax'
    )
])

# COMPILE MODEL

model.compile(
    optimizer=Adam(
        learning_rate=0.0001
    ),
    loss=weighted_cce(class_weights),
    metrics=['accuracy']
)

# MODEL SUMMARY

model.summary()

# EARLY STOPPING

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True
)

# TRAIN MODEL

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

# SAVE MODEL

model.save(
    "models/fruit_model.h5"
)

print("\nModel saved successfully!")

# ACCURACY GRAPH

plt.figure(figsize=(8,5))

plt.plot(
    history.history['accuracy'],
    label='Training Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.title("VGG16 Fruit Recognition Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()

plt.show()

# LOSS GRAPH

plt.figure(figsize=(8,5))

plt.plot(
    history.history['loss'],
    label='Training Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.title("VGG16 Fruit Recognition Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()

plt.show()
