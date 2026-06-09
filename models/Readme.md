# Models Directory

This folder contains trained model files generated during training.

## Files

### fruit_model.h5

Trained VGG16-based fruit classification model.

Purpose:

* Store learned weights
* Load model without retraining
* Perform inference on new images

Example:

model = tf.keras.models.load_model("fruit_model.h5")

---

## Model Architecture

VGG16
→ GlobalAveragePooling2D
→ Dense(256)
→ Dropout(0.5)
→ Dense(Output Classes)

---

## Training Parameters

Optimizer: Adam

Learning Rate: 0.0001

Loss: Weighted Categorical Cross Entropy

Metric: Accuracy
