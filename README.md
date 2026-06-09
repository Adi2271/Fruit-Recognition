# Fruit Recognition Model using VGG16 Transfer Learning

## Overview

This project implements a Fruit Recognition System using Transfer Learning with the VGG16 Convolutional Neural Network. The model is trained on a Kaggle Fruit Recognition dataset and is capable of classifying fruit images into their corresponding categories.

The project uses TensorFlow/Keras in Google Colab and leverages pretrained ImageNet weights for efficient training and improved accuracy.

Google colab link : https://colab.research.google.com/drive/1HSJ3aEY23YnWvVpz73i0_lQJ6ZGIT92W?authuser=4#scrollTo=A65iLI9XCFf_
---

## Features

* Transfer Learning using VGG16
* Data Augmentation
* Class Imbalance Handling
* Custom Weighted Loss Function
* Early Stopping
* Accuracy and Loss Visualization
* Model Saving and Reusability

---

## Dataset

Dataset Source:

Fruit Recognition Dataset from Kaggle

Dataset ID:

chrisfilo/fruit-recognition

Dataset contains images organized into folders where each folder represents a fruit class.

---

## Project Workflow

Dataset Download
→ Dataset Exploration
→ Image Preprocessing
→ Data Augmentation
→ Custom-loss Function
→ Training/Validation Split
→ Class Weight Computation
→ VGG16 Feature Extraction
→ Custom Classification Layers
→ Model Training
→ Evaluation
→ Model Saving

---

## Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Matplotlib
* Scikit-Learn
* KaggleHub
* pillow
* h5py

---

## Model Architecture

Input Image (224×224×3)

↓

VGG16 Base Model

↓

GlobalAveragePooling2D

↓

Dense(256, ReLU)

↓

Dropout(0.5)

↓

Dense(Output Classes, Softmax)

---

## Training Configuration

Batch Size: 32
Image Size: 224×224
Optimizer: Adam
Learning Rate: 0.0001
Epochs Trained: 10
Loss Function: Weighted Categorical Cross Entropy
Metric: Accuracy

---

## Results

The model training process includes:

* Training Accuracy
* Validation Accuracy
* Training Loss
* Validation Loss

Performance graphs are generated for monitoring learning behavior and overfitting.

---

## Future Improvements

* Fine-tuning VGG16 layers
* ResNet50 implementation
* EfficientNet implementation
* Streamlit deployment
* TensorFlow Lite conversion

---

## Authors

Aditya Patil

Akash Patra

Balkirshna Goswami


IIIT Raichur
