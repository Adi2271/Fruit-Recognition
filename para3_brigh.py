import os
import cv2
import numpy as np

dataset_path = "dataset"

brightness_values = []

for root, dirs, files in os.walk(dataset_path):

    for file in files:

        if file.lower().endswith((".png", ".jpg", ".jpeg")):

            img_path = os.path.join(root, file)

            img = cv2.imread(img_path)

            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            brightness_values.append(
                np.mean(gray)
            )

if len(brightness_values) == 0:
    print("No images found!")
else:
    print("Total Images:", len(brightness_values))
    print("Average Brightness:", round(np.mean(brightness_values), 2))
    print("Minimum Brightness:", round(np.min(brightness_values), 2))
    print("Maximum Brightness:", round(np.max(brightness_values), 2))