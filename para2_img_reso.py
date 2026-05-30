from PIL import Image
import os
import pandas as pd

dataset_path = "dataset"

widths = []
heights = []

for cls in os.listdir(dataset_path):

    cls_path = os.path.join(dataset_path, cls)

    if not os.path.isdir(cls_path):
        continue

    for img_name in os.listdir(cls_path):

        img_path = os.path.join(cls_path, img_name)

        try:
            img = Image.open(img_path)

            widths.append(img.width)
            heights.append(img.height)

        except:
            pass

print("Average Width :", sum(widths)/len(widths))
print("Average Height:", sum(heights)/len(heights))

print("Min Width :", min(widths))
print("Max Width :", max(widths))   