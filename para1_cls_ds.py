import os

dataset_path = "dataset"

class_counts = {}

for fruit in os.listdir(dataset_path):

    fruit_path = os.path.join(dataset_path, fruit)

    if not os.path.isdir(fruit_path):
        continue

    count = 0

    for root, dirs, files in os.walk(fruit_path):

        count += len([
            f for f in files
            if f.lower().endswith(
                (".png", ".jpg", ".jpeg")
            )
        ])

    class_counts[fruit] = count

for cls, cnt in sorted(
        class_counts.items(),
        key=lambda x:x[1],
        reverse=True):

    print(f"{cls:15} {cnt}")