import os
import shutil
import random

random.seed(42)  # for reproducibility

# Original dataset
source_root = "/kaggle/input/lung-and-colon-cancer-histopathological-images/lung_colon_image_set"
# New dataset
target_root = "/kaggle/working/"
train_dir = os.path.join(target_root, "train")
val_dir = os.path.join(target_root, "val")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Iterate over all classes
for main_folder in os.listdir(source_root):
    main_path = os.path.join(source_root, main_folder)
    if not os.path.isdir(main_path):
        continue

    for class_folder in os.listdir(main_path):
        class_path = os.path.join(main_path, class_folder)
        if not os.path.isdir(class_path):
            continue

        # Create corresponding folders in train/val
        train_class_folder = os.path.join(train_dir, class_folder)
        val_class_folder = os.path.join(val_dir, class_folder)
        os.makedirs(train_class_folder, exist_ok=True)
        os.makedirs(val_class_folder, exist_ok=True)

        # List all images
        images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        random.shuffle(images)

        split_idx = int(0.8 * len(images))  # 80% train, 20% val
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]

        # Copy images
        for img in train_imgs:
            shutil.copy(os.path.join(class_path, img), os.path.join(train_class_folder, img))
        for img in val_imgs:
            shutil.copy(os.path.join(class_path, img), os.path.join(val_class_folder, img))

print("Dataset restructured successfully!")