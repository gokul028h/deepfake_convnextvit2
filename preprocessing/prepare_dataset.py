import os
import uuid
import random
import shutil
from tqdm import tqdm


# ===== PATHS =====

SOURCE_DATASET = r"C:\Users\Pranesh\Downloads\dataset"

DEST_DATASET = r"C:\Users\Pranesh\Desktop\paper\deepfake_convnextvit2\deep_fakes_explain\dataset"


# ===== SPLIT RATIO =====

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


# ===== COLLECT IMAGES =====

real_images = []
fake_images = []

print("Scanning REAL images...")

real_path = os.path.join(SOURCE_DATASET, "real")

for root, dirs, files in os.walk(real_path):
    for f in files:
        if f.lower().endswith((".jpg", ".png", ".jpeg")):
            real_images.append(os.path.join(root, f))


print("Scanning FAKE images...")

fake_path = os.path.join(SOURCE_DATASET, "fake")

for root, dirs, files in os.walk(fake_path):
    for f in files:
        if f.lower().endswith((".jpg", ".png", ".jpeg")):
            fake_images.append(os.path.join(root, f))


print(f"REAL images found: {len(real_images)}")
print(f"FAKE images found: {len(fake_images)}")


# ===== SHUFFLE =====

random.shuffle(real_images)
random.shuffle(fake_images)


# ===== SPLIT FUNCTION =====

def split_data(images):

    train_end = int(len(images) * TRAIN_RATIO)
    val_end = int(len(images) * (TRAIN_RATIO + VAL_RATIO))

    train = images[:train_end]
    val = images[train_end:val_end]
    test = images[val_end:]

    return train, val, test


real_train, real_val, real_test = split_data(real_images)
fake_train, fake_val, fake_test = split_data(fake_images)


# ===== COPY FUNCTION =====

def copy_images(image_list, destination):

    os.makedirs(destination, exist_ok=True)

    for img in tqdm(image_list):
        filename = str(uuid.uuid4()) + "_" + os.path.basename(img)
        shutil.copy(img, os.path.join(destination, filename))


# ===== COPY REAL =====

print("\nCopying REAL training images...")
copy_images(real_train, os.path.join(DEST_DATASET, "training_set", "real"))

print("\nCopying REAL validation images...")
copy_images(real_val, os.path.join(DEST_DATASET, "validation_set", "real"))

print("\nCopying REAL test images...")
copy_images(real_test, os.path.join(DEST_DATASET, "test_set", "real"))


# ===== COPY FAKE =====


print("\nCopying FAKE training images...")
copy_images(fake_train, os.path.join(DEST_DATASET, "training_set", "fake"))

print("\nCopying FAKE validation images...")
copy_images(fake_val, os.path.join(DEST_DATASET, "validation_set", "fake"))

print("\nCopying FAKE test images...")
copy_images(fake_test, os.path.join(DEST_DATASET, "test_set", "fake"))


print("\nDONE — Dataset prepared successfully.")