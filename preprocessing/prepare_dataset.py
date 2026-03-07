import os
import uuid
import random
import shutil
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# ================================
# PATHS
# ================================

SOURCE_DATASET = r"D:\dataset"

DEST_DATASET = r"C:\Users\HP\Desktop\paper\deepfake_convnextvit2\deep_fakes_explain\dataset"


# ================================
# SPLIT RATIO
# ================================

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


# ================================
# IMAGE COLLECTION
# ================================

def collect_images(path):

    images = []

    for root, dirs, files in os.walk(path):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                images.append(os.path.join(root, f))

    return images


print("Scanning REAL images...")
real_images = collect_images(os.path.join(SOURCE_DATASET, "real"))

print("Scanning FAKE images...")
fake_images = collect_images(os.path.join(SOURCE_DATASET, "fake"))

print(f"REAL images found: {len(real_images)}")
print(f"FAKE images found: {len(fake_images)}")


# ================================
# SHUFFLE DATA
# ================================

random.shuffle(real_images)
random.shuffle(fake_images)


# ================================
# SPLIT FUNCTION
# ================================

def split_data(images):

    train_end = int(len(images) * TRAIN_RATIO)
    val_end = int(len(images) * (TRAIN_RATIO + VAL_RATIO))

    train = images[:train_end]
    val = images[train_end:val_end]
    test = images[val_end:]

    return train, val, test


real_train, real_val, real_test = split_data(real_images)
fake_train, fake_val, fake_test = split_data(fake_images)


# ================================
# PARALLEL COPY FUNCTION
# ================================

def copy_worker(args):

    src, dst = args

    filename = f"{uuid.uuid4()}_{os.path.basename(src)}"
    dst_path = os.path.join(dst, filename)

    try:
        shutil.copy(src, dst_path)
    except Exception:
        pass


def copy_images_parallel(image_list, destination):

    os.makedirs(destination, exist_ok=True)

    tasks = [(img, destination) for img in image_list]

    workers = max(cpu_count() - 1, 1)

    with Pool(workers) as pool:
        list(tqdm(pool.imap(copy_worker, tasks), total=len(tasks)))


# ================================
# COPY REAL
# ================================

print("\nCopying REAL training images...")
copy_images_parallel(real_train, os.path.join(DEST_DATASET, "training_set", "real"))

print("\nCopying REAL validation images...")
copy_images_parallel(real_val, os.path.join(DEST_DATASET, "validation_set", "real"))

print("\nCopying REAL test images...")
copy_images_parallel(real_test, os.path.join(DEST_DATASET, "test_set", "real"))


# ================================
# COPY FAKE
# ================================

print("\nCopying FAKE training images...")
copy_images_parallel(fake_train, os.path.join(DEST_DATASET, "training_set", "fake"))

print("\nCopying FAKE validation images...")
copy_images_parallel(fake_val, os.path.join(DEST_DATASET, "validation_set", "fake"))

print("\nCopying FAKE test images...")
copy_images_parallel(fake_test, os.path.join(DEST_DATASET, "test_set", "fake"))


print("\nDONE — Dataset prepared successfully.")