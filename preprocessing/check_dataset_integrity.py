import os
import cv2
import glob
from tqdm import tqdm

BASE_DIR = "../deep_fakes_explain/dataset"

DATASETS = [
    "training_set",
    "validation_set",
    "test_set"
]

CLASSES = [
    "real",
    "fake"
]


def check_images(folder):

    images = glob.glob(folder + "/*.jpg") + glob.glob(folder + "/*.png")

    bad_files = []

    for img_path in tqdm(images):

        img = cv2.imread(img_path)

        if img is None:
            bad_files.append(img_path)
            continue

        h, w = img.shape[:2]

        if h < 32 or w < 32:
            bad_files.append(img_path)

    return bad_files


def main():

    total_bad = []

    for dataset in DATASETS:

        for cls in CLASSES:

            folder = os.path.join(BASE_DIR, dataset, cls)

            print("\nChecking:", folder)

            bad = check_images(folder)

            total_bad.extend(bad)

    print("\nBad images found:", len(total_bad))

    for f in total_bad:
        print(f)

    if len(total_bad) > 0:

        print("\nRemoving corrupted files...")

        for f in total_bad:
            os.remove(f)

        print("Removed corrupted images")

    print("\nDataset check complete")


if __name__ == "__main__":
    main()
