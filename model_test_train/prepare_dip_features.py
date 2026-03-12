import cv2
import numpy as np
import os
from PIL import Image, ImageChops
from io import BytesIO
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

DATASET_DIR = "../deep_fakes_explain/dataset"
OUTPUT_DIR = "../deep_fakes_explain/dataset_with_dip"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_dip(path):

    try:
        image = cv2.imread(path)
        if image is None:
            return

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        edge = cv2.Laplacian(gray, cv2.CV_64F)
        edge = cv2.convertScaleAbs(edge).astype(np.float32) / 255.0

        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        fft_map = np.log(np.abs(fshift) + 1)
        fft_map = cv2.normalize(fft_map, None, 0, 1, cv2.NORM_MINMAX)

        pil = Image.fromarray(image)
        buffer = BytesIO()
        pil.save(buffer, "JPEG", quality=90)
        buffer.seek(0)
        compressed = Image.open(buffer)

        ela = ImageChops.difference(pil, compressed)
        ela = np.array(ela.convert("L")).astype(np.float32) / 255.0

        relative_path = os.path.relpath(path, DATASET_DIR)
        save_path = os.path.join(OUTPUT_DIR, relative_path + ".npz")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        np.savez_compressed(
            save_path,
            rgb=image,
            edge=edge,
            fft=fft_map,
            ela=ela
        )

    except:
        pass


if __name__ == "__main__":

    print("Scanning dataset...")

    all_images = []

    for root, dirs, files in os.walk(DATASET_DIR):
        for f in files:
            if f.lower().endswith(("jpg", "jpeg", "png")):
                all_images.append(os.path.join(root, f))

    print(f"Total images: {len(all_images)}")
    print(f"CPU cores: {cpu_count()}")

    with Pool(cpu_count()) as p:
        list(tqdm(p.imap_unordered(compute_dip, all_images),
                  total=len(all_images)))

    print("Done.")