import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageChops, ImageEnhance
import os

IMAGE_PATH = "examples/fake.jpg"   # change if needed


def perform_ela(image_path, quality=90):

    original = Image.open(image_path).convert("RGB")

    temp_path = "temp_ela.jpg"
    original.save(temp_path, "JPEG", quality=quality)

    compressed = Image.open(temp_path)

    ela_image = ImageChops.difference(original, compressed)

    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])

    scale = 255.0 / max_diff if max_diff != 0 else 1

    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    os.remove(temp_path)

    return original, ela_image


original, ela = perform_ela(IMAGE_PATH)

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(original)
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Error Level Analysis")
plt.imshow(ela)
plt.axis("off")

plt.show()