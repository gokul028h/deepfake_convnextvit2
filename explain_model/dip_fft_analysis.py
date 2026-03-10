import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGE_PATH = "examples/real.jpg"   # change if needed


def show_fft_spectrum(image_path):

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Fourier Transform
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.title("Original Image")
    plt.imshow(gray, cmap='gray')
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.title("FFT Magnitude Spectrum")
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.axis("off")

    plt.show()


show_fft_spectrum(IMAGE_PATH)