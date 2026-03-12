import cv2
import matplotlib.pyplot as plt

IMAGE_PATH = "examples/real_1.jpg"   # change image if needed


def show_edge_map(image_path):

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 100, 200)

    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.title("Canny Edge Map")
    plt.imshow(edges, cmap="gray")
    plt.axis("off")

    plt.show()


show_edge_map(IMAGE_PATH)