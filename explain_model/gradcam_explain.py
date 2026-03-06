import torch
import cv2
import numpy as np
import os
import sys

sys.path.append("../model_test_train")

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from convnext_crossvit import ConvNeXtCrossViT

IMAGE_SIZE = 224

MODEL_PATH = "../deep_fakes_explain/models/best_model.pth"
IMAGE_PATH = "examples/fake.jpg"
OUTPUT_PATH = "explanation/gradcam_result.jpg"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_image(image):

    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = image.astype(np.float32) / 255.0

    input_tensor = torch.from_numpy(image).permute(2,0,1).unsqueeze(0)

    return input_tensor


def load_model():

    model = ConvNeXtCrossViT()

    if os.path.exists(MODEL_PATH):

        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint["model"])
        print("Loaded trained model")

    else:

        print("WARNING: best_model.pth not found. Using untrained model.")

    model.to(device)
    model.eval()

    return model


def generate_gradcam():

    model = load_model()

    image = cv2.imread(IMAGE_PATH)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    rgb_img = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    rgb_img = rgb_img.astype(np.float32) / 255.0

    input_tensor = preprocess_image(image).to(device)

    target_layers = [model.backbone.stages[-1]]

    cam = GradCAM(
        model=model,
        target_layers=target_layers,
    )

    targets = [ClassifierOutputTarget(0)]

    with GradCAM(model=model, target_layers=target_layers) as cam: grayscale_cam = cam( input_tensor=input_tensor, targets=targets )[0]

    visualization = show_cam_on_image(
        rgb_img,
        grayscale_cam,
        use_rgb=True
    )

    os.makedirs("explanation", exist_ok=True)

    cv2.imwrite(
        OUTPUT_PATH,
        cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
    )

    print("GradCAM saved to:", OUTPUT_PATH)


if __name__ == "__main__":

    generate_gradcam()
