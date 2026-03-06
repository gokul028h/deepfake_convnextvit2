import torch
import cv2
import numpy as np
import os
import sys

# Allow importing model from parent folder
sys.path.append(os.path.abspath("../model_test_train"))

from convnext_crossvit import ConvNeXtCrossViT


BASE_DIR = "../deep_fakes_explain"
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pth")

IMAGE_PATH = "examples/fake_0.jpg"

OUTPUT_DIR = "explanation"

IMAGE_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def preprocess_image(image):

    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

    image = image.astype(np.float32) / 255.0

    tensor = torch.from_numpy(image).permute(2,0,1).unsqueeze(0)

    return tensor


def visualize_attention():

    model = load_model()

    image = cv2.imread(IMAGE_PATH)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    rgb = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE))

    rgb_norm = rgb.astype(np.float32) / 255.0

    input_tensor = preprocess_image(image).to(device)

    with torch.no_grad():

        features, attention = model.forward_attention(input_tensor)

    attention_map = attention.mean(dim=1).squeeze().cpu().numpy()

    attention_map = cv2.resize(attention_map, (IMAGE_SIZE, IMAGE_SIZE))

    attention_map = (attention_map - attention_map.min()) / (
        attention_map.max() - attention_map.min()
    )

    heatmap = cv2.applyColorMap(
        np.uint8(255 * attention_map),
        cv2.COLORMAP_JET
    )

    heatmap = np.float32(heatmap) / 255

    overlay = heatmap + rgb_norm

    overlay = overlay / overlay.max()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    save_path = os.path.join(
        OUTPUT_DIR,
        "cross_attention_map.jpg"
    )

    cv2.imwrite(
        save_path,
        cv2.cvtColor(
            np.uint8(255 * overlay),
            cv2.COLOR_RGB2BGR
        )
    )

    print("Saved:", save_path)


if __name__ == "__main__":

    visualize_attention()
