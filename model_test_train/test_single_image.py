import torch
import cv2
import numpy as np
import yaml
import argparse
import os
from convnext_crossvit import ConvNeXtCrossViT
from albumentations import Compose, PadIfNeeded
from transforms.albu import IsotropicResize

def compute_dip_features(image_rgb):
    """
    Replication of the 6-channel forensic map generation from deepfakes_dataset.py
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    # 1. FUSED EDGE DETECTION
    canny = cv2.Canny(gray, 100, 200).astype(np.float32) / 255.0
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian).astype(np.float32) / 255.0
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    sobel_mag = np.clip(sobel_mag / (sobel_mag.max() + 1e-8), 0, 1).astype(np.float32)
    edge_fused = (0.4 * canny + 0.3 * laplacian + 0.3 * sobel_mag).astype(np.float32)

    # 2. HIGH-PASS EMPHASIZED FFT
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    max_dist = np.sqrt(cx ** 2 + cy ** 2)
    emphasis = 0.5 + 0.5 * (dist / (max_dist + 1e-8))
    magnitude = magnitude * emphasis
    fft_map = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)

    # 3. MULTI-QUALITY ELA
    ela_fused = np.zeros((h, w), dtype=np.float32)
    for quality in [90, 75]:
        _, enc = cv2.imencode('.jpg', image_rgb, [cv2.IMWRITE_JPEG_QUALITY, quality])
        compressed = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        compressed = cv2.cvtColor(compressed, cv2.COLOR_BGR2RGB)
        ela_diff = cv2.absdiff(image_rgb, compressed)
        ela_gray = cv2.cvtColor(ela_diff, cv2.COLOR_RGB2GRAY).astype(np.float32)
        max_diff = ela_gray.max()
        scale = 255.0 / max_diff if max_diff != 0 else 1
        ela_gray = np.clip(ela_gray * scale, 0, 255) / 255.0
        ela_fused += ela_gray * 0.5
    
    return edge_fused, fft_map, ela_fused

def predict(image_path, model_path, config_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    size = config['model']['image-size']

    # Load Model
    model = ConvNeXtCrossViT().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    # Check if checkpoint contains state_dict or is the state_dict itself
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Load and Preprocess Image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize and Pad
    transform = Compose([
        IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
    ])
    image = transform(image=image)['image']

    # Compute Forensic Maps
    edge, fft, ela = compute_dip_features(image)

    # RGB Normalization (Parity with training)
    rgb = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    rgb = (rgb - mean) / std

    # Stack into 6-channel input
    input_stack = np.concatenate([
        rgb, 
        edge[..., np.newaxis], 
        fft[..., np.newaxis], 
        ela[..., np.newaxis]
    ], axis=-1)

    # Convert to Tensor (N, C, H, W)
    input_tensor = torch.from_numpy(input_stack).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()

    prediction = "FAKE" if prob > 0.5 else "REAL"
    confidence = prob if prob > 0.5 else (1 - prob)

    print(f"\nResults for: {os.path.basename(image_path)}")
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence*100:.2f}%")
    print(f"Raw Probability: {prob:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a single image for deepfakes")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, default="../deep_fakes_explain/models/best_model.pth", help="Path to model weights")
    parser.add_argument("--config", type=str, default="configs/explained_architecture.yaml", help="Path to config file")

    args = parser.parse_args()
    predict(args.image, args.model, args.config)
