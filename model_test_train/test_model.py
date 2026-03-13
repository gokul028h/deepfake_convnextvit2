import torch
import cv2
import numpy as np
import yaml
import argparse
import os
import sys
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from convnext_crossvit import ConvNeXtCrossViT
from albumentations import Compose, PadIfNeeded
from transforms.albu import IsotropicResize

# forensic signal computation
def compute_dip_features(image_rgb):
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

def save_roc_curve(correct_labels, preds, model_name, output_dir):
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    fpr, tpr, _ = metrics.roc_curve(correct_labels, preds)
    model_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{model_name} (AUC={model_auc:.3f})")
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC Curve')
    plt.legend(loc='best')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{model_name}_roc.png"))
    plt.clf()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--output_dir', type=str, default='eval_results')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    size = config['model']['image-size']

    # Initialize Model
    model = ConvNeXtCrossViT().to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print(f"Model loaded from {args.model_path}")

    # Transform
    transform = Compose([
        IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
    ])

    # Dataset Scan
    TEST_DIR = "../deep_fakes_explain/dataset/test_set"
    dataset = []
    for label, folder in [(0, "real"), (1, "fake")]:
        folder_path = os.path.join(TEST_DIR, folder)
        if not os.path.exists(folder_path):
            continue
        for f in os.listdir(folder_path):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                dataset.append((os.path.join(folder_path, f), label))
    
    print(f"Total test samples: {len(dataset)}")
    
    all_probs = []
    all_labels = []
    
    # Simple batching (manual to avoid excessive Dataloader complexity for forensic compute)
    for i in tqdm(range(0, len(dataset), args.batch_size)):
        batch_info = dataset[i : i + args.batch_size]
        batch_images = []
        batch_labels = []
        
        for img_path, label in batch_info:
            image = cv2.imread(img_path)
            if image is None: continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = transform(image=image)['image']
            
            # Forensic features
            edge, fft, ela = compute_dip_features(image)
            
            # RGB Normalization (Parity with training)
            rgb = image.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            rgb = (rgb - mean) / std
            
            # Combine into 6-channel stack
            input_stack = np.concatenate([
                rgb, 
                edge[..., np.newaxis], 
                fft[..., np.newaxis], 
                ela[..., np.newaxis]
            ], axis=-1)
            
            batch_images.append(input_stack)
            batch_labels.append(label)
            
        if not batch_images: continue
        
        # (N, H, W, 6) -> (N, 6, H, W)
        input_tensor = torch.from_numpy(np.stack(batch_images)).permute(0, 3, 1, 2).float().to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.sigmoid(output).cpu().numpy().flatten()
            
        all_probs.extend(probs)
        all_labels.extend(batch_labels)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = (all_probs > 0.5).astype(int)

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    print("\n" + "="*30)
    print("      TEST SET RESULTS")
    print("="*30)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("="*30)

    save_roc_curve(all_labels, all_probs, "ConvNeXt_TestSet", args.output_dir)
    print(f"ROC curve saved to {args.output_dir}")