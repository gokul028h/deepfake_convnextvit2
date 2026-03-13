import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

from albumentations import Compose, RandomBrightnessContrast, \
    HorizontalFlip, FancyPCA, HueSaturationValue, ToGray, \
    Affine, ImageCompression, PadIfNeeded, GaussNoise

from transforms.albu import IsotropicResize

class DeepFakesDataset(Dataset):

    def __init__(self, images, labels, image_size, mode='train'):

        self.x = images
        self.y = torch.from_numpy(labels)
        self.image_size = image_size
        self.mode = mode
        self.n_samples = len(images)

        # create transforms once
        if mode == "train":
            self.transform = self.create_train_transforms(image_size)
        else:
            self.transform = self.create_val_transform(image_size)


    def create_train_transforms(self, size):

        return Compose([
            ImageCompression(p=0.2),
            GaussNoise(p=0.3),
            HorizontalFlip(p=0.5),

            IsotropicResize(
                max_side=size,
                interpolation_down=cv2.INTER_AREA,
                interpolation_up=cv2.INTER_CUBIC
            ),

            PadIfNeeded(
                min_height=size,
                min_width=size,
                border_mode=cv2.BORDER_CONSTANT
            ),

            RandomBrightnessContrast(p=0.3),
            HueSaturationValue(p=0.3),
            FancyPCA(p=0.05),
            ToGray(p=0.2),

            Affine(
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                scale=(0.8, 1.2),
                rotate=(-5, 5),
                p=0.2
            ),
        ])


    def create_val_transform(self, size):

        return Compose([
            IsotropicResize(
                max_side=size,
                interpolation_down=cv2.INTER_AREA,
                interpolation_up=cv2.INTER_CUBIC
            ),

            PadIfNeeded(
                min_height=size,
                min_width=size,
                border_mode=cv2.BORDER_CONSTANT
            ),
        ])


    def compute_dip_features(self, image_rgb):
        """
        Compute 3 enhanced DIP feature maps on-the-fly for deepfake detection.
        Optimized version using OpenCV-only operations for speed.

        Args:
            image_rgb: numpy array (H, W, 3), RGB uint8

        Returns:
            edge_fused: Fused edge map (Canny + Laplacian + Sobel), float32 [0,1]
            fft_map: High-pass emphasized FFT magnitude, float32 [0,1]
            ela_fused: Multi-quality ELA map, float32 [0,1]
        """

        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape

        # 1. FUSED EDGE DETECTION (Speed: OpenCV Optimized)
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

        # 3. MULTI-QUALITY ELA (Speed: OpenCV Buffer-based)
        ela_fused = np.zeros((h, w), dtype=np.float32)
        for quality in [90, 75]:
            # Encode and decode in-memory (no disk, much faster than PIL/BytesIO)
            _, enc = cv2.imencode('.jpg', image_rgb, [cv2.IMWRITE_JPEG_QUALITY, quality])
            compressed = cv2.imdecode(enc, cv2.IMREAD_COLOR)

            # Map back to RGB if necessary (imdecode returns BGR by default)
            compressed = cv2.cvtColor(compressed, cv2.COLOR_BGR2RGB)

            # Abs difference (forensic signal)
            ela_diff = cv2.absdiff(image_rgb, compressed)
            ela_gray = cv2.cvtColor(ela_diff, cv2.COLOR_RGB2GRAY).astype(np.float32)

            # Scaled brightness per research requirements
            max_diff = ela_gray.max()
            scale = 255.0 / max_diff if max_diff != 0 else 1
            ela_gray = np.clip(ela_gray * scale, 0, 255) / 255.0
            
            ela_fused += ela_gray / 2.0

        return edge_fused, fft_map, ela_fused


    def __getitem__(self, index):

        image_path = self.x[index]

        image = cv2.imread(image_path)

        if image is None:
            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentation
        image = self.transform(image=image)['image']

        # Compute DIP features on-the-fly
        edge, fft_map, ela = self.compute_dip_features(image)

        # Normalize RGB (ImageNet stats)
        rgb = image.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        rgb = (rgb - mean) / std
        rgb = rgb.transpose(2, 0, 1)  # HWC -> CHW

        # Combine: 6 channels = 3 RGB + 3 DIP
        dip_maps = np.stack([edge, fft_map, ela])
        combined = np.concatenate([rgb, dip_maps], axis=0)

        return torch.tensor(combined).float(), self.y[index]

    def __len__(self):
        return self.n_samples