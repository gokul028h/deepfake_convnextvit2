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
            ImageCompression(quality_lower=60, quality_upper=100, p=0.2),
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


    def __getitem__(self, index):

        image_path = self.x[index]

        image = cv2.imread(image_path)

        if image is None:
            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentation
        image = self.transform(image=image)['image']

        # Normalize RGB (ImageNet stats)
        image = image.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        image = (image - mean) / std
        image = image.transpose(2, 0, 1)  # HWC -> CHW

        return torch.tensor(image).float(), self.y[index]

    def __len__(self):
        return self.n_samples