import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

from albumentations import Compose, RandomBrightnessContrast, \
    HorizontalFlip, FancyPCA, HueSaturationValue, ToGray, \
    ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise

from transforms.albu import IsotropicResize


class DeepFakesDataset(Dataset):

    def __init__(self, images, labels, image_size, mode='train'):

        self.x = images
        self.y = torch.from_numpy(labels)
        self.image_size = image_size
        self.mode = mode
        self.n_samples = images.shape[0]

        # Create transforms once (faster)
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
            FancyPCA(p=0.2),
            ToGray(p=0.2),

            ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=5,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5
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

        # Read image properly
        image = cv2.imread(image_path)

        # Handle corrupted images
        if image is None:
            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = self.transform(image=image)['image']

        image = image.transpose(2, 0, 1)

        return torch.from_numpy(image).float(), self.y[index]


    def __len__(self):
        return self.n_samples