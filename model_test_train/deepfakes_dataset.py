import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

from albumentations import Compose, RandomBrightnessContrast, \
    HorizontalFlip, FancyPCA, HueSaturationValue, ToGray, \
    ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise

from transforms.albu import IsotropicResize
from PIL import Image, ImageChops


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

        image = cv2.imread(image_path)

        # handle corrupted images
        if image is None:
            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = self.transform(image=image)['image']


        # ===== DIP FEATURES =====

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Edge map
        edge = cv2.Canny(gray, 100, 200)

        # FFT spectrum
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        fft_map = np.log(np.abs(fshift) + 1)
        fft_map = cv2.normalize(fft_map, None, 0, 255, cv2.NORM_MINMAX)

        # Error Level Analysis
        pil_img = Image.fromarray(image)
        temp_path = "temp.jpg"
        pil_img.save(temp_path, "JPEG", quality=90)
        compressed = Image.open(temp_path)
        ela = ImageChops.difference(pil_img, compressed)
        ela = np.array(ela.convert("L"))

        # resize maps
        edge = cv2.resize(edge, (self.image_size, self.image_size))
        fft_map = cv2.resize(fft_map, (self.image_size, self.image_size))
        ela = cv2.resize(ela, (self.image_size, self.image_size))

        # convert RGB
        rgb = image.transpose(2, 0, 1)

        # stack DIP maps
        dip_maps = np.stack([edge, fft_map, ela])

        combined = np.concatenate([rgb, dip_maps], axis=0)

        return torch.tensor(combined).float(), self.y[index]


    def __len__(self):
        return self.n_samples