import lmdb
import pickle
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset


class LMDBDataset(Dataset):

    def __init__(self, path, image_size=224):

        self.env = lmdb.open(path, readonly=True, lock=False)
        self.txn = self.env.begin()

        self.length = int(self.txn.get(b'length').decode())

        self.image_size = image_size

    def __len__(self):

        return self.length

    def __getitem__(self, index):

        key = f"{index:08}".encode()

        img,label = pickle.loads(self.txn.get(key))

        img = cv2.resize(img,(self.image_size,self.image_size))

        img = img.astype(np.float32)/255.0

        img = torch.from_numpy(img)

        return img,label