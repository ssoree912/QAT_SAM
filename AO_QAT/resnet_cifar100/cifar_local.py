"""Local CIFAR-100 loader from npz (converted from HuggingFace uoft-cs/cifar100).
Avoids the slow toronto.edu download. Returns a PIL image before transform, so
the standard torchvision transforms (RandomCrop, etc.) work unchanged.
"""
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class CIFAR100Local(Dataset):
    def __init__(self, root, train=True, transform=None):
        path = os.path.join(root, f"cifar100_{'train' if train else 'test'}.npz")
        d = np.load(path)
        self.images = d["images"]  # [N, 32, 32, 3] uint8
        self.labels = d["labels"].astype("int64")
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = Image.fromarray(self.images[i])
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[i])
