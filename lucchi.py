import torch
from torch.utils.data import Dataset
import cv2
import os
from glob import glob
from tqdm import tqdm
import numpy as np


class LucchiPPDataset(Dataset):
    def __init__(self, train=True, transforms=None):
        self.subset = 'Train' if train else 'Test'
        self.files = glob('dataset/Lucchi++/'+self.subset+'_In/*.png')
        self.files.sort(key=lambda x:int(os.path.basename(x).split('.')[0].replace('mask','')))
        self.labels = glob('dataset/Lucchi++/'+self.subset+'_Out/*.png')
        self.labels.sort(key=lambda x:int(os.path.basename(x).split('.')[0]))
        assert len(self.files) == len(self.labels)
        assert len(self.files) > 0
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        x = cv2.imread(self.files[item])
        y = cv2.imread(self.labels[item])
        if self.transforms is not None:
            x = self.transforms(x)
            y = self.transforms(y)
        return torch.from_numpy(x.astype(np.float32).transpose(2,0,1)), torch.from_numpy(y.astype(np.float32).transpose(2,0,1))


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        return cv2.resize(img, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class Scale(object):
    def __init__(self):
        pass

    def __call__(self, img):
        return img / 255.

    def __repr__(self):
        return self.__class__.__name__
