import os
import sys
import time
import numpy as np
import pandas as pd
import idx2numpy
import numpy as np

# PyTorch for ML
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor, RandomHorizontalFlip, RandomRotation, ToPILImage


class ImageDataset(Dataset):

    def __init__(self, data, split, im_size=28):
        self.split = split
        self.data = data[self.split]
        print(im_size)
        
        if split == "train":

            self.transform = Compose([
                Resize((im_size, im_size)),
                RandomHorizontalFlip(p=0.5),
                RandomRotation(degrees=45),
            ])
         
        elif self.split == "eval" or split == "test":
            self.transform = Compose([
                Resize((im_size, im_size)),
            ])
        
        
        if split == 'train':
            self.transform = Compose([
                Resize((im_size, im_size)),
                RandomHorizontalFlip(p=0.5),
                RandomRotation(degrees=45),
                ToTensor(),
            ])
            self.image_ixs = self.image_ixs[:num_train]

        elif split == 'eval':
            self.transform = Compose([
                Resize((im_size, im_size)),
                ToTensor(),
            ])

        elif split == 'test':
            self.transform = Compose([
                Resize((im_size, im_size)),
                ToTensor(),
            ])

    def __len__(self):
        return self.data["images"].shape[0]

    def __getitem__(self, index):
        return {
            'image': s(self.data["images"][index,:,:]),
            'label': torch.tensor(self.data["labels"][index], dtype=torch.float32)
        }