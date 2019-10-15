import torch
import numpy as np
import cv2
import glob
import os
import re
import csv
from torch.utils.data import Dataset
from albumentations import Compose
from albumentations.pytorch.functional import img_to_tensor
from albumentations.augmentations.transforms import RandomCrop


class AerialDataset(Dataset):
    def __init__(self, filepath, to_augment=False, transform=None, mode='train'):
        self.file_names = []
        with open(filepath) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=' ')
            line_count = 0
            for row in csv_reader:
                self.file_names.append(row[0])

        self.to_augment = to_augment
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_file_name = self.file_names[idx]
        image = load_image(img_file_name)

        if self.mode == 'train':
            mask_path = re.sub('.tif', 'segcls.tif', img_file_name)
            mask = load_mask(mask_path)
            data = {"image": image, "mask": mask}
            augmented = self.transform(**data)
            image, mask = augmented["image"], augmented["mask"]
            return img_to_tensor(image), torch.from_numpy(np.expand_dims(mask, 0)).float()
        else:
            data = {"image": image}
            augmented = self.transform(**data)
            image = augmented["image"]

            return img_to_tensor(image), str(img_file_name)


class AerialCombinedDataset(Dataset):
    def __init__(self, filepath, to_augment=False, transform=None, mode='train', image_cols=416, image_rows=416, sequential=True):
        self.file_names = []
        with open(filepath) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=' ')
            line_count = 0
            for row in csv_reader:
                self.file_names.append(row[0])
        self.mask_dir = os.path.join(os.path.dirname(filepath), 'annotations')
        self.to_augment = to_augment
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_file_name = self.file_names[idx]
        image = load_image(img_file_name)

        if self.mode == 'train':
            name_root = img_file_name.split('_')[-1].split('.')[0]
            mask_path = os.path.join(
                self.mask_dir, 'mask_' + name_root + '.png')
            mask = load_mask(mask_path)

            data = {"image": image, "mask": mask}
            augmented = self.transform(**data)
            image, mask = augmented["image"], augmented["mask"]

            return img_to_tensor(image), torch.from_numpy(mask).to(torch.long)
        else:
            data = {"image": image}
            augmented = self.transform(**data)
            image = augmented["image"]
            return img_to_tensor(image), str(img_file_name)


class AerialRoadDataset(Dataset):
    def __init__(self, filepath, to_augment=False, transform=None, mode='train', image_cols=416, image_rows=416, sequential=True):
        self.file_names = []
        with open(filepath) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=' ')
            line_count = 0
            for row in csv_reader:
                self.file_names.append(row[0])
            self.mask_dir = os.path.join(os.path.dirname(
                os.path.dirname(row[0])), 'masks_2m_clip')
        self.to_augment = to_augment
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):

        img_file_name = self.file_names[idx]
        image = load_image(img_file_name)

        if self.mode == 'train':
            name_root = img_file_name.split('_')[-1].split('.')[0]
            mask_path = os.path.join(
                self.mask_dir, 'mask_' + name_root + '.png')
            mask = load_mask(mask_path)

            data = {"image": image, "mask": mask}
            augmented = self.transform(**data)
            image, mask = augmented["image"], augmented["mask"]

            #return img_to_tensor(image), torch.from_numpy(mask).long()
            return img_to_tensor(image), torch.from_numpy(np.expand_dims(mask, 0)).float()
        else:
            data = {"image": image}
            augmented = self.transform(**data)
            image = augmented["image"]
            return img_to_tensor(image), str(img_file_name)


def load_image(path):
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(mask_path):
    factor = 100
    mask = cv2.imread(mask_path, 0)
    return (mask / factor).astype(np.uint8)
