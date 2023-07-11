import os.path
import numpy as np
from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection
import pandas as pd

import torch
from torch.utils.data import Dataset
from PIL import Image


class MuraDataset(Dataset):
    def __init__(self, path, transform, body_part=None):
        self.dataframe = pd.read_pickle(path)
        self.transform = transform  # Add any additional image transformations here
        if body_part:
            self.dataframe = self.dataframe[self.dataframe.body_part == body_part].reset_index(drop=True)

        print("loaded MURA Dataset", self.dataframe.shape)

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, index):
        while True:
            index = index % self.dataframe.shape[0]
            image_path = self.dataframe.loc[index, 'colab_path']
            label = self.dataframe.loc[index, 'label']

            if not os.path.exists(image_path):
                print("path doesn't exists", image_path)
                index = np.random.randint(self.dataframe.shape[0])
                continue

            try:
                # Load the image using PIL
                image = Image.open(image_path).convert("RGB")
                break
            except Exception as e:
                print(e)
                index = np.random.randint(self.dataframe.shape[0])
                continue

        # Apply transformations if any
        if self.transform is not None:
            image = self.transform(image)

        return image, label


class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([
                                              transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )
        return data_transforms

    def get_dataset(self, name, n_views, body_part=None):
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  self.get_simclr_pipeline_transform(32),
                                                                  n_views),
                                                              download=True),

                          'stl10': lambda: datasets.STL10(self.root_folder, split='unlabeled',
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(96),
                                                              n_views),
                                                          download=True),
                          'mura': lambda: MuraDataset(self.root_folder,
                                                      transform=ContrastiveLearningViewGenerator(
                                                          self.get_simclr_pipeline_transform(128),
                                                          n_views),
                                                      body_part=body_part
                                                      )}

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()
