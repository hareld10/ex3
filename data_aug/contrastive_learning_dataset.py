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
    def __init__(self, path, transform):
        self.dataframe = pd.read_pickle(path)
        self.transform = transform  # Add any additional image transformations here
        print("loaded MURA Dataset", self.dataframe.shape)
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        index = index % self.dataframe.shape[0]
        image_path = self.dataframe.loc[index, 'img_path']
        label = self.dataframe.loc[index, 'label']

        # Load the image using PIL
        image = Image.open(image_path)

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
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])
        return data_transforms

    def get_dataset(self, name, n_views):
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
                                                          self.get_simclr_pipeline_transform(96),
                                                          n_views),
                                                      )}

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()
