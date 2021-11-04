import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

import torch
from functools import reduce
from PIL import Image

PATH_DATA = os.path.join('..', '..', 'data', 'processed')

standard_datasets_info = {
    'CIFAR10': {
        'dataset': datasets.CIFAR10,
        'mean': (0.485, 0.456, 0.406),
        'std': (0.229, 0.224, 0.225),
    }
}

custom_datasets_info = {
    'SAMPLE_DATASET': {
        'dataset_path': os.path.join(PATH_DATA, 'sample_dataset'),
        'mean': (0., 0., 0.),
        'std': (1., 1., 1.),
        'colour': True
    },
    'SAMPLE_DATASET_grayscale': {
        'dataset_path': os.path.join(PATH_DATA, 'sample_dataset_grayscale'),
        'mean': (0.,),
        'std': (1.,)
    }
}


def get_mean(dataset_train):
    num_samples = len(dataset_train)
    to_tensor = transforms.ToTensor()
    sample_shape = to_tensor(dataset_train[0][0]).shape
    num_pixels = reduce(lambda x, y: x * y, sample_shape)
    num_channels = sample_shape[0]

    total_intensity = torch.zeros(num_channels)
    # Compute global mean of the whole sample (both train and test) set:
    for img in dataset_train:
        for channel in range(num_channels):
            total_intensity[channel] += torch.sum(to_tensor(img[0])[channel, :])
    dataset_mean = total_intensity / (num_samples * num_pixels)
    return dataset_mean


def get_std(dataset_train, dataset_mean):
    num_samples = len(dataset_train)
    to_tensor = transforms.ToTensor()
    sample_shape = to_tensor(dataset_train[0][0]).shape
    num_pixels = reduce(lambda x, y: x * y, sample_shape)
    num_channels = sample_shape[0]

    # Compute global standard deviation
    total_diff = torch.zeros(num_channels)
    for img in dataset_train:
        for channel in range(num_channels):
            total_diff[channel] += torch.sum(torch.pow(to_tensor(img[0])[channel, :] - dataset_mean[channel], 2))
    dataset_std = torch.sqrt(total_diff / (num_samples * num_pixels))
    return dataset_std


def load_dataset(dataset_name, batch_size=32, train=True, train_proportion=0.8, val=True, num_workers=1, pin_memory=True):
    if dataset_name in standard_datasets_info.keys():
        return load_standard_dataset(dataset_name, batch_size, train, train_proportion, val, num_workers, pin_memory)
    elif dataset_name in custom_datasets_info.keys():
        return load_custom_dataset(dataset_name, batch_size, train, train_proportion, val, num_workers, pin_memory)
    else:
        raise Exception('No entry for dataset {}'.format(dataset_name))


def load_standard_dataset(dataset_name, batch_size=32, train=True, train_proportion=0.8, val=True, num_workers=1, pin_memory=True):

    if dataset_name not in standard_datasets_info.keys():
        raise Exception('No entry for dataset {}: Provided dataset must be one of {}'.format(
            dataset_name, standard_datasets_info.keys()))

    # 1.-Prepare transformations to be applied to each set:
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(standard_datasets_info[dataset_name]['mean'], standard_datasets_info[dataset_name]['std'])]
    )
    # 2.-Prepare the datasets:
    if train:
        train_dataset = standard_datasets_info[dataset_name]['dataset'](
            root=os.path.join('..', '..', 'data', 'external'), train=True,
            download=True, transform=transform,
        )
    else:
        test_dataset = standard_datasets_info[dataset_name]['dataset'](
            root=os.path.join('..', '..', 'data', 'external'), train=False,
            download=True, transform=transform,
        )
    # 3.-Prepare the DataLoaders using the previous Datasets
    if train:
        if val:
            # Split the train dataset into train and validation sets:
            # Get a random split with a proportion of train_proportion samples for the train subset and the remaining
            # ones for the validation subset:
            num_train = len(train_dataset)
            indices = list(range(num_train))
            split = int(np.floor(train_proportion * num_train))
            np.random.shuffle(indices)
            # Generate some SubsetRandomSampler with the indexes of the images corresponding to each subset:
            train_idx, val_idx = indices[:split], indices[split:]
            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)
            # Generate DataLoader for the images:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                                      num_workers=num_workers,
                                      pin_memory=pin_memory)
            val_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers,
                                    pin_memory=pin_memory)
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                                      pin_memory=pin_memory, shuffle=True)
    else:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    # 4.-Return the requested DataLoaders:
    if train:
        if val:
            return train_loader, val_loader
        else:
            return train_loader
    else:
        return test_loader


def load_custom_dataset(dataset_name, batch_size=32, train=True, train_proportion=0.8, val=True, num_workers=1, pin_memory=True):

    # Auxiliar functions which facilitate image loading:
    def pil_loader_RGB(path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')  # RGB image

    def pil_loader_grayscale(path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')  # Grayscale

    if dataset_name not in custom_datasets_info.keys():
        raise Exception('No entry for dataset {}: Provided dataset must be one of {}'.format(
            dataset_name, custom_datasets_info.keys()))

    dataset_info = custom_datasets_info[dataset_name]

    if dataset_info['colour']:
        loader = pil_loader_RGB
        distribution_stats = ((dataset_info['mean']), (dataset_info['std']))
        loader = pil_loader_grayscale
        distribution_stats = ((dataset_info['mean']), (dataset_info['std']))

    # 1.-Prepare transformations to be applied to each set:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(distribution_stats[0], distribution_stats[1])
    ])
    # 2.-Prepare the datasets:
    if train:
        train_dataset = datasets.DataFolder(root=os.path.join(dataset_info['dataset_path'], 'train'), 
            loader=loader, transform=transform, extensions=('jpeg', 'png', 'jpg'))
    else:
        test_dataset = datasets.DataFolder(root=os.path.join(dataset_info['dataset_path'], 'test'), 
            loader=loader, transform=transform, extensions=('jpeg', 'png', 'jpg'))
    # 3.-Prepare the DataLoaders using the previous Datasets
    if train:
        if val:
            # Split the train dataset into train and validation sets:
            # Get a random split with a proportion of train_proportion samples for the train subset and the remaining
            # ones for the validation subset:
            num_train = len(train_dataset)
            indices = list(range(num_train))
            split = int(np.floor(train_proportion * num_train))
            np.random.shuffle(indices)
            # Generate some SubsetRandomSampler with the indexes of the images corresponding to each subset:
            train_idx, val_idx = indices[:split], indices[split:]
            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)
            # Generate DataLoader for the images:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                                      num_workers=num_workers,
                                      pin_memory=pin_memory)
            val_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers,
                                    pin_memory=pin_memory)
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                                      pin_memory=pin_memory, shuffle=True)
    else:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    # 4.-Return the requested DataLoaders:
    if train:
        if val:
            return train_loader, val_loader
        else:
            return train_loader
    else:
        return test_loader
