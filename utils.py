from ops import Identity, Sep_Conv, Conv, Stacked_conv, Pooling, Dil_Conv
from functools import partial
import itertools
import torch
from torchvision import datasets, transforms

kernel_sizes = [3, 5, 7]
strides = [1, 2]
in_channels_options = [32, 64] 
num_channels_options = [32, 64, 128]

conv_combinations = [
    partial(Conv, num_channels=num_channels, kernel=kernel, strides=stride)
    for num_channels, kernel, stride in itertools.product(num_channels_options, kernel_sizes, strides)
]

sep_conv_combinations = [
    partial(Sep_Conv, in_channels=in_channels, num_channels=num_channels, kernel=kernel, strides=stride)
    for in_channels, num_channels, kernel, stride in itertools.product(in_channels_options, num_channels_options, kernel_sizes, strides)
]

available_ops = conv_combinations + sep_conv_combinations


def load_cifar10_data():
    """Load and preprocess CIFAR-10 data."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load CIFAR-10 data
    train_data_full = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_size = int(0.8 * len(train_data_full))
    validation_size = len(train_data_full) - train_size
    train_data, validation_data = torch.utils.data.random_split(train_data_full, [train_size, validation_size])

    return train_data, validation_data, test_data


