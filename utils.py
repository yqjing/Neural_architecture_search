from ops import Identity, Sep_Conv, Conv, Stacked_conv, Pooling, Dil_Conv
from functools import partial
import itertools
import torch
from torchvision import datasets, transforms
import numpy as np
import copy
import random
import matplotlib.pyplot as plt


action_ls_full = ["identity", "3*3 dconv",  "5*5 dconv", "3*3 conv", "5*5 conv", "1*7-7*1 conv", "3*3 dil conv", "3*3 maxpool", "3*3 avgpool"]
action_ls_input = ["identity", "3*3 dconv",  "5*5 dconv", "3*3 conv", "5*5 conv"]



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



def input_ed_generator(random_pre):
    ed_input = []
    action_ls = copy.deepcopy(action_ls_input)
    random_p = random.random()
    if random_p >= random_pre:
        ed_input.append('identity')
        del action_ls[0]
    else:
        del action_ls[0]
    len_remain = 3 - len(ed_input)
    ed_input = ed_input + [random.choice(action_ls) for _ in range(len_remain)]
    return ed_input

def normal_ed_generator(random_pre):
    ed_input = []
    action_ls = copy.deepcopy(action_ls_full)
    random_p = random.random()
    if random_p >= random_pre:
        ed_input.append('identity')
        del action_ls[0]
    else:
        del action_ls[0]
    num_ls = [2, 3, 4]
    num_actions = random.choice(num_ls)
    len_remain = num_actions - len(ed_input)
    ed_input = ed_input + [random.choice(action_ls) for _ in range(len_remain)]
    return ed_input

def full_ed_generator(random_pre):
    """
    Net encoding generator

    Parameters
    ----------

    random_pre : float
        [0, 1],
        if randomly chosen probability random_p > random_pre,
        assign "identity" operation to the first position of the action list

    Variables in the encoding of a cell
    -----------------------------------

    [num_channels: int, num_blocks: int, action_list: ["identity", "3*3 avgpool", "1*7-7*1 conv"]]

    num_channel : int
        randomly chosen from the list [24, 40, 64, 80, 128, 256]
    action_list : List[str]
        a list of string, e.g. ["identity", "3*3 avgpool", "1*7-7*1 conv"],
        randomly initialized.

    """

    net_ed = []
    channel_list = [24, 40, 64, 80, 128, 256]
    ed_cell_0 = [np.inf, 1, input_ed_generator(random_pre)]
    ed_cell_1 = [random.choice(channel_list), 1, normal_ed_generator(random_pre)]
    ed_cell_2 = [random.choice(channel_list), 1, normal_ed_generator(random_pre)]
    ed_cell_3 = [random.choice(channel_list), 1, normal_ed_generator(random_pre)]
    ed_cell_4 = [random.choice(channel_list), 1, normal_ed_generator(random_pre)]

    net_ed.append(ed_cell_0)
    net_ed.append(ed_cell_1)
    net_ed.append(ed_cell_2)
    net_ed.append(ed_cell_3)
    net_ed.append(ed_cell_4)

    return net_ed

import matplotlib.pyplot as plt
import numpy as np

def plot_best_gen(list1, list2, outer, inner, patience):
    fig, ax1 = plt.subplots(figsize=(8, 8))
    prev_value = None
    
    for i, value in enumerate(list1):
        if i == 0 or value != prev_value:
            ax1.scatter(i, value, c='r', zorder=5, s=13)
        prev_value = value
    
    line1, = ax1.plot(list1, 'r--', linewidth=1, label="Best Generation")
    x = np.linspace(0, len(list1), len(list1))
    ax1.plot(x, x, 'b-.', linewidth=0.5)
    
    ax1.axis('equal')
    ax1.grid(True, linestyle='--')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Generation best individual from')

    ax2 = ax1.twinx() 
    for i, value in enumerate(list2):
        if i == 0 or value != prev_value:
            ax2.scatter(i, value, c='g', marker='v', zorder=5, s=13)
        prev_value = value
    line2, = ax2.plot(list2, 'g-.', linewidth=1, label="Best Accuracy")
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(ymin=min(list2) - 0.2, ymax=max(list2) + 0.2)
    
    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper left")

    plt.savefig(f"plotting/best_gen_{outer}_{inner}_{patience}.png", dpi=300)
    plt.close()

  
def parallelize(model):
    return torch.nn.DataParallel(model)
