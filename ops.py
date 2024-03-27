import numpy as np
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.nn import functional as F
# from gpu import get_gpu_status
from torchsummary import summary
import gc
from PIL import Image
import glob
import wandb
import os
import copy
import time

class Op(torch.nn.Module):
    """ 
    Op object, basic object
    Each of them operates on a single tensor

    Methods
    -------

    forward : method
        Parameters
        ---------

        input : tensor

        Returns
        -------

        x : tensor

    Usage
    -----

    operation = Op()
    inputs = torch.randn(32, 3, 32, 32)
    output = Op(inputs)


    Note
    ----

    The Op only applies to CIFAR10 image format i.e. 4d tensor 
    with shape [batch_size, num_channel, 32, 32]
    """
    def __init__(self):
        super().__init__()


class Identity(Op):
    """ 
    Identity operation, which is None

    Parameters
    ----------

    num_channels : int
        channel size
    strides : int
        1, or 2
    """

    def __init__(self, num_channels, strides):
        super().__init__()

        if strides == 2:
            self.op = nn.LazyConv2d(num_channels, kernel_size=1, stride=strides, padding=16)
        else:
            self.op = lambda x: x
        self.out_channels = num_channels

    def forward(self, inputs):
        return self.op(inputs)


class Sep_Conv(Op):
    """ 
    Seperable convolution, (depthwise conv -> pointwise conv) -> batchnorm -> relu

    Parameters
    ----------

    in_channels : int
        in_channel size of input
    num_channels : int
        channel size, num of filters
    kernel : int
        kernel size, {3, 5, 7}
    strides : int
        either 1 or 2
    """
    def __init__(self, in_channels, num_channels, kernel, strides):
        super().__init__()
        assert kernel in [3, 5, 7], "kernel not in the range {3, 5, 7}"
        
        if kernel == 3:
            if strides == 1:
                self.conv_d = nn.Conv2d(in_channels, in_channels, kernel_size=kernel, groups = in_channels, stride=strides, padding="same")
            else:
                self.conv_d = nn.Conv2d(in_channels, in_channels, kernel_size=kernel, groups = in_channels, stride=strides, padding=17)
        elif kernel == 5:
            if strides == 1:
                self.conv_d = nn.Conv2d(in_channels, in_channels, kernel_size=kernel, groups = in_channels, stride=strides, padding="same")
            else:
                self.conv_d = nn.Conv2d(in_channels, in_channels, kernel_size=kernel, groups = in_channels, stride=strides, padding=18)
        elif kernel == 7:
            if strides == 1:
                self.conv_d = nn.Conv2d(in_channels, in_channels, kernel_size=kernel, groups = in_channels, stride=strides, padding="same")
            else:
                self.conv_d = nn.Conv2d(in_channels, in_channels, kernel_size=kernel, groups = in_channels, stride=strides, padding=19)
        
        self.conv_p = nn.LazyConv2d(num_channels, kernel_size=1)
        self.bn = nn.LazyBatchNorm2d()
        self.relu = nn.ReLU(inplace=True)
        self.out_channels = num_channels

    def forward(self, inputs):
        x = self.conv_d(inputs)
        x = self.conv_p(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Conv(Op):
    """ 
    Base convolution object 

    Parameters
    ----------

    num_channels : int
        output channels
    kernel : int
        kernel size, {3, 7}
    strides : int
        stride size, {1, 2}
    """

    def __init__(self, num_channels, kernel, strides):
        super().__init__()

        if kernel == 3:
            if strides == 1:
                self.conv = nn.LazyConv2d(num_channels, kernel_size=kernel, stride=strides, padding="same")
            else:
                self.conv = nn.LazyConv2d(num_channels, kernel_size=kernel, stride=strides, padding=17)
        elif kernel == (1, 7):
            if strides == 1:
                self.conv = nn.LazyConv2d(num_channels, kernel_size=kernel, stride=strides, padding="same")
            else:
                self.conv = nn.LazyConv2d(num_channels, kernel_size=kernel, stride=strides, padding=(16, 19))
        elif kernel == (7, 1):
            if strides == 1:
                self.conv = nn.LazyConv2d(num_channels, kernel_size=kernel, stride=strides, padding="same")
            else:
                self.conv = nn.LazyConv2d(num_channels, kernel_size=kernel, stride=strides, padding=(19, 16))        
        self.bn = nn.LazyBatchNorm2d()
        self.relu = nn.ReLU(inplace=True)
        self.out_channels = num_channels

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Stacked_conv(Op):
    """ 
    Stacked convolution of 1 * 7 followed by 7 * 1 convolution

    Parameters
    ----------

    channel_list : list[int]
        e.g. [64, 128]
    stride_list : list[int]
        e.g. [1, 2]

    """
    def __init__(self, channel_list, stride_list, kernel_list=[(1, 7), (7, 1)]):
        super().__init__()
        assert kernel_list == [(1, 7), (7, 1)], "kernel list must be [(1, 7), (7, 1)]"
        assert len(channel_list) == len(kernel_list) and len(kernel_list) == len(stride_list), "List lengths must match"
        self.convs = nn.ModuleList([])
        for _, (c, k, s) in enumerate(zip(channel_list, kernel_list, stride_list)):
            convolution = Conv(c, k, s)
            self.convs.append(convolution)
        self.out_channels = channel_list[1]
    
    def forward(self, inputs):
        x = inputs
        for op in self.convs:
            x = op(x)
        return x


class Pooling(Op):
    """ 
    Pooling operation, two variations
    1. 3 by 3 average pooling
    2. 3 by 3 max pooling

    Parameters
    ----------
    in_channels : int
        input channel number
    type : str
        "max" or "average"
    strides : int
        1 or 2
    """

    def __init__(self, in_channels, type, strides, size = 3):
        super().__init__()
        assert size == 3, "kernel size must be 3"
        self.strides = strides
        if type == "max":
            if strides == 1:
                self.pool = nn.MaxPool2d(size, strides, padding = int(np.floor(size / 2)))
            else:
                self.pad = nn.ZeroPad2d(17)
                self.pool = nn.MaxPool2d(size, strides, padding = 1)
        elif type == "average":
            if strides == 1:
                self.pool = nn.AvgPool2d(size, strides, padding = int(np.floor(size / 2)))
            else:
                self.pad = nn.ZeroPad2d(17)
                self.pool = nn.AvgPool2d(size, strides, padding = 1)
        self.out_channels = in_channels
    
    def forward(self, inputs):
        if self.strides == 2:
            x = self.pad(inputs)
            x = self.pool(x)
        else:
            x = self.pool(inputs)
        return x


class Dil_Conv(Op):
    """ 
    3 by 3Seperable dilated convolution, (depthwise conv -> pointwise conv) -> batchnorm -> relu

    Parameters
    ----------

    in_channels : int
        in_channel size of input
    num_channels : int
        channel size, num of filters
    kernel : int
        kernel size, {3, 5, 7}
    strides : int
        either 1 or 2
    """
    def __init__(self, in_channels, num_channels, strides, kernel=3, dilation=2):
        super().__init__()
        assert kernel == 3, "kernel not equal to 3"
        
        if strides == 1:
            self.conv_d = nn.Conv2d(in_channels, in_channels, kernel_size=kernel, groups = in_channels, stride=strides, dilation=dilation, padding="same")
        else:
            self.conv_d = nn.Conv2d(in_channels, in_channels, kernel_size=kernel, groups = in_channels, stride=strides, dilation=dilation, padding=18)
            
        self.conv_p = nn.LazyConv2d(num_channels, kernel_size=1)
        self.bn = nn.LazyBatchNorm2d()
        self.relu = nn.ReLU(inplace=True)
        self.out_channels = num_channels

    def forward(self, inputs):
        x = self.conv_d(inputs)
        x = self.conv_p(x)
        x = self.bn(x)
        x = self.relu(x)
        return x




# testing

# op_1 = Stacked_conv([64, 128], [2, 1])

# x = torch.randn(32, 3, 32, 32)
# x = op_1(x)

# print(x.shape)

# op_2 = Sep_Conv(op_1.out_channels, 64, 5, 2)
# x = op_2(x)

# op_3 = Identity(128, 2)
# x = op_3(x)

# op_4 = Pooling(op_3.out_channels, "max", 2)
# x = op_4(x)

# op_5 = Dil_Conv(op_4.out_channels, 128, 1)
# x = op_5(x)
# x.shape




        


        

