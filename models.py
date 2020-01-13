from __future__ import division          # for python2 users

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.parse_config import *
from utils.utils import build_targets, to_cpu, non_max_supression

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def create_modules(module_defs):
    """
    Construct module list of layer blocks from module configuration in module_defs
    :param module_defs:  utils.parse_config.parse_model_config函数返回值
    :return:
    """
    hyperparams = module_defs.pop(0)                 # 取出返回值的第一个元素，即超参数配置信息，是一个dict
    output_filters = [int(hyperparams["channels"])]
    module_list = nn.ModuleList()
    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def['type'] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2
            modules.add_module(
                f"conv_{module_i}",
                nn.Conv2d(
                    in_channels = output_filters[-1],
                    out_channels = filters,
                    kernel_size = kernel_size,
                    stride = int(module_def["stride"]),
                    padding = pad,
                    bias =not bn
                )
            )
            if bn:
                modules.add_module(f"batch_norm_{module_i}", nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5))
            if module_def["activation"] == "leaky":
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))

        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                modules.add_module(f"_debug_padding_{module_i}", nn.ZeroPad2d((0,1,0,1)))
            maxpool = nn.MaxPool2d(kernel_size = kernel_size, stride = stride, padding = int((kernel_size-1) // 2))
            modules.add_module(f"maxpool{module_i}", maxpool)





