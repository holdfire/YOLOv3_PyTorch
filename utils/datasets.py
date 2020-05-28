import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.function as F

from utils.augmentations import horisontal_flip
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def pad_to_square(image, pad_value):

    pass