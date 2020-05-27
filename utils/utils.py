from __future__ import division

import math
import time
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlin.patches as patches

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def to_cpu(tensor):
    return tensor.detach().cpu()         # Returns a copy of this object in CPU memory


def load_classes(path):
    """
    Loads class labels at "path"
    :param path:
    :return:
    """
    fp = open(path, 'r')
    names = fp.read().split('\n')[:-1]
    return names


def weight_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def rescale_boxes(boxes, current_dim, original_shape):
    """
    Rescales bounding boxes to the original shape
    :param boxes:
    :param current_dim:
    :param original_dim:
    :return:
    """
    orig_h, orig_w = original_shape
    # the amount of padding that was used
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes


def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def pa_per_class(tp, conf, pred_cls, target_cls):
    """
    Compute hte average precision, givern the recall and precision curves.
    Source: http://github.com/rafaelpadilla/Object-Detection-Metrics.
    :param tp: True positives (list).
    :param conf: Objectness value from 0-1 (list).
    :param pred_cls: Predicted object classes (list).
    :param target_cls: True object classes (list).
    :return: The average precision as computed in py-faster-rcnn.
    """



