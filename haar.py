import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math


def LL(x):
    n, c, h, w = x.size()
    filter =  np.array([[1,1],[1,1]], np.float32)
    filter = np.tile(filter, (c, c, 1, 1))
    filter = torch.as_tensor(filter)
    LL_x = F.conv2d(x, filter.to(x.device), stride=2) / 2
    return LL_x

def LH(x):
    n, c, h, w = x.size()
    filter =  np.array([[1,1],[-1,-1]], np.float32)
    filter = np.tile(filter, (c, c, 1, 1))
    filter = torch.as_tensor(filter)
    LH_x = F.conv2d(x, filter.to(x.device), stride=2) / 2
    return LH_x

def HL(x):
    n, c, h, w = x.size()
    filter =  np.array([[1,-1],[1,-1]], np.float32)
    filter = np.tile(filter, (c, c, 1, 1))
    filter = torch.as_tensor(filter)
    HL_x = F.conv2d(x, filter.to(x.device), stride=2) / 2
    return HL_x

def HH(x):
    n, c, h, w = x.size()
    filter =  np.array([[1,-1],[-1,1]], np.float32)
    filter = np.tile(filter, (c, c, 1, 1))
    filter = torch.as_tensor(filter)
    HH_x = F.conv2d(x, filter.to(x.device), stride=2) / 2
    return HH_x

def inverse_haar(LL, HL, LH, HH):
    n, c, h, w = LL.size()
    filter_LL = np.array([[1, 1], [1, 1]], np.float32)
    filter_HL = np.array([[1, -1], [1, -1]], np.float32)
    filter_LH = np.array([[1, 1], [-1, -1]], np.float32)
    filter_HH = np.array([[1, -1], [-1, 1]], np.float32)
    filter_LL = np.tile(filter_LL, (c, c, 1, 1))
    filter_HL = np.tile(filter_HL, (c, c, 1, 1))
    filter_LH = np.tile(filter_LH, (c, c, 1, 1))
    filter_HH = np.tile(filter_HH, (c, c, 1, 1))
    filter_LL = torch.tensor(filter_LL)
    filter_HL = torch.tensor(filter_HL)
    filter_LH = torch.tensor(filter_LH)
    filter_HH = torch.tensor(filter_HH)

    LL_upsampled = F.conv_transpose2d(LL / 2, filter_LL.to(LL.device), stride=2)
    HL_upsampled = F.conv_transpose2d(HL / 2, filter_HL.to(LL.device), stride=2)
    LH_upsampled = F.conv_transpose2d(LH / 2, filter_LH.to(LL.device), stride=2)
    HH_upsampled = F.conv_transpose2d(HH / 2, filter_HH.to(LL.device), stride=2)
    x_reconstructed = LL_upsampled + HL_upsampled + LH_upsampled + HH_upsampled
    return x_reconstructed
