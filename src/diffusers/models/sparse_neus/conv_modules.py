import torch

# ! amazing!!!! autograd.grad with set_detect_anomaly(True) will cause memory leak
# ! https://github.com/pytorch/pytorch/issues/51349
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from inplace_abn import InPlaceABN

#############################################     MVS Net models        ################################################
class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1,
                 norm_act=InPlaceABN):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride=stride, padding=pad, bias=False)
        # self.bn = norm_act(out_channels)
        # self.bn = nn.BatchNorm2d(out_channels * 8)
        self.bn = nn.GroupNorm(min(out_channels // 2, 16), out_channels, eps=1e-6, affine=True)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        # x = self.conv(x)
        # bv, c, h, w = x.shape
        # x = x.view(bv//8, 8*c, h, w)
        # x = self.act(self.bn(x)).view(bv, c, h, w)
        # return x
        # return self.bn(self.conv(x))
        return self.act(self.bn(self.conv(x)))


class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1,
                 norm_act=InPlaceABN):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels,
                              kernel_size, stride=stride, padding=pad, bias=False)
        # self.bn = norm_act(out_channels)
        # self.bn = nn.ReLU()
        self.bn = nn.BatchNorm3d(out_channels * 8)
        # self.bn = nn.GroupNorm(min(out_channels // 2, 16), out_channels, eps=1e-6, affine=True)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        # x = self.conv(x)
        # bv, c, *sp = x.shape
        # x = x.view(bv//8, 8*c, *sp)
        # x = self.act(self.bn(x)).view(bv, c, *sp)
        # return x
        # return self.bn(self.conv(x))
        return self.act(self.bn(self.conv(x)))