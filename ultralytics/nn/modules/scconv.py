import torch
import torch.nn as nn
import torch.nn.functional as F

class SCConv(nn.Module):
    """
    Simplified SCConv block for YOLOv11 integration.
    Reference: Selective Convolutional Kernel Networks (SCNet)
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, pooling_r=4):
        super(SCConv, self).__init__()
        # Pooling branch: downsamples feature maps to get context
        self.pooling = nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r)
        # Main convolution branches
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        # Normalization + activation
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()  # same activation as YOLOv11

    def forward(self, x):
        # Residual connection (original input)
        residual = x
        # Main convolution path
        x1 = self.conv1(x)
        # Context path (pooled + upsampled)
        x2 = self.pooling(x)
        x2 = F.interpolate(x2, size=x1.shape[-2:], mode='nearest')
        # Selective combination
        x = x1 * torch.sigmoid(x2) + residual
        # Second conv + normalization + activation
        x = self.conv2(x)
        return self.act(self.bn(x))
