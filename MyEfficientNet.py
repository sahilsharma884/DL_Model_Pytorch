import torch
import torch.nn as nn
from torchvision.ops import StochasticDepth

from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
import math
import os
import shutil

# Depthwise Convolution Module
class DepthWise_Conv(nn.Module):
    def __init__(self, in_fts, k, stride=(1,1), padding=(1,1)) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_fts, in_fts, kernel_size=(k,k), stride=stride, padding=padding, groups=in_fts, bias=False),
            nn.BatchNorm2d(in_fts),
            nn.SiLU(inplace=True)
        )

    def forward(self, input_image):
        x = self.conv(input_image)
        return x

class Pointwise_Conv(nn.Module):
    def __init__(self, in_fts, out_fts):
        super(Pointwise_Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_fts, out_fts, kernel_size=(1, 1)),
            nn.BatchNorm2d(out_fts),
            nn.SiLU(inplace=True)
        )

    def forward(self, input_image):
        x = self.conv(input_image)
        return x

class MBConvConfig:
    def __init__(self) -> None:
        pass
 
    def make_divisible(self, channels, divisor=8, min_value=None):
        """
        Makes the number of channels divisible by a divisor.

        Args:
            channels: The original number of channels.
            divisor: The desired divisibility factor (e.g., 8).
            min_value: The minimum allowed value for the adjusted channel count
            (defaults to the divisor).

        Returns:
            The adjusted number of channels, which is divisible by the divisor.
        """
        if min_value is None:
            min_value = divisor

        new_channels = max(min_value, int(math.ceil(channels / (divisor)) * divisor))

        # Clamp to avoid excessive reduction in channel count
        if new_channels < 0.9 * channels:
            new_channels = channels

        return new_channels

    def adjust_channels(self, in_fts, width_mult, min_value=None):
        return self.make_divisible(in_fts * width_mult, 8, min_value)
    
    def adjust_depth(self, num_layers, depth_mult):
       """
       A method that adjusts the number of layers based on the depth multiplier and returns the result as an integer.
       """
       return int(math.ceil(num_layers * depth_mult))

class MBConv(nn.Module):
    def __init__(self, config: MBConvConfig, in_fts, out_fts, k, s, expand_ratio, st_prob) -> None:
        super().__init__()

        self.use_residual = in_fts == out_fts and s == 1

        self.layers = []

        # Expansion Phase
        # Increase the number of channels
        expanded_channels = config.adjust_channels(in_fts, expand_ratio)
        # print(in_fts , expanded_channels)
        if expanded_channels != in_fts:
            self.layers.append(nn.Conv2d(in_fts, expanded_channels, kernel_size=1, groups=2, bias=False))
            self.layers.append(nn.SiLU())

        # Depthwise
        # Applies a lightweight depthwise convolution for feature extraction.
        self.layers.append(DepthWise_Conv(expanded_channels, k, stride=(s, s), padding=(k//2, k//2)))

        # Squeeze and Excitation
        # Optionally uses an SE block to adaptively recalibrate channel importance.
        squeeze_channel = max(1, in_fts // 8)
        self.layers.append(nn.Sequential(
            nn.Conv2d(expanded_channels, squeeze_channel, kernel_size=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(squeeze_channel, out_fts, kernel_size=1, bias=False),
            )
        )

        # Pointwise
        # Applies a lightweight pointwise convolution.
        self.layers.append(Pointwise_Conv(out_fts, out_fts))

        self.blocks = nn.Sequential(*self.layers)

        # Stochastic Depth
        self.st_depth = StochasticDepth(st_prob, "row")

    def forward(self, x):
        x = self.blocks(x)
        if self.use_residual:
            x = x + self.st_depth(x)
        return x

class MyEfficientNet(nn.Module):
    def __init__(self, in_fts=3, config: MBConvConfig = MBConvConfig(), alpha=1.2, beta=1.1, piece=1, st_peice = 0.8, n_classes=1000) -> None:
        super().__init__()
        self.alpha = alpha ** piece # width (channels)
        self.beta = beta ** piece # depth (layers)
        
        self.conv1 = nn.Conv2d(in_fts, config.adjust_channels(in_fts=32, width_mult=self.alpha), kernel_size=3, stride=2, padding=1, bias=False)

        self.st_prob = st_peice

        self.mbconv = [
            # in_fts, out_fts, k, s, expansions, layers
            [32, 16, 3, 1, 1, 1],
            [16, 24, 3, 2, 6, 2],
            [24, 40, 5, 2, 6, 2],
            [40, 80, 3, 2, 6, 3],
            [80, 112, 5, 1, 6, 3],
            [112, 192, 5, 2, 6, 4],
            [192, 320, 3, 1, 6, 1]
        ]

        self.layers = self.layerConstruct(config)

        in_fts = config.adjust_channels(self.mbconv[-1][1], self.alpha, 8)
        out_fts = config.adjust_channels(1280, self.alpha, 8)
        self.final_conv =nn.Sequential(
            nn.Conv2d(in_fts, out_fts, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            nn.AdaptiveAvgPool2d(1)
        )

        self.output = nn.Linear(out_fts, n_classes)

    def layerConstruct(self, config: MBConvConfig):
        block = OrderedDict()

        # list -- 
        # l[0] : in_fts
        # l[1] : out_fts
        # l[2] : k
        # l[3] : s
        # l[4] : expansions
        # l[5] : layers

        for l in self.mbconv:
            # print(l[0], l[1], l[5])
            in_fts = config.adjust_channels(l[0], self.alpha, 8)
            out_fts = config.adjust_channels(l[1], self.alpha, 8)
            numLayers = config.adjust_depth(l[5], self.beta)
            # print(f'After {in_fts}, {out_fts}, {numLayers}')
            for layer_number in range(numLayers):
                if l[3] == 2:
                    block[str(l[0])+str(out_fts)+str(layer_number)] = MBConv(config, in_fts, out_fts, l[2], l[3], l[4], self.st_prob)
                    l[3] = 1
                else:
                    block[str(l[0])+str(out_fts)+str(layer_number)] = MBConv(config, in_fts, out_fts, l[2], 1, l[4], self.st_prob)
                in_fts = out_fts

        return nn.Sequential(block)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layers(x)
        x = self.final_conv(x).view(x.shape[0], -1)
        x = self.output(x)
        return x
    
def drawGrapgh(model, logDir, device, input_size=(1,3, 224, 224)):
    if os.path.exists(logDir):
        shutil.rmtree(logDir)

    x = torch.randn(input_size).to(device)
    writer = SummaryWriter(logDir)

    writer.add_graph(model, x)
    writer.close()

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device connected to', device)

    st_peice = 0.2 # survival probability
    for i in range(1,9):
        m = MyEfficientNet(st_peice=st_peice, piece=i).to(device)
        print(f'EfficientNet B{i-1} : {sum(p.numel() for p in m.parameters())}')
        st_peice += 0.05 # increase survival probability
        del m # to save memory

    # drawGrapgh(m, 'logs/efficientnet_b0',device)