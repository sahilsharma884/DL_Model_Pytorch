from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class TransitionLayer(nn.Module):
    """
    Essential part of convolutional networks is down-sampling layers that change the size of feature-maps
    """

    def __init__(self, in_fts, out_fts):
        super(TransitionLayer, self).__init__()
        out_fts = int(out_fts)

        self.transition = nn.Sequential(
            nn.BatchNorm2d(in_fts),
            nn.Conv2d(in_channels=in_fts, out_channels=out_fts//2, kernel_size=(1, 1), stride=(1, 1)),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

    def forward(self, input_img):
        x = self.transition(input_img)

        return x


class CompositeFunction(nn.Module):
    """
    To define H_l(.) inside Denseblock
    """

    def __init__(self, in_fts, out_fts):
        super(CompositeFunction, self).__init__()
        self.composite_fn = nn.Sequential(
            nn.BatchNorm2d(in_fts),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_fts, out_channels=4*out_fts, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(4*out_fts),
            nn.ReLU(),
            nn.Conv2d(in_channels=4*out_fts, out_channels=out_fts, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )

    def forward(self, input_img):
        x = self.composite_fn(input_img)

        return x


class DenseBlock(nn.Module):
    def __init__(self, in_fts=3, out_fts=32, n_times=6):
        super(DenseBlock, self).__init__()

        layer_ = []
        for i in range(1, n_times + 1):
            layer_.append(CompositeFunction(in_fts, out_fts))
            in_fts = in_fts + out_fts

        self.n_times = n_times
        self.Network = nn.Sequential(*layer_)

    def forward(self, input_img):
        for i in range(self.n_times):
            x = self.Network[i](input_img)
            input_img = torch.cat((input_img, x), dim=1)

        return input_img


class MyDenseNet(nn.Module):
    def __init__(self, in_fts=3, k=32, L='121', num_classes=1000, compress_ratio=1.0):
        super(MyDenseNet, self).__init__()
        self.DenseNet = dict()
        self.DenseNet['121'] = [6, 12, 24, 16]
        self.DenseNet['169'] = [6, 12, 32, 32]
        self.DenseNet['201'] = [6, 12, 48, 32]
        self.DenseNet['264'] = [6, 12, 64, 48]

        self.conv = nn.Conv2d(in_channels=in_fts, out_channels=2 * k, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        out_dim = k*2

        self.DenseBlock1 = DenseBlock(out_dim, k, self.DenseNet[L][0])
        out_dim += k*self.DenseNet[L][0]
        self.TransitionLayer1 = TransitionLayer(out_dim, compress_ratio * out_dim)
        out_dim = int(compress_ratio * (out_dim//2))

        self.DenseBlock2 = DenseBlock(out_dim, k, self.DenseNet[L][1])
        out_dim += k*self.DenseNet[L][1]
        self.TransitionLayer2 = TransitionLayer(out_dim, compress_ratio * out_dim)
        out_dim = int(compress_ratio * (out_dim//2))

        self.DenseBlock3 = DenseBlock(out_dim, k, self.DenseNet[L][2])
        out_dim += k * self.DenseNet[L][2]
        self.TransitionLayer3 = TransitionLayer(out_dim, compress_ratio * out_dim)
        out_dim = int(compress_ratio * (out_dim//2))

        self.DenseBlock4 = DenseBlock(out_dim, k, self.DenseNet[L][3])
        out_dim += k * self.DenseNet[L][3]

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))

        self.fc = nn.Sequential(
            nn.Linear(out_dim * 7 * 7, num_classes)
        )

    def forward(self, input_img):
        N = input_img.shape[0]
        x = self.conv(input_img)
        x = self.maxpool(x)
        x = self.DenseBlock1(x)
        x = self.TransitionLayer1(x)
        x = self.DenseBlock2(x)
        x = self.TransitionLayer2(x)
        x = self.DenseBlock3(x)
        x = self.TransitionLayer3(x)
        x = self.DenseBlock4(x)
        x = self.avgpool(x)
        x = x.reshape(N, -1)
        x = self.fc(x)

        return x


if __name__ == '__main__':
    x = torch.randn((3, 3, 224, 224))
    num_class = 1000

    m = MyDenseNet(L='121', compress_ratio=0.5)
    # writer = SummaryWriter('logs/densenet_121')
    # print(m)
    # writer.add_graph(m, x)
    # writer.close()
    print(m(x).shape)
