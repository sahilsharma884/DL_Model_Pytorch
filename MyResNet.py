from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary


class BottleNeckBlock(nn.Module):
    def __init__(self, in_fts, out_fts, expansion=4, shortcut=False):
        super(BottleNeckBlock, self).__init__()
        # Conv 1x1, n_filters
        self.conv1 = nn.Conv2d(in_channels=in_fts, out_channels=out_fts, kernel_size=(1, 1), stride=(1, 1))
        self.bn1 = nn.BatchNorm2d(out_fts)
        self.relu = nn.ReLU()

        # Conv 3x3 n_filters
        self.conv2 = nn.Conv2d(in_channels=out_fts, out_channels=out_fts, kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(out_fts)

        # Conv 1x1 expansion times n_filters
        self.conv3 = nn.Conv2d(in_channels=out_fts, out_channels=expansion * out_fts, kernel_size=(1, 1), stride=(1, 1))
        self.bn3 = nn.BatchNorm2d(expansion * out_fts)

        self.shortcut = shortcut
        if self.shortcut:
            # When one block of conv2_x run first time,
            # we get input_img.shape = (N,56,56,64) and output_shape = (N,56,56,256)
            # But num_filters are not same. In order to add both, we need to make expansion times num_filters
            if in_fts == 64:
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_channels=in_fts, out_channels=expansion * out_fts, kernel_size=(1, 1), stride=(1, 1)),
                    nn.BatchNorm2d(expansion * out_fts)
                )
            else:
                # afterwards from 2nd Conv block, first time in each block, we need to reduce input size to half it.
                self.conv2 = nn.Conv2d(in_channels=out_fts, out_channels=out_fts, kernel_size=(3, 3), stride=(2, 2),
                                       padding=(1, 1))

                # The projection shortcut is used to match dimensions (done by 1×1 convolutions).
                # When the shortcuts go across feature maps of two sizes, they are performed with a stride of 2.
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_channels=in_fts, out_channels=expansion * out_fts, kernel_size=(1, 1), stride=(2, 2)),
                    nn.BatchNorm2d(expansion * out_fts)
                )

    def forward(self, input_img):
        if self.shortcut:
            y = self.downsample(input_img)

        x = self.conv1(input_img)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.shortcut:
            x = F.relu(x + y)
        else:
            x = F.relu(x + input_img)
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_fts, out_fts, shortcut=False):
        super(BasicBlock, self).__init__()
        # Conv 3x3, n_filters
        self.conv1 = nn.Conv2d(in_channels=in_fts, out_channels=out_fts, kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(out_fts)
        self.relu = nn.ReLU()

        # Conv 3x3, n_filters
        self.conv2 = nn.Conv2d(in_channels=out_fts, out_channels=out_fts, kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(out_fts)

        # The projection shortcut is used to match dimensions (done by 1×1 convolutions).
        # When the shortcuts go across feature maps of two sizes, they are performed with a stride of 2.
        self.shortcut = shortcut
        if self.shortcut:
            self.conv1 = nn.Conv2d(in_channels=in_fts, out_channels=out_fts, kernel_size=(3, 3), stride=(2, 2),
                                   padding=(1, 1))
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_fts, out_channels=out_fts, kernel_size=(1, 1), stride=(2, 2)),
                nn.BatchNorm2d(out_fts)
            )

    def forward(self, input_img):
        # One conv block to another conv block
        if self.shortcut:
            x_1 = self.downsample(input_img)
            x = self.conv1(input_img)
            x = self.bn1(x)
            input_img = x_1
        else:
            x = self.conv1(input_img)
            x = self.bn1(x)

        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        x = F.relu(x + input_img)

        return x


class MyResNet(nn.Module):
    def __init__(self, num_classes=1000, layer_name='18', in_fts=3, num_filter=64, expansion=4):
        super(MyResNet, self).__init__()

        if layer_name == '18' or layer_name == '34':
            self.n_blocks = 2
            self.fc_in = 512
        else:
            self.n_blocks = 3
            self.expansion = expansion
            self.fc_in = 2048

        self.n_filter = num_filter

        self.ResNet = dict()
        self.ResNet['18'] = [2, 2, 2, 2]
        self.ResNet['34'] = [3, 4, 6, 3]
        self.ResNet['50'] = [3, 4, 6, 3]
        self.ResNet['101'] = [3, 4, 23, 3]
        self.ResNet['152'] = [3, 8, 36, 3]

        self.in_fts = 64
        self.conv = nn.Conv2d(in_channels=in_fts, out_channels=64, kernel_size=(7, 7), stride=(2, 2),
                              padding=(3, 3))
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        self.layer1 = self.layer_construct(self.ResNet[layer_name][0])
        self.layer2 = self.layer_construct(self.ResNet[layer_name][1])
        self.layer3 = self.layer_construct(self.ResNet[layer_name][2])
        self.layer4 = self.layer_construct(self.ResNet[layer_name][3])

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(self.fc_in, num_classes)

    def forward(self, input_img):
        N = input_img.shape[0]
        x = self.conv(input_img)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(N, -1)
        x = self.fc(x)

        return x

    def layer_construct(self, n_times):
        # For 18 and 34 layers of ResNet
        if self.n_blocks == 2:
            block = OrderedDict()
            for i in range(n_times):
                if self.in_fts == self.n_filter:
                    block[str(i)] = BasicBlock(self.in_fts, self.n_filter, False)
                else:
                    block[str(i)] = BasicBlock(self.in_fts, self.n_filter, True)

                self.in_fts = self.n_filter

        # For 50,101,152 layers of ResNet
        else:
            block = OrderedDict()
            for i in range(n_times):
                if self.in_fts == (self.expansion * self.n_filter):
                    block[str(i)] = BottleNeckBlock(self.in_fts, self.n_filter, self.expansion, False)
                else:
                    block[str(i)] = BottleNeckBlock(self.in_fts, self.n_filter, self.expansion, True)

                self.in_fts = self.expansion * self.n_filter

        self.n_filter *= 2
        return nn.Sequential(block)


if __name__ == '__main__':
    x = torch.randn((5, 3, 224, 224))

    m = MyResNet(layer_name='50')
    writer = SummaryWriter('logs/resnet_50')
    writer.add_graph(m, x)
    writer.close()
    # print(m)
    print(m(x).shape)
    print(summary(m, (3,224,224)))