from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


class BottleNeckBlock(nn.Module):
    def __init__(self, in_fts, out_fts, expansion=4, if_downsample=False):
        super(BottleNeckBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_fts, out_channels=out_fts, kernel_size=(1, 1), stride=(1, 1))
        self.bn1 = nn.BatchNorm2d(out_fts)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_fts, out_channels=out_fts, kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1))

        if if_downsample and in_fts != 64:
            self.conv2 = nn.Conv2d(in_channels=out_fts, out_channels=out_fts, kernel_size=(3, 3), stride=(2, 2),
                                   padding=(1, 1))

        self.bn2 = nn.BatchNorm2d(out_fts)
        self.conv3 = nn.Conv2d(in_channels=out_fts, out_channels=expansion * out_fts, kernel_size=(1, 1), stride=(1, 1))
        self.bn3 = nn.BatchNorm2d(expansion * out_fts)

        self.if_downsample = if_downsample
        if if_downsample and in_fts == 64:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_fts, out_channels=expansion * out_fts, kernel_size=(1, 1), stride=(1, 1)),
                nn.BatchNorm2d(expansion * out_fts)
            )
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_fts, out_channels=expansion * out_fts, kernel_size=(1, 1), stride=(2, 2)),
                nn.BatchNorm2d(expansion * out_fts)
            )

    def forward(self, input_img):
        if self.if_downsample:
            y = self.downsample(input_img)

        x = self.conv1(input_img)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.if_downsample:
            x = F.relu(x + y)
        else:
            x = F.relu(x + input_img)

        return x


class BasicBlock(nn.Module):
    def __init__(self, in_fts, out_fts, if_downsample=False):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_fts, out_channels=out_fts, kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(out_fts)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_fts, out_channels=out_fts, kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(out_fts)

        self.if_downsample = if_downsample
        if if_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_fts, out_channels=out_fts, kernel_size=(1, 1), stride=(2, 2)),
                nn.BatchNorm2d(out_fts)
            )

    def forward(self, input_img):
        if self.if_downsample:
            x = self.downsample(input_img)
            input_img = x
        else:
            x = self.conv1(input_img)
            x = self.bn1(x)

        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        x = F.relu(x + input_img)

        return x


class MyResNet(nn.Module):
    def __init__(self, num_classes=1000, layer_name='18', in_fts=3, expansion=4):
        super(MyResNet, self).__init__()

        if layer_name == '18' or layer_name == '34':
            self.n_blocks = 2
            self.fc_in = 512
        else:
            self.n_blocks = 3
            self.expansion = expansion
            self.fc_in = 2048

        self.n_filter = [64, 64 * 2, 64 * 4, 64 * 8]
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

        self.layer1 = self.layer_construct(0, self.ResNet[layer_name][0])
        self.layer2 = self.layer_construct(1, self.ResNet[layer_name][1])
        self.layer3 = self.layer_construct(2, self.ResNet[layer_name][2])
        self.layer4 = self.layer_construct(3, self.ResNet[layer_name][3])

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

    def layer_construct(self, idx, n_times):
        if self.n_blocks == 2:
            block = OrderedDict()
            for i in range(n_times):
                if self.in_fts == self.n_filter[idx]:
                    block[str(i)] = BasicBlock(self.in_fts, self.n_filter[idx], False)
                else:
                    block[str(i)] = BasicBlock(self.in_fts, self.n_filter[idx], True)

                self.in_fts = self.n_filter[idx]
        else:
            block = OrderedDict()
            for i in range(n_times):
                if self.in_fts != (self.expansion * self.n_filter[idx]):
                    block[str(i)] = BottleNeckBlock(self.in_fts, self.n_filter[idx], self.expansion, True)
                else:
                    block[str(i)] = BottleNeckBlock(self.in_fts, self.n_filter[idx], self.expansion, False)

                self.in_fts = self.expansion * self.n_filter[idx]

        return nn.Sequential(block)


if __name__ == '__main__':
    x = torch.randn((5, 3, 224, 224))

    m = MyResNet(layer_name='152')
    writer = SummaryWriter('logs/resnet_152')
    writer.add_graph(m, x)
    writer.close()
    # print(m)
    print(m(x).shape)
