import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary


class ConvBlock(nn.Module):
    def __init__(self, in_fts, out_fts, k, s, p, act_relu=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_fts, out_channels=out_fts, kernel_size=(k, k), stride=(s, s), padding=p)
        self.bn = nn.BatchNorm2d(out_fts)
        self.act_relu = act_relu
        if self.act_relu:
            self.relu = nn.ReLU()

    def forward(self, input_img):
        x = self.conv(input_img)
        x = self.bn(x)
        if self.act_relu:
            x = self.relu(x)

        return x


class CardinalityBlock(nn.Module):
    def __init__(self, in_fts, out_fts, downsample=False, C=32, expansion=2):
        super(CardinalityBlock, self).__init__()
        list_conv = []
        if out_fts % C == 0:
            c_out_fts = int(int(out_fts / C) / expansion)
        else:
            raise Exception('out_ft should be divisible by C')
        for i in range(C):
            if downsample:
                list_conv.append(
                    nn.Sequential(
                        ConvBlock(in_fts, c_out_fts, 1, 2, 0),
                        ConvBlock(c_out_fts, c_out_fts, 3, 1, 1)
                    )
                )
            else:
                list_conv.append(
                    nn.Sequential(
                        ConvBlock(in_fts, c_out_fts, 1, 1, 0),
                        ConvBlock(c_out_fts, c_out_fts, 3, 1, 1)
                    )
                )

        self.downsample = downsample

        self.conv = ConvBlock(int(out_fts / expansion), out_fts, 1, 1, 0, False)
        self.list_conv = nn.Sequential(*list_conv)
        self.C = C

    def forward(self, input_img):
        x = []
        for i in range(self.C):
            x.append(self.list_conv[i](input_img))
        x = torch.cat(*[x], dim=1)
        x = self.conv(x)

        return x


class MyResNeXt(nn.Module):
    def __init__(self, in_fts=3, num_times=[3, 4, 6, 3], num_classes=1000):
        super(MyResNeXt, self).__init__()
        self.conv1 = ConvBlock(in_fts, 64, 7, 2, 3)
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.num_times = num_times

        num_filters = 128
        expansion = 2

        out_fts = num_filters * expansion

        self.conv2 = self.architecture(64, out_fts, num_times[0])
        self.conv_channel_2 = ConvBlock(64, out_fts, 1, 1, 0, False)

        in_fts = out_fts
        out_fts *= 2
        self.conv3 = self.architecture(in_fts, out_fts, num_times[1], True)
        self.conv_channel_3 = ConvBlock(in_fts, out_fts, 1, 2, 0, False)

        in_fts = out_fts
        out_fts *= 2
        self.conv4 = self.architecture(in_fts, out_fts, num_times[2], True)
        self.conv_channel_4 = ConvBlock(in_fts, out_fts, 1, 2, 0, False)

        in_fts = out_fts
        out_fts *= 2
        self.conv5 = self.architecture(in_fts, out_fts, num_times[3], True)
        self.conv_channel_5 = ConvBlock(in_fts, out_fts, 1, 2, 0, False)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, input_img):
        N = input_img.shape[0]

        x = self.conv1(input_img)
        x = self.maxpool(x)

        y = self.conv_channel_2(x)
        x = self.conv2[0](x)
        x = F.relu(x + y)
        for i in range(1, self.num_times[0]):
            input_img = x
            x = self.conv2[i](x)
            x = F.relu(x + input_img)

        y = self.conv_channel_3(x)
        x = self.conv3[0](x)
        x = F.relu(x + y)
        for i in range(1, self.num_times[1]):
            input_img = x
            x = self.conv3[i](x)
            x = F.relu(x + input_img)

        y = self.conv_channel_4(x)
        x = self.conv4[0](x)
        x = F.relu(x + y)
        for i in range(1, self.num_times[2]):
            input_img = x
            x = self.conv4[i](x)
            x = F.relu(x + input_img)

        y = self.conv_channel_5(x)
        x = self.conv5[0](x)
        x = F.relu(x + y)
        for i in range(1, self.num_times[3]):
            input_img = x
            x = self.conv5[i](x)
            x = F.relu(x + input_img)

        x = self.avgpool(x)
        x = x.reshape(N, -1)
        x = self.fc(x)

        return x

    def architecture(self, in_fts, out_fts, num_times, downsample=False):
        list_conv = [CardinalityBlock(in_fts, out_fts, downsample)]

        for i in range(1, num_times):
            list_conv.append(CardinalityBlock(out_fts, out_fts))

        return nn.Sequential(*list_conv)


if __name__ == '__main__':
    x = torch.randn((5, 3, 224, 224))
    writer = SummaryWriter('logs/resnext_101')
    m = MyResNeXt(num_times=[3, 4, 23, 3], num_classes=10)
    writer.add_graph(m, x)
    writer.close()
    # print(m)
    print(m(x).shape)
    print(summary(m, (3, 224, 224)))
