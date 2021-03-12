import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


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
    def __init__(self, in_fts, out_fts, C=32, expansion=2):
        super(CardinalityBlock, self).__init__()
        list_conv = []
        if out_fts % C == 0:
            c_out_fts = int(out_fts / C)
        else:
            raise Exception('out_ft should be divisible by C')
        for i in range(C):
            list_conv.append(
                nn.Sequential(
                    ConvBlock(in_fts, c_out_fts, 1, 1, 0),
                    ConvBlock(c_out_fts, c_out_fts, 3, 1, 1)
                )
            )
        self.conv = ConvBlock(out_fts, expansion * out_fts, 1, 1, 0, False)
        self.list_conv = nn.Sequential(*list_conv)
        # self.list_conv = list_conv
        self.C = C

    def forward(self, input_img):
        x = []
        for i in range(self.C):
            x.append(self.list_conv[i](input_img))

        x = torch.cat(*[x], dim=1)
        x = self.conv(x)

        return x


class MyResNeXt(nn.Module):
    def __init__(self, in_fts=3):
        super(MyResNeXt, self).__init__()
        self.conv1 = ConvBlock(in_fts, 64, 7, 2, 3)
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.conv1_x = CardinalityBlock(64, 128)

    def forward(self, input_img):
        x = self.conv1(input_img)
        x = self.maxpool(x)
        x = self.conv1_x(x)

        return x


if __name__ == '__main__':
    x = torch.randn((5, 3, 224, 224))
    writer = SummaryWriter('logs/resnext_50')
    m = MyResNeXt()
    writer.add_graph(m, x)
    writer.close()
    print(m)
    print(m(x).shape)
