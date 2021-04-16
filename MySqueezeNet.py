import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.utils.prune as prune


class FireModule(nn.Module):
    def __init__(self, in_fts, s_1x1, e_1x1, e_3x3):
        super().__init__()
        self.squeeze_conv = nn.Conv2d(in_channels=in_fts, out_channels=s_1x1, kernel_size=(1, 1))
        self.expand_conv1 = nn.Conv2d(in_channels=s_1x1, out_channels=e_1x1, kernel_size=(1, 1))
        self.expand_conv3 = nn.Conv2d(in_channels=s_1x1, out_channels=e_3x3, kernel_size=(3, 3), stride=(1, 1),
                                      padding=(1, 1))

    def forward(self, inp_image):
        x = self.squeeze_conv(inp_image)
        x1 = self.expand_conv1(x)
        x3 = self.expand_conv3(x)
        x = torch.cat([x1, x3], dim=1)
        return x


class MySqueezeNet(nn.Module):
    def __init__(self, in_fts=3, num_classes=1000):
        super(MySqueezeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_fts, out_channels=96, kernel_size=(7, 7), stride=(2, 2), padding=2)
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.fire2 = FireModule(96, 16, 64, 64)
        self.fire3 = FireModule(128, 16, 64, 64)
        self.fire4 = FireModule(128, 32, 128, 128)
        self.fire5 = FireModule(256, 32, 128, 128)
        self.fire6 = FireModule(256, 48, 192, 192)
        self.fire7 = FireModule(384, 48, 192, 192)
        self.fire8 = FireModule(384, 64, 256, 256)
        self.fire9 = FireModule(512, 64, 256, 256)
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=(1, 1))
        self.avg10 = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, inp_img):
        x = self.conv1(inp_img)
        x = self.maxpool(x)
        x = self.fire2(x)
        x = self.fire3(x)
        x = self.fire4(x)
        x = self.maxpool(x)
        x = self.fire5(x)
        x = self.fire6(x)
        x = self.fire7(x)
        x = self.fire8(x)
        x = self.maxpool(x)
        x = self.fire9(x)
        x = self.conv10(x)
        x = self.avg10(x)
        return x


if __name__ == '__main__':
    x = torch.rand((1, 3, 224, 224))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MySqueezeNet().to(device)
    # print(model(x).shape)
    print(summary(model, (3, 224, 224)))

    # To prune some of the layers
    # Docs: https://pytorch.org/tutorials/intermediate/pruning_tutorial.html

