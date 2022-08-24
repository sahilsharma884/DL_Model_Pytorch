import torch
import torch.nn as nn

from torchinfo import summary

from collections import OrderedDict

# Depthwise Convolution Module
class DepthWise_Conv(nn.Module):
    def __init__(self, in_fts, stride=(1,1)) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_fts, in_fts, kernel_size=(3,3), stride=stride, padding=(1,1), groups=in_fts, bias=False),
            nn.BatchNorm2d(in_fts),
            nn.ReLU6(inplace=True)
        )

    def forward(self, input_image):
        x = self.conv(input_image)
        return x

# PointWise Convolution Module
class Pointwise_Conv(nn.Module):
    def __init__(self, in_fts, out_fts) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_fts, out_fts, kernel_size=(1,1), bias=False),
            nn.BatchNorm2d(out_fts)
        )
    def forward(self, input_image):
        x = self.conv(input_image)
        return x

# Bottleneck Layer when Stride is 1
class NetForStrideOne(nn.Module):
    def __init__(self, in_fts, out_fts, expansion) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_fts, expansion*in_fts, kernel_size=(1,1), bias=False),
            nn.BatchNorm2d(expansion*in_fts),
            nn.ReLU6(inplace=True)
        )
        self.dw = DepthWise_Conv(expansion*in_fts)
        self.pw = Pointwise_Conv(expansion*in_fts, out_fts)

        self.in_fts = in_fts
        self.out_fts = out_fts
        self.expansion = expansion

    def forward(self, input_image):
        if self.expansion == 1:
            x = self.dw(input_image)
            x = self.pw(x)
        else:
            x = self.conv1(input_image)
            x = self.dw(x)
            x = self.pw(x)

        # If input channel and output channel are same, then perform add
        # residual part
        if self.in_fts == self.out_fts:
            x = input_image + x          

        return x

# Bottleneck layer when Stride is 2
class NetForStrideTwo(nn.Module):
    def __init__(self, in_fts, out_fts, expansion) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_fts, expansion*in_fts, kernel_size=(1,1), bias=False),
            nn.BatchNorm2d(expansion*in_fts),
            nn.ReLU6(inplace=True)
        )
        self.dw = DepthWise_Conv(expansion*in_fts, stride=(2,2))
        self.pw = Pointwise_Conv(expansion*in_fts, out_fts)

        self.in_fts = in_fts
        self.out_fts = out_fts
        self.expansion = expansion

    def forward(self, input_image):
        if self.expansion == 1:
            x = self.dw(input_image)
            x = self.pw(x)
        else:
            x = self.conv1(input_image)
            x = self.dw(x)
            x = self.pw(x)      

        return x

# MobileNetV2 architecture
class MyMobileNet_v2(nn.Module):
    def __init__(self, bottleneckLayerDetails, in_fts=3, numClasses=1000, width_multiplier=1) -> None:
        super().__init__()
        self.bottleneckLayerDetails = bottleneckLayerDetails
        self.width_multiplier = width_multiplier

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_fts, round(width_multiplier*32), kernel_size=(3,3), stride=(2,2), padding=(1,1), bias=False),
            nn.BatchNorm2d(round(width_multiplier*32)),
            nn.ReLU6(inplace=True)
        )
        self.in_fts = round(width_multiplier*32)
        
        # Defined bottleneck layer as per Table 2
        self.layerConstructed = self.constructLayer()

        # Top layers after bottleneck
        self.feature = nn.Sequential(
            nn.Conv2d(self.in_fts, round(width_multiplier*1280), kernel_size=(1,1), bias=False),
            nn.BatchNorm2d(round(width_multiplier*1280)),
            nn.ReLU6(inplace=True)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))

        self.outputLayer = nn.Sequential(
            nn.Dropout2d(),
            nn.Conv2d(round(width_multiplier*1280), numClasses, kernel_size=(1,1)),
        )

    def forward(self, input_image):
        x = self.conv1(input_image)
        x = self.layerConstructed(x)
        x = self.feature(x)
        x = self.avgpool(x)
        x = self.outputLayer(x)
        return x

    # Defined function to construct the layer based on bottleneck layer defined in Table 2
    def constructLayer(self):
        itemIndex = 0
        block = OrderedDict()
        # iterating the defined layer details
        for lItem in self.bottleneckLayerDetails:
            # each items assigned corresponding values
            t, out_fts, n, stride = lItem
            # If width multipler is mentioned then perform this line
            out_fts = round(self.width_multiplier*out_fts)
            # for stride value 1
            if stride == 1:
                # constructedd the NetForStrideOne module by n times
                for nItem in range(n):
                    block[str(itemIndex)+"_"+str(nItem)] = NetForStrideOne(self.in_fts, out_fts, t)
                    self.in_fts = out_fts
            # for stride value 2
            elif stride == 2:
                # First layer constructed for NetForStrideTwo module once only
                block[str(itemIndex)+"_"+str(0)] = NetForStrideTwo(self.in_fts, out_fts, t)
                self.in_fts = out_fts
                # Remaining will be NetForStrideOne module (n-1) times
                for nItem in range(1,n):
                    block[str(itemIndex)+"_"+str(nItem)] = NetForStrideOne(self.in_fts, out_fts, t)
            itemIndex += 1

        return nn.Sequential(block)

if __name__ == "__main__":
    # as per research paper
    bottleneckLayerDetails = [
        # (expansion, out_dimension, number_of_times, stride)
            (1,16,1,1),
            (6,24,2,2),
            (6,32,3,2),
            (6,64,4,2),
            (6,96,3,1),
            (6,160,3,2),
            (6,320,1,1)
        ]

    model = MyMobileNet_v2(bottleneckLayerDetails, width_multiplier=1)
    summary(model, (1,3,224,224))
