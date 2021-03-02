import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class MyVGG(nn.Module):
    def __init__(self, num_classes, ConvNet='A', in_fts=3):
        super(MyVGG, self).__init__()

        self.VGG_Model = dict()
        self.VGG_Model['A'] = [1, 1, 2, 2, 2]
        self.VGG_Model['B'] = [2, 2, 2, 2, 2]
        self.VGG_Model['C'] = [(3, 2), 'M', (3, 2), 'M', (3, 2), (1, 1), 'M', (3, 2), (1, 1), 'M', (3, 2), (1, 1), 'M']
        self.VGG_Model['D'] = [2, 2, 3, 3, 3]
        self.VGG_Model['E'] = [2, 2, 4, 4, 4]

        self.in_fts = in_fts

        self.features = self.feature_architecture(self.VGG_Model[ConvNet], ConvNet)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)

        )

    def forward(self, input_image):
        N = input_image.shape[0]
        x = self.features(input_image)
        x = self.avgpool(x)
        x = x.reshape(N, -1)
        x = self.classifier(x)
        return x

    def feature_architecture(self, list_param, model='C'):

        list_layer = []

        if model != 'C':
            num_filters = [64, 64 * 2, 64 * 4, 64 * 8, 64 * 8]

            for idx, num_times in enumerate(list_param):
                for _ in range(num_times):
                    list_layer.append(
                        nn.Conv2d(in_channels=self.in_fts, out_channels=num_filters[idx], kernel_size=(3, 3),
                                  stride=(1, 1), padding=(1, 1)))
                    list_layer.append(
                        nn.ReLU()
                    )
                    self.in_fts = num_filters[idx]

                list_layer.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        else:
            num_filters = [64, 64 * 2, 64 * 4, 64 * 4, 64 * 8, 64 * 8, 64 * 8, 64 * 8]
            idx = 0

            for i in list_param:
                if isinstance(i, str) and i == 'M':
                    list_layer.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
                else:
                    (k, num_times) = i
                    for _ in range(num_times):
                        list_layer.append(
                            nn.Conv2d(in_channels=self.in_fts, out_channels=num_filters[idx], kernel_size=(k, k),
                                      stride=(1, 1),
                                      padding=(1, 1)))
                        list_layer.append(
                            nn.ReLU()
                        )
                        self.in_fts = num_filters[idx]
                    idx += 1

        return nn.Sequential(*list_layer)


if __name__ == '__main__':
    x = torch.randn((5, 3, 224, 224))
    num_class = 1000
    writer = SummaryWriter('logs/vgg_C')
    m = MyVGG(num_class, 'C')
    writer.add_graph(m, x)
    writer.close()
    # print(m)
    print(m(x).shape)
