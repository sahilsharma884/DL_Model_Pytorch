import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

class MyAlexNet(nn.Module):
    def __init__(self, num_classes, in_fts=3):
        super(MyAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=in_fts, out_channels=64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256 * 6 * 6, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=num_classes)
        )

    def forward(self, input_image):
        N = input_image.shape[0]
        x = self.features(input_image)
        x = self.avgpool(x)
        x = x.reshape((N, -1))
        x = self.classifier(x)

        return x


if __name__ == '__main__':
    x = torch.randn((5, 3, 224, 224))
    num_class = 10
    writer = SummaryWriter('logs/alexnet')

    m = MyAlexNet(num_class)
    writer.add_graph(m, x)
    writer.close()
    print(m(x).shape)
    # print(m)
