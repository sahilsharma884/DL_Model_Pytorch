import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary as summary


class MyVGG(nn.Module):
    def __init__(self, num_classes, ConvNet='A', in_fts=3):
        super(MyVGG, self).__init__()

        # Number of Conv Layer in each block
        self.VGG_Model = dict()
        self.VGG_Model['A'] = [1, 1, 2, 2, 2]
        self.VGG_Model['B'] = [2, 2, 2, 2, 2]
        self.VGG_Model['C'] = [(3, 2), 'M', (3, 2), 'M', (3, 2), (1, 1), 'M', (3, 2), (1, 1), 'M', (3, 2), (1, 1), 'M']
        self.VGG_Model['D'] = [2, 2, 3, 3, 3]
        self.VGG_Model['E'] = [2, 2, 4, 4, 4]

        # Set as in_channel as per argument
        self.in_fts = in_fts

        # Make a Conv layer depending on VGG model
        self.features = self.feature_architecture(self.VGG_Model[ConvNet], ConvNet)
        # Average Pool
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        # First 2 FC's layer with dropout
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
        # Store the number of samples (In case for batch size)
        N = input_image.shape[0]
        # Pass the input image into Stack of Conv layers
        x = self.features(input_image)
        # Pass to Average Pooling
        x = self.avgpool(x)
        # Make it into (Batch, Features)
        x = x.reshape(N, -1)
        # Pass these into FC layers
        x = self.classifier(x)
        # Return result value of that VGG model
        return x

    def feature_architecture(self, list_param, model='C'):

        list_layer = []
        num_filters = 64
        # If VGG is not 'C'
        # Has a special case: perform 1x1 and 3x3. There padding value will be affected
        if model != 'C':
            # Iterating each block
            for num_times in list_param:
                # Stack Conv as per number of times appear in each block
                for _ in range(num_times):
                    # Added it into the list_layer list Conv followed by ReLU
                    list_layer.append(
                        nn.Conv2d(in_channels=self.in_fts, out_channels=num_filters, kernel_size=(3, 3),
                                  stride=(1, 1), padding=(1, 1)))
                    list_layer.append(
                        nn.ReLU()
                    )
                    # Set in_channel as out_channel (i.e. num_filters)
                    self.in_fts = num_filters

                # After each block over, it followed by maxpooling (added into list_layer list)
                list_layer.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
                # We know, that after maxpooling, num_filters in increase by factor 2 until 512
                if num_filters != 512:
                    num_filters *= 2
        else:
            # Iterating over list elements
            for i in list_param:
                # Is list element is 'M' denote Maxpooling, then added into the list_layer list
                if isinstance(i, str) and i == 'M':
                    list_layer.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
                    # We know, that after maxpooling, num_filters in increase by factor 2 until 512
                    if num_filters != 512:
                        num_filters *= 2
                else:
                    # Otherwise list element (kernel_size, num_of_times)
                    (k, num_times) = i
                    # Stack Conv as per number of times appear in each block
                    for _ in range(num_times):
                        # Added it into the list_layer list Conv followed by ReLU
                        list_layer.append(
                            nn.Conv2d(in_channels=self.in_fts, out_channels=num_filters, kernel_size=(k, k),
                                      stride=(1, 1),
                                      padding=(1, 1)))
                        list_layer.append(
                            nn.ReLU()
                        )
                        # Set in_channel as out_channel (i.e. num_filters)
                        self.in_fts = num_filters

        # Make it all list_layer list into Sequential
        return nn.Sequential(*list_layer)


if __name__ == '__main__':
    x = torch.randn((5, 3, 224, 224))
    num_class = 1000
    writer = SummaryWriter('logs/vgg_E')
    m = MyVGG(num_class, 'E')
    writer.add_graph(m, x)
    writer.close()
    # print(m)
    print(m(x).shape)
    print(summary(m, (3,224,224)))
