# Author: Nils Froehling

# Imports
import torch
import torch.nn as nn

###################################
# Define different neural networks
###################################


class CNN(nn.Module):
    def __init__(self, num_classes=1, in_channels=1, dropout=0.25):
        """
        Create simple CNN
        :param num_classes: number of classes
        :param in_channels: number of incoming channels
        """
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, out_channels=16, kernel_size=10, padding=0),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=10, padding=0),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.BatchNorm1d(16),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=10, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=10, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(2208, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256,num_classes)
        )

    def forward(self, x):
        """
        forward pass
        :param x: input x
        :type x: torch.tensor
        :return: x
        """
        x = self.features(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)

        return x


class VGG19_1D(nn.Module):
    def __init__(self, num_classes, in_channels=1, dropout=0.5):
        """
        Create VGG19 model in 1D
        :param num_classes: number of classes
        :param in_channels: number of incoming channels
        :param dropout: dropout probability
        """
        super(VGG19_1D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool1d(7)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        """
        forward pass
        :param x: input x
        :type x: torch.tensor
        :return: x
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class CNN_try(nn.Module):
    def __init__(self, num_classes=1, in_channels=1, dropout=0.5):
        super(CNN_try, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(8),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.BatchNorm1d(8),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(16),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.BatchNorm1d(16),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.BatchNorm1d(32),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.BatchNorm1d(64),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.BatchNorm1d(128)
        )
        self.classifier = nn.Sequential(
            nn.Linear(6*128, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(p=dropout),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(32),
            nn.Dropout(p=dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# CNN after paper from Amir Zargari
class CNN_Amir_Zargari(nn.Module):
    def __init__(self, num_classes=1, in_channels=1):
        super(CNN_Amir_Zargari, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, out_channels=70, kernel_size=10),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=70, out_channels=70, kernel_size=10),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3),
            nn.Conv1d(in_channels=70, out_channels=140, kernel_size=10),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=140, out_channels=140, kernel_size=10),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2)
            )
        self.classifier = nn.Sequential(
            nn.Linear(140*20, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x


# 2D-CNN used for RecurrencePlots
class CNN_2D(nn.Module):
    def __init__(self, num_classes=1, in_channels=1, dropout=0.5):
        super(CNN_2D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=32, kernel_size=2, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64*47*47, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class CNN_big_kernel(nn.Module):
    def __init__(self, num_classes=1, in_channels=1, dropout=0.5):
        super(CNN_big_kernel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, 12, 2, 5, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, 12, 2, 5, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 12, 2, 5, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 12, 2, 5, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 32, 1, 1, 0, bias=False),
            nn.BatchNorm1d(32)
        )
        self.classifier = nn.Sequential(
            nn.Linear(12*32, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


class InceptionNet(nn.Module):
    def __init__(self, num_classes=1, in_channels=1):
        super(InceptionNet, self).__init__()

        self.conv1 = conv_block(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv2 = conv_block(64, 192, kernel_size=3, stride=2, padding=1)
        self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # order = in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool
        self.inception3a = inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.inception4a = inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.inception5a = inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inception_block(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AvgPool1d(kernel_size=3, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc1(x)

        return x


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, **kwargs),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class inception_block(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool):
        super(inception_block, self).__init__()
        self.branch1 = conv_block(in_channels, out_1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            conv_block(in_channels, red_3x3, kernel_size=1),
            conv_block(red_3x3, out_3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, red_5x5, kernel_size=1),
            conv_block(red_5x5, out_5x5, kernel_size=5, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            conv_block(in_channels, out_1x1pool, kernel_size=1)
        )

    def forward(self, x):
        return torch.cat(
            [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1
        )
