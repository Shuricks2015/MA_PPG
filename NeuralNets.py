# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x


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
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x
