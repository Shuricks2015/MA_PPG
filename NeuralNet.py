# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from Utils import ppgDaLiA_Dataset, check_accuracy, save_checkpoint, load_checkpoint
from pathlib import Path
import os
from tqdm import tqdm


# import matplotlib.pyplot as plt
# import numpy as np


class CNN(nn.Module):
    def __init__(self, num_classes, in_channels=1):
        """
        Create simple CNN
        :param num_classes: number of classes
        :param in_channels: number of incoming channels
        """
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 18, num_classes)

    def forward(self, x):
        """
        forward pass
        :param x: input x
        :type x: torch.tensor
        :return: x
        """
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

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
    def __init__(self, num_classes, in_channels=1, dropout=0.5):
        super(CNN_try, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
in_channel = 1
num_classes = 1
learning_rate = 1e-5
batch_size = 128
num_epochs = 300

# Load Data
train_set = ppgDaLiA_Dataset(str(Path(os.getcwd())) + '/dataset/new_PPG_DaLiA_train/processed_dataset/')
test_set = ppgDaLiA_Dataset(str(Path(os.getcwd())) + '/dataset/new_PPG_DaLiA_test/processed_dataset/')
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

# Initialize Network
model = CNN_try(in_channels=in_channel, num_classes=num_classes).to(device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((12966 / 30084)))
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)

# Train network
for epoch in range(num_epochs):
    losses = []

    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    for batch_idx, (data, targets) in loop:
        # Get data to cuda
        data = data.to(device=device)
        targets = targets.to(device=device)
        targets = targets.reshape((targets.shape[0], 1))

        # forward
        scores = model(data)
        loss = criterion(scores, targets)
        losses.append(loss)

        # backward
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

        optimizer.zero_grad()

        # update progress bar
        loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
        loop.set_postfix(loss=loss.item())

    # calculate loss over whole epoch
    mean_loss = sum(losses) / len(losses)
    scheduler.step(mean_loss)
    print("Loss for Epoch {} was {}".format(epoch, mean_loss))

    # check accuracy of model after epoch for best performance
    check_accuracy(test_loader, model, device)

    checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "mean_loss": mean_loss}
    save_checkpoint(checkpoint, "checkpoints/checkpoint{}.pth.tar".format(epoch))

# Check accuracy on training & test set
check_accuracy(train_loader, model, device)
