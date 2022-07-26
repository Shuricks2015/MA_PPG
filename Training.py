# Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Utils import ppgDaLiA_Dataset, check_accuracy, save_checkpoint
from pathlib import Path
import os
from tqdm import tqdm
from NeuralNets import CNN, CNN_try, VGG19_1D, CNN_Amir_Zargari, CNN_2D
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import ImageGrid


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
# Hyperparameters
in_channels = 1
num_classes = 1
learning_rate = 1e-4
batch_size = 64
num_epochs = 100
recurrence = False

# Accuracy
accuracy = torch.empty(num_epochs)

# Load Data
train_set = ppgDaLiA_Dataset(str(Path(os.getcwd())) + '/dataset/new_PPG_DaLiA_train/processed_dataset/', recurrence)
test_set = ppgDaLiA_Dataset(str(Path(os.getcwd())) + '/dataset/new_PPG_DaLiA_test/processed_dataset/', recurrence)
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

# Initialize Network
if recurrence:
    model = CNN_2D(in_channels=1, num_classes=num_classes).to(device)
else:
    model = CNN_try(in_channels=in_channels, num_classes=num_classes, dropout=0.75).to(device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((10001 / 24359)))
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)

"""
# Visualization
mpl.rcParams['axes.linewidth'] = 3
data, targets = next(iter(train_loader))
data = data.reshape((batch_size, 192, 192))

fig = plt.figure(figsize=(20,10))
grid = ImageGrid(fig, 111, nrows_ncols=(4, int(batch_size/4)), axes_pad=0.2, share_all=True)
for i, ax in enumerate(grid):
    ax.imshow(data[i], origin='lower')
    plt.setp(ax.spines.values(), color='red' if targets[i] == 1 else 'green')
grid[0].get_yaxis().set_ticks([])
grid[0].get_xaxis().set_ticks([])
plt.show()

exit()
"""

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
    accuracy[epoch] = check_accuracy(test_loader, model, device)

    checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'mean_loss': mean_loss}
    save_checkpoint(checkpoint, "checkpoints/checkpoint{}.pth.tar".format(epoch))

# Check accuracy on training & test set
print(torch.max(accuracy), torch.argmax(accuracy))
check_accuracy(train_loader, model, device)
check_accuracy(test_loader, model, device)
