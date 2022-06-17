from torch.utils.data import Dataset
import torch
import numpy as np
import torch.nn as nn


class ppgDaLiA_Dataset(Dataset):
    """
    Helper class to import PPG_Dalia
    """
    def __init__(self, root_dir):
        """
        Init Dataset and normalize it
        """
        self.x = torch.tensor(np.load(root_dir + 'MA_segmented_ppg.npy'), dtype=torch.float32)
        self.y = torch.tensor(np.load(root_dir + 'MA_labels.npy'), dtype=torch.float32)
        self.x = torch.reshape(self.x, (self.x.shape[0], 1, self.x.shape[1]))
        std, mean = torch.std_mean(self.x)
        self.x = (self.x - mean) / std

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def check_accuracy(loader, model, device):
    """
    Calculate binary accuracy with added sigmoid
    :param loader: DataLoader
    :param model: nn.Module
    :return: none
    """
    num_correct = 0
    num_samples = 0

    # Toggle for evaluation
    model.eval()

    with torch.no_grad():
        sig = nn.Sigmoid()
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            y = y.reshape((y.shape[0], 1))

            scores = model(x)
            scores = sig(scores)
            scores = torch.round(scores)

            # _,predictions= scores.max(1)  # only for multiclass classification
            num_correct += (scores == y).sum()
            num_samples += scores.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')

    # Switch back to training
    model.train()
