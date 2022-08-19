import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np
import torch.nn as nn
from scipy import signal
import os
import socket


class ppgDaLiA_Dataset(Dataset):
    """
    Helper class to import PPG_Dalia
    """

    def __init__(self, root_dir, recurrence):
        """
        Init Dataset and normalize it
        """
        if recurrence:
            self.x = torch.tensor(np.load(root_dir + 'MA_segmented_recurrence_ppg.npy'), dtype=torch.float32)
            self.y = torch.tensor(np.load(root_dir + 'MA_labels_recurrence.npy'), dtype=torch.float32)
            self.x = torch.reshape(self.x, (self.x.shape[0], 1, self.x.shape[1], self.x.shape[2]))
        else:
            self.x = torch.tensor(np.load(root_dir + 'MA_segmented_ppg.npy'), dtype=torch.float32)
            self.y = torch.tensor(np.load(root_dir + 'MA_labels.npy'), dtype=torch.float32)
            self.x = torch.reshape(self.x, (self.x.shape[0], 1, self.x.shape[1]))
        # std, mean = torch.std_mean(self.x)
        # self.x = (self.x - mean) / std

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class study_Dataset(Dataset):
    """
    Helper class to import study_dataset
    """

    def __init__(self, root_dir, csv):
        """
        Init Dataset
        """
        self.annotations = pd.read_csv(root_dir + csv, header=None)
        self.root_dir = root_dir
        self.filter = filter_creation2()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        data_path = os.path.join(self.root_dir, "Proband_{}/sample{}.npy".format(int(self.annotations.iloc[index, 2]),
                                                                                 int(self.annotations.iloc[index, 0])))
        data = np.load(data_path)
        data = band_filter2(data[:, 0], self.filter)
        data = torch.tensor(data, dtype=torch.float32).view(1, -1)
        label = torch.tensor(self.annotations.iloc[index, 1], dtype=torch.float32)
        return data, label


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

        accuracy = float(num_correct) / float(num_samples) * 100
        print(f'Got {num_correct} / {num_samples} with accuracy {accuracy:.2f}')

    # Switch back to training
    model.train()
    return accuracy


def save_checkpoint(state, file="checkpoint.pth.tar"):
    # print("Saving Checkpoint")
    torch.save(state, file)


def load_checkpoint_w_opti(checkpoint, model, optimizer):
    print("Loading Checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


def load_checkpoint(checkpoint, model):
    print("Loading Checkpoint")
    model.load_state_dict(checkpoint['state_dict'])


def oxygen_estimation(ppg_red, ppg_ir):
    sp_o2 = 110 - 25 * (
                (np.sqrt(np.mean(ppg_red ** 2)) / np.mean(ppg_red)) / (np.sqrt(np.mean(ppg_ir ** 2)) / np.mean(ppg_ir)))
    return sp_o2


def minmax_normalization(ppg):
    if min(ppg) == max(ppg):
        # signal is only 1 value, returns 1 for the whole signal
        return ppg / max(ppg)
    else:
        # returns values in range [0, 1]
        ppg_norm = (ppg - min(ppg)) / (max(ppg) - min(ppg))
        return ppg_norm


def normalization_customrange(data_signal, newmin, newmax):
    if min(data_signal) == max(data_signal):
        print("whole signal was 1 value, normalized to 1")
        return data_signal / max(data_signal)
    else:
        # returns values in range [newmin, newmax]
        a = (newmax - newmin) / (max(data_signal) - min(data_signal))
        b = newmax - a * max(data_signal)
        newsignal = a * data_signal + b
        return newsignal


def band_filter(ppg, sos):
    filtered_ppg = signal.filtfilt(sos, 1.0, ppg)
    ppg_norm = minmax_normalization(filtered_ppg)
    return ppg_norm


def filter_creation():
    fs = 100
    low_end = 0.9
    high_end = 5
    ntaps = 16
    sos = signal.firwin(ntaps, [low_end, high_end], fs=fs, pass_zero='bandpass')
    return sos


def filter_creation2():
    fs = 100
    low_end = 0.9
    high_end = 5
    order = 2
    sos = signal.butter(order, [low_end, high_end], btype='bandpass', fs=fs, output='sos')
    return sos


def band_filter2(ppg, sos):
    filtered_ppg = signal.sosfiltfilt(sos, ppg)
    ppg_norm = normalization_customrange(filtered_ppg, -1, 1)
    return ppg_norm
