from torch.utils.data import Dataset
import torch
import numpy as np


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

