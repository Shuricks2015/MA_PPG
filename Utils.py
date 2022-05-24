from torch.utils.data import Dataset
import pandas as pd
import os
import wfdb
import torch
import numpy as np


class customDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.annotations = pd.read_csv(csv_file)
        length_anno = len(self.annotations)
        self.record = torch.empty(size=(length_anno, 1, 300))
        self.y_label = torch.empty(size=(length_anno, 1))
        for idx in np.arange(length_anno):
            signal_path = os.path.join(root_dir, str(self.annotations.iloc[idx, 0]),
                                       str(self.annotations.iloc[idx, 0]) + '_PPG')
            self.record[idx] = torch.tensor(wfdb.rdrecord(signal_path).p_signal, dtype=torch.float32)
            self.y_label[idx] = torch.tensor(float(self.annotations.iloc[idx, 1]))

        std, mean = torch.std_mean(self.record)
        self.record = (self.record - mean) / std

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        return self.record[index], self.y_label[index]
