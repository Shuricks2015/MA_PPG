import os
import numpy as np
from pathlib import Path

data_dir_train = str(Path(os.getcwd())) + '/dataset/new_PPG_DaLiA_train/processed_dataset/'
data_dir_test = str(Path(os.getcwd())) + '/dataset/new_PPG_DaLiA_test/processed_dataset/'

label_train = np.empty(34360)
label_test = np.empty(8690)

x_train = np.load(data_dir_train + 'scaled_ppgs.npy')
y_train = np.load(data_dir_train + 'seg_labels.npy')
x_test = np.load(data_dir_test + 'scaled_ppgs.npy')
y_test = np.load(data_dir_test + 'seg_labels.npy')

# Now take 3s intervals from 30s signals (Frequency: 64Hz, samples in one 30s signal: 1920)
x_train = np.reshape(x_train, (34360, 192))
y_train = np.reshape(y_train, (34360, 192))
x_test = np.reshape(x_test, (8690, 192))
y_test = np.reshape(y_test, (8690, 192))

# If there are more than 19 samples marked as artifact choose bad quality for segment (SNR > 90%)
for index, data in enumerate(y_train[:]):
    if np.count_nonzero(data == 1) > 19:
        label_train[index] = 1
    else:
        label_train[index] = 0

for index, data in enumerate(y_test[:]):
    if np.count_nonzero(data == 1) > 19:
        label_test[index] = 1
    else:
        label_test[index] = 0

# save segments and respective labels
np.save(data_dir_train + 'MA_segmented_ppg.npy', x_train)
np.save(data_dir_train + 'MA_labels.npy', label_train)
np.save(data_dir_test + 'MA_segmented_ppg.npy', x_test)
np.save(data_dir_test + 'MA_labels.npy', label_test)

print(np.count_nonzero(label_train == 1) + np.count_nonzero(label_test == 1))
print(len(label_train))
print(np.count_nonzero(label_train == 1))
print(len(label_test))
print(np.count_nonzero(label_test==1))
