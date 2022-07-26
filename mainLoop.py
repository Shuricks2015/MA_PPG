import Utils
import matplotlib.pyplot as plt
import socket
import numpy as np
import torch
import torch.nn as nn
from NeuralNets import CNN_try
import time
from scipy import signal
import threading

Reading_time = 3
numToResample = 64 * Reading_time
numOfSamples = 100 * Reading_time
halfNumOfSamples = int(numOfSamples / 2)
loops = 19
numOfChannels = 4
sos = Utils.filter_creation()

# Load model
model = CNN_try()
Utils.load_checkpoint(torch.load("checkpoints/checkpoint9.pth.tar"), model)
print(sum(p.numel() for p in model.parameters()))

data = np.loadtxt("samplesAndVideos/2022-07-14_h17-m25-s26_PPGdataFromEthernet.csv", delimiter=",")

"""
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
ip_address = '192.168.137.2'
ip_port = 4000
s.connect((ip_address, ip_port))
print('Connection established')

x = s.makefile("r")
"""

ppg = np.empty((numOfSamples, numOfChannels))
ppg_filtered = np.empty((numOfSamples, numOfChannels))
ppg_resa = np.empty((numToResample, numOfChannels))
ppg_resa_norm = np.empty((numToResample, numOfChannels))
complete_ppg = np.empty((numOfSamples + halfNumOfSamples * (loops - 1), numOfChannels))
complete_resa_ppg = np.empty((numToResample + int(numToResample/2) * (loops - 1), numOfChannels))
prediction = torch.empty((loops, numOfChannels))

for count in range(loops):
    if count == 0:
        #for i in range(numOfSamples):
            #messageStr = x.readline().split(",")
            #ppg[i, :] = messageStr[:numOfChannels]
        ppg = data[:numOfSamples, :4]
        complete_ppg[0:numOfSamples] = ppg
    else:
        ppg[:halfNumOfSamples] = old_ppg
        #for i in range(halfNumOfSamples):
            #messageStr = x.readline().split(",")
            #ppg[halfNumOfSamples + i, :] = messageStr[:numOfChannels]
        ppg[halfNumOfSamples:, :] = data[numOfSamples + (count-1)*halfNumOfSamples:numOfSamples+count*halfNumOfSamples, :4]
        complete_ppg[numOfSamples + (count - 1) * halfNumOfSamples:numOfSamples + count * halfNumOfSamples] = ppg[
                                                                                                              halfNumOfSamples:]

    st = time.time()
    # print(Utils.oxygen_estimation(ppg[:, 2], ppg[:, 3]))

    # Normalize Data using min-max Normalization
    for index in range(numOfChannels):
        ppg_filtered[:, index] = Utils.band_filter(ppg[:, index], sos)
        ppg_resa[:, index] = signal.resample(ppg_filtered[:, index], numToResample)
        ppg_resa_norm[:, index] = Utils.minmax_normalization(ppg_resa[:,index])
        """
        if count == 0:
            complete_resa_ppg[:numToResample, index] = ppg_resa_norm[:, index]
        else:
            complete_resa_ppg[numToResample + (count - 1) * int(numToResample/2):numToResample + count * int(numToResample/2), index] = ppg_resa_norm[int(numToResample/2):,index]
        """

    # Prep data for model input
    ppgTensor = ((torch.tensor(ppg_resa_norm, dtype=torch.float32)).permute(1, 0)).reshape(numOfChannels, 1,
                                                                                           numToResample)

    # Predict Signal Quality
    with torch.no_grad():
        sig = nn.Sigmoid()
        prediction[count] = sig(model(ppgTensor)).flatten()
        prediction[count] = torch.round(prediction[count])
        # (prediction[count])

    et1 = time.time()
    old_ppg = ppg[halfNumOfSamples:, :]
    # print("Execution time for code was {}".format(et1 - st))

# Utils.get_edges(prediction)
for index in range(numOfChannels):
    complete_ppg[:, index] = Utils.band_filter(complete_ppg[:, index], sos)
    complete_resa_ppg[:, index] = signal.resample(complete_ppg[:, index], numToResample*10)
    complete_resa_ppg[:, index] = Utils.minmax_normalization(complete_resa_ppg[:, index])

# Plot the extracted signal
ppgRRED = ppg[:, 0]
ppgRIR = ppg[:, 1]
ppgLRED = ppg[:, 2]
ppgLIR = ppg[:, 3]
Fs = 100
FsR = 64
taxis = np.linspace(0, (len(ppgLRED) - 1) / Fs, len(ppgLRED))
taxis2 = np.linspace(0, (len(ppg_resa[:, 0]) - 1) / FsR, len(ppg_resa[:, 0]))

fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('PPG - RED')
# plt.plot(taxis, ppgRRED, color="green")
plt.plot(taxis2, ppg_resa_norm[:, 0], color="red")
ax2 = fig.add_subplot(2, 2, 3)
ax2.set_ylabel('PPG - IR')
ax2.set_xlabel('Time [s]')
# plt.plot(taxis, ppgLIR)
plt.plot(taxis2, ppg_resa_norm[:, 1], color="red")
ax3 = fig.add_subplot(1, 2, 2)
ax3.set_ylabel('PPG Amplitude')
ax3.set_xlabel('Time [s]')
plt.plot(taxis, ppgRRED, label="RED", color="red")
plt.plot(taxis, ppgRIR, label="IR")
plt.legend(loc="upper right")
fig.tight_layout()

taxis3 = np.linspace(0, (len(complete_resa_ppg[:, 2]) - 1) / FsR, len(complete_resa_ppg[:, 2]))
taxis4 = np.linspace(0, (len(complete_resa_ppg[:, 2]) - 1) / FsR, num=loops)
fig2 = plt.figure()
ax1 = fig2.add_subplot(2, 1, 1)
ax1.set_xlabel('Time [s]')
par1 = ax1.twinx()
ax1.set_ylabel('PPG - RED')
ax1.set_ylim(int(np.min(complete_resa_ppg[:, 0])), int(np.max(complete_resa_ppg[:, 0])))
par1.set_ylim(-0.2, 1.2)
ax1.plot(taxis3, complete_resa_ppg[:, 0], color="red")
par1.plot(taxis4, prediction[:, 0], color="blue", drawstyle="steps-mid")
ax2 = fig2.add_subplot(2, 1, 2)
ax2.set_ylabel('PPG - IR')
ax2.set_xlabel('Time [s]')
par2 = ax2.twinx()
ax2.set_ylim(int(np.min(complete_resa_ppg[:, 1])), int(np.max(complete_resa_ppg[:, 1])))
par2.set_ylim(-0.2, 1.2)
ax2.plot(taxis3, complete_resa_ppg[:, 1])
par2.plot(taxis4, prediction[:, 1], color="red", drawstyle="steps-mid")
fig2.tight_layout()
plt.show()
