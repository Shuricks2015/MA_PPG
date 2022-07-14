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

Reading_time = 20
numToResample = 64 * Reading_time
numOfSamples = 100 * Reading_time
halfNumOfSamples = int(numOfSamples/2)
loops = 1
numOfChannels = 4

# Load model
model = CNN_try()
Utils.load_checkpoint(torch.load("checkpoints/checkpoint7.pth.tar"), model)
print(sum(p.numel() for p in model.parameters()))

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
ip_address = '192.168.137.2'
ip_port = 4000
s.connect((ip_address, ip_port))
print('Connection established')

x = s.makefile("r")

ppg = np.empty((numOfSamples, numOfChannels))
ppg_resa_norm = np.empty((numToResample, numOfChannels))
complete_ppg = np.empty((numOfSamples + halfNumOfSamples * (loops-1), numOfChannels))
prediction = torch.empty((loops, numOfChannels))

for count in range(loops):
    if count == 0:
        for i in range(numOfSamples):
            messageStr = x.readline().split(",")
            ppg[i, :] = messageStr[:numOfChannels]
        complete_ppg[0:numOfSamples] = ppg
    else:
        ppg[:halfNumOfSamples] = old_ppg
        for i in range(halfNumOfSamples):
            messageStr = x.readline().split(",")
            ppg[halfNumOfSamples+i, :] = messageStr[:numOfChannels]
        complete_ppg[numOfSamples+(count-1)*halfNumOfSamples:numOfSamples+count*halfNumOfSamples] = ppg[halfNumOfSamples:]

    st = time.time()
    print(Utils.oxygen_estimation(ppg[:, 2], ppg[:, 3]))

    resampled_ppg = signal.resample(ppg, numToResample)
    # Normalize Data using min-max Normalization
    for index in range(numOfChannels):
        ppg_resa_norm[:, index] = signal.detrend(resampled_ppg[:, index])

    # Prep data for model input
    ppgTensor = ((torch.tensor(ppg_resa_norm, dtype=torch.float32)).permute(1, 0)).reshape(numOfChannels, 1, numToResample)

    """
    # Predict Signal Quality
    with torch.no_grad():
        sig = nn.Sigmoid()
        prediction[count] = sig(model(ppgTensor)).flatten()
        prediction[count] = torch.round(prediction[count])
        print(prediction[count])
    """
    et1 = time.time()
    old_ppg = ppg[halfNumOfSamples:, :]
    print("Execution time for code was {}".format(et1 - st))

# Utils.get_edges(prediction)

# Plot the extracted signal
ppgRRED = ppg[:, 0]
ppgRIR = ppg[:, 1]
ppgLRED = ppg[:, 2]
ppgLIR = ppg[:, 3]
Fs = 100
taxis = np.linspace(0, (len(ppgLRED) - 1) / Fs, len(ppgLRED))
taxis2 = np.linspace(0, (len(resampled_ppg[:, 2]) - 1) / 64, len(resampled_ppg[:, 2]))

fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('PPG - RED')
#plt.plot(taxis, ppgRRED, color="green")
plt.plot(taxis2, ppg_resa_norm[:, 0], color="red")
ax2 = fig.add_subplot(2, 2, 3)
ax2.set_ylabel('PPG - IR')
ax2.set_xlabel('Time [s]')
plt.plot(taxis, ppgRIR)
plt.plot(taxis2, resampled_ppg[:, 1], color="red")
ax3 = fig.add_subplot(1, 2, 2)
ax3.set_ylabel('PPG Amplitude')
ax3.set_xlabel('Time [s]')
plt.plot(taxis, ppgRRED, label="RED", color="red")
plt.plot(taxis, ppgRIR, label="IR")
plt.legend(loc="upper right")
fig.tight_layout()


taxis3 = np.linspace(0, (len(complete_ppg[:, 2]) - 1) / Fs, len(complete_ppg[:, 2]))
taxis4 = np.linspace(0, (len(complete_ppg[:, 2]) - 1) / Fs, num=loops)
fig2 = plt.figure()
ax1 = fig2.add_subplot(2, 1, 1)
ax1.set_xlabel('Time [s]')
par1 = ax1.twinx()
ax1.set_ylabel('PPG - RED')
ax1.set_ylim(int(np.min(complete_ppg[:,0])), int(np.max(complete_ppg[:,0])))
par1.set_ylim(-0.2, 1.2)
ax1.plot(taxis3, complete_ppg[:, 0], color="red")
par1.plot(taxis4, prediction[:, 0], color="blue", drawstyle="steps-mid")
ax2 = fig2.add_subplot(2, 1, 2)
ax2.set_ylabel('PPG - IR')
ax2.set_xlabel('Time [s]')
par2 = ax2.twinx()
ax2.set_ylim(int(np.min(complete_ppg[:,1])), int(np.max(complete_ppg[:,1])))
par2.set_ylim(-0.2, 1.2)
ax2.plot(taxis3, complete_ppg[:, 1])
par2.plot(taxis4, prediction[:, 1], color="red", drawstyle="steps-mid")
fig2.tight_layout()
plt.show()
