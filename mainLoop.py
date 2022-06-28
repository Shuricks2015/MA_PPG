from Utils import load_checkpoint, oxygen_estimation
import matplotlib.pyplot as plt
import socket
import numpy as np
import torch
import torch.nn as nn
from NeuralNets import CNN_try

numOfSamples = 192

# Load model
model = CNN_try()
load_checkpoint(torch.load("checkpoints/checkpoint199.pth.tar"), model)

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
ip_address = '192.168.137.2'
ip_port = 4000
s.connect((ip_address, ip_port))
print('Connection established')
x = s.makefile("r")

while True:
    ppg = np.empty((numOfSamples, 4))
    for i in range(numOfSamples):
        messageStr = x.readline().split(",")
        ppg[i, :] = messageStr

    # Normalize Data using min-max Normalization
    for index in range(3, 4):
        ppg[:, index] = (ppg[:, index] - np.min(ppg[:, index])) / (np.max(ppg[:, index]) - np.min(ppg[:, index]))

    # Prep data for model input
    ppgTensor = ((torch.tensor(ppg, dtype=torch.float32)).permute(1, 0)).reshape(4, 1, numOfSamples)

    # Predict Signal Quality
    with torch.no_grad():
        sig = nn.Sigmoid()
        prediction = torch.round(sig(model(ppgTensor)))
        print(prediction)

    print(oxygen_estimation(ppg[:, 2], ppg[:, 3]))

# Plot the extracted signal
ppgRRED = ppg[:, 0]
ppgRIR = ppg[:, 1]
ppgLRED = ppg[:, 2]
ppgLIR = ppg[:, 3]
Fs = 100
taxis = np.linspace(0, (len(ppgLRED) - 1) / Fs, len(ppgLRED))

fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('PPG - RED')
plt.plot(taxis, ppgLRED, color="red")
ax2 = fig.add_subplot(2, 2, 3)
ax2.set_ylabel('PPG - IR')
ax2.set_xlabel('Time [s]')
plt.plot(taxis, ppgLIR)
ax3 = fig.add_subplot(1, 2, 2)
ax3.set_ylabel('PPG Amplitude')
ax3.set_xlabel('Time [s]')
plt.plot(taxis, ppgLRED, label="RED", color="red")
plt.plot(taxis, ppgLIR, label="IR")
plt.legend(loc="upper right")
fig.tight_layout()
plt.show()
