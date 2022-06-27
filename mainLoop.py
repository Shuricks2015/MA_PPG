from Utils import establish_connection, read_segment, check_accuracy
import matplotlib.pyplot as plt
import socket
import numpy as np

numOfSamples = 192

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
ip_address = '192.168.137.2'
ip_port = 4000
s.connect((ip_address, ip_port))
print('Connection established')
x = s.makefile("r")

ppg = np.empty((numOfSamples, 4))
for i in range(numOfSamples):
    messageStr = x.readline().split(",")
    ppg[i, :] = messageStr

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
