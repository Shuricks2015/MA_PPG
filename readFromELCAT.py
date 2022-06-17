import socket
import time
import numpy as np

# Read from ETHERNET
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
IP_ADDRESS = '192.168.137.2'
IP_PORT = 4000
s.connect((IP_ADDRESS, IP_PORT))
print('Connection established')
# To read each line of message from Ethernet
x = s.makefile("rb")
# Create PPG array to save data
ppgL = []
# Initialize parameters to find each signal in message (Transmissive, Reflective, R and L)
commaPosL = []
startPos = 1
endPos = -3

# READ THE FIRST LINE
messageStr = str(x.readline())
#  Find positions of commas in message to find each signal in message
for pos,char in enumerate(messageStr):
    if(char == ','):
        commaPosL.append(pos)    
posAll = np.concatenate((startPos,np.array(commaPosL),endPos),axis=None)

ppg = np.empty([1,4]) 
# Fill PPG matrix with each signal  
for i in range(len(posAll)-1):
    ppg[0][i] = int(messageStr[posAll[i]+1:posAll[i+1]])
    
print('Acquiring 10 seconds of signal')   
# READ THE REST OF THE LINES
j=2
while j<=1000:
    # print(i)
    # print ("%s" % (time.ctime(time.time())))
    messageStr = str(x.readline())
    commaPosL = []
    #  Find positions of commas in message to find each signal in message
    for pos,char in enumerate(messageStr):
        if(char == ','):
            commaPosL.append(pos)    
    posAll = np.concatenate((startPos,np.array(commaPosL),endPos),axis=None)
    ppgLine = np.empty([1,4])
    # Fill PPG matrix with each signal  
    for i in range(len(posAll)-1):
        ppgLine[0][i] = int(messageStr[posAll[i]+1:posAll[i+1]])
        
    ppg = np.concatenate((ppg,ppgLine),axis=0)
    j=j+1
        
print('Message received')


# PLOT SIGNALS
import matplotlib.pyplot as plt
from matplotlib import gridspec

ppgRRED = ppg[:,0]
ppgRIR = ppg[:,1]
ppgLRED = ppg[:,2]
ppgLIR = ppg[:,3]
Fs = 100
taxis = np.linspace(0, (len(ppgLRED)-1)/Fs, len(ppgLRED))

fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('PPG - RED')
plt.plot(taxis, ppgLRED,color="red")
ax2 = fig.add_subplot(2,2,3)
ax2.set_ylabel('PPG - IR')
ax2.set_xlabel('Time [s]')
plt.plot(taxis, ppgLIR)
ax3 = fig.add_subplot(1,2,2)
ax3.set_ylabel('PPG Amplitude')
ax3.set_xlabel('Time [s]')
plt.plot(taxis, ppgLRED, label="RED",color="red")
plt.plot(taxis, ppgLIR, label="IR")
plt.legend(loc="upper right")
fig.tight_layout()
plt.show()