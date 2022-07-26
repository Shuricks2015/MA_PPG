# -------------------------------------------------------------------------------------------------
# AUTHOR: Idoia Badiola
# Created: 12.07.2022
# Human study for PPGI+VMPT
# -------------------------------------------------------------------------------------------------

# Import all packages
import threading as thr
import socket
import time
import numpy as np
# import cv2
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.animation as animation
import queue
from datetime import datetime
from scipy import signal
import torch
import Utils
import torch.nn as nn
import NeuralNets

# Define global parameters
# Buffer for arriving PPG message from Serial
global message
message = queue.Queue()
# Buffer for arriving time (in milliseconds) of PPG messages 
global messageTime
messageTime = queue.Queue()
# Vector with deencrypted PPG data
global ppg
# PPG vector buffer
global ppgQueue
ppgQueue = queue.Queue()
# Time buffer for PPG data based on the counter and the frequency
global ppgTime
# ppgTime = queue.Queue()
# Total amount of samples (depends on number of seconds given by user in command window)
global samplesT
# Flag to indicate if the message was deencrypted (0=not deencrypted yet, 1=already deencrypted)
global analysisFinished
analysisFinished = 0
# Flag to indicate if the message has already arrived completely from the device (0=not completely, 1=measurement fnished)
global messageReady
messageReady = 0
# Filename with date and time of each measurement (everytime the program runs)
global filename
now = datetime.now()
dateTime = now.strftime('%Y-%m-%d_h%H-m%M-s%S')
filename = 'samplesAndVideos/' + dateTime
global millisecondsVid
millisecondsVid = []
time_segment = 3
numOfSamples = 100 * time_segment
numToResample = 64 * time_segment
halfNumOfSamples = int(numOfSamples / 2)
numOfChannels = 4
model = NeuralNets.CNN_try()
Utils.load_checkpoint(torch.load("checkpoints/checkpoint99.pth.tar"), model)
prediction = torch.Tensor()

"""
# THREAD TO PLOT AND SAVE VIDEO
class App1(thr.Thread):
    global messageReady
    global millisecondsVid
    def run(self):
        # Define parameters for cameras
        # Camera input
        capture = cv2.VideoCapture(2)
        # Sampling frequency (30)
        capture.set(cv2.CAP_PROP_FPS,30)
        # Resolution
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        frame_width = int(capture.get(3))
        frame_height = int(capture.get(4))
        # To save video
        fourcc = cv2.VideoWriter_fourcc('X','V','I','D')        
        filenameVid = filename+'_video.avi'
        videoWriter = cv2.VideoWriter(filenameVid, fourcc, 30.0, (frame_width,frame_height))
        # Create lists for the milliseconds and frame count
        countFrame = []
        self.j = 1
        while (True): 
            ret, frame = capture.read()               
            if ret:                
                countFrame.append(self.j)
                millisecondsVid.append(int(round(time.time() * 1000)))
                videoWriter.write(frame)
                cv2.imshow('video', frame) 
                self.j=self.j+1
            # The camera stops recording it ESC is pressed or when the recording ends
            if cv2.waitKey(1) == 27 or messageReady == 1:        
                break
        # Release video frames and file
        capture.release()
        videoWriter.release()
        # Close video window
        cv2.destroyAllWindows()
        
        # Save milliseconds and count of each frame in file
        filenameMilli = filename+'_video_FrameMilliseconds.csv'
        milliArr=np.array(millisecondsVid)
        countArr=np.array(countFrame)
        videoFrameMilli = np.column_stack((countArr,milliArr))
        np.savetxt(filenameMilli,videoFrameMilli,delimiter=',', newline='\n',
              header='Frame Count, Milliseconds')
              """


# THREAD TO READ FROM ETHERNET
class App2(thr.Thread):
    def run(self):
        global samplesT
        global message
        global messageTime
        global messageReady
        global ppgTime

        # Read from ETHERNET
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        IP_ADDRESS = '192.168.137.2'
        IP_PORT = 4000
        s.connect((IP_ADDRESS, IP_PORT))

        # To read each line of message from Ethernet
        x = s.makefile("r")
        # Read each message that arrives from Ethernet        
        for j in range(samplesT):
            if j == 0:
                timeNowMillis = int(round(time.time() * 1000))
            messageStr = str(x.readline())
            message.put(messageStr)
            # if j==1:

        # ppgTimeList = range(timeNowMillis,samplesT)
        messageReady = 1
        # Create numpy array from list
        ppgTime = t = np.linspace(timeNowMillis, timeNowMillis + (samplesT * 10), num=samplesT)


# THREAD TO DEENCRYPT MESSAGE AND PUT IN BUFFER
class App3(thr.Thread):
    def run(self):
        global ppg
        ppg = np.empty([1, numOfChannels])
        global ppgTime
        ppgTime = np.empty([1, 1])
        global ppgQueue
        global analysisFinished
        global message
        global messageTime

        j = 0
        while j <= samplesT - 1:
            # Deencrypt message and save in ppg vector            
            messageStr = message.get()
            messageStrSplit = messageStr.replace('b', '').replace('\\n', '').replace('"', '').replace("'", '').split(
                ",")
            messageStrList = [int(x) for x in messageStrSplit]
            messageStrArray = np.reshape(np.array(messageStrList), (1, numOfChannels))
            # Fill the PPG matrix
            ppg = np.concatenate((ppg, np.array(messageStrArray)), axis=0)

            # Write in Buffer for real-time plot
            ppgQueue.put(messageStrArray)

            j = j + 1

        # Notify STOP time
        print("STOP: %s" % (time.ctime(time.time())))
        # Delete first sample
        ppg = ppg[1:]
        # ppgTime = ppgTime[1:]
        # Set flag to 1
        analysisFinished = 1


class App4(thr.Thread):
    def run(self):
        # Definition on thread start
        global prediction
        count = 0
        ppg_sliced_resa = np.empty((numToResample, numOfChannels))
        sig = nn.Sigmoid()

        # Filter creation
        sos = Utils.filter_creation()

        while analysisFinished == 0:
            time.sleep(0.5)
            if len(ppg[halfNumOfSamples * count:]) >= numOfSamples:
                start = time.time()
                ppg_sliced = ppg[halfNumOfSamples * count:halfNumOfSamples * count + numOfSamples]
                for index in range(numOfChannels):
                    ppg_sliced[:, index] = Utils.band_filter(ppg_sliced[:, index], sos)
                    ppg_sliced_resa[:, index] = signal.resample(ppg_sliced[:, index], numToResample)

                # Reshape for model input
                ppgTensor = ((torch.tensor(ppg_sliced_resa, dtype=torch.float32)).permute(1, 0)).reshape(numOfChannels, 1, numToResample)

                # Make Predictions
                with torch.no_grad():
                    prediction = torch.cat((prediction, torch.round(sig(model(ppgTensor)).flatten())))

                # Increment counter when if statement was True
                count = count + 1
                end = time.time()
                diff = end-start
                print("Time was {}".format(diff))


# Define all threads
# app1 = App1()
app2 = App2()
app3 = App3()
app4 = App4()

# Ask per command window how many seconds of recording you wish
txt = input("How many seconds would you like to record?: ")
# txt = 10
# Calculate number of samples (100 Hz)
samplesT = int(txt) * 100

# Start all threads
# app1.start()
app2.start()
app3.start()
app4.start()
# Notify START time
print("START: %s" % (time.ctime(time.time())))
time.sleep(0.2)
# ---------------------------------------------------
# PLOT DURING ANALYSIS
# ---------------------------------------------------
# ---------------------------------------------------
# SOLUTION1
# Works
# To improve:
# - xaxes and yaxes do not update automatically (do not adapt)
# - Still to synchronize
# ---------------------------------------------------
# Define how many samples you wish to show on plot 
n = 1000
fig = plt.figure()  # figsize=(12,9))
# Define one channel (tPPG-R-IR, in this case)
ax1 = fig.add_subplot(1, 1, 1)
# ax2 = fig.add_subplot(2,1,2)

ch1, = ax1.plot([], [], 'b', label='PPG - R - IR')
# ch2, = ax2.plot([], [], 'r', label = 'Channel 2')

# axes = [ax1, ax2]
axes = [ax1]

for ax in axes:
    ax.set_xlim(0, n + 1)
    # ax.set_ylim(150000,200000)
    ax.set_ylim(ppg[1, 1] * 0.7, ppg[1, 1] * 1.3)
    ax.set_ylabel('PPG Amplitude')
    ax.set_xlabel('Samples (100Hz)')
    ax.legend(loc='upper right')
    ax.grid(True)

ax1.set_title('PPG values from ELCAT vasoport')
# ax2.set_xlabel('Values')

t = list(range(0, n))
t1 = list(range(0, n))
channel1 = [0] * n
tchannel1 = [0] * n


# channel2 = [0] * n

def init():
    ch1.set_data([], [])
    # ch2.set_data([], [])

    return ch1,  # ch2,


def animate(i):
    # ax.set_ylim(auto=True)
    ppgValues = ppgQueue.get()
    data = ppgValues[:, 1]
    # ax1.set_ylim(0,auto=True)
    channel1.append(float(data))
    # channel2.append(float(data))
    # ax1.set_ylim(max(channel1)*0.8,max(channel1)*1.2)
    channel1.pop(0)
    # channel2.pop(0)
    # tnewdata = ppgTime.get()
    # tchannel1.append(float(tnewdata))
    # tchannel1.pop(0)
    if i >= n + 1:
        t1.append(i)
        t1.pop(0)

        ax1.set_xlim([t1[0], t1[-1]])

        # plt.xticks(t1,range(len(t1)))

    ch1.set_data(t1, channel1)

    # ch2.set_data(t, channel2)
    return ch1,  # ch2


delay = 0
anim = animation.FuncAnimation(fig, animate, frames=samplesT - 1, init_func=init, interval=delay, repeat=False,
                               blit=True)
plt.show()

# ppgTime = np.reshape(ppgTime, (np.size(ppgTime), 1))
# timeAndPPG = np.concatenate([ppgTime,ppg],axis=1)

# # Save complete deencrypted message in Excel table
filenamePPG = filename + '_PPGdataFromEthernet.csv'
# np.savetxt(filenamePPG,timeAndPPG,delimiter=',', newline='\n',
# header='Time [ms],tPPG-R-RED,tPPG-R-IR,tPPR-L-RED,tPPG-L-IR,NELLCOR-RED,NELLCOR-IR,rPPG-R-RED,rPPG-R-IR,rPPR-L-RED,rPPG-L-IR,accel-R-X,accel-L-X,accel-R-Y,accel-L-Y,accel-R-Z,accel-L-Z,Temp-R,Temp-L,PressureTank,Pressure-R,Pressure-L,Dummy')
np.savetxt(filenamePPG, ppg, delimiter=',', newline='\n',
           header='tPPG-R-RED,tPPG-R-IR,tPPR-L-RED,tPPG-L-IR,NELLCOR-RED,NELLCOR-IR,rPPG-R-RED,rPPG-R-IR,rPPR-L-RED,rPPG-L-IR,accel-R-X,accel-L-X,accel-R-Y,accel-L-Y,accel-R-Z,accel-L-Z,Temp-R,Temp-L,PressureTank,Pressure-R,Pressure-L,Dummy')

"""
a=ppgTime
print('PPG0ms = ',a[0],'// PPG-1ms = ',a[-1])
# msVid = np.array(millisecondsVid)
# print(msVid)
# b = np.reshape(msVid,(np.size(msVid),1))
# print('Video0ms = ',b[0],'// Video-1ms = ',b[-1])
# print(b)
xy, x_ind, y_ind=np.intersect1d(a,b,return_indices=True)
print(xy)


# FIGURE 2
tm = list(range(0,np.size(b)))
fig3 = plt.figure()
ax3 = fig3.add_subplot(1,1,1)
plt.plot(tm, b,label="all",color='black')
plt.plot(y_ind, xy,'o',label="shared",color='red')
plt.legend(loc="upper right")
ax3.set_ylabel('Milliseconds')
ax3.set_xlabel('Sample')
fig3.tight_layout()
plt.show()
"""

# ---------------------------------------------------
# PLOT AFTER FINISHING THE ANALYSIS
# ---------------------------------------------------
# Do not perform until reading is finished
while analysisFinished == 0:
    pass

# Define tPPG variables 
ppgRRED = ppg[:, 0]
ppgRIR = ppg[:, 1]
ppgLRED = ppg[:, 2]
ppgLIR = ppg[:, 3]
# Define sampling frequency
Fs = 100
# Create xaxis
taxis = np.linspace(0, (len(ppgRRED) - 1) / Fs, len(ppgRRED))

# FIGURE 2
fig2 = plt.figure()
ax1 = fig2.add_subplot(2, 2, 1)
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('PPG - RED')
plt.plot(taxis, ppgRRED, color="red")
ax2 = fig2.add_subplot(2, 2, 3)
ax2.set_ylabel('PPG - IR')
ax2.set_xlabel('Time [s]')
plt.plot(taxis, ppgRIR)
ax3 = fig2.add_subplot(1, 2, 2)
ax3.set_ylabel('PPG Amplitude')
ax3.set_xlabel('Time [s]')
plt.plot(taxis, ppgRRED, label="RED", color="red")
plt.plot(taxis, ppgRIR, label="IR")
plt.legend(loc="upper right")
fig2.tight_layout()
plt.show()

prediction = prediction.reshape((int(len(prediction)/numOfChannels), int(numOfChannels)))
print(prediction)
