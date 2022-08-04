# -------------------------------------------------------------------------------------------------
# AUTHOR: Idoia Badiola
# Created: 12.07.2022
# Study for MotionArtifacts
# -------------------------------------------------------------------------------------------------

# Import all packages
import threading as thr
import socket
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
import queue
from datetime import datetime
import sys
import os
import subprocess as sp
import shlex
import playsound
from PIL import ImageTk, Image
from threading import Timer

global fig12
# Define global parameters
global timeTotal
timeTotal = -1
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
analysisFinished=0
# Flag to indicate if the message has already arrived completely from the device (0=not completely, 1=measurement fnished)
global messageReady
messageReady = 0
# Filename with date and time of each measurement (everytime the program runs)

now=datetime.now()
dateTime = now.strftime('%Y-%m-%d_h%H-m%M-s%S')            
# filename = 'samplesAndVideos/'+dateTime
global filename
filename = 'samplesAndVideos/MotionArts_'+dateTime
global filenamePPG 
filenamePPG = filename+'_PPGdataFromEthernet.csv'
global millisecondsVid
millisecondsVid=[]
global ipaddressPhilips
beeping = 2
firstBeep = 0
beepCount = 0

# ----------------------------------
 # SAVE FILE WITH DATA
# ----------------------------------
def savePPGfiles(ppgTime,ppg,filename): 
    global filenamePPG   
    ppgTime = np.reshape(ppgTime,(np.size(ppgTime),1))
    timeAndPPG = np.concatenate([ppgTime,ppg],axis=1)
    # Save complete deencrypted message in Excel table    
    np.savetxt(filenamePPG,timeAndPPG,delimiter=',', newline='\n',
                header='Time [ms],tPPG-R-RED,tPPG-R-IR,tPPR-L-RED,tPPG-L-IR,NELLCOR-RED,NELLCOR-IR,rPPG-R-RED,rPPG-R-IR,rPPR-L-RED,rPPG-L-IR,accel-R-X,accel-L-X,accel-R-Y,accel-L-Y,accel-R-Z,accel-L-Z,Temp-R,Temp-L,PressureTank,Pressure-R,Pressure-L,Dummy')
    print("INFO: PPG data saved!")
    
# ---------------------------------------------------
# THREADS 0
# ---------------------------------------------------  
# COUNTDOWN AND ACOUSTIC AIDS
# define the countdown func.

class Countdown(thr.Thread): 
    global timeTotal
    def create_countdown_timer(self,time):
        global firstBeep
        global beepCount
        mins, secs = divmod(int(time), 60)
        timerT = 'Verbleibende Zeit: {:02d}:{:02d}'.format(mins, secs)
        print(timerT, end='\n')
        if firstBeep == 0:
            firstBeep = 1
        else:
            if beeping == 1:
                if secs % 5 == 0 and beepCount < 6:
                    beepCount = beepCount + 1
                    playsound.playsound("beep.wav")
            elif beeping == 2 and beepCount < 4:
                if (secs+5) % 10 == 0:
                    beepCount = beepCount + 1
                    playsound.playsound("beep.wav")

        # restingTime.config(text=timerT)
        
    def run(self):
        global timeTotal
        global imageSelected
        self.time_in_sec = timeTotal
        self.count = 0
        t = timeTotal
        # For the first time we will call the function manually
        self.create_countdown_timer(self.time_in_sec) 
        for times in range(1,self.time_in_sec): 
            # calling the Timer class every second
            t = Timer(1,self.create_countdown_timer,[str(self.time_in_sec-times)])
            t.start()
            time.sleep(1)
            
class PhilipsMonitor(thr.Thread): 
    def run(self):
        global filename
        global filenamePPG
        global ipaddressPhilips
        currentPath = os.getcwd()
        appPath = currentPath+"\\VSCaptureMP_IB\\bin\\Debug\\net6.0\\VSCaptureMP.exe"
        #print("AppPath = ", appPath)
        commando = appPath+" "+filename+" "+filename+'_Running.csv' + " "+ipaddressPhilips
        os.system(commando)
        # if ipaddressPhilips =="":
        #     commando = [appPath"," ", filenamePPG]
        #     os.system(commando)
        # else:
        #     commando = [appPath," ", ipaddressPhilips, " ", filenamePPG]
        #     os.system(commando)

# ---------------------------------------------------
# THREADS 1
# ---------------------------------------------------  
# THREAD TO PLOT AND SAVE VIDEO
class App1(thr.Thread):
    global messageReady
    global millisecondsVid
    def run(self):        
        width, height, fps = 640, 480, 25  # 50 frames, resolution 1344x756, and 25 fps

        filenameVid = filename +'_video.avi'  
        devnull = open(os.devnull, 'wb')
        process = sp.Popen(shlex.split(f'ffmpeg -y -s {width}x{height} -pixel_format bgr24 -f rawvideo -r {fps} -i pipe: -vcodec libx264 -pix_fmt yuv420p -crf 0 {filenameVid}'), stdin=sp.PIPE, stderr=devnull)

        # Define parameters for cameras
        # Camera input
        capture = cv2.VideoCapture(2)
        # Sampling frequency (30)
        capture.set(cv2.CAP_PROP_FPS,fps)    # 30
        # WHITE BALANCE: From https://iopscience.iop.org/article/10.1088/1361-6579/ab87b3/pdf
        capture.set(cv2.CAP_PROP_AUTO_WB,0)
        capture.set(cv2.CAP_PROP_WHITE_BALANCE_RED_V,550)   
        capture.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U,810)  
        # BRIGHTNESS
        # capture.set(cv2.CAP_PROP_BRIGHTNESS,150)

        # Resolution
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width) # 1280
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height) # 480
        capture.set(cv2.CAP_PROP_BACKLIGHT,0)
        frame_width = int(capture.get(3))
        frame_height = int(capture.get(4))
        fourcc = cv2.VideoWriter_fourcc('H','F','Y','U') # Recommended in https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8666757
        capture.set(cv2.CAP_PROP_FOURCC,fourcc)

        # Create lists for the milliseconds and frame count
        millisecondsVid = []
        countFrame = []
        j = 1
        while (True): 
            ret, frame = capture.read()               
            if ret:                
                countFrame.append(j)
                millisecondsVid.append(int(round(time.time() * 1000)))
                process.stdin.write(frame)
                cv2.imshow('video', frame) 
                j=j+1
            # The camera stops recording it ESC is pressed or when the recording ends
            if cv2.waitKey(1) == 27 or messageReady == 1 or app2.errorFlag == 1:        
                break

        # Close and flush stdin
        process.stdin.close()
        # Wait for sub-process to finish
        process.wait()
        # Terminate the sub-process
        process.terminate()

        # Close video window
        cv2.destroyAllWindows()
        # Release video frames and file
        capture.release()
        
        # Save video frames info
        if app2.errorFlag !=1:    
            # Save milliseconds and count of each frame in file
            filenameMilli = filename+'_video_FrameMilliseconds.csv'
            milliArr=np.array(millisecondsVid)
            countArr=np.array(countFrame)
            videoFrameMilli = np.column_stack((countArr,milliArr))
            np.savetxt(filenameMilli,videoFrameMilli,delimiter=',', newline='\n',
                header='Frame Count, Milliseconds')
            print('INFO: Video and frame information saved!')
            
        else:
            os.remove(filenameVid)
            print('INFO: Video file deleted')

# ---------------------------------------------------
# THREAD 2
# ---------------------------------------------------  
# THREAD TO PLOT AND SAVE VIDEO# THREAD TO READ FROM ETHERNET
class App2(thr.Thread):      
    def run(self):
        self.errorFlag = -1  
        global samplesT
        global message
        global messageTime
        global messageReady
        global ppgTime
        # Read from ETHERNET
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        IP_ADDRESS = '192.168.137.2'
        IP_PORT = 4000 
        try:
            s.connect((IP_ADDRESS, IP_PORT))
            print('Connection established!')
        except:
            print("ERROR: Device not found. Please, check the connection.")
            self.errorFlag = 1
        try:
            # To read each line of message from Ethernet
            x = s.makefile("rb")
            # Read each message that arrives from Ethernet        
            for self.j in range(samplesT):                                    
                    messageStr = str(x.readline())
                    message.put(messageStr)                
                    if self.j==1:
                        timeNowMillis = int(round(time.time() * 1000))
                        # Notify START time
                        print("START: %s" % (time.ctime(time.time())))
                        countdown.start()   
                        f = open(filename+'_Running.csv', 'w') 
                        f.close()
                        philipsM.start() 
                    self.errorFlag = 0
                    # print("APP2: ",j)                        
            # ppgTimeList = range(timeNowMillis,samplesT)
            messageReady = 1
            # Create numpy array from list
            ppgTime = np.arange(timeNowMillis,timeNowMillis+(samplesT)*10,10)
        except:
            self.errorFlag = 1
            if self.j==0:
                print("ERROR: The device is not sending any data. Please start the measurement on the device.")
            else:
                print("ERROR: The measurement stopped abruptly on the device.")
                print ("STOP: %s" % (time.ctime(time.time())))
        
# ---------------------------------------------------
# THREAD 3
# ---------------------------------------------------  
# THREAD TO DEENCRYPT MESSAGE AND PUT IN BUFFER
class App3(thr.Thread):    
    def run(self):  
        global ppg
        ppg = np.empty([1,22])
        global ppgTime
        # ppgTime = np.empty([1,1])
        global ppgQueue
        # global ppgTime
        global analysisFinished        
        global message
        global messageTime  
        
        j=0
        while j<=samplesT-1:
            # print("APP3: ",j)
            try:
            # Deencrypt message and save in ppg vector            
                messageStr = message.get()
            except:
                print("kaka")
            messageStrSplit = messageStr.replace('b','').replace('\\n','').replace('"','').replace("'",'').split(",")            
            messageStrList = [int(x) for x in messageStrSplit]
            messageStrArray = np.reshape(np.array(messageStrList),(1,22))
            # Fill the PPG matrix
            ppg = np.concatenate((ppg,np.array(messageStrArray)),axis=0)
            
            # Write in Buffer for real-time plot
            ppgQueue.put(messageStrArray)            
            j=j+1            
        
        # Notify STOP time
        print ("STOP: %s" % (time.ctime(time.time())))
        # Delete first sample
        ppg = ppg[1:]
        # ppgTime = ppgTime[1:]
        # Set flag to 1
        analysisFinished = 1
        os.remove(filename+'_Running.csv')
        return analysisFinished
    
# ---------------------------------------------------
# REAL-TIME PLOT DURING ANALYSIS
# ---------------------------------------------------
# ---------------------------------------------------
# Works
# To improve:
# - xaxes and yaxes do not update automatically (do not adapt)
# ---------------------------------------------------
class RealTimePlot: 
    global ppg
    global fig12
    def __init__(self):
        # Define how many samples you wish to show on plot 
        self.n = 3000
        # Create plot
        self.fig1 = fig12
        # self.fig1 = plt.figure(figsize=(12,5))
        # self.fig1.suptitle('PPG values from ELCAT vasoport device (Real-time)')
        # Define lists for variables to plot
        self.t = list(range(0,self.n))
        self.channel1 = [0] * self.n
        self.channel2 = [0] * self.n
        # Define channel axes (tPPG-R-RED and tPPG-R-IR)
        self.ax1 = self.fig1.add_subplot(1,2,1)    # RED
        self.ax2 = self.fig1.add_subplot(1,2,2)    # IR
        # Define line color and legend label for each channel
        self.ch1, = self.ax1.plot([], [], 'r', label = 'PPG - R - RED')
        self.ch2, = self.ax2.plot([], [], 'b', label = 'PPG - R - IR')
        # Y-axis limits
        self.ax1.set_ylim(0,ppg[1,0]*3)
        self.ax2.set_ylim(0,ppg[1,1]*3)
        # ax.set_ylim(ppg[1,1]*0.7,ppg[1,1]*1.3)
        # Y-axis label (only left, cause they are the same for both)
        self.ax1.set_ylabel('PPG Amplitude')
        self.axes = [self.ax1, self.ax2]
        for ax in self.axes:
            # x axis limit
            ax.set_xlim(0, self.n+1)
            # x-axis labels 
            ax.set_xlabel('Samples (100Hz)')
            ax.legend(loc='upper right')
            ax.grid(True)
        self.ch1.set_data([], [])
        self.ch2.set_data([], [])
        self.done = 0
        delay = 0
        self.anim = animation.FuncAnimation(self.fig1, self.animate, frames=samplesT-2,interval=delay, repeat=False, blit=True)

    def animate(self,i):   
        if app2.errorFlag == 1:
            plt.close(rt.fig1)
            time.sleep(1)
            sys.exit(1)
        # ax.set_ylim(auto=True)
        while ppgQueue.qsize()<=1:
            if app2.errorFlag == 1:
                plt.close(rt.fig1)
                # To delete video file
                time.sleep(1)
                sys.exit(1)
            if i == samplesT-3 and analysisFinished == 1:
                # label.config(text="ENDE DER MESSUNG")
                restingTime.config(text="ENDE")
                savePPGfiles(ppgTime,ppg,filename) 
                # print('hamen')
                # Delete the real-time plot when it finishes
                plt.clf()                
                plt.draw()
                # Plot the complete signal
                timeaxis = np.linspace(0,(samplesT-1)/100,samplesT)                         
                # Define channel axes (tPPG-R-RED and tPPG-R-IR)
                self.ax1 = self.fig1.add_subplot(1,2,1)    # RED
                self.ax2 = self.fig1.add_subplot(1,2,2)    # IR
                # self.fig1.suptitle('PPG values from ELCAT vasoport device')
                self.ax1.set_ylabel('PPG Amplitude')
                self.ax1.set_xlabel('Time [s]')
                self.ax2.set_xlabel('Time [s]')
                self.ax1.set_xlim([timeaxis[0], timeaxis[-1]+0.01])
                self.ax2.set_xlim([timeaxis[0], timeaxis[-1]+0.01])
                self.ax1.plot(timeaxis, ppg[:,0], 'r', label = 'PPG - R - RED')
                self.ax2.plot(timeaxis, ppg[:,1], 'b', label = 'PPG - R - IR')  
                self.axes = [self.ax1, self.ax2]
                for ax in self.axes:
                    # x-axis labels 
                    ax.set_xlabel('Samples (100Hz)')
                    ax.legend(loc='upper right')
                    ax.grid(True)          
                plt.draw()                
                self.done = 1
                break
            
        ppgValues = ppgQueue.get()
        data1 = ppgValues[:,0]
        data2 = ppgValues[:,1]
        # ax1.set_ylim(0,auto=True)
        self.channel1.append(float(data1))
        self.channel2.append(float(data2))
        # ax1.set_ylim(max(channel1)*0.8,max(channel1)*1.2)
        self.channel1.pop(0)
        self.channel2.pop(0)
        if self.done != 1:
            if i>=self.n+1:
                self.t.append(i)
                self.t.pop(0)        
                self.ax1.set_xlim([self.t[0], self.t[-1]])
                self.ax2.set_xlim([self.t[0], self.t[-1]])
                # plt.xticks(t1,range(len(t1)))
            self.ch1.set_data(self.t, self.channel1)
            self.ch2.set_data(self.t, self.channel2)        
        return self.ch1,self.ch2,

# ---------------------------------------------------
# MAIN
# ---------------------------------------------------  

# Ask per command window how many seconds of recording you wish
txt = input("How many seconds would you like to record?: ")
timeTotal = int(txt)
# Calculate number of samples (100 Hz)
samplesT = timeTotal*100

ipaddressPhilips = input("IP Address of the Philips Monitor: ")


import tkinter as Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Image.MAX_IMAGE_PIXELS = None
fig12 = plt.figure(figsize=(10,4))

root = Tk.Tk()
root.title('Human study - Motion artifacts')
root.attributes('-topmost',True)

# label = Tk.Label(root,text="Bitte, bleiben Sie ruhig, bis Sie drei Beeps hören.",font=("Arial", 20))
# label.grid(column=0, row=0,columnspan=10)

# labelNum =[]
# for i in range(1,11):
#     label1=Tk.Label(root,text="",font=("Arial", 20))
#     label1.grid(column=i-1, row=1,columnspan=1)
#     labelNum.append(label1)

# imagefootDown = Image.open("samplesAndVideos/downFootIcon.png").resize((180,200), Image.ANTIALIAS)
# imageFootDown = ImageTk.PhotoImage(imagefootDown)
# imagefootUp = Image.open("samplesAndVideos/upFootIcon.png").resize((180,200), Image.ANTIALIAS)
# imageFootUp = ImageTk.PhotoImage(imagefootUp)
# global imageSelected
# imageSelected = imageFootDown
# canvas2 = Tk.Label(root,image = imageFootDown, width=200,height=200)
# canvas2.grid(column=0, row=2,columnspan=10)

# picture = FigureCanvasTkAgg(image = imageFoot, master=root).grid(column=0, row=1)
canvas = FigureCanvasTkAgg(fig12, master=root)
canvas.get_tk_widget().grid(column=0,row=0)
toolbar = NavigationToolbar2Tk(canvas, root,pack_toolbar=False)
toolbar.grid(column=0,row=1,columnspan=10)

restingTime = Tk.Label(root,text="PPG-Right-Hand",font=("Arial", 20))
restingTime.grid(column=0, row=2, columnspan=10)

# Define all threads
countdown = Countdown()
# app1 = App1()
app2 = App2()
app3 = App3()
philipsM = PhilipsMonitor()

# Finish threads when main finishes
countdown.daemon = True
# app1.daemon = True
app2.daemon = True
app3.daemon = True
philipsM.daemon = True
# Start all threads
# app1.start()
app2.start()
app3.start()

# time.sleep(0.5)
while (app2.errorFlag == -1):
    pass

if(app2.errorFlag==1):
    sys.exit(1)

# Show but without blocking the rest of
rt = RealTimePlot()
# plt.show()


def _quit():
    root.quit()
    root.destroy()


root.protocol("WM_DELETE_WINDOW", _quit)
Tk.mainloop()
print("Finish")
if analysisFinished != 1:
    os.remove(filename+'_Running.csv')
