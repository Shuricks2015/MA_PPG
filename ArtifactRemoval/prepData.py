# Author: Nils Froehling

import pandas as pd
import os
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np

#######################################
# SETUP STUDY PATH
#######################################

# Current working directory (ArtifactRemoval)
wd = os.getcwd()
# Folder where study measurements are found
filename = "STUDY_MotionArtifacts_NilsFroehling/"
# Directory above current working directory (filename should be found here)
dirUp, _ = os.path.split(wd)
# Path to study data
pathStudy = os.path.join(dirUp, filename)
# Number of Proband to start with
probandNumber = 1
# Create directory for first proband if not there
if not os.path.exists('study_dataset/Proband_{}'.format(probandNumber)):
    os.makedirs('study_dataset/Proband_{}'.format(probandNumber))
    print('Created directory for next proband')
# size of signal interval
interval = 300

# Plot to decide for label
fig1 = plt.figure(figsize=(15, 6))
fig2 = plt.subplot(211)
fig3 = plt.subplot(212)

# counts for extracting data
count = 0
count_plot = 0
sample_count = 0
start = 0

# To store labels for every proband individually
label = []


# Action to take when Buttons pressed
def actiongood():
    label.append([sample_count, 0])
    general_action()


def actionpoor():
    label.append([sample_count, 1])
    general_action()


def general_action():
    # These values from outer scope are changed in this function therefore global
    global sample_count
    temparray = np.array(data.iloc[start:start + interval, [1, 3]].to_numpy())
    np.save('study_dataset/Proband_{}/sample'.format(probandNumber) + str(sample_count), temparray)
    sample_count = sample_count + 1
    plot_next()


# Helper functions
def plot_next():
    # These values from outer scope are changed in this function therefore global
    global data
    global count_plot
    global start
    # Use slicing here
    if max(data.shape) < (count_plot + 2) * int(interval/2):
        data = get_next_data()
        count_plot = 1
    else:
        count_plot = count_plot + 1

    start = (count_plot - 1) * int(interval/2)

    fig2.clear()
    fig3.clear()
    fig2.plot(data.iloc[start:start + interval, 1], label='PPG-R-RED')
    fig2.legend(loc='upper right')
    fig2.set_xlabel('# Samples')
    fig2.set_ylabel('PPG Amplitude')
    fig3.plot(data.iloc[start:start + interval, 3], label='PPG-L-RED')
    fig3.legend(loc='upper right')
    fig3.set_xlabel('# Samples')
    fig3.set_ylabel('PPG Amplitude')

    canvas1.draw()


def get_next_data():
    # These values from outer scope are changed in this function therefore global
    global probandNumber
    global count
    global label
    global sample_count

    # Check if data from one Proband is finished
    if count > 9:
        count = 0
        # Save labels for Proband
        np.save('study_dataset/Proband_{}/labels'.format(probandNumber), label)
        # Reset labels to empty list
        label = []
        # Switch to next proband
        probandNumber = probandNumber + 1
        # Reset sample_count
        sample_count = 0
        # Check if there are still probands with data (no data after 18)
        if probandNumber == 19:
            exit()

        if not os.path.exists('study_dataset/Proband_{}'.format(probandNumber)):
            os.makedirs('study_dataset/Proband_{}'.format(probandNumber))
            print('Created directory for next proband')

    # Use elif statement instead of match-case statements
    # for more compatibility
    if count == 0:
        path_to_file = os.path.join(pathStudy, 'Proband_{}/NOMOVE/x01/'.format(probandNumber))
        # Info starts at 1 for better readability
        movement = 'NOMOVE {}'.format(count+1)
    elif 0 < count < 4:
        path_to_file = os.path.join(pathStudy, 'Proband_{}/VERTICAL/x0{}/'.format(probandNumber, count))
        # Info reflects all 3 repetitions with count starting at 1
        movement = 'VERTICAL {}'.format(count)
    elif 3 < count < 7:
        path_to_file = os.path.join(pathStudy, 'Proband_{}/HORIZONTAL/x0{}/'.format(probandNumber, count - 3))
        # Info reflects all 3 repetitions with count starting at 1
        movement = 'HORIZONTAL {}'.format(count-3)
    elif 6 < count < 10:
        path_to_file = os.path.join(pathStudy, 'Proband_{}/RANDOMNAME/x0{}/'.format(probandNumber, count - 6))
        # Info reflects all 3 repetitions with count starting at 1
        movement = 'RANDOMNAME {}'.format(count-6)

    # Get name of all files in directory
    dirs = os.listdir(path_to_file)
    for file in dirs:
        # Search for ELCAT file
        if 'PPGdataFromEthernet' in file:
            dataElcat = pd.read_csv(path_to_file + file,
                                    usecols=["# Time [ms]", "tPPG-R-RED", "tPPG-R-IR", "tPPR-L-RED", "tPPG-L-IR",
                                             "rPPG-R-RED", "rPPG-R-IR", "rPPR-L-RED", "rPPG-L-IR"])

    # Update info on GUI
    infolabel.configure(text='Proband ' + str(probandNumber) + ' ' + movement)
    # Increase counter because new file was read
    count = count + 1
    return dataElcat


root = tk.Tk()
root.title('PPG Signal quality labeling')
root.attributes('-topmost', True)

canvas1 = FigureCanvasTkAgg(fig1, master=root)
canvas1.get_tk_widget().pack()

infolabel = tk.Label(root, text='temp', font=("Arial", 20))
infolabel.pack()

data = get_next_data()
plot_next()

tk.Button(root, text='Good quality', font=("Arial", 20), command=actiongood, background='green').pack(side='left', ipadx=30, ipady=30, expand=True)
tk.Button(root, text='Poor quality', font=("Arial", 20), command=actionpoor, background='red').pack(side='right', ipadx=30, ipady=30, expand=True)

# Function definition if window doesn't close
# def _quit():
#     root.quit()
#     root.destroy()
#
#
# root.protocol("WM_DELETE_WINDOW", _quit)

root.mainloop()
