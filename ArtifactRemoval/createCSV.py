# Author: Nils Froehling

# This script is used to generate a csv-file for all poor-labeled data from study

# Imports
import os
import numpy as np
import glob

# constants
probands = 18

# helper variable for initializing the array
notConcatenate = True

# setup path to study
root_dir = os.getcwd() + "/study_dataset/"

# Get the label data from all probands
for i in range(1, probands+1):
    for file in glob.glob("labels.npy", root_dir=root_dir + "Proband_{}/".format(i)):
        temp = np.load(root_dir + "Proband_{}/".format(i) + file)
        temp = np.c_[temp, np.full(max(temp.shape), i)]
        if notConcatenate:
            csv = np.array(temp)
            notConcatenate = False
        else:
            csv = np.concatenate((csv, temp), axis=0)

# Save all samples in a csv list to index from with sampleNumber, label and probandNumber
np.savetxt(root_dir+"all_samples.csv", csv, delimiter=',')

# Save all poor samples in a csv list to index from with sampleNumber, label and probandNumber
csv_only_poor = csv[csv[:, 1] == 1]
np.savetxt(root_dir + "poor_samples.csv", csv_only_poor, delimiter=',')
