# Author: Nils Froehling

import pandas as pd
import os
import datetime

#######################################
# SETUP STUDY PATH
#######################################

# Current working directory (ArtifactRemoval)
wd = os.getcwd()
# Folder where study measurements are found
filename = "STUDY_MotionArtifacts_NilsFroehling/Proband_"
# Directory above current working directory (filename should be found here)
dirUp, _ = os.path.split(wd)
# Path to study data
pathStudy = os.path.join(dirUp, filename)

# temporary
probandNumber = 15


def get_elcat_ppg(dataELCAT):
    # Extract only Time and PPG data from ELCAT measurement
    # Typo in second-last index while performing study
    return dataELCAT.loc[:, ["# Time [ms]", "tPPG-R-RED", "tPPG-R-IR", "tPPR-L-RED", "tPPG-L-IR", "rPPG-R-RED",
                             "rPPG-R-IR", "rPPR-L-RED", "rPPG-L-IR"]]


# Read in csv data using pandas
try:
    dataElcat = pd.read_csv(pathStudy + "{}".format(probandNumber) + "/NOMOVE/x01/MotionArts_2022-08-02_h14-m34-s18_PPGdataFromEthernet.csv")
except:
    print("CSV-file could not be opened. Check if location is right or file is already open")

ppgDataElcat = get_elcat_ppg(dataElcat)

dateStamp = datetime.datetime.fromtimestamp(ppgDataElcat.iloc[0, 0]/1000.0)
print(dateStamp)
