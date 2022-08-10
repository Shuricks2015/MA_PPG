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
probandNumber = 1

# Read in csv data using pandas
try:
    dataElcat = pd.read_csv(pathStudy + "{}".format(
        probandNumber) + "/NOMOVE/x01/MotionArts_2022-08-01_h14-m12-s35_PPGdataFromEthernet.csv",
                            usecols=["# Time [ms]", "tPPG-R-RED", "tPPG-R-IR", "tPPR-L-RED", "tPPG-L-IR", "rPPG-R-RED",
                                     "rPPG-R-IR", "rPPR-L-RED", "rPPG-L-IR"])

    dataO2philips = pd.read_csv(pathStudy + "{}".format(
        probandNumber) + "/NOMOVE/x01/MotionArts_2022-08-01_h14-m12-s35_Philips_MPDataExport.csv", usecols=[2, 3, 4])

    # PPG is mirrored!!!
    dataPPGphilips = pd.read_csv(pathStudy + "{}".format(
        probandNumber) + "/NOMOVE/x01/MotionArts_2022-08-01_h14-m12-s35_Philips_NOM_PLETHWaveExport.csv",
                                 usecols=[2, 3], names=['SystemLocalTime', 'PPG'])
except:
    print("CSV-file could not be opened. Check if location is right or file is already open")

# SystemLocalTime is corresponding to the time from ELCAT
timeO2Philips = dataO2philips.loc[:, "SystemLocalTime"]
dateStamp = datetime.datetime.fromtimestamp(dataElcat.iloc[0, 0] / 1000.0)
print(dateStamp)
