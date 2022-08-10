import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import os
import matplotlib.pyplot as plt
from Utils import *

sos = filter_creation2()

filename = "STUDY_MotionArtifacts_NilsFroehling/Proband_"
probandNumber = 1

dataElcat = pd.read_csv(filename + "{}".format(
    probandNumber) + "/NOMOVE/x01/MotionArts_2022-08-01_h14-m12-s35_PPGdataFromEthernet.csv",
                        usecols=["# Time [ms]", "tPPG-R-RED", "tPPG-R-IR", "tPPR-L-RED", "tPPG-L-IR", "rPPG-R-RED",
                                 "rPPG-R-IR", "rPPR-L-RED", "rPPG-L-IR"])

dataPPGphilips = pd.read_csv(filename + "{}".format(
    probandNumber) + "/NOMOVE/x01/MotionArts_2022-08-01_h14-m12-s35_Philips_NOM_PLETHWaveExport.csv",
                             usecols=[2, 3], names=['SystemLocalTime', 'PPG'])

for i in range(len(dataPPGphilips['PPG'])):
    dataPPGphilips.iat[i, 1] = - dataPPGphilips.iat[i, 1]

x = band_filter2(np.array(dataElcat['tPPG-R-IR']), sos)
y = minmax_normalization(np.array(dataPPGphilips['PPG']))

distance, path = fastdtw(x, y, dist=euclidean)

result = []

for i in range(0, len(path)):
    result.append([dataElcat['# Time [ms]'].iloc[path[i][0]], dataElcat['tPPG-R-IR'].iloc[path[i][0]],
                   dataPPGphilips['PPG'].iloc[path[i][1]]])


df_synchronized = pd.DataFrame(data=result,columns=['Time','PPG_ELCAT','PPG_Philips']).dropna()
df_synchronized = df_synchronized.drop_duplicates(subset=['Time'])
df_synchronized = df_synchronized.sort_values(by='Time')
df_synchronized = df_synchronized.reset_index(drop=True)

df_unsynchronized = dataElcat[['# Time [ms]', 'tPPG-R-IR']].copy()
df_unsynchronized['PPG'] = dataPPGphilips['PPG']
df_unsynchronized = df_unsynchronized.drop_duplicates(subset=['# Time [ms]'])
df_unsynchronized = df_unsynchronized.sort_values(by='# Time [ms]')
df_unsynchronized = df_unsynchronized.reset_index(drop=True)

df_unsynchronized['tPPG-R-IR'] = minmax_normalization(np.array(df_unsynchronized['tPPG-R-IR']))
df_unsynchronized['PPG'] = minmax_normalization(np.array(df_unsynchronized['PPG']))
df_synchronized['PPG_ELCAT'] = minmax_normalization(np.array(df_synchronized['PPG_ELCAT']))
df_synchronized['PPG_Philips'] = minmax_normalization(np.array(df_synchronized['PPG_Philips']))

fig1 = plt.figure()
plt.plot(dataElcat['# Time [ms]'], x)
plt.plot(dataElcat['# Time [ms]'], y[:2500])

df_synchronized.plot(x='Time')
df_unsynchronized.plot(x='# Time [ms]')

plt.show()
