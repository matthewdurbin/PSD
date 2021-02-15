# -*- coding: utf-8 -*-
"""
Parse a binary (.dat) file of CAEN Digitizer Waveforms 
Extract varios input features 
"""
import numpy as np
import matplotlib.pyplot as plt
import struct

# Parameter set up
# Recorld length, late pulse threshold, time constants,
# fwtm min/max, calibration parameters
rl, lp, t0, t1, t2, fwtm_min, fwtm_max, x = (
    1200,
    140,
    50,
    110,
    510,
    100,
    400,
    np.array([0, 164584, 426845]),
)

# Load Binary File (takes some RAM and time)
with open("Ham6075_stil_Cf_612.dat", mode="rb") as file:
    fileContent = file.read()
    raw = struct.unpack("h" * int((len(fileContent) / 2)), fileContent[:])

print("Loaded")

# reshape to have each row a pulse
gp = np.reshape(raw, (int(len(raw) / rl), rl))

# save first 30 waveforms
raw_save = gp[:30, :]
np.savetxt("Ham6075_stil_Cf_612_30waveforms.csv", raw_save, delimiter=",", comments="")

# vector with the index of max of each pulse
mxi = np.argmax(gp, axis=1)

# throw up late pulses/pile ups etc..
for l in range(len(mxi)):
    if mxi[l] > lp:
        mxi[l] = 0
        gp[l, :] = np.zeros(rl)
    if mxi[l] == 0:
        gp[l, :] = np.zeros(rl)
    if mxi[l] < 32:
        mxi[l] = 0
        gp[l, :] = np.zeros(rl)

mxi = mxi[mxi != 0]
gp = gp[~np.all(gp == 0, axis=1)]

# Baseline subtraction
for i in range(len(gp)):
    avg = int(np.mean(gp[i, :30]))
    gp[i, :] = gp[i, :] - avg

baseline_save = gp[:30, :]
baseline = mxi - t0

# Calculate areas
# head defined as baseline: peak+head

headarea = np.zeros(len(mxi))
for pii in range(len(mxi)):
    headarea[pii] = sum(gp[pii, baseline[pii] : mxi[pii] + t1])

# tail defined as end of head+700
tailarea = np.zeros(len(mxi))
for ti in range(len(mxi)):
    tailarea[ti] = sum(gp[ti, mxi[ti] + t1 : mxi[ti] + t2])

# total area from baseline end to end of pulse
totalarea = np.zeros(len(mxi))
for toi in range(len(mxi)):
    totalarea[toi] = sum(gp[toi, baseline[toi] : mxi[toi] + t2])

ttt = tailarea / totalarea

# area matrix: head, tail, total, ttt
area = np.column_stack((headarea, tailarea, totalarea, ttt, mxi, baseline))
print("Areas calculated")

# Calibrate
y = np.array([0, 341, 1061])

# fit to a quatratic
quad = np.polyfit(x, y, 2)
area[:, 2] = (quad[0] * (area[:, 2] ** 2)) + (quad[1] * area[:, 2]) + quad[2]

print("Calibrated")

# throw out negative pusles/ bad typcial psp and re run
for k in range(len(area)):
    if area[k, 3] < 0:
        area[k, :] = np.zeros(6)
        gp[k, :] = np.zeros(rl)
    if area[k, 3] > 1:
        area[k, :] = np.zeros(6)
        gp[k, :] = np.zeros(rl)
    if area[k, 3] == 0:
        area[k, :] = np.zeros(6)
        gp[k, :] = np.zeros(rl)

area = area[~np.all(area == 0, axis=1)]
gp = gp[~np.all(gp == 0, axis=1)]


#    time intergrals 10
# each time bin will be tail/5 long, with the addition of the rise time
time = int(t2 / 9)

# anchors
ts1_10 = area[:, 5]
ts2_10 = area[:, 4]
ts3_10 = area[:, 4] + time
ts4_10 = ts3_10 + time
ts5_10 = ts4_10 + time
ts6_10 = ts5_10 + time
ts7_10 = ts6_10 + time
ts8_10 = ts7_10 + time
ts9_10 = ts8_10 + time
ts10_10 = ts9_10 + time
end_10 = ts10_10 + time

# areas
ts1a_10 = np.zeros(len(area))  # baseline to mxi
ts2a_10 = np.zeros(len(area))
ts3a_10 = np.zeros(len(area))
ts4a_10 = np.zeros(len(area))
ts5a_10 = np.zeros(len(area))
ts6a_10 = np.zeros(len(area))
ts7a_10 = np.zeros(len(area))
ts8a_10 = np.zeros(len(area))
ts9a_10 = np.zeros(len(area))
ts10a_10 = np.zeros(len(area))

for ts1i in range(len(area)):
    ts1a_10[ts1i] = sum(gp[ts1i, int(ts1_10[ts1i]) : int(ts2_10[ts1i])])

for ts2i in range(len(area)):
    ts2a_10[ts2i] = sum(gp[ts2i, int(ts2_10[ts2i]) : int(ts3_10[ts2i])])

for ts3i in range(len(area)):
    ts3a_10[ts3i] = sum(gp[ts3i, int(ts3_10[ts3i]) : int(ts4_10[ts3i])])

for ts4i in range(len(area)):
    ts4a_10[ts4i] = sum(gp[ts4i, int(ts4_10[ts4i]) : int(ts5_10[ts4i])])

for ts5i in range(len(area)):
    ts5a_10[ts5i] = sum(gp[ts5i, int(ts5_10[ts5i]) : int(ts6_10[ts5i])])

for ts6i in range(len(area)):
    ts6a_10[ts6i] = sum(gp[ts6i, int(ts6_10[ts6i]) : int(ts7_10[ts6i])])

for ts7i in range(len(area)):
    ts7a_10[ts7i] = sum(gp[ts7i, int(ts7_10[ts7i]) : int(ts8_10[ts7i])])

for ts8i in range(len(area)):
    ts8a_10[ts8i] = sum(gp[ts8i, int(ts8_10[ts8i]) : int(ts9_10[ts8i])])

for ts9i in range(len(area)):
    ts9a_10[ts9i] = sum(gp[ts9i, int(ts9_10[ts9i]) : int(ts10_10[ts9i])])

for ts10i in range(len(area)):
    ts10a_10[ts10i] = sum(gp[ts10i, int(ts10_10[ts10i]) : int(end_10[ts10i])])

print("Time intergrals calculated -10 ")

# time intergrals 8
# each time bin will be tail/5 long, with the addition of the rise time
time = int(t2 / 7)

# anchors
ts1_8 = area[:, 5]
ts2_8 = area[:, 4]
ts3_8 = area[:, 4] + time
ts4_8 = ts3_8 + time
ts5_8 = ts4_8 + time
ts6_8 = ts5_8 + time
ts7_8 = ts6_8 + time
ts8_8 = ts7_8 + time
end_8 = ts8_8 + time

# areas
ts1a_8 = np.zeros(len(area))  # baseline to mxi
ts2a_8 = np.zeros(len(area))
ts3a_8 = np.zeros(len(area))
ts4a_8 = np.zeros(len(area))
ts5a_8 = np.zeros(len(area))
ts6a_8 = np.zeros(len(area))
ts7a_8 = np.zeros(len(area))
ts8a_8 = np.zeros(len(area))


for ts1i in range(len(area)):
    ts1a_8[ts1i] = sum(gp[ts1i, int(ts1_8[ts1i]) : int(ts2_8[ts1i])])

for ts2i in range(len(area)):
    ts2a_8[ts2i] = sum(gp[ts2i, int(ts2_8[ts2i]) : int(ts3_8[ts2i])])

for ts3i in range(len(area)):
    ts3a_8[ts3i] = sum(gp[ts3i, int(ts3_8[ts3i]) : int(ts4_8[ts3i])])

for ts4i in range(len(area)):
    ts4a_8[ts4i] = sum(gp[ts4i, int(ts4_8[ts4i]) : int(ts5_8[ts4i])])

for ts5i in range(len(area)):
    ts5a_8[ts5i] = sum(gp[ts5i, int(ts5_8[ts5i]) : int(ts6_8[ts5i])])

for ts6i in range(len(area)):
    ts6a_8[ts6i] = sum(gp[ts6i, int(ts6_8[ts6i]) : int(ts7_8[ts6i])])

for ts7i in range(len(area)):
    ts7a_8[ts7i] = sum(gp[ts7i, int(ts7_8[ts7i]) : int(ts8_8[ts7i])])

for ts8i in range(len(area)):
    ts8a_8[ts8i] = sum(gp[ts8i, int(ts8_8[ts8i]) : int(end_8[ts8i])])

print("Time intergrals calculated - 8 ")

# time intergrals 6
# each time bin will be tail/5 long, with the addition of the rise time
time = int(t2 / 5)

# anchors
ts1_6 = area[:, 5]
ts2_6 = area[:, 4]
ts3_6 = area[:, 4] + time
ts4_6 = ts3_6 + time
ts5_6 = ts4_6 + time
ts6_6 = ts5_6 + time
end_6 = ts6_6 + time

# areas
ts1a_6 = np.zeros(len(area))
ts2a_6 = np.zeros(len(area))
ts3a_6 = np.zeros(len(area))
ts4a_6 = np.zeros(len(area))
ts5a_6 = np.zeros(len(area))
ts6a_6 = np.zeros(len(area))

for ts1i in range(len(area)):
    ts1a_6[ts1i] = sum(gp[ts1i, int(ts1_6[ts1i]) : int(ts2_6[ts1i])])

for ts2i in range(len(area)):
    ts2a_6[ts2i] = sum(gp[ts2i, int(ts2_6[ts2i]) : int(ts3_6[ts2i])])

for ts3i in range(len(area)):
    ts3a_6[ts3i] = sum(gp[ts3i, int(ts3_6[ts3i]) : int(ts4_6[ts3i])])

for ts4i in range(len(area)):
    ts4a_6[ts4i] = sum(gp[ts4i, int(ts4_6[ts4i]) : int(ts5_6[ts4i])])

for ts5i in range(len(area)):
    ts5a_6[ts5i] = sum(gp[ts5i, int(ts5_6[ts5i]) : int(ts6_6[ts5i])])

for ts6i in range(len(area)):
    ts6a_6[ts6i] = sum(gp[ts6i, int(ts6_6[ts6i]) : int(end_6[ts6i])])

print("Time intergrals calculated - 6 ")
# time intergrals 4
# each time bin will be tail/5 long, with the addition of the rise time
time = int(t2 / 3)

# anchors
ts1 = area[:, 5]
ts2 = area[:, 4]
ts3 = area[:, 4] + time
ts4 = ts3 + time
end = ts4 + time

# areas
ts1a = np.zeros(len(area))
ts2a = np.zeros(len(area))
ts3a = np.zeros(len(area))
ts4a = np.zeros(len(area))

for ts1i in range(len(area)):
    ts1a[ts1i] = sum(gp[ts1i, int(ts1[ts1i]) : int(ts2[ts1i])])

for ts2i in range(len(area)):
    ts2a[ts2i] = sum(gp[ts2i, int(ts2[ts2i]) : int(ts3[ts2i])])

for ts3i in range(len(area)):
    ts3a[ts3i] = sum(gp[ts3i, int(ts3[ts3i]) : int(ts4[ts3i])])

for ts4i in range(len(area)):
    ts4a[ts4i] = sum(gp[ts4i, int(ts4[ts4i]) : int(end[ts4i])])

print("Time intergrals calculated - 4")
#
# vector with max value of each pulse
mx = np.zeros(len(area))
for m in range(len(area)):
    mx[m] = gp[m, int(area[m, 4])]

# FWTM Calculations
mxi = np.argmax(gp, axis=1)
fwtm = np.zeros(len(mx))


def near(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


for fi in range(len(mx)):
    fwtm[fi] = near(gp[fi, :], mx[fi] / 10) - mxi[fi]

print("FWTM Calculated")
# Make parsed file
B = np.column_stack(
    (
        ts1a,
        ts2a,
        ts3a,
        ts4a,
        ts1a_6,
        ts2a_6,
        ts3a_6,
        ts4a_6,
        ts5a_6,
        ts6a_6,
        ts1a_8,
        ts2a_8,
        ts3a_8,
        ts4a_8,
        ts5a_8,
        ts6a_8,
        ts7a_8,
        ts8a_8,
        ts1a_10,
        ts2a_10,
        ts3a_10,
        ts4a_10,
        ts5a_10,
        ts6a_10,
        ts7a_10,
        ts8a_10,
        ts9a_10,
        ts10a_10,
        area[:, 0],
        area[:, 1],
        area[:, 3],
        area[:, 2],
        mx,
    )
)


# Save file
np.savetxt(
    "Ham6075_stil_Cf_mixed_var3.csv",
    B,
    delimiter=",",
    header="ts1a,ts2a,ts3a,ts4a,ts1a_6,ts2a_6,ts3a_6,ts4a_6,ts5a_6,ts6a_6,ts1a_8,ts2a_8,ts3a_8,ts4a_8,ts5a_8,ts6a_8,ts7a_8,ts8a_8,ts1a_10,ts2a_10,ts3a_10,ts4a_10,ts5a_10,ts6a_10,ts7a_10,ts8a_10,ts9a_10,ts10a_10,head,tail,ttt,tot,voltage",
    comments="",
)


print("fin")
#%% Basic plots
# plt.plot(mx,area[:,3],'.')
# plt.figure()
# plt.plot(mx,fwtm,'.')

##%% Reload from previoiusly saved
# import pandas as pd
# data=pd.read_csv('Ham6075_ej_Cf_mixed.csv')
# data=np.array(data)
#
#
#
# B=data

##%% Calibrate and cut
# weight=0.004399632
# bias=-54.28918773
#
# data[:,9]=(data[:,9]*weight)+bias
#
# cut_min, cut_max = 200, 1000
##for i in range (len(fwtm)):
##    if  fwtm[k] < fwtm_min :
# for i in range(len(data)):
#    if data[i,9] < cut_min:
#        data[i,:]=np.zeros(len(data[0,:]))
#    if data[i,9] > cut_max:
#        data[i,:]=np.zeros(len(data[0,:]))
#
# data=data[~np.all(data == 0, axis=1)]
