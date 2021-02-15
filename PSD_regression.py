# -*- coding: utf-8 -*-
"""
Expirement for PSD Regression INMM Work - 2020

Perform KNN regression for various input features
Perform FOM analysis 
@author: Matthew Durbin
"""

import numpy as np
from sklearn import neighbors
from sklearn.utils import shuffle
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit, KFold
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.stats as norm
from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy
import os

# Load Data
path = "E:\\MattD\\Covoid\\PSD\\alg"
os.chdir(path)
# t,t1,t2,t3,t4,t5,t6,t7,t8,t9,w0,w1,w2,w3,w4,w5,w6,w7,w8,f0,f1,f2,f3,f4,volt,total,cpsp,wpsp,fpsp # form
data = np.array(pd.read_csv("Ham_stil_Cf_829_master.csv"))
exec(open("psd_functions_w.py").read())
# Energy cut
ecut_h, ecut_l = 1000, 200
for i in range(len(data)):
    if data[i, -4] < ecut_l:
        data[i, :] = np.zeros(len(data[0]))
    if data[i, -4] > ecut_h:
        data[i, :] = np.zeros(len(data[0]))

data = data[~np.all(data == 0, axis=1)]
# first split
data_s = shuffle(data, random_state=0)  # shuffle data
d_test = data_s[: int(0.2 * len(data_s)), :]
d_train_val = data_s[int(0.2 * len(data_s)) :, :]

x_tv, y_tv, e_tv = d_train_val[:, :24], d_train_val[:, -2], d_train_val[:, :-4]
mms = MinMaxScaler()
mms.fit(x_tv)
d_tv = mms.transform(x_tv)

# K Fold KNN Regression Loop
nn = 800
foms = np.zeros((5, 8))
kf_s = KFold(n_splits=5, shuffle=True)
for fno, (t_in, v_in) in enumerate(kf_s.split(x_tv, y_tv)):
    x_t_c, x_t_w, x_t_f, x_t_cw, x_t_cf, x_t_wf, x_t_cwf = (
        x_tv[t_in, :10],
        x_tv[t_in, 10:19],
        x_tv[t_in, 19:24],
        np.column_stack(
            (x_tv[t_in, :10], x_tv[t_in, 13], x_tv[t_in, 16], x_tv[t_in, 19])
        ),
        np.column_stack((x_tv[t_in, :10], x_tv[t_in, 19:22])),
        np.column_stack(
            (x_tv[t_in, 13], x_tv[t_in, 16], x_tv[t_in, 19], x_tv[t_in, 19:22])
        ),
        np.column_stack(
            (
                x_tv[t_in, :10],
                x_tv[t_in, 13],
                x_tv[t_in, 16],
                x_tv[t_in, 19],
                x_tv[t_in, 19:22],
            )
        ),
    )
    x_v_c, x_v_w, x_v_f, x_v_cw, x_v_cf, x_v_wf, x_v_cwf = (
        x_tv[v_in, :10],
        x_tv[v_in, 10:19],
        x_tv[v_in, 19:24],
        np.column_stack(
            (x_tv[v_in, :10], x_tv[v_in, 13], x_tv[v_in, 16], x_tv[v_in, 19])
        ),
        np.column_stack((x_tv[v_in, :10], x_tv[v_in, 19:22])),
        np.column_stack(
            (x_tv[v_in, 13], x_tv[v_in, 16], x_tv[v_in, 19], x_tv[v_in, 19:22])
        ),
        np.column_stack(
            (
                x_tv[v_in, :10],
                x_tv[v_in, 13],
                x_tv[v_in, 16],
                x_tv[v_in, 19],
                x_tv[v_in, 19:22],
            )
        ),
    )
    y_t, e_t, y_v, e_v = y_tv[t_in], e_tv[t_in], y_tv[v_in], e_tv[v_in]
    print("folding", fno + 1, "/5")
    foms[fno, 0] = tFOM(y_v)
    pred_c = kr(nn, x_t_c, y_t, x_v_c, y_v)
    foms[fno, 1] = kFOM(pred_c)
    pred_w = kr(nn, x_t_w, y_t, x_v_w, y_v)
    foms[fno, 2] = kFOM(pred_w)
    pred_f = kr(nn, x_t_f, y_t, x_v_f, y_v)
    foms[fno, 3] = kFOM(pred_f)
    pred_cw = kr(nn, x_t_cw, y_t, x_v_cw, y_v)
    foms[fno, 4] = kFOM(pred_cw)
    pred_cf = kr(nn, x_t_cf, y_t, x_v_cf, y_v)
    foms[fno, 5] = kFOM(pred_cf)
    pred_wf = kr(nn, x_t_wf, y_t, x_v_wf, y_v)
    foms[fno, 6] = kFOM(pred_wf)
    pred_cwf = kr(nn, x_t_cwf, y_t, x_v_cwf, y_v)
    foms[fno, 7] = kFOM(pred_cwf)
