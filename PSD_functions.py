# -*- coding: utf-8 -*-
"""
Support functions for KNN PSD work for INMM 2020
Author: Matthew Durbin
"""

import numpy as np
from sklearn import neighbors
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.stats as stats
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D

# Detector/Light sensor specific parameters
# Hammamatsu and Stilbene
# histbins,nlb,nub, glb, gub, histbins_f, nlb_f,nub_f, glb_f, gub_f = np.linspace(0,.5, 501), .28, .4, .1, .275, np.linspace(0,400,401), 210, 300, 101, 200
histbins, nlb, nub, glb, gub, histbins_f, nlb_f, nub_f, glb_f, gub_f = (
    np.linspace(0.1, 0.3, 201),
    0.17,
    0.24,
    0.13,
    0.16,
    np.linspace(0, 400, 401),
    210,
    300,
    101,
    200,
)
# ham and EJ
# histbins,nlb,nub, glb, gub, histbins_f, nlb_f,nub_f, glb_f, gub_f = np.linspace(0.1,.6, 501), .35, .45, .2, .33, np.linspace(100,400,301), 215, 380, 160, 200
# Sensl and Stilbene
# histbins,nlb,nub, glb, gub, histbins_f, nlb_f,nub_f, glb_f, gub_f = np.linspace(0,.5, 501), .28, .4, .1, .275, np.linspace(100,500,401), 328, 410, 250, 325
# Sensl and Ej
# histbins,nlb,nub, glb, gub, histbins_f, nlb_f,nub_f, glb_f, gub_f = np.linspace(0.1,.6, 501), .285, .41, .1, .27, np.linspace(100,500,401), 336, 400, 230, 324
# PMT and Stil
# histbins,nlb,nub, glb, gub= np.linspace(0,.5, 501), .22, .36, 0, .19
# PMT and Ej 299
# histbins,nlb,nub, glb, gub= np.linspace(0,.5, 501), .185, .36, 0, .17
# nlb,nub = 0.0, .19 # pmt stil floor glb, gub = 0.22, 0.32 # pmt stil

# knn
def kr(nn, x_t, y_t, x_v, y_v):
    """
    Perform KNN Regression
    nn - nearest neighbors
    x_t/y_t - training
    x_v/y_v - testing
    """
    knnr = neighbors.KNeighborsRegressor(n_neighbors=nn)
    knnr.fit(x_t, y_t)
    pred = knnr.predict(x_v)
    return pred


# Gauss
def func(x, a, x0, sigma):
    # gaussian fit
    return a * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))


# traditional fom anlsysis
def tFOM(Y_test):
    """
    Perform the conventional FOM analysis
    Y_test - data to perform analysis on
    """
    delta = histbins[1] - histbins[0]
    nt_lnspc = np.linspace(
        nlb, nub, int(round((nub - nlb) / delta))
    )  # create hist bins for bounded neturon region
    test_n_hist, vals = np.histogram(
        Y_test, bins=histbins
    )  # create hist for bounded neturon region
    test_n_hist = test_n_hist[
        int(round(nlb / delta))
        - int(round(histbins[0] / delta)) : int(round(nub / delta))
        - int(round(histbins[0] / delta))
    ]

    # create new vector that has these PSP values between bounds
    Y_test_n = np.zeros(len(Y_test))

    for i in range(len(Y_test)):
        if nlb <= Y_test[i] <= nub:
            Y_test_n[i] = Y_test[i]

    Y_test_n = Y_test_n[Y_test_n != 0]

    # Truth fitting
    popt, pcov = curve_fit(
        func,
        nt_lnspc,
        test_n_hist,
        p0=[sum(test_n_hist), np.mean(nt_lnspc), np.mean(nt_lnspc) / 2],
        maxfev=1000000,
    )
    test_n_gauss = func(nt_lnspc, popt[0], popt[1], popt[2])
    a, mu_nt, sigma_nt = abs(popt)

    # gamma
    gt_lnspc = np.linspace(
        glb, gub, int(round((gub - glb) / delta))
    )  # create hist bins for bounded gamma region
    test_g_hist, vals = np.histogram(
        Y_test, bins=histbins
    )  # create hist for bounded neturon region
    test_g_hist = test_g_hist[
        int(round(glb / delta))
        - int(round(histbins[0] / delta)) : int(round(gub / delta))
        - int(round(histbins[0] / delta))
    ]

    Y_test_g = np.zeros(len(Y_test))

    for i in range(len(Y_test)):
        if glb <= Y_test[i] <= gub:
            Y_test_g[i] = Y_test[i]

    Y_test_g = Y_test_g[Y_test_g != 0]

    popt, pcov = curve_fit(
        func,
        gt_lnspc,
        test_g_hist,
        p0=[sum(test_g_hist), np.mean(gt_lnspc), np.mean(gt_lnspc) / 2],
        maxfev=1000000,
    )
    test_g_gauss = func(gt_lnspc, popt[0], popt[1], popt[2])
    a, mu_gt, sigma_gt = abs(popt)

    fwhm_nt = 2.355 * sigma_nt
    fwhm_gt = 2.355 * sigma_gt
    FOM_t = (mu_nt - mu_gt) / (fwhm_nt + fwhm_gt)
    return FOM_t

    # FOM anayslis for knn
    """
    Perform the KNN Regressed FOM analysis
    Y_test - data to perform analysis on
    """


def kFOM(prediction):
    delta = histbins[1] - histbins[0]
    pred_n_hist, vals = np.histogram(
        prediction, bins=histbins
    )  # create hist for bounded neturon region
    pred_n_hist = pred_n_hist[
        int(round(nlb / delta))
        - int(round(histbins[0] / delta)) : int(round(nub / delta))
        - int(round(histbins[0] / delta))
    ]

    nub_p = nub
    # nub_p=sum(pred_n_hist*nt_lnspc)/sum(pred_n_hist)
    # nub_p=nt_lnspc[np.argmax(pred_n_hist)]+nd*delta
    nt_lnspc_p = np.linspace(
        nlb, nub_p, int(round((nub_p - nlb) / delta))
    )  # create hist bins for bounded neturon region
    pred_n_hist, vals = np.histogram(
        prediction, bins=histbins
    )  # create hist for bounded neturon region
    pred_n_hist = pred_n_hist[
        int(round(nlb / delta))
        - int(round(histbins[0] / delta)) : int(round(nub_p / delta))
        - int(round(histbins[0] / delta))
    ]

    prediction_n = np.zeros(len(prediction))
    for i in range(len(prediction)):
        if nlb <= prediction[i] <= nub_p:
            prediction_n[i] = prediction[i]

    prediction_n = prediction_n[prediction_n != 0]

    # prediction fitting
    popt, pcov = curve_fit(
        func,
        nt_lnspc_p,
        pred_n_hist,
        p0=[sum(pred_n_hist), np.mean(nt_lnspc_p), np.mean(nt_lnspc_p) / 2],
        maxfev=1000000,
    )
    pred_n_gauss = func(nt_lnspc_p, popt[0], popt[1], popt[2])
    a, mu_np, sigma_np = abs(popt)

    # gamma
    pred_g_hist, vals = np.histogram(
        prediction, bins=histbins
    )  # create hist for bounded neturon region
    pred_g_hist = pred_g_hist[
        int(round(glb / delta))
        - int(round(histbins[0] / delta)) : int(round(gub / delta))
        - int(round(histbins[0] / delta))
    ]
    # glb_p=mu_gt-delta
    glb_p = glb
    # glb_p=gt_lnspc[np.argmax(pred_g_hist)]-(gd*delta)
    gt_lnspc_p = np.linspace(
        glb_p, gub, int(round((gub - glb_p) / delta))
    )  # create hist bins for bounded gamma region
    pred_g_hist, vals = np.histogram(
        prediction, bins=histbins
    )  # create hist for bounded neturon region
    pred_g_hist = pred_g_hist[
        int(round(glb_p / delta))
        - int(round(histbins[0] / delta)) : int(round(gub / delta))
        - int(round(histbins[0] / delta))
    ]

    prediction_g = np.zeros(len(prediction))

    for i in range(len(prediction)):
        if glb_p <= prediction[i] <= gub:
            prediction_g[i] = prediction[i]

    prediction_g = prediction_g[prediction_g != 0]

    popt, pcov = curve_fit(
        func,
        gt_lnspc_p,
        pred_g_hist,
        p0=[sum(pred_g_hist), np.mean(gt_lnspc_p), np.mean(gt_lnspc_p) / 2],
        maxfev=1000000,
    )
    pred_g_gauss = func(gt_lnspc_p, popt[0], popt[1], popt[2])
    a, mu_gp, sigma_gp = abs(popt)
    fwhm_np = 2.355 * sigma_np
    fwhm_gp = 2.355 * sigma_gp
    FOM_p = (mu_np - mu_gp) / (fwhm_np + fwhm_gp)
    return FOM_p
