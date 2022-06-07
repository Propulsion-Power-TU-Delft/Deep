#! /usr/bin/env python
# -*- coding: utf-8 -*-

########################################################################################################################
# DeepData: data-driven equation of state application
# Authors: ir. A. Giuffre', ir. E. Bunschoten, Dr. ir. A. Cappiello, ir. M. Majer, Dr. ir. M. Pini
# Content: Thermodynamic relationships between s and primary properties
# 2022 - TU Delft - All rights reserved
########################################################################################################################

import numpy as np


def multi_output_primary_props(model, X, X_scaler, Y_scaler):
    """
    Evaluate the primary thermodynamic properties by resorting to a multi-output regression model
    :param model: pre-trained data-driven model, featuring 6 outputs
    :param X: array of input features (m x 2)
    :param X_scaler: pre-fitted scaler used for input features
    :param Y_scaler: pre-fitted scaler used for labels
    :return:
        - Y: array of model predictions after inverse transformation (m x 6)
        - primary_props: array pf primary thermodynamic properties (m x 4)
    """
    rho = X[:, 0]
    e = X[:, 1]

    # features encoding
    X_norm = X_scaler.transform(X)

    # model prediction
    Y_norm = model.predict(X_norm)

    # labels decoding
    Y = Y_scaler.inverse_transform(Y_norm)
    Y[:, 2] = - np.exp(Y[:, 2])
    Y[:, 5] = np.exp(Y[:, 5])

    # evaluation of primary thermodynamic properties
    s = Y[:, 0]
    ds_de = Y[:, 1]
    ds_drho = Y[:, 2]
    d2s_dedrho = Y[:, 3]
    d2s_de2 = Y[:, 4]
    d2s_drho2 = Y[:, 5]

    blue_term = (ds_drho * (2 - rho * (ds_de ** (-1)) * d2s_dedrho) + rho * d2s_drho2)
    green_term = (- (ds_de ** (-1)) * d2s_de2 * ds_drho + d2s_dedrho)
    c = np.sqrt(- rho * (ds_de ** (-1)) * (blue_term - rho * green_term * (ds_drho / ds_de)))
    T = 1 / ds_de
    P = - (rho ** 2) * T * ds_drho
    h = e + P / rho

    primary_props = np.vstack((P, T, h, c)).T

    return Y, primary_props