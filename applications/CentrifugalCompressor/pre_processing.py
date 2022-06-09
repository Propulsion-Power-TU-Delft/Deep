#! /usr/bin/env python
# -*- coding: utf-8 -*-

########################################################################################################################
# DeepData: single-stage centrifugal compressor application
# Authors: ir. A. Giuffre', Dr. ir. M. Pini
# Content: Utilities for loading and cleaning the dataset
# 2022 - TU Delft - All rights reserved
########################################################################################################################

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def get_data(data_folders):
    """
    Load data and create two arrays collecting the input features and the labels
    :param data_folders: list of folders collecting the data to load
    :return:
        - X: array of input features (m, n_features)
        - Y: array of labels (m, n_labels)
    """
    for ii, folder in enumerate(data_folders):
        ds = np.load(os.path.join('data', folder), allow_pickle=True)
        print("Number of individuals in folder %s: %d" % (folder, ds["X"].shape[0]))

        if ii == 0:
            X = ds["X"]
            Y = ds["Y"]
        else:
            X = np.concatenate((X, ds["X"]), axis=0)
            Y = np.concatenate((Y, ds["Y"]), axis=0)

    print("\nShape of X:", X.shape)
    print("Shape of Y:", Y.shape)

    return X, Y


def pre_processing(X, Y, features, labels, dev_size, test_size, X_scaler_type, Y_scaler_type, encoding, plotClass,
                   data_dir, eta_min=0.5, OR_min=0.05, beta_deviation_max=0.2):
    """
    Delete meaningless individuals, normalize the dataset, and split into train, dev and test sets
    :param X: raw vector of input features (m, n_features)
    :param Y: raw vector of labels (m, n_labels)
    :param features: name of the input features to be considered
    :param labels: name of the labels to be considered
    :param dev_size: percentage of the dataset used for the dev set
    :param test_size: percentage of the dataset used for the test set
    :param X_scaler_type: type of scaler used to normalize the input features --> 'min max', 'standard', or 'none'
    :param Y_scaler_type: type of scaler used to normalize the labels --> 'min max', 'standard', or 'none'
    :param encoding: True or False --> use label encoding or not
    :param plotClass:
    :param data_dir: directory where data are stored
    :param eta_min: minimum threshold of eta_tt used for data filtering
    :param beta_deviation_max: maximum acceptable deviation between beta_tt_target and beta_tt used for data filtering
    :return:
        - X_train: array of input features used for the training set (m_train, n_features)
        - X_test: array of input features used for the test set (m_test, n_features)
        - Y_train: array of labels used for the training set (m_train, n_labels)
        - Y_test: array of labels used for the test set (m_test, n_labels)
    """
    # remove meaningless individuals from the dataset: min eta_tt, OR != 0, max % beta_tt deviation, max M3
    beta_deviation = np.abs(X[:, 8] - Y[:, 0]) / X[:, 8]
    mask = np.logical_and(np.logical_and(
        np.logical_and(np.greater_equal(Y[:, 2], eta_min), np.greater_equal(Y[:, 4], OR_min)),
        np.less_equal(beta_deviation, beta_deviation_max)), np.less_equal(Y[:, -2], 0.9))
    X_filtered = X[mask, :]
    Y_filtered = Y[mask, :]

    # remove undesired input features and labels
    features_tot = ['phi_t1', 'psi_is', 'alpha2 [deg]', 'R3 / R2', 'k', 'Nbl', 'Hr_pinch', 'Rr_pinch', 'beta_tt_target',
                    'm [kg/s]', 'N', 'Pr', 'Tr', 't_le [m]', 't_te [m]', 'R_shaft [m]', 'roughness [m]',
                    'rel_tip_clearance', 'rel_bf_clearance', 'gamma_Pv_mean']
    labels_tot = ['beta_tt', 'beta_ts', 'eta_tt', 'eta_ts', 'OR', 'm_choke [kg/s]', 'Omega [rpm]', 'F_ax [N]',
                  'R1_hub [m]', 'H2 [m]', 'beta2_bl [deg]', 'R4 [m]', 'M3', 'Mw1,s']
    X_idx = []
    Y_idx = []

    for feature in features:
        X_idx.append(features_tot.index(feature))

    for label in labels:
        Y_idx.append(labels_tot.index(label))

    X_filtered = X_filtered[:, X_idx]
    Y_filtered = Y_filtered[:, Y_idx]

    print("\nShape of X after cleaning and filtering:", X_filtered.shape)
    print("Shape of Y after cleaning and filtering:", Y_filtered.shape)

    # shuffle filtered dataset
    X_filtered, Y_filtered = shuffle(X_filtered, Y_filtered, random_state=61)

    # initialize scaler for input features
    if X_scaler_type == 'min-max':
        X_scaler = MinMaxScaler()
    elif X_scaler_type == 'standard':
        X_scaler = StandardScaler()
    else:
        X_scaler = None

    # initialize scaler for labels
    if Y_scaler_type == 'min-max':
        Y_scaler = MinMaxScaler()
    elif Y_scaler_type == 'standard':
        Y_scaler = StandardScaler()
    else:
        Y_scaler = None

    # labels encoding
    if encoding:
        Y_encoded, beta2_bl_min, Fax_min = labels_encoding(labels, Y_filtered)
        pickle.dump(beta2_bl_min, open(os.path.join(data_dir, 'beta2_bl_min'), 'wb'))
        pickle.dump(Fax_min, open(os.path.join(data_dir, 'Fax_min'), 'wb'))
    else:
        Y_encoded = Y_filtered

    # split the dataset into train, dev and test sets
    n_dev = int(X_filtered.shape[0] * dev_size / 100)
    n_test = int(X_filtered.shape[0] * test_size / 100)
    n_train = X_filtered.shape[0] - n_dev - n_test
    X_train = X_filtered[:n_train, :]
    Y_train = Y_encoded[:n_train, :]
    X_dev = X_filtered[n_train:(n_train + n_dev), :]
    Y_dev = Y_encoded[n_train:(n_train + n_dev), :]
    X_test = X_filtered[(n_train + n_dev):, :]
    Y_test = Y_encoded[(n_train + n_dev):, :]

    # normalize input features
    if X_scaler is None:
        X_train_norm = X_train
        X_dev_norm = X_dev
        X_test_norm = X_test
    else:
        X_scaler.fit(X_train)
        X_train_norm = X_scaler.transform(X_train)
        X_dev_norm = X_scaler.transform(X_dev)
        X_test_norm = X_scaler.transform(X_test)

    # normalize labels
    if Y_scaler is None:
        Y_train_norm = Y_train
        Y_dev_norm = Y_dev
        Y_test_norm = Y_test
    else:
        Y_scaler.fit(Y_train)
        Y_train_norm = Y_scaler.transform(Y_train)
        Y_dev_norm = Y_scaler.transform(Y_dev)
        Y_test_norm = Y_scaler.transform(Y_test)

    print("\nShape of X_train after scaling:", X_train_norm.shape)
    print("Shape of X_dev after scaling  :", X_dev_norm.shape)
    print("Shape of X_test after scaling :", X_test_norm.shape)
    print("\nShape of Y_train after scaling:", Y_train_norm.shape)
    print("Shape of Y_dev after scaling  :", Y_dev_norm.shape)
    print("Shape of Y_test after scaling :", Y_test_norm.shape)

    # plot input features and labels distribution
    plotClass.plot_feature_distribution(X_train_norm)

    if labels[0] == 'beta_tt':
        plotClass.plot_objectives_distribution(Y_train_norm)
        eta_reg = max_eta_trend(plotClass, X_filtered[:, 8], X_filtered[:, 9], Y_filtered[:, 1])
    elif labels[0] == 'Omega [rpm]':
        plotClass.plot_constraints_distribution(Y_train_norm)
        eta_reg = None

        # plot individual x-y relationships to analyze the dataset
    while True:
        X_idx, Y_idx = [int(x) for x in input("\nSpecify indexes of input feature and label you want to plot. "
                                              "Enter -1 -1 to quit plotting: ").split()]
        if (X_idx == -1) and (Y_idx == -1):
            break
        else:
            plt.figure()
            plt.scatter(X_filtered[:, X_idx], Y_filtered[:, Y_idx], s=1)
            plt.xlabel(features[X_idx])
            plt.ylabel(labels[Y_idx])
            plt.show()

    return X_train_norm, X_dev_norm, X_test_norm, Y_train_norm, Y_dev_norm, Y_test_norm, X_scaler, Y_scaler, eta_reg


def labels_encoding(Y_labels, Y):
    """
    Transform the labels to achieve a distribution of data that resembles more closely a Gaussian distribution,
    aiming to improve the performance of the ANN.
    :param Y_labels: array of strings identifying the labels associated to each column of Y
    :param Y: original labels
    :return: Y_encoded, beta2_bl_min, Fax_min
    """
    Y_encoded = Y
    beta2_bl_min = 0
    Fax_min = 0

    for ii, label in enumerate(Y_labels):
        if (label == 'beta_tt') or (label == 'Omega [rpm]'):
            Y_encoded[:, ii] = np.log(Y[:, ii])

        elif label == 'OR':
            Y_encoded[:, ii] = np.log(Y[:, ii] * 100)

        elif label == 'R1_hub [m]':
            Y_encoded[:, ii] = Y[:, ii] * 1000

        elif label == 'R4 [m]':
            Y_encoded[:, ii] = np.log(Y[:, ii] * 1000)

        elif label == 'H2 [m]':
            Y_encoded[:, ii] = np.log1p(Y[:, ii] * 1000)

        elif label == 'beta2_bl [deg]':
            beta2_bl_min = np.min(Y[:, ii])
            Y_encoded[:, ii] = Y[:, ii] - (beta2_bl_min - 1)

        elif label == 'F_ax [N]':
            Fax_min = np.min(Y[:, ii])
            Y_encoded[:, ii] = np.log(Y[:, ii] - (Fax_min * 1.01))

    return Y_encoded, beta2_bl_min, Fax_min


def labels_decoding(Y_labels, Y_encoded, beta2_bl_min, Fax_min):
    """
    Inverse transformation of labels, used when testing and deploying the ANN to predict an unknown output.
    :param Y_labels: array of strings identifying the labels associated to each column of Y
    :param Y_encoded: encoded labels
    :return:original labels
    """
    Y = Y_encoded

    for ii, label in enumerate(Y_labels):
        if (label == 'beta_tt') or (label == 'Omega [rpm]'):
            Y[:, ii] = np.exp(Y_encoded[:, ii])

        elif label == 'OR':
            Y[:, ii] = np.exp(Y_encoded[:, ii]) / 100

        elif label == 'R1_hub [m]':
            Y[:, ii] = Y_encoded[:, ii] / 1000

        elif label == 'R4 [m]':
            Y[:, ii] = np.exp(Y_encoded[:, ii]) / 1000

        elif label == 'H2 [m]':
            Y[:, ii] = np.expm1(Y_encoded[:, ii]) / 1000

        elif label == 'beta2_bl [deg]':
            Y[:, ii] = Y_encoded[:, ii] + (beta2_bl_min - 1)

        elif label == 'F_ax [N]':
            Y[:, ii] = np.exp(Y_encoded[:, ii]) + (Fax_min * 1.01)

    return Y


def max_eta_trend(plotClass, beta_tt, mass_flow, eta_tt, samples=30):
    """
    :param beta_tt: target total-to-total compression ratio dataset
    :param mass_flow: mass flow rate dataset
    :param eta_tt: total-to-total efficiency dataset
    :return:
    """
    beta_tt_vec = np.linspace(min(beta_tt), max(beta_tt), samples)
    mass_flow_vec = np.linspace(min(mass_flow), max(mass_flow), samples)
    eta_max = np.zeros((samples, samples))
    eta_max_fit = np.zeros((samples, samples))
    d_beta = 0.45 * (beta_tt_vec[1] - beta_tt_vec[0])
    d_mass = 0.45 * (mass_flow_vec[1] - mass_flow_vec[0])
    X = np.zeros((int(samples ** 2), 2))
    count = 0

    for ii, mass in enumerate(mass_flow_vec):
        mask_mass = np.logical_and(mass_flow > mass - d_mass, mass_flow < mass + d_mass)
        for jj, beta in enumerate(beta_tt_vec):
            mask_beta = np.logical_and(beta_tt > beta - d_beta, beta_tt < beta + d_beta)
            mask = np.logical_and(mask_mass, mask_beta)
            eta_max[ii, jj] = np.max(eta_tt[mask])
            X[count, :] = [mass, beta]
            count += 1

    eta_reg = LinearRegression().fit(X, eta_max.flatten())

    for ii, mass in enumerate(mass_flow_vec):
        for jj, beta in enumerate(beta_tt_vec):
            eta_max_fit[ii, jj] = eta_reg.predict(np.array([mass, beta]).reshape(1, -1))

    plotClass.plot_eta_max(beta_tt_vec, mass_flow_vec, eta_max, eta_max_fit)

    return eta_reg


