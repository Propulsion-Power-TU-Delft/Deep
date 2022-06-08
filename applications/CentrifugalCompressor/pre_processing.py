#! /usr/bin/env python
# -*- coding: utf-8 -*-

########################################################################################################################
# DeepData: single-stage centrifugal compressor application
# Authors: ir. A. Giuffre', Dr. ir. M. Pini
# Content: Utilities for loading and cleaning the dataset
# 2022 - TU Delft - All rights reserved
########################################################################################################################

import os
import numpy as np
import scipy.stats
from sklearn.utils import shuffle
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


def pre_processing(X, Y, features, labels, dev_size, test_size, X_scaler_type, Y_scaler_type, plotClass,
                   eta_min=0.5, OR_min=0.05, beta_deviation_max=0.2):
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
    :param plotClass:
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
    if Y_scaler is None:
        Y_encoded = Y_filtered
    else:
        Y_encoded = labels_encoding(labels, Y_filtered)

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
    elif labels[0] == 'Omega [rpm]':
        plotClass.plot_constraints_distribution(Y_train_norm)

    return X_train_norm, X_dev_norm, X_test_norm, Y_train_norm, Y_dev_norm, Y_test_norm, X_scaler, Y_scaler


def labels_encoding(labels, Y):
    """
    Transform the labels to simplify the regression task
    :param labels: array of strings identifying the labels associated to each column of Y
    :param Y: original labels
    :return: encoded labels
    """
    Y_encoded = Y

    for ii, label in enumerate(labels):
        if (label == 'beta_tt') or (label == 'Omega [rpm]') or \
                (label == 'OR') or (label == 'R4 [m]') or (label == 'H2 [m]'):
            Y_encoded[:, ii] = np.log(Y[:, ii])

    return Y_encoded


def labels_decoding(labels, Y_encoded):
    """
    Inverse transformation of labels, used when testing and deploying the ANN to predict an unknown output
    :param labels: array of strings identifying the labels associated to each column of Y
    :param Y_encoded: encoded labels
    :return: original labels
    """
    Y = Y_encoded

    for ii, label in enumerate(labels):
        if (label == 'beta_tt') or (label == 'Omega [rpm]') or \
                (label == 'OR') or (label == 'R4 [m]') or (label == 'H2 [m]'):
            Y[:, ii] = np.exp(Y_encoded[:, ii])

    return Y

