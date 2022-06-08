#! /usr/bin/env python
# -*- coding: utf-8 -*-

########################################################################################################################
# DeepData: single-stage centrifugal compressor application
# Authors: ir. A. Giuffre', Dr. ir. M. Pini
# Content: Load and pre-process the dataset, train and save the selected model
# 2022 - TU Delft - All rights reserved
########################################################################################################################

import plot
import pickle
from utils.gp import *
import tensorflow as tf
from utils.mlp import *
from pre_processing import *
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader


# user-defined input
dev_size = 6                              # percentage of total dataset used for the dev set
test_size = 6                             # percentage of total dataset used for the test set
model_type = 'GP'                         # 'MLP' or 'GP'
output_dir = "test"      # name of the output directory, where results will be saved

# paths of dataset to be loaded
dataset = ["R1233zd(E)_Pr0.015_Tr0.65", "Propane_Pr0.015_Tr0.65", "Isobutane_Pr0.015_Tr0.65",
           "R134a_Pr0.015_Tr0.65", "R134a_Pr0.15_Tr0.81", "R1233zd(E)_Pr0.15_Tr0.82",
           "CO2_Pr0.2_Tr0.85", "H2_Pr0.08_Tr2.0"]

# name of the input features to be considered
features = ['phi_t1', 'psi_is', 'alpha2 [deg]', 'R3 / R2', 'k', 'Nbl', 'Hr_pinch', 'Rr_pinch', 'beta_tt_target',
            'm [kg/s]', 'N', 'R_shaft [m]',  'rel_tip_clearance', 'rel_bf_clearance', 'gamma_Pv_mean']

# list of labels to be considered
labels = ['beta_tt', 'eta_tt', 'OR', 'm_choke [kg/s]']
# labels = ['Omega [rpm]', 'F_ax [N]', 'R1_hub [m]', 'H2 [m]', 'beta2_bl [deg]', 'R4 [m]']

"""
Available features: 
'phi_t1', 'psi_is', 'alpha2 [deg]', 'R3 / R2', 'k', 'Nbl', 'Hr_pinch', 'Rr_pinch', 'beta_tt_target', 'm [kg/s]', 
'N', 'Pr', 'Tr', 't_le [m]', 't_te [m]', 'R_shaft [m]', 'roughness [m]', 'rel_tip_clearance', 'rel_bf_clearance',
'gamma_Pv_mean'
     
Available labels:
'beta_tt', 'beta_ts', 'eta_tt', 'eta_ts', 'OR', 'm_choke [kg/s]', 'Omega [rpm]', 'F_ax [N]', 'R1_hub [m]', 'H2 [m]',
 'beta2_bl [deg]', 'R4 [m]', 'M3', 'Mw1,s'
"""

# ------------------------------------------------------------------------------------------------------------------- #

# Pre-Processing

print("\n# ------------------------------------------- Pre-Processing ------------------------------------------- #\n")

if labels[0] == 'beta_tt':
    plot_dir = os.path.join('plots', model_type, output_dir, 'obj')
    model_dir = os.path.join('models', model_type, output_dir, 'obj')
    data_dir = os.path.join('data', model_type, output_dir, 'obj')

elif labels[0] == 'Omega [rpm]':
    plot_dir = os.path.join('plots', model_type, output_dir, 'con')
    model_dir = os.path.join('models', model_type, output_dir, 'con')
    data_dir = os.path.join('data', model_type, output_dir, 'con')

if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

if not os.path.isdir(plot_dir):
    os.makedirs(plot_dir)

if not os.path.isdir(data_dir):
    os.makedirs(data_dir)

plotClass = plot.Plot(plot_dir)
X, Y = get_data(dataset)
pickle.dump(X, open(os.path.join(data_dir, 'X'), 'wb'))
pickle.dump(Y, open(os.path.join(data_dir, 'Y'), 'wb'))

# plot individual x-y relationships to analyze the dataset
while True:
    X_idx, Y_idx = [int(x) for x in input("\nSpecify indexes of input feature and label you want to plot. "
                                          "Enter -1 -1 to quit plotting: ").split()]
    if (X_idx == -1) and (Y_idx == -1):
        break
    else:
        plt.figure()
        plt.scatter(X[:, X_idx], Y[:, Y_idx], s=1)
        plt.xlabel(features[X_idx])
        plt.ylabel(labels[Y_idx])
        plt.show()

# pre-processing
X_train_norm, X_dev_norm, X_test_norm, Y_train_norm, Y_dev_norm, Y_test_norm, X_scaler, Y_scaler = \
    pre_processing(X, Y, features, labels, dev_size, test_size, 'min-max', 'none', plotClass)

# save the train, dev, and test sets
pickle.dump(features, open(os.path.join(data_dir, 'features'), 'wb'))
pickle.dump(X_scaler, open(os.path.join(data_dir, 'X_scaler'), 'wb'))
pickle.dump(X_train_norm, open(os.path.join(data_dir, 'X_train_norm'), 'wb'))
pickle.dump(X_dev_norm, open(os.path.join(data_dir, 'X_dev_norm'), 'wb'))
pickle.dump(X_test_norm, open(os.path.join(data_dir, 'X_test_norm'), 'wb'))
pickle.dump(labels, open(os.path.join(data_dir, 'labels'), 'wb'))
pickle.dump(Y_scaler, open(os.path.join(data_dir, 'Y_scaler'), 'wb'))
pickle.dump(Y_train_norm, open(os.path.join(data_dir, 'Y_train_norm'), 'wb'))
pickle.dump(Y_dev_norm, open(os.path.join(data_dir, 'Y_dev_norm'), 'wb'))
pickle.dump(Y_test_norm, open(os.path.join(data_dir, 'Y_test_norm'), 'wb'))

# ------------------------------------------------------------------------------------------------------------------- #

# Multi-Layer Perceptron

if model_type == 'MLP':
    print("\n# ------------------------------------------- MLP Training ------------------------------------------- #")

    # hyper-parameters
    L = 5
    nl = [199, 199, 200, 144, 42]
    n_epochs = 10
    alpha = 2.75
    lr_decay = 1.0
    decay_steps = 10000
    batch_size = 6
    batch_norm = 0
    reg = 0
    dropout_prob = [0.99, 0.8, 0.8, 0.8, 0.8]
    staircase = False
    max_norm = 4

    # performance settings suggested by Intel (for fpp2)
    n_cores_socket = 24  # number of cores per socket installed on the current machine
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(n_cores_socket)
    tf.config.set_soft_device_placement(True)
    os.environ["OMP_NUM_THREADS"] = "%d" % n_cores_socket
    os.environ["KMP_BLOCKTIME"] = "30"
    os.environ["KMP_SETTINGS"] = "1"
    os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"

    # train the model
    model = MLP(X_train_norm, X_dev_norm, Y_train_norm, Y_dev_norm, L, nl, n_epochs,
                alpha=alpha, lr_decay=lr_decay, decay_steps=decay_steps, staircase=staircase, batch_size=batch_size,
                batch_norm=batch_norm, regularization=reg, max_norm=max_norm, dropout_prob=dropout_prob)
    model.set_model_architecture()
    model.model.summary()
    model.train_model()

    # save and plot results
    pickle.dump(model.history.history, open(os.path.join(model_dir, 'training_history'), 'wb'))
    model.model.save(model_dir)
    model.plot_training_history(plot_dir)

# ------------------------------------------------------------------------------------------------------------------- #

# Gaussian Process: train a stochastic variational GP for each output

if model_type == 'GP':
    print("\n# ------------------------------------------- GP Training ------------------------------------------- #")

    # hyper-parameters
    epochs = 10
    alpha = 2
    batch_size = 10
    n_inducing = 500
    kernel = 'matern'

    for ii, label in enumerate(labels):

        print('\nTraining Gaussian Process to predict: ' + label + ' ...')

        # tensors definition
        train_x = torch.Tensor(X_train_norm)
        train_y = torch.Tensor(Y_train_norm[:, ii])
        test_x = torch.Tensor(X_test_norm)
        test_y = torch.Tensor(Y_test_norm[:, ii])

        # define dataset and mini-batches
        train_dataset = TensorDataset(train_x, train_y)
        train_loader = DataLoader(train_dataset, batch_size=int(2 ** batch_size), shuffle=True)
        test_dataset = TensorDataset(test_x, test_y)
        test_loader = DataLoader(test_dataset, batch_size=int(2 ** batch_size), shuffle=False)

        # define model
        inducing_points = train_x[:n_inducing, :].contiguous()
        model = StochasticVariationalGP(inducing_points, X_train_norm.shape[1], kernel)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        # train the model
        train_variational_gp(model, likelihood, train_loader, X_train_norm.shape[0], epochs, alpha)

        # save trained model
        torch.save(model.state_dict(), os.path.join(model_dir, 'model_state_' + label.split(' ')[0] + '.pth'))

# ------------------------------------------------------------------------------------------------------------------- #
