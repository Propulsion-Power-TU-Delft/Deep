#! /usr/bin/env python
# -*- coding: utf-8 -*-

########################################################################################################################
# DeepData: data-driven equation of state application
# Authors: ir. A. Giuffre', ir. E. Bunschoten, Dr. ir. A. Cappiello, ir. M. Majer, Dr. ir. M. Pini
# Content: Load and pre-process the dataset, train and save the selected model
# 2022 - TU Delft - All rights reserved
########################################################################################################################

import pickle
import numpy as np
import seaborn as sns
from utils.gp import *
import tensorflow as tf
from utils.mlp import *
from matplotlib import cm
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader


# user-defined input
dev_size = 5                # percentage of total dataset used for the dev set
test_size = 5               # percentage of total dataset used for the test set
model_type = 'GP'           # 'MLP' or 'GP'
data_folder = 'MM_250k'     # name of the folder collecting the dataset
data_type = '1phase'        # '1phase', '2phase', or 'full'

# ------------------------------------------------------------------------------------------------------------------- #

# Pre-Processing

X = pickle.load(open(os.path.join('data', data_folder, 'X_' + data_type + '.pkl'), 'rb'))
Y = pickle.load(open(os.path.join('data', data_folder, 'Y_' + data_type + '.pkl'), 'rb'))

# split into train, dev, test sets
X, Y = shuffle(X, Y, random_state=61)
n_dev = int(X.shape[0] * dev_size / 100)
n_test = int(X.shape[0] * test_size / 100)
n_train = X.shape[0] - n_dev - n_test
X_train = X[:n_train, :]
Y_train = Y[:n_train, :]
X_dev = X[n_train:(n_train + n_dev), :]
Y_dev = Y[n_train:(n_train + n_dev), :]
X_test = X[(n_train + n_dev):, :]
Y_test = Y[(n_train + n_dev):, :]

# save train, dev and test sets
pickle.dump(X_train, open(os.path.join('data', data_folder, 'X_train_' + data_type + '.pkl'), 'wb'))
pickle.dump(Y_train, open(os.path.join('data', data_folder, 'Y_train_' + data_type + '.pkl'), 'wb'))
pickle.dump(X_dev, open(os.path.join('data', data_folder, 'X_dev_' + data_type + '.pkl'), 'wb'))
pickle.dump(Y_dev, open(os.path.join('data', data_folder, 'Y_dev_' + data_type + '.pkl'), 'wb'))
pickle.dump(X_test, open(os.path.join('data', data_folder, 'X_test_' + data_type + '.pkl'), 'wb'))
pickle.dump(Y_test, open(os.path.join('data', data_folder, 'Y_test_' + data_type + '.pkl'), 'wb'))

# normalize input features of train, dev and test sets
X_scaler = MinMaxScaler()
X_scaler.fit(X_train)
X_train_norm = X_scaler.transform(X_train)
X_dev_norm = X_scaler.transform(X_dev)
X_test_norm = X_scaler.transform(X_test)
pickle.dump(X_scaler, open(os.path.join('data', data_folder, 'X_scaler_' + data_type), 'wb'))

# transform and normalize labels of train, dev and test sets
Y_train_new = np.copy(Y_train)
Y_dev_new = np.copy(Y_dev)
Y_test_new = np.copy(Y_test)
Y_train_new[:, 2] = np.log(-Y_train_new[:, 2])
Y_train_new[:, 5] = np.log(Y_train_new[:, 5])
Y_dev_new[:, 2] = np.log(-Y_dev_new[:, 2])
Y_dev_new[:, 5] = np.log(Y_dev_new[:, 5])
Y_test_new[:, 2] = np.log(-Y_test_new[:, 2])
Y_test_new[:, 5] = np.log(Y_test_new[:, 5])

Y_scaler = MinMaxScaler()
Y_scaler.fit(Y_train_new)
Y_train_norm = Y_scaler.transform(Y_train_new)
Y_dev_norm = Y_scaler.transform(Y_dev_new)
Y_test_norm = Y_scaler.transform(Y_test_new)
pickle.dump(Y_scaler, open(os.path.join('data', data_folder, 'Y_scaler_' + data_type), 'wb'))

# ------------------------------------------------------------------------------------------------------------------- #

# Plotting

sns.set_style("darkgrid")
sns.set_context("paper")
plot_dir = os.path.join('plots', model_type, data_folder + '_' + data_type)
if not os.path.isdir(plot_dir):
    os.makedirs(plot_dir)

# plot input features distribution
fig1, axs1 = plt.subplots(2, 2, figsize=(8, 10))
n_plots = 2
new_colors = [plt.get_cmap(cm.viridis)(1. * i / n_plots) for i in range(n_plots)]

ax00 = sns.histplot(X_train[:, 0], ax=axs1[0, 0], color=new_colors[0])
ax00.set_xlabel('rho [kg/m3]')
ax00.set_yticklabels([])
ax00.set_ylabel('')
ax01 = sns.histplot(X_train[:, 1] / 1e3, ax=axs1[0, 1], color=new_colors[1])
ax01.set_xlabel('e [kJ/kg]')
ax01.set_yticklabels([])
ax01.set_ylabel('')
ax10 = sns.histplot(X_train_norm[:, 0], ax=axs1[1, 0], color=new_colors[0])
ax10.set_xlabel('rho norm [-]')
ax10.set_yticklabels([])
ax10.set_ylabel('')
ax11 = sns.histplot(X_train_norm[:, 1], ax=axs1[1, 1], color=new_colors[1])
ax11.set_xlabel('e norm [-]')
ax11.set_yticklabels([])
ax11.set_ylabel('')
fig1.savefig(os.path.join(plot_dir, 'features_distribution.jpeg'), dpi=400)
plt.close(fig1)

# plot labels distribution
fig2, axs2 = plt.subplots(2, 3, figsize=(10, 8))
n_plots = 6
new_colors = [plt.get_cmap(cm.viridis)(1. * i / n_plots) for i in range(n_plots)]

ax00 = sns.histplot(Y_train[:, 0], ax=axs2[0, 0], color=new_colors[0])
ax00.set_xlabel('s')
ax00.set_yticklabels([])
ax00.set_ylabel('')
ax01 = sns.histplot(Y_train[:, 1], ax=axs2[0, 1], color=new_colors[1])
ax01.set_xlabel('ds/de')
ax01.set_yticklabels([])
ax01.set_ylabel('')
ax02 = sns.histplot(Y_train[:, 2], ax=axs2[0, 2], color=new_colors[2])
ax02.set_xlabel('ds/drho')
ax02.set_yticklabels([])
ax02.set_ylabel('')
ax10 = sns.histplot(Y_train[:, 3], ax=axs2[1, 0], color=new_colors[3])
ax10.set_xlabel('d2s/de.drho')
ax10.set_yticklabels([])
ax10.set_ylabel('')
ax11 = sns.histplot(Y_train[:, 4], ax=axs2[1, 1], color=new_colors[4])
ax11.set_xlabel('d2s/de2')
ax11.set_yticklabels([])
ax11.set_ylabel('')
ax12 = sns.histplot(Y_train[:, 5], ax=axs2[1, 2], color=new_colors[5])
ax12.set_xlabel('d2s/drho2')
ax12.set_yticklabels([])
ax12.set_ylabel('')
fig2.savefig(os.path.join(plot_dir, 'labels_distribution.jpeg'), dpi=400)
plt.close(fig2)

# plot normalized labels distribution
fig3, axs3 = plt.subplots(2, 3, figsize=(10, 8), sharex=True, sharey=True)
n_plots = 6
new_colors = [plt.get_cmap(cm.viridis)(1. * i / n_plots) for i in range(n_plots)]

ax00 = sns.histplot(Y_train_norm[:, 0], ax=axs3[0, 0], color=new_colors[0])
ax00.set_xlabel('s norm')
ax00.set_yticklabels([])
ax00.set_ylabel('')
ax01 = sns.histplot(Y_train_norm[:, 1], ax=axs3[0, 1], color=new_colors[1])
ax01.set_xlabel('ds/de norm')
ax01.set_yticklabels([])
ax01.set_ylabel('')
ax02 = sns.histplot(Y_train_norm[:, 2], ax=axs3[0, 2], color=new_colors[2])
ax02.set_xlabel('ds/drho norm')
ax02.set_yticklabels([])
ax02.set_ylabel('')
ax10 = sns.histplot(Y_train_norm[:, 3], ax=axs3[1, 0], color=new_colors[3])
ax10.set_xlabel('d2s/de.drho norm')
ax10.set_yticklabels([])
ax10.set_ylabel('')
ax11 = sns.histplot(Y_train_norm[:, 4], ax=axs3[1, 1], color=new_colors[4])
ax11.set_xlabel('d2s/de2 norm')
ax11.set_yticklabels([])
ax11.set_ylabel('')
ax12 = sns.histplot(Y_train_norm[:, 5], ax=axs3[1, 2], color=new_colors[5])
ax12.set_xlabel('d2s/drho2 norm')
ax12.set_yticklabels([])
ax12.set_ylabel('')
fig3.savefig(os.path.join(plot_dir, 'norm_labels_distribution.jpeg'), dpi=400)
plt.close(fig3)

# ------------------------------------------------------------------------------------------------------------------- #

# Multi-Layer Perceptron

if model_type == 'MLP':

    # hyper-parameters
    L = 4
    nl = [30, 50, 50, 30]
    n_epochs = 10
    alpha = 4
    lr_decay = 0.95
    decay_steps = 10000
    batch_size = 7
    batch_norm = 0
    reg = 0
    dropout_prob = [0.99, 0.8, 0.8, 0.8]
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
                alpha=alpha, lr_decay=lr_decay, decay_steps=decay_steps, staircase=staircase,
                batch_size=batch_size, batch_norm=batch_norm)
    model.set_model_architecture()
    model.model.summary()
    model.train_model()

    # save and plot results
    model_dir = os.path.join('models', 'MLP', data_folder + '_' + data_type)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    pickle.dump(model.history.history, open(os.path.join(model_dir, 'training_history'), 'wb'))
    model.model.save(model_dir)
    model.plot_training_history(plot_dir)

# ------------------------------------------------------------------------------------------------------------------- #

# Gaussian Process: train a stochastic variational GP for each output

if model_type == 'GP':

    # hyper-parameters
    epochs = 10
    alpha = 1
    batch_size = 10
    n_inducing = 500
    kernel = 'rbf'

    # iterate over labels
    labels = ['s', 'ds_de', 'ds_drho', 'd2s_de_drho', 'd2s_de2', 'd2s_drho2']

    for ii, label in enumerate(labels):

        print('\nTraining Gaussian Process to predict: ' + label + ' ...')

        # tensors definition
        train_x = torch.Tensor(X_train_norm)
        train_y = torch.Tensor(Y_train[:, ii])
        test_x = torch.Tensor(X_test_norm)
        test_y = torch.Tensor(Y_test[:, ii])

        # define dataset and mini-batches
        train_dataset = TensorDataset(train_x, train_y)
        train_loader = DataLoader(train_dataset, batch_size=int(2 ** batch_size), shuffle=True)
        test_dataset = TensorDataset(test_x, test_y)
        test_loader = DataLoader(test_dataset, batch_size=int(2 ** batch_size), shuffle=False)

        # define model
        inducing_points = train_x[:n_inducing, :].contiguous()
        model = StochasticVariationalGP(inducing_points, X_train.shape[1], kernel)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        # train the model
        train_variational_gp(model, likelihood, train_loader, X_train.shape[0], epochs, alpha)

        # save trained model
        model_dir = os.path.join('models', 'GP', data_folder + '_' + data_type)
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)

        torch.save(model.state_dict(), os.path.join(model_dir, 'model_state_' + label + '.pth'))

# ------------------------------------------------------------------------------------------------------------------- #
