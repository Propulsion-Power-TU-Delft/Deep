#! /usr/bin/env python
# -*- coding: utf-8 -*-

########################################################################################################################
# Deep: data-driven equation of state application
# Authors: ir. A. Giuffre', ir. E. Bunschoten, Dr. ir. A. Cappiello, ir. M. Majer, Dr. ir. M. Pini
# Content: Optimize the set of hyper-parameters for the MLP architecture
# 2022 - TU Delft - All rights reserved
########################################################################################################################

import pickle
import PyNomad
import numpy as np
import seaborn as sns
import tensorflow as tf
from utils.mlp import *
from matplotlib import cm
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader


# user-defined input
dev_size = 5                                 # percentage of total dataset used for the dev set
test_size = 5                                # percentage of total dataset used for the test set
model_type = 'MLP'                           # 'MLP' or 'GP'
data_folder = 'MLP_MM_250k_rho_above_10'     # name of the folder collecting the dataset
data_type = '1phase'                         # '1phase', '2phase', or 'full'
L = 2                                        # number of hidden layers
n_epochs = 250                               # number of epochs used for training
n_obj = 2                                    # number of objectives considered for the hyper-parameters search
regularization = 0                           # regularization strategy
w_init = 'he_uniform'                        # weight initialization strategy
lr_decay = 1.0                               # learning rate decay
decay_steps = 100000                         # the learning rate drops significantly every decay_steps
staircase = False                            # if True, the learning rate drops following a staircase
max_norm = 4                                 # value of max-norm weight constraint used in each layer (inverted dropout)
dropout_prob = []                            # list of dropout probabilities used in each layer, except for the output
x0 = [10,  10,  6,  0, 4, 6]                 # guess values of hyper-parameters used to start the optimization
lb = [2,   2,   2,  0, 1, 0]                 # lower bounds of hyper-parameters
ub = [100, 100, 10, 2, 5, 6]                 # upper bounds of hyper-parameters
max_eval = 100                               # maximum number of evaluations set for the optimization

# ------------------------------------------------------------------------------------------------------------------- #

# Pre-Processing

X = pickle.load(open(os.path.join('data', data_folder, 'X_' + data_type + '.pkl'), 'rb'))
Y = pickle.load(open(os.path.join('data', data_folder, 'Y_' + data_type + '.pkl'), 'rb'))

# split into train, dev, test sets
X, Y = shuffle(X, Y, random_state=29)
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

# Optimization

input_type = "BB_INPUT_TYPE ( "

for layer in range(L):
    input_type += " I"

input_type += " I I R I )"

model_dir = os.path.join('models', 'MLP', data_folder + '_' + data_type)
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

params = [input_type, "BB_OUTPUT_TYPE OBJ", "MAX_BB_EVAL " + str(max_eval),
          "SGTELIB_MODEL_SEARCH yes", "SGTELIB_MODEL_DEFINITION TYPE KRIGING", "SGTELIB_MODEL_SEARCH_TRIALS 5",
          "QUAD_MODEL_SEARCH yes", "SPECULATIVE_SEARCH yes", "CS_OPTIMIZATION true",
          "NB_THREADS_OPENMP 1", "DISPLAY_DEGREE 2", "DISPLAY_ALL_EVAL false", "DISPLAY_STATS BBE OBJ",
          "HISTORY_FILE " + model_dir + "/black_box_evaluations.txt",
          "SOLUTION_FILE " + model_dir + "/solution.txt",
          "STATS_FILE " + model_dir + "/convergence_history.txt",
          "CACHE_FILE " + model_dir + "/cache.txt"]

problem = HyperSearch(X_train_norm, X_dev_norm, Y_train_norm, Y_dev_norm, L, n_epochs, n_obj,
                      regularization=regularization, w_init=w_init, lr_decay=lr_decay, decay_steps=decay_steps,
                      staircase=staircase, max_norm=max_norm, dropout_prob=dropout_prob)
res = PyNomad.optimize(problem.evaluate, x0, lb, ub, params)

fmt = ["{} = {}".format(n, v) for (n, v) in res.items()]
output = "\n".join(fmt)
print("\nNOMAD results \n" + output + " \n")

# ------------------------------------------------------------------------------------------------------------------- #
