#! /usr/bin/env python
# -*- coding: utf-8 -*-

########################################################################################################################
# Deep: data-driven equation of state application
# Authors: ir. A. Giuffre', ir. E. Bunschoten, Dr. ir. A. Cappiello, ir. M. Majer, Dr. ir. M. Pini
# Content: Optimize the set of hyper-parameters for the MLP architecture
# 2022 - TU Delft - All rights reserved
########################################################################################################################

from plot import *
from utils.mlp import *
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from smt.surrogate_models import KRG
from smt.applications.mixed_integer import MixedIntegerSurrogateModel, FLOAT, INT, ENUM


# user-defined input
dev_size = 5                                 # percentage of total dataset used for the dev set
test_size = 5                                # percentage of total dataset used for the test set
model_type = 'MLP'                           # 'MLP' or 'GP'
data_folder = 'MLP_MM_250k_rho_above_10'     # name of the folder collecting the dataset
data_type = '1phase'                         # '1phase', '2phase', or 'full'
L = 2                                        # number of hidden layers
n_epochs = 250                               # number of epochs used for training
batch_norm = 0                               # batch normalization strategy
regularization = 0                           # regularization strategy
w_init = 'he_uniform'                        # weight initialization strategy
lr_decay = 1.0                               # learning rate decay
decay_steps = 100000                         # the learning rate drops significantly every decay_steps
staircase = False                            # if True, the learning rate drops following a staircase
max_norm = 4                                 # value of max-norm weight constraint used in each layer (inverted dropout)
dropout_prob = []                            # list of dropout probabilities used in each layer, except for the output
samples = 100                                # number of samples used for the design of experiments
flag_doe = False                             # if True, run design of experiments; otherwise, load results

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

model_dir = os.path.join('models', 'MLP', data_folder + '_' + data_type)
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

if flag_doe:
    opt = HyperSearch(X_train_norm, X_dev_norm, Y_train_norm, Y_dev_norm, L, n_epochs, batch_norm=batch_norm,
                      regularization=regularization, w_init=w_init, lr_decay=lr_decay, decay_steps=decay_steps,
                      staircase=staircase, max_norm=max_norm, dropout_prob=dropout_prob)
    opt.doe(samples, model_dir)

doe = pickle.load(open(os.path.join(model_dir, 'doe.pkl'), 'rb'))

# remove from the dataset the MLP architectures featuring low accuracy
mask = np.logical_and(doe[:, -2] != 0, doe[:, -2] < 1e-5)

# define and train the surrogate model
xtypes = [INT, INT, FLOAT, (ENUM, 7)]
bounds = [[2, 100], [4, 10], [1.0, 5.0], [0, 1, 2, 3, 4, 5, 6]]

sm_accuracy = MixedIntegerSurrogateModel(xtypes=xtypes, xlimits=bounds, surrogate=KRG(theta0=[1e-2]))
sm_accuracy.set_training_values(doe[mask, :-2], doe[mask, -2])
sm_accuracy.train()
accuracy_hat = sm_accuracy.predict_values(doe[mask, :-2])

sm_comp_cost = MixedIntegerSurrogateModel(xtypes=xtypes, xlimits=bounds, surrogate=KRG(theta0=[1e-2]))
sm_comp_cost.set_training_values(doe[mask, :-2], doe[mask, -1])
sm_comp_cost.train()
comp_cost_hat = sm_comp_cost.predict_values(doe[mask, :-2])

# plot the space of objectives: accuracy, computational cost, and predictions of the surrogate model
plot = Plot(os.path.join('plots', 'MLP', data_folder + '_' + data_type))
plot.plot_objectives(doe[mask, -2], doe[mask, -1], accuracy_hat, comp_cost_hat)


# ------------------------------------------------------------------------------------------------------------------- #
