#! /usr/bin/env python
# -*- coding: utf-8 -*-

########################################################################################################################
# A Consistent Data-Driven Thermodynamic Model for NICFD
# Authors: ir. A. Giuffre', ir. E. B, Dr. ir. A. Cappiello, ir. M. Majer, Dr. ir. M. Pini
# Content: Load pre-trained model and evaluate its accuracy on the test set
# 2021 - TU Delft - All rights reserved
########################################################################################################################

import pickle
import numpy as np
from gp_model import *
from ann_model import *
import tensorflow as tf
from matplotlib import cm

# TODO: IMPLEMENT dx_norm_dx FOR MIN-MAX NORMALIZATION AND FOR STANDARDIZATION

# user-defined input
model_type = 'MLP'            # 'MLP' or 'GP'
data_type = 'full'           # '1phase', '2phase', or 'full'
fluid = 'MM'
library = 'REFPROP'

# load data and scaler
X_train = pickle.load(open(os.path.join('data', model_type, 'X_train_' + data_type + '.pkl'), 'rb'))
Y_train = pickle.load(open(os.path.join('data', model_type, 'Y_train_' + data_type + '.pkl'), 'rb'))
X_dev = pickle.load(open(os.path.join('data', model_type, 'X_dev_' + data_type + '.pkl'), 'rb'))
Y_dev = pickle.load(open(os.path.join('data', model_type, 'Y_dev_' + data_type + '.pkl'), 'rb'))
X_test = pickle.load(open(os.path.join('data', model_type, 'X_test_' + data_type + '.pkl'), 'rb'))
Y_test = pickle.load(open(os.path.join('data', model_type, 'Y_test_' + data_type + '.pkl'), 'rb'))
X_scaler = pickle.load(open(os.path.join('data', model_type, 'X_scaler_' + data_type), 'rb'))
Y_scaler = pickle.load(open(os.path.join('data', model_type, 'Y_scaler_' + data_type), 'rb'))

# load primary properties for comparison purposes
rho_sat = pickle.load(open(os.path.join('data', model_type, 'rho_sat.pkl'), 'rb'))
e_sat = pickle.load(open(os.path.join('data', model_type, 'e_sat.pkl'), 'rb'))
primary_props = pickle.load(open(os.path.join('data', model_type, 'primary_props.pkl'), 'rb'))


def eval_primary_props(model, X, X_scaler, Y_scaler):
    """

    :param model:
    :param X:
    :param X_scaler:
    :param Y_scaler:
    :return:
    """
    rho = X[:, 0]
    e = X[:, 1]

    X_norm = X_scaler.transform(X)
    Y_norm = model.predict(X_norm)
    Y = Y_scaler.inverse_transform(Y_norm)
    Y[:, 2] = - np.exp(Y[:, 2])
    Y[:, 5] = np.exp(Y[:, 5])

    s = Y[:, 0]
    ds_de = Y[:, 1]
    ds_drho = Y[:, 2]
    d2s_dedrho = Y[:, 3]
    d2s_de2 = Y[:, 4]
    d2s_drho2 = Y[:, 5]

    BlueTerm = (ds_drho * (2 - rho * (ds_de ** (-1)) * d2s_dedrho) + rho * d2s_drho2)
    GreenTerm = (- (ds_de ** (-1)) * d2s_de2 * ds_drho + d2s_dedrho)
    c = np.sqrt(- rho * (ds_de ** (-1)) * (BlueTerm - rho * GreenTerm * (ds_drho / ds_de)))
    T = 1 / ds_de
    P = - (rho ** 2) * T * ds_drho
    h = e + P / rho

    primary_props = np.vstack((P, T, h, c)).T

    return Y, primary_props


# load pre-trained model
if model_type == 'MLP':
    model = tf.keras.models.load_model(os.path.join('models', 'MLP', data_type))
    model.summary()

    Y_hat_test, _ = eval_primary_props(model, X_test, X_scaler, Y_scaler)
    error = np.abs(Y_hat_test - Y_test)
    pct_error = error / np.abs(Y_test) * 100

    print('\nMean Absolute Percentage Error [%]')
    print('s          : %10.4f' % np.mean(pct_error[:, 0]))
    print('ds/de      : %10.4f' % np.mean(pct_error[:, 1]))
    print('ds/drho    : %10.4f' % np.mean(pct_error[:, 2]))
    print('d2s/de.drho: %10.4f' % np.mean(pct_error[:, 3]))
    print('d2s/de2    : %10.4f' % np.mean(pct_error[:, 4]))
    print('d2s/drho2  : %10.4f' % np.mean(pct_error[:, 5]))

elif model_type == 'GP':
    # Convert train set in GPyTorch format
    X_train_norm = torch.Tensor(X_scaler.transform(X_train))
    Y_train = torch.Tensor(Y_train)
    Y_test = torch.Tensor(Y_test)

    lh = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=3)
    model = GPModelWithDerivatives(X_train_norm, Y_train, lh, 2)
    state_dict = torch.load(os.path.join('models', 'GP', data_type, 'model_state.pth'))
    model.load_state_dict(state_dict)

    Y_hat_test_mean, Y_hat_test_var = predict_gp(model, lh, X_scaler, X_test)
    pct_error = torch.abs(Y_test - Y_hat_test_mean) / Y_test * 100


_, primary_hat = eval_primary_props(model, primary_props[:, :2], X_scaler, Y_scaler)
error_primary = np.abs(primary_hat - primary_props[:, 2:])
pct_error_primary = error_primary / np.abs(primary_props[:, 2:]) * 100
print('P          : %10.4f' % np.mean(pct_error_primary[:, 0]))
print('T          : %10.4f' % np.mean(pct_error_primary[:, 1]))
print('h          : %10.4f' % np.mean(pct_error_primary[:, 2]))
print('c          : %10.4f' % np.mean(pct_error_primary[:, 3]))

rho_matrix = primary_props[:, 0].reshape(500, 500)
e_matrix = primary_props[:, 1].reshape(500, 500)

fig, ax = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
cs00 = ax[0, 0].contourf(rho_matrix, e_matrix / 1e3, pct_error_primary[:, 0].reshape(500, 500), cmap=cm.viridis)
cbar00 = fig.colorbar(cs00, shrink=1.0, format='%.2f', ax=ax[0, 0])
ax[0, 0].plot(rho_sat, e_sat / 1e3, 'black')
ax[0, 0].set_xlim(np.min(rho_matrix), np.max(rho_matrix))
ax[0, 0].set_ylim(np.min(e_matrix / 1e3), np.max(e_matrix / 1e3))
ax[0, 0].set_title('P error [%]')
ax[0, 0].set_xlabel('rho [kg/m3]')
ax[0, 0].set_ylabel('e [J/kg]')

cs01 = ax[0, 1].contourf(rho_matrix, e_matrix / 1e3, pct_error_primary[:, 1].reshape(500, 500), cmap=cm.viridis)
cbar01 = fig.colorbar(cs01, shrink=1.0, format='%.2f', ax=ax[0, 1])
ax[0, 1].plot(rho_sat, e_sat / 1e3, 'black')
ax[0, 1].set_xlim(np.min(rho_matrix), np.max(rho_matrix))
ax[0, 1].set_ylim(np.min(e_matrix / 1e3), np.max(e_matrix / 1e3))
ax[0, 1].set_title('T error [%]')
ax[0, 1].set_xlabel('rho [kg/m3]')
ax[0, 1].set_ylabel('e [J/kg]')

cs10 = ax[1, 0].contourf(rho_matrix, e_matrix / 1e3, pct_error_primary[:, 2].reshape(500, 500), cmap=cm.viridis)
cbar10 = fig.colorbar(cs10, shrink=1.0, format='%.2f', ax=ax[1, 0])
ax[1, 0].plot(rho_sat, e_sat / 1e3, 'black')
ax[1, 0].set_xlim(np.min(rho_matrix), np.max(rho_matrix))
ax[1, 0].set_ylim(np.min(e_matrix / 1e3), np.max(e_matrix / 1e3))
ax[1, 0].set_title('h error [%]')
ax[1, 0].set_xlabel('rho [kg/m3]')
ax[1, 0].set_ylabel('e [J/kg]')

cs11 = ax[1, 1].contourf(rho_matrix, e_matrix / 1e3, pct_error_primary[:, 3].reshape(500, 500), cmap=cm.viridis)
cbar11 = fig.colorbar(cs11, shrink=1.0, format='%.2f', ax=ax[1, 1])
ax[1, 1].plot(rho_sat, e_sat / 1e3, 'black')
ax[1, 1].set_xlim(np.min(rho_matrix), np.max(rho_matrix))
ax[1, 1].set_ylim(np.min(e_matrix / 1e3), np.max(e_matrix / 1e3))
ax[1, 1].set_title('c error [%]')
ax[1, 1].set_xlabel('rho [kg/m3]')
ax[1, 1].set_ylabel('e [J/kg]')

fig.savefig('plots/pct_error.jpeg', dpi=400)
plt.close(fig)
