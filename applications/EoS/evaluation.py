#! /usr/bin/env python
# -*- coding: utf-8 -*-

########################################################################################################################
# DeepData: data-driven equation of state application
# Authors: ir. A. Giuffre', ir. E. Bunschoten, Dr. ir. A. Cappiello, ir. M. Majer, Dr. ir. M. Pini
# Content: Load pre-trained model and evaluate its accuracy on the test set
# 2022 - TU Delft - All rights reserved
########################################################################################################################

import time
import pickle
from utils.gp import *
import seaborn as sns
from utils.mlp import *
import tensorflow as tf
from matplotlib import cm
from thermodynamics import *

# TODO: IMPLEMENT EVALUATION FOR GPs

# user-defined input
model_type = 'MLP'           # 'MLP' or 'GP'
data_folder = 'MM_250k'     # name of the folder collecting the dataset
data_type = '1phase'        # '1phase', '2phase', or 'full'

# ------------------------------------------------------------------------------------------------------------------- #

# Loading

# dataset and scaler(s)
X_train = pickle.load(open(os.path.join('data', data_folder, 'X_train_' + data_type + '.pkl'), 'rb'))
Y_train = pickle.load(open(os.path.join('data', data_folder, 'Y_train_' + data_type + '.pkl'), 'rb'))
X_dev = pickle.load(open(os.path.join('data', data_folder, 'X_dev_' + data_type + '.pkl'), 'rb'))
Y_dev = pickle.load(open(os.path.join('data', data_folder, 'Y_dev_' + data_type + '.pkl'), 'rb'))
X_test = pickle.load(open(os.path.join('data', data_folder, 'X_test_' + data_type + '.pkl'), 'rb'))
Y_test = pickle.load(open(os.path.join('data', data_folder, 'Y_test_' + data_type + '.pkl'), 'rb'))
X_scaler = pickle.load(open(os.path.join('data', data_folder, 'X_scaler_' + data_type), 'rb'))
Y_scaler = pickle.load(open(os.path.join('data', data_folder, 'Y_scaler_' + data_type), 'rb'))

# primary properties for comparison purposes
rho_sat = pickle.load(open(os.path.join('data', data_folder, 'rho_sat.pkl'), 'rb'))
e_sat = pickle.load(open(os.path.join('data', data_folder, 'e_sat.pkl'), 'rb'))
primary_props = pickle.load(open(os.path.join('data', data_folder, 'primary_props.pkl'), 'rb'))

# pre-trained model
if model_type == 'MLP':
    model = tf.keras.models.load_model(os.path.join('models', 'MLP', data_folder + '_' + data_type))
    model.summary()

elif model_type == 'GP':
    raise NotImplementedError("Evaluation with GP not implemented yet")
#     lh = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=6)
#     model = GPModelWithDerivatives(None, None, lh, 2)
#     state_dict = torch.load(os.path.join('models', 'GP', data_type, 'model_state.pth'))
#     model.load_state_dict(state_dict)
#
#     Y_hat_test_mean, _ = predict_gp(model, lh, torch.Tensor(X_test), X_scaler, Y_scaler)
#     pct_error = (Y_test - Y_hat_test_mean) / Y_test * 100

# ------------------------------------------------------------------------------------------------------------------- #

# Evaluation

if model_type == 'MLP':

    # evaluate accuracy with respect to ground true labels of test set
    Y_hat_test, _ = multi_output_primary_props(model, X_test, X_scaler, Y_scaler)
    error = Y_hat_test - Y_test
    abs_error = np.abs(error)
    pct_error = error / Y_test * 100
    abs_pct_error = abs_error / np.abs(Y_test) * 100

    print('\nMean Absolute Percentage Error [%]')
    print('\nLabels')
    print('s          : %10.4f' % np.mean(abs_pct_error[:, 0]))
    print('ds/de      : %10.4f' % np.mean(abs_pct_error[:, 1]))
    print('ds/drho    : %10.4f' % np.mean(abs_pct_error[:, 2]))
    print('d2s/de.drho: %10.4f' % np.mean(abs_pct_error[:, 3]))
    print('d2s/de2    : %10.4f' % np.mean(abs_pct_error[:, 4]))
    print('d2s/drho2  : %10.4f' % np.mean(abs_pct_error[:, 5]))

    # evaluate accuracy and computational cost in terms of evaluation of primary properties
    start = time.time()
    _, primary_hat = multi_output_primary_props(model, primary_props[:, :2], X_scaler, Y_scaler)
    end = time.time()

    error_primary = primary_hat - primary_props[:, 3:]
    abs_error_primary = np.abs(error_primary)
    pct_error_primary = error_primary / primary_props[:, 3:] * 100
    abs_pct_error_primary = abs_error_primary / np.abs(primary_props[:, 3:]) * 100

    if data_type == '1phase':

        # remove two-phase region from evaluation of accuracy over the entire dataset
        mask = primary_props[:, 2] >= 1
        error_primary[np.logical_not(mask)] = np.nan
        abs_error_primary[np.logical_not(mask)] = np.nan
        pct_error_primary[np.logical_not(mask)] = np.nan
        abs_pct_error_primary[np.logical_not(mask)] = np.nan
        print('\nPrimary properties')
        print('P          : %10.4f' % np.mean(abs_pct_error_primary[mask, 0]))
        print('T          : %10.4f' % np.mean(abs_pct_error_primary[mask, 1]))
        print('h          : %10.4f' % np.mean(abs_pct_error_primary[mask, 2]))
        print('c          : %10.4f' % np.mean(abs_pct_error_primary[mask, 3]))
    else:
        print('\nPrimary properties')
        print('P          : %10.4f' % np.mean(abs_pct_error_primary[:, 0]))
        print('T          : %10.4f' % np.mean(abs_pct_error_primary[:, 1]))
        print('h          : %10.4f' % np.mean(abs_pct_error_primary[:, 2]))
        print('c          : %10.4f' % np.mean(abs_pct_error_primary[:, 3]))

    print('\nComputational cost [s]')
    print('Global     : %10.6f' % (end - start))
    print('Single call: %10.6f' % ((end - start) / primary_props.shape[0]))

elif model_type == 'GP':
    raise NotImplementedError("Evaluation with GP not implemented yet")

# ------------------------------------------------------------------------------------------------------------------- #

# Plotting

plot_dir = os.path.join('plots', model_type, data_folder + '_' + data_type)
rho_matrix = primary_props[:, 0].reshape(int(np.sqrt(primary_props.shape[0])), int(np.sqrt(primary_props.shape[0])))
e_matrix = primary_props[:, 1].reshape(int(np.sqrt(primary_props.shape[0])), int(np.sqrt(primary_props.shape[0])))

# mean absolute percentage error with respect to primary properties
fig1, ax1 = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
cs00 = ax1[0, 0].contourf(rho_matrix, e_matrix / 1e3,
                          abs_pct_error_primary[:, 0].reshape(int(np.sqrt(primary_props.shape[0])),
                                                              int(np.sqrt(primary_props.shape[0]))), cmap=cm.viridis)
cbar00 = fig1.colorbar(cs00, shrink=1.0, format='%.2f', ax=ax1[0, 0])
ax1[0, 0].plot(rho_sat, e_sat / 1e3, 'black')
ax1[0, 0].set_xlim(np.min(rho_matrix), np.max(rho_matrix))
ax1[0, 0].set_ylim(np.min(e_matrix / 1e3), np.max(e_matrix / 1e3))
ax1[0, 0].set_title('P abs error [%]')
ax1[0, 0].set_xlabel('rho [kg/m3]')
ax1[0, 0].set_ylabel('e [J/kg]')

cs01 = ax1[0, 1].contourf(rho_matrix, e_matrix / 1e3,
                          abs_pct_error_primary[:, 1].reshape(int(np.sqrt(primary_props.shape[0])),
                                                              int(np.sqrt(primary_props.shape[0]))), cmap=cm.viridis)
cbar01 = fig1.colorbar(cs01, shrink=1.0, format='%.2f', ax=ax1[0, 1])
ax1[0, 1].plot(rho_sat, e_sat / 1e3, 'black')
ax1[0, 1].set_xlim(np.min(rho_matrix), np.max(rho_matrix))
ax1[0, 1].set_ylim(np.min(e_matrix / 1e3), np.max(e_matrix / 1e3))
ax1[0, 1].set_title('T abs error [%]')
ax1[0, 1].set_xlabel('rho [kg/m3]')
ax1[0, 1].set_ylabel('e [J/kg]')

cs10 = ax1[1, 0].contourf(rho_matrix, e_matrix / 1e3,
                          abs_pct_error_primary[:, 2].reshape(int(np.sqrt(primary_props.shape[0])),
                                                              int(np.sqrt(primary_props.shape[0]))), cmap=cm.viridis)
cbar10 = fig1.colorbar(cs10, shrink=1.0, format='%.2f', ax=ax1[1, 0])
ax1[1, 0].plot(rho_sat, e_sat / 1e3, 'black')
ax1[1, 0].set_xlim(np.min(rho_matrix), np.max(rho_matrix))
ax1[1, 0].set_ylim(np.min(e_matrix / 1e3), np.max(e_matrix / 1e3))
ax1[1, 0].set_title('h abs error [%]')
ax1[1, 0].set_xlabel('rho [kg/m3]')
ax1[1, 0].set_ylabel('e [J/kg]')

cs11 = ax1[1, 1].contourf(rho_matrix, e_matrix / 1e3,
                          abs_pct_error_primary[:, 3].reshape(int(np.sqrt(primary_props.shape[0])),
                                                              int(np.sqrt(primary_props.shape[0]))), cmap=cm.viridis)
cbar11 = fig1.colorbar(cs11, shrink=1.0, format='%.2f', ax=ax1[1, 1])
ax1[1, 1].plot(rho_sat, e_sat / 1e3, 'black')
ax1[1, 1].set_xlim(np.min(rho_matrix), np.max(rho_matrix))
ax1[1, 1].set_ylim(np.min(e_matrix / 1e3), np.max(e_matrix / 1e3))
ax1[1, 1].set_title('c abs error [%]')
ax1[1, 1].set_xlabel('rho [kg/m3]')
ax1[1, 1].set_ylabel('e [J/kg]')

fig1.savefig(os.path.join(plot_dir, 'mape_primary.jpeg'), dpi=400)
plt.close(fig1)

# mean percentage error with respect to ground true labels
sns.set_context("paper")
n_plots = 6
fig2, axs2 = plt.subplots(2, 3, figsize=(10, 10))
new_colors = [plt.get_cmap(cm.viridis)(1. * i / n_plots) for i in range(n_plots)]

ax00 = sns.histplot(pct_error[:, 0], ax=axs2[0, 0], color=new_colors[0])
ax00.set_xlabel('s error [%]')
ax00.set_yticklabels([])
ax00.set_ylabel('')
ax01 = sns.histplot(pct_error[:, 1], ax=axs2[0, 1], color=new_colors[1])
ax01.set_xlabel('ds/de error [%]')
ax01.set_yticklabels([])
ax01.set_ylabel('')
ax02 = sns.histplot(pct_error[:, 2], ax=axs2[0, 2], color=new_colors[2])
ax02.set_xlabel('ds/drho error [%]')
ax02.set_yticklabels([])
ax02.set_ylabel('')
ax10 = sns.histplot(pct_error[:, 3], ax=axs2[1, 0], color=new_colors[3])
ax10.set_xlabel('d2s/de.drho error [%]')
ax10.set_yticklabels([])
ax10.set_ylabel('')
ax11 = sns.histplot(pct_error[:, 4], ax=axs2[1, 1], color=new_colors[4])
ax11.set_xlabel('d2s/de2 [%]')
ax11.set_yticklabels([])
ax11.set_ylabel('')
ax12 = sns.histplot(pct_error[:, 5], ax=axs2[1, 2], color=new_colors[5])
ax12.set_xlabel('d2s/drho2 [%]')
ax12.set_yticklabels([])
ax12.set_ylabel('')

fig2.tight_layout()
fig2.savefig(os.path.join(plot_dir, 'mpe_labels.jpeg'), dpi=400)
plt.close(fig2)

# mean percentage error with respect to primary properties
n_plots = 4
fig3, axs3 = plt.subplots(2, 2, figsize=(8, 10))
new_colors = [plt.get_cmap(cm.viridis)(1. * i / n_plots) for i in range(n_plots)]

ax00 = sns.histplot(pct_error_primary[:, 0], ax=axs3[0, 0], color=new_colors[0])
ax00.set_xlabel('P error [%]')
ax00.set_yticklabels([])
ax00.set_ylabel('')
ax01 = sns.histplot(pct_error_primary[:, 1], ax=axs3[0, 1], color=new_colors[1])
ax01.set_xlabel('T error [%]')
ax01.set_yticklabels([])
ax01.set_ylabel('')
ax10 = sns.histplot(pct_error_primary[:, 2], ax=axs3[1, 0], color=new_colors[2])
ax10.set_xlabel('h error [%]')
ax10.set_yticklabels([])
ax10.set_ylabel('')
ax11 = sns.histplot(pct_error_primary[:, 3], ax=axs3[1, 1], color=new_colors[3])
ax11.set_xlabel('c error [%]')
ax11.set_yticklabels([])
ax11.set_ylabel('')

fig3.tight_layout()
fig3.savefig(os.path.join(plot_dir, 'mpe_primary.jpeg'), dpi=400)
plt.close(fig3)
