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
import seaborn as sns
import torch

from utils.gp import *
import tensorflow as tf
from matplotlib import cm
from pre_processing import *

# TODO: IMPLEMENT EVALUATION FOR GPs

# user-defined input
model_type = 'GP'                         # 'MLP' or 'GP'
labels_type = 'obj'     # 'obj' or 'con'
output_dir = "test"      # name of the output directory, where results will be saved

# ------------------------------------------------------------------------------------------------------------------- #

# Loading

plot_dir = os.path.join('plots', model_type, output_dir, labels_type)
model_dir = os.path.join('models', model_type, output_dir, labels_type)
data_dir = os.path.join('data', model_type, output_dir, labels_type)

# dataset and scaler(s)
features = pickle.load(open(os.path.join(data_dir, 'features'), 'rb'))
labels = pickle.load(open(os.path.join(data_dir, 'labels'), 'rb'))
X_train_norm = pickle.load(open(os.path.join(data_dir, 'X_train_norm'), 'rb'))
Y_train_norm = pickle.load(open(os.path.join(data_dir, 'Y_train_norm'), 'rb'))
X_dev_norm = pickle.load(open(os.path.join(data_dir, 'X_dev_norm'), 'rb'))
Y_dev_norm = pickle.load(open(os.path.join(data_dir, 'Y_dev_norm'), 'rb'))
X_test_norm = pickle.load(open(os.path.join(data_dir, 'X_test_norm'), 'rb'))
Y_test_norm = pickle.load(open(os.path.join(data_dir, 'Y_test_norm'), 'rb'))
X_scaler = pickle.load(open(os.path.join(data_dir, 'X_scaler'), 'rb'))
Y_scaler = pickle.load(open(os.path.join(data_dir, 'Y_scaler'), 'rb'))

print('\nTrain set size: %d' % X_train_norm.shape[0])
print('Dev set size: %d' % X_dev_norm.shape[0])
print('Test set size: %d' % X_test_norm.shape[0])

# pre-trained model
if model_type == 'MLP':
    model = tf.keras.models.load_model(model_dir)
    model.summary()

elif model_type == 'GP':

    # user-defined input
    n_inducing = 500        # number of inducing points used to train the GPs
    kernel = 'matern'          # kernel used to define the GPs during training

    # initialize objects used to define models
    inducing_points = torch.Tensor(X_train_norm[:n_inducing, :])
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    models = []

    # load pre-trained models
    for ii, label in enumerate(labels):
        models.append(StochasticVariationalGP(inducing_points, X_train_norm.shape[1], kernel))
        models[ii].load_state_dict(torch.load(os.path.join(model_dir, 'model_state_' + label.split(' ')[0] + '.pth')))

# ------------------------------------------------------------------------------------------------------------------- #

# Evaluation

if X_scaler is not None:
    X_test = X_scaler.inverse_transform(X_test_norm)
else:
    X_test = X_test_norm

if model_type == 'MLP':

    # evaluate accuracy with respect to ground true labels of test set
    res = model.predict(X_test_norm)

    if Y_scaler is not None:
        Y_hat_encoded = Y_scaler.inverse_transform(res)
        Y_hat = labels_decoding(labels, Y_hat_encoded)
        Y_test = Y_scaler.inverse_transform(Y_test_norm)
    else:
        Y_hat = res
        Y_test = Y_test_norm

    error = Y_hat - Y_test
    abs_error = np.abs(error)
    pct_error = error / Y_test * 100
    abs_pct_error = abs_error / np.abs(Y_test) * 100

    pickle.dump(error, open(os.path.join(model_dir, 'error'), 'wb'))
    pickle.dump(abs_error, open(os.path.join(model_dir, 'abs_error'), 'wb'))
    pickle.dump(pct_error, open(os.path.join(model_dir, 'pct_error'), 'wb'))
    pickle.dump(abs_pct_error, open(os.path.join(model_dir, 'abs_pct_error'), 'wb'))

    # print errors
    print('\n***********************************************')
    print('ERRORS COMPUTED OVER THE TEST SET')
    for ii, label in enumerate(labels):
        print('\nMean absolute error - ' + label + '         : %10.4f' % np.mean(abs_error[:, ii]))
        print('Max absolute error - ' + label + '            : %10.4f' % np.max(abs_error[:, ii]))
        print('Mean absolute percentage error - ' + label + ': %10.4f' % np.mean(abs_pct_error[:, ii]))
        print('Max absolute percentage error - ' + label + ' : %10.4f' % np.max(abs_pct_error[:, ii]))

    print('\nMean absolute error - overall: %10.4f' % np.mean(abs_error))
    print('Max absolute error - overall: %10.4f' % np.max(abs_error))
    print('Mean absolute percentage error - overall: %10.4f' % np.mean(abs_pct_error))
    print('Max absolute percentage error - overall: %10.4f' % np.max(abs_pct_error))

    while True:
        idx = int(input("\nSpecify indexes of input feature and label you want to print. "
                        "Enter -1 to exit: "))
        if idx == -1:
            break
        elif idx > (X_test.shape[0] - 1):
            print("\nThe index must be smaller than %d!" % (X_test.shape[0] - 1))
        else:
            print('\nInput features: ', features)
            print('X             : ', X_test[idx, :])
            print('\nLabels        : ', labels)
            print('Y predicted : ', Y_hat[idx, :])
            print('Y true label: ', Y_test[idx, :])

elif model_type == 'GP':

    # initialize arrays where results are stored
    Y_hat_mean = np.zeros((X_test_norm.shape[0], len(models)))
    Y_hat_var = np.zeros((X_test_norm.shape[0], len(models)))

    # evaluate accuracy with respect to ground true labels of test set
    for ii, model in enumerate(models):
        model.eval()
        likelihood.eval()

        # Make predictions
        with torch.no_grad(), gpytorch.settings.fast_computations(log_prob=False, covar_root_decomposition=False):
            predictions = likelihood(model(torch.Tensor(X_test_norm)))
            res_mean = predictions.mean
            res_var = predictions.variance
            Y_hat_var[:, ii] = res_var      # TODO: understand how to convert var in case of normalization

            if Y_scaler is not None:
                Y_hat_mean_encoded = Y_scaler.inverse_transform(res_mean)
                Y_hat_mean[:, ii] = labels_decoding(labels, Y_hat_mean_encoded)
                Y_test = Y_scaler.inverse_transform(Y_test_norm)
            else:
                Y_hat_mean[:, ii] = res_mean
                Y_test = Y_test_norm

    error = Y_hat_mean - Y_test
    abs_error = np.abs(error)
    pct_error = error / Y_test * 100
    abs_pct_error = abs_error / np.abs(Y_test) * 100

    pickle.dump(error, open(os.path.join(model_dir, 'error'), 'wb'))
    pickle.dump(abs_error, open(os.path.join(model_dir, 'abs_error'), 'wb'))
    pickle.dump(pct_error, open(os.path.join(model_dir, 'pct_error'), 'wb'))
    pickle.dump(abs_pct_error, open(os.path.join(model_dir, 'abs_pct_error'), 'wb'))

    # print errors
    print('\n***********************************************')
    print('ERRORS COMPUTED OVER THE TEST SET')
    for ii, label in enumerate(labels):
        print('\nMean absolute error - ' + label + '         : %10.4f' % np.mean(abs_error[:, ii]))
        print('Max absolute error - ' + label + '            : %10.4f' % np.max(abs_error[:, ii]))
        print('Mean absolute percentage error - ' + label + ': %10.4f' % np.mean(abs_pct_error[:, ii]))
        print('Max absolute percentage error - ' + label + ' : %10.4f' % np.max(abs_pct_error[:, ii]))

    print('\nMean absolute error - overall: %10.4f' % np.mean(abs_error))
    print('Max absolute error - overall: %10.4f' % np.max(abs_error))
    print('Mean absolute percentage error - overall: %10.4f' % np.mean(abs_pct_error))
    print('Max absolute percentage error - overall: %10.4f' % np.max(abs_pct_error))

    while True:
        idx = int(input("\nSpecify indexes of input feature and label you want to print. "
                        "Enter -1 to exit: "))
        if idx == -1:
            break
        elif idx > (X_test.shape[0] - 1):
            print("\nThe index must be smaller than %d!" % (X_test.shape[0] - 1))
        else:
            print('\nInput features: ', features)
            print('X             : ', X_test[idx, :])
            print('\nLabels        : ', labels)
            print('Y predicted : ', Y_hat_mean[idx, :])
            print('Y true label: ', Y_test[idx, :])

# ------------------------------------------------------------------------------------------------------------------- #

# Plotting

# plot_dir = os.path.join('plots', model_type, data_folder + '_' + data_type)
# rho_matrix = primary_props[:, 0].reshape(int(np.sqrt(primary_props.shape[0])), int(np.sqrt(primary_props.shape[0])))
# e_matrix = primary_props[:, 1].reshape(int(np.sqrt(primary_props.shape[0])), int(np.sqrt(primary_props.shape[0])))
#
# # mean absolute percentage error with respect to primary properties
# fig1, ax1 = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
# cs00 = ax1[0, 0].contourf(rho_matrix, e_matrix / 1e3,
#                           abs_pct_error_primary[:, 0].reshape(int(np.sqrt(primary_props.shape[0])),
#                                                               int(np.sqrt(primary_props.shape[0]))), cmap=cm.viridis)
# cbar00 = fig1.colorbar(cs00, shrink=1.0, format='%.2f', ax=ax1[0, 0])
# ax1[0, 0].plot(rho_sat, e_sat / 1e3, 'black')
# ax1[0, 0].set_xlim(np.min(rho_matrix), np.max(rho_matrix))
# ax1[0, 0].set_ylim(np.min(e_matrix / 1e3), np.max(e_matrix / 1e3))
# ax1[0, 0].set_title('P abs error [%]')
# ax1[0, 0].set_xlabel('rho [kg/m3]')
# ax1[0, 0].set_ylabel('e [J/kg]')
#
# cs01 = ax1[0, 1].contourf(rho_matrix, e_matrix / 1e3,
#                           abs_pct_error_primary[:, 1].reshape(int(np.sqrt(primary_props.shape[0])),
#                                                               int(np.sqrt(primary_props.shape[0]))), cmap=cm.viridis)
# cbar01 = fig1.colorbar(cs01, shrink=1.0, format='%.2f', ax=ax1[0, 1])
# ax1[0, 1].plot(rho_sat, e_sat / 1e3, 'black')
# ax1[0, 1].set_xlim(np.min(rho_matrix), np.max(rho_matrix))
# ax1[0, 1].set_ylim(np.min(e_matrix / 1e3), np.max(e_matrix / 1e3))
# ax1[0, 1].set_title('T abs error [%]')
# ax1[0, 1].set_xlabel('rho [kg/m3]')
# ax1[0, 1].set_ylabel('e [J/kg]')
#
# cs10 = ax1[1, 0].contourf(rho_matrix, e_matrix / 1e3,
#                           abs_pct_error_primary[:, 2].reshape(int(np.sqrt(primary_props.shape[0])),
#                                                               int(np.sqrt(primary_props.shape[0]))), cmap=cm.viridis)
# cbar10 = fig1.colorbar(cs10, shrink=1.0, format='%.2f', ax=ax1[1, 0])
# ax1[1, 0].plot(rho_sat, e_sat / 1e3, 'black')
# ax1[1, 0].set_xlim(np.min(rho_matrix), np.max(rho_matrix))
# ax1[1, 0].set_ylim(np.min(e_matrix / 1e3), np.max(e_matrix / 1e3))
# ax1[1, 0].set_title('h abs error [%]')
# ax1[1, 0].set_xlabel('rho [kg/m3]')
# ax1[1, 0].set_ylabel('e [J/kg]')
#
# cs11 = ax1[1, 1].contourf(rho_matrix, e_matrix / 1e3,
#                           abs_pct_error_primary[:, 3].reshape(int(np.sqrt(primary_props.shape[0])),
#                                                               int(np.sqrt(primary_props.shape[0]))), cmap=cm.viridis)
# cbar11 = fig1.colorbar(cs11, shrink=1.0, format='%.2f', ax=ax1[1, 1])
# ax1[1, 1].plot(rho_sat, e_sat / 1e3, 'black')
# ax1[1, 1].set_xlim(np.min(rho_matrix), np.max(rho_matrix))
# ax1[1, 1].set_ylim(np.min(e_matrix / 1e3), np.max(e_matrix / 1e3))
# ax1[1, 1].set_title('c abs error [%]')
# ax1[1, 1].set_xlabel('rho [kg/m3]')
# ax1[1, 1].set_ylabel('e [J/kg]')
#
# fig1.savefig(os.path.join(plot_dir, 'mape_primary.jpeg'), dpi=400)
# plt.close(fig1)
#
# # mean percentage error with respect to ground true labels
# sns.set_context("paper")
# n_plots = 6
# fig2, axs2 = plt.subplots(2, 3, figsize=(10, 10))
# new_colors = [plt.get_cmap(cm.viridis)(1. * i / n_plots) for i in range(n_plots)]
#
# ax00 = sns.histplot(pct_error[:, 0], ax=axs2[0, 0], color=new_colors[0])
# ax00.set_xlabel('s error [%]')
# ax00.set_yticklabels([])
# ax00.set_ylabel('')
# ax01 = sns.histplot(pct_error[:, 1], ax=axs2[0, 1], color=new_colors[1])
# ax01.set_xlabel('ds/de error [%]')
# ax01.set_yticklabels([])
# ax01.set_ylabel('')
# ax02 = sns.histplot(pct_error[:, 2], ax=axs2[0, 2], color=new_colors[2])
# ax02.set_xlabel('ds/drho error [%]')
# ax02.set_yticklabels([])
# ax02.set_ylabel('')
# ax10 = sns.histplot(pct_error[:, 3], ax=axs2[1, 0], color=new_colors[3])
# ax10.set_xlabel('d2s/de.drho error [%]')
# ax10.set_yticklabels([])
# ax10.set_ylabel('')
# ax11 = sns.histplot(pct_error[:, 4], ax=axs2[1, 1], color=new_colors[4])
# ax11.set_xlabel('d2s/de2 [%]')
# ax11.set_yticklabels([])
# ax11.set_ylabel('')
# ax12 = sns.histplot(pct_error[:, 5], ax=axs2[1, 2], color=new_colors[5])
# ax12.set_xlabel('d2s/drho2 [%]')
# ax12.set_yticklabels([])
# ax12.set_ylabel('')
#
# fig2.tight_layout()
# fig2.savefig(os.path.join(plot_dir, 'mpe_labels.jpeg'), dpi=400)
# plt.close(fig2)
#
# # mean percentage error with respect to primary properties
# n_plots = 4
# fig3, axs3 = plt.subplots(2, 2, figsize=(8, 10))
# new_colors = [plt.get_cmap(cm.viridis)(1. * i / n_plots) for i in range(n_plots)]
#
# ax00 = sns.histplot(pct_error_primary[:, 0], ax=axs3[0, 0], color=new_colors[0])
# ax00.set_xlabel('P error [%]')
# ax00.set_yticklabels([])
# ax00.set_ylabel('')
# ax01 = sns.histplot(pct_error_primary[:, 1], ax=axs3[0, 1], color=new_colors[1])
# ax01.set_xlabel('T error [%]')
# ax01.set_yticklabels([])
# ax01.set_ylabel('')
# ax10 = sns.histplot(pct_error_primary[:, 2], ax=axs3[1, 0], color=new_colors[2])
# ax10.set_xlabel('h error [%]')
# ax10.set_yticklabels([])
# ax10.set_ylabel('')
# ax11 = sns.histplot(pct_error_primary[:, 3], ax=axs3[1, 1], color=new_colors[3])
# ax11.set_xlabel('c error [%]')
# ax11.set_yticklabels([])
# ax11.set_ylabel('')
#
# fig3.tight_layout()
# fig3.savefig(os.path.join(plot_dir, 'mpe_primary.jpeg'), dpi=400)
# plt.close(fig3)
