#! /usr/bin/env python
# -*- coding: utf-8 -*-

########################################################################################################################
# DeepData: data-driven regression based on artificial intelligence
# Author: ir. A. Giuffre'
# Content: Classes to create, train, and optimize a Gaussian Process
# 2022 - TU Delft - All rights reserved
########################################################################################################################

import torch
import gpytorch
import numpy as np
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

# TODO: FIX predict_gp, ADD FUNCTION TO SAVE TRAINING HISTORY


class StochasticVariationalGP(ApproximateGP):
    """
    Create a Stochastic Variational Gaussian Process with custom architecture
    """
    def __init__(self, inducing_points, n_features, kernel_type, nu_matern=2.5, custom_kernel=None):
        """
        :param n_features: number of input features
        :param inducing_points: first guess of inducing points --> the only relevant information is their number
        :param kernel_type: type of kernel defining the prior covariance of the Gaussian Process
        :param nu_matern: smoothness parameter for the Matern kernel: 0.5, 1.5, or 2.5
        :param custom_kernel: custom kernel object prescribed by the user
        """
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution,
                                                   learn_inducing_locations=True)

        super(StochasticVariationalGP, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()

        if kernel_type == 'rbf':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=n_features,
                                                                                        learn_length_scales=True))
        elif kernel_type == 'matern':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(ard_num_dims=n_features,
                                                                                           learn_length_scales=True,
                                                                                           nu=nu_matern))
        elif kernel_type == 'custom':
            self.covar_module = custom_kernel
        else:
            raise NotImplementedError("Currently, the available options for the parameter kernel_type are: "
                                      "'rbf', 'matern', or 'custom'")

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_variational_gp(model, likelihood, train_loader, n_train, epochs, alpha, eps=2):
    """
    Train a prescribed stochastic variational Gaussian Process, given the training set
    :param model: gaussian process model
    :param likelihood:
    :param train_loader: training set (features + labels), already structured in mini-batches
    :param n_train: number of training examples
    :param epochs: number of epochs used for training
    :param alpha: learning rate
    :param eps: prescribed tolerance over the loss function, used to determine convergence and stop training
    """
    # Set training mode
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=10 ** (- alpha))  # Includes GaussianLikelihood parameters

    # Loss function for variational GPs
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=n_train)

    # Iterate over the prescribed number of epochs
    for i in range(epochs):

        # Within each iteration, iterate over the prescribed mini-batches of data
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            loss.backward()
            optimizer.step()

        # Check the convergence criterion at the end of each epoch
        print("Iter %d/%d - Loss: %.3f" % (i + 1, epochs, loss.item()))
        if loss < -eps:
            break


def predict_gp(model, likelihood, X, X_scaler, Y_scaler):
    """
    :param model:
    :param likelihood:
    :param X:
    :param X_scaler:
    :param Y_scaler:
    :return:
    """
    # Set into eval mode
    model.eval()
    likelihood.eval()

    # Normalize X and convert it in GPyTorch format
    X_norm = torch.Tensor(X_scaler.transform(X))

    # Make predictions
    with torch.no_grad(), gpytorch.settings.fast_computations(log_prob=False, covar_root_decomposition=False):
        predictions = likelihood(model(X_norm))
        mean = predictions.mean
        var = predictions.variance

    mean = Y_scaler.inverse_transform(mean)
    mean[:, 2] = - np.exp(mean[:, 2])
    mean[:, 5] = np.exp(mean[:, 5])

    return mean, var
