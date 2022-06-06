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


class GPModelWithDerivatives(gpytorch.models.ExactGP):
    """
    Define a Multi-Output Regression Gaussian Process that will be trained with function values and derivatives
    """
    def __init__(self, train_x, train_y, likelihood, nx):
        """
        :param train_x: input features from the training set
        :param train_y: labels from the training set
        :param likelihood:
        :param nx: number of input features
        """
        super(GPModelWithDerivatives, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMeanGrad()                    # define mean
        self.base_kernel = gpytorch.kernels.RBFKernelGrad(ard_num_dims=nx)      # define kernel
        self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


def train_gp(model, likelihood, train_x, train_y, epochs, alpha):
    """
    :param model: gaussian process model
    :param likelihood:
    :param train_x: input features from the training set
    :param train_y: labels from the training set
    :param epochs: number of epochs used for training
    :param alpha: learning rate
    :return:
    """
    # Set training mode
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=alpha)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(epochs):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        print("Iter %d/%d - Loss: %.3f   lengthscales: %.3f, %.3f   noise: %.3f" % (
            i + 1, epochs, loss.item(),
            model.covar_module.base_kernel.lengthscale.squeeze()[0],
            model.covar_module.base_kernel.lengthscale.squeeze()[1],
            model.likelihood.noise.item()
        ))
        optimizer.step()


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
