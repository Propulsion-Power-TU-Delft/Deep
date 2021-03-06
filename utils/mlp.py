#! /usr/bin/env python
# -*- coding: utf-8 -*-

########################################################################################################################
# Deep: data-driven regression based on artificial intelligence
# Author: ir. A. Giuffre'
# Content: Classes to create, train, and optimize a Multi-Layer Perceptron artificial neural network
# 2022 - TU Delft - All rights reserved
########################################################################################################################

import os
import time
import pickle
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from smt.sampling_methods import LHS
from smt.applications.mixed_integer import (FLOAT, INT, ENUM, MixedIntegerSamplingMethod)


class MLP:
    """
    Create a Multi-Layer Perceptron model with custom architecture
    """
    def __init__(self, X_train, X_dev, Y_train, Y_dev, L, nl, n_epochs, activation='relu', w_init='he_uniform',
                 alpha=3, lr_decay=1.0, decay_steps=100000, staircase=False, batch_size=6, batch_norm=0,
                 regularization=0, max_norm=4, dropout_prob=[]):
        """
        :param X_train: array of input features used for the training set
        :param X_dev: array of input features used for the dev set
        :param Y_train: array of labels used for the training set
        :param Y_dev: array of labels used for the dev set
        :param L: number of layers
        :param nl: list of integers corresponding to the number of neurons in each layer
        :param n_epochs: number of epochs used to train the neural network
        :param activation: activation function used in each layer of the ann, except for the output layer
        :param w_init: method used to initialize the weights of the ann
        :param alpha: (initial) learning rate used in Adam optimizer --> 10 ** (- alpha)
        :param lr_decay: learning rate decay used to define the exponential decay schedule; if 1, there is no decay
        :param decay_steps: the learning rate drops significantly every decay_steps
        :param staircase: if True, the learning rate drops following a staircase instead of a continuous function
        :param batch_size: number of samples evaluated during training before updating the weights of the neural network
            --> 2 ** batch_size
        :param batch_norm:
            if 0, don't batch normalization;
            if 1, apply batch normalization to each hidden layer (before the activation function);
            if 2, apply batch normalization to each hidden layer (after the activation function)
        :param regularization:
            if 0, don't apply regularization;
            if 1, apply L1 regularization to each layer;
            if 2, apply L2 regularization to each layer;
            if 3, apply dropout to each layer
        :param max_norm: value of max-norm weight constraint used in each layer, when resorting to inverted dropout
        :param dropout_prob: list of dropout probabilities used in each layer, except for the output layer
        """
        self.X_train = X_train
        self.X_dev = X_dev
        self.Y_train = Y_train
        self.Y_dev = Y_dev
        self.L = L
        self.nl = nl
        self.n_epochs = n_epochs
        self.activation = activation
        self.w_init = w_init
        self.alpha = 10 ** (- alpha)
        self.lr_decay = lr_decay
        self.decay_steps = decay_steps
        self.staircase = staircase
        self.batch_size = 2 ** batch_size
        self.batch_norm = batch_norm
        self.regularization = regularization
        self.max_norm = max_norm
        self.dropout_prob = dropout_prob
        self.model = keras.models.Sequential()
        self.history = None

        # input sanity check
        if len(self.nl) != self.L:
            raise ValueError("The length of the list 'nl' must be equal to the prescribed number of layers 'L'")

        if (self.regularization == 3) and (len(self.dropout_prob) != self.L):
            raise ValueError("When using dropout, the length of the 'dropout_prob' list must be equal to 'L'")

    def set_model_architecture(self):
        """ Define the model architecture based on the prescribed set of hyper-parameters """
        # visible layer
        if self.regularization == 0:
            # no regularization
            if self.activation == 'prelu':
                # use advanced feature: parametric rectified linear unit defined as custom layer
                self.model.add(keras.layers.Dense(self.nl[0], input_dim=self.X_train.shape[1],
                                                  kernel_initializer=self.w_init))
                self.model.add(keras.layers.PReLU(alpha_initializer='zeros'))
            else:
                # use prescribed activation function
                self.model.add(keras.layers.Dense(self.nl[0], input_dim=self.X_train.shape[1],
                                                  activation=self.activation,
                                                  kernel_initializer=self.w_init))

        elif self.regularization == 1:
            # L1 regularization
            if self.activation == 'prelu':
                # use advanced feature: parametric rectified linear unit defined as custom layer
                self.model.add(keras.layers.Dense(self.nl[0], input_dim=self.X_train.shape[1],
                                                  kernel_initializer=self.w_init,
                                                  kernel_regularizer=keras.regularizers.l1()))
                self.model.add(keras.layers.PReLU(alpha_initializer='zeros'))
            else:
                # use prescribed activation function
                self.model.add(keras.layers.Dense(self.nl[0], input_dim=self.X_train.shape[1],
                                                  activation=self.activation,
                                                  kernel_initializer=self.w_init,
                                                  kernel_regularizer=keras.regularizers.l1()))

        elif self.regularization == 2:
            # L2 regularization
            if self.activation == 'prelu':
                # use advanced feature: parametric rectified linear unit defined as custom layer
                self.model.add(keras.layers.Dense(self.nl[0], input_dim=self.X_train.shape[1],
                                                  kernel_initializer=self.w_init,
                                                  kernel_regularizer=keras.regularizers.l2()))
                self.model.add(keras.layers.PReLU(alpha_initializer='zeros'))
            else:
                # use prescribed activation function
                self.model.add(keras.layers.Dense(self.nl[0], input_dim=self.X_train.shape[1],
                                                  activation=self.activation,
                                                  kernel_initializer=self.w_init,
                                                  kernel_regularizer=keras.regularizers.l2()))

        elif self.regularization == 3:
            # use dropout and max-norm
            if self.activation == 'prelu':
                # use advanced feature: parametric rectified linear unit defined as custom layer
                self.model.add(keras.layers.Dropout(self.dropout_prob[0], input_shape=(self.X_train.shape[1],)))
                self.model.add(keras.layers.Dense(self.nl[0], kernel_initializer=self.w_init,
                                                  kernel_constraint=keras.constraints.max_norm(self.max_norm)))
                self.model.add(keras.layers.PReLU(alpha_initializer='zeros'))
            else:
                # use prescribed activation function
                self.model.add(keras.layers.Dropout(self.dropout_prob[0], input_shape=(self.X_train.shape[1],)))
                self.model.add(keras.layers.Dense(self.nl[0], activation=self.activation,
                                                  kernel_initializer=self.w_init,
                                                  kernel_constraint=keras.constraints.max_norm(self.max_norm)))

        # iterate over L hidden layers
        layer = 1
        while layer < self.L:

            # no batch normalization
            if self.batch_norm == 0:

                if self.regularization == 0:
                    # no regularization
                    if self.activation == 'prelu':
                        # use advanced feature: parametric rectified linear unit defined as custom layer
                        self.model.add(keras.layers.Dense(self.nl[layer], kernel_initializer=self.w_init))
                        self.model.add(keras.layers.PReLU(alpha_initializer='zeros'))
                    else:
                        # use prescribed activation function
                        self.model.add(keras.layers.Dense(self.nl[layer], activation=self.activation,
                                                          kernel_initializer=self.w_init))

                elif self.regularization == 1:
                    # L1 regularization
                    if self.activation == 'prelu':
                        # use advanced feature: parametric rectified linear unit defined as custom layer
                        self.model.add(keras.layers.Dense(self.nl[layer], kernel_initializer=self.w_init,
                                                          kernel_regularizer=keras.regularizers.l1()))
                        self.model.add(keras.layers.PReLU(alpha_initializer='zeros'))
                    else:
                        # use prescribed activation function
                        self.model.add(keras.layers.Dense(self.nl[layer], activation=self.activation,
                                                          kernel_initializer=self.w_init,
                                                          kernel_regularizer=keras.regularizers.l1()))

                elif self.regularization == 2:
                    # L2 regularization
                    if self.activation == 'prelu':
                        # use advanced feature: parametric rectified linear unit defined as custom layer
                        self.model.add(keras.layers.Dense(self.nl[layer], kernel_initializer=self.w_init,
                                                          kernel_regularizer=keras.regularizers.l2()))
                        self.model.add(keras.layers.PReLU(alpha_initializer='zeros'))
                    else:
                        # use prescribed activation function
                        self.model.add(keras.layers.Dense(self.nl[layer], activation=self.activation,
                                                          kernel_initializer=self.w_init,
                                                          kernel_regularizer=keras.regularizers.l2()))

                elif self.regularization == 3:
                    # use dropout and max-norm
                    if self.activation == 'prelu':
                        # use advanced feature: parametric rectified linear unit defined as custom layer
                        self.model.add(keras.layers.Dropout(self.dropout_prob[layer]))
                        self.model.add(keras.layers.Dense(self.nl[layer], kernel_initializer=self.w_init,
                                                          kernel_constraint=keras.constraints.max_norm(self.max_norm)))
                        self.model.add(keras.layers.PReLU(alpha_initializer='zeros'))
                    else:
                        # use prescribed activation function
                        self.model.add(keras.layers.Dropout(self.dropout_prob[layer]))
                        self.model.add(keras.layers.Dense(self.nl[layer], activation=self.activation,
                                                          kernel_initializer=self.w_init,
                                                          kernel_constraint=keras.constraints.max_norm(self.max_norm)))

            # batch normalization before the activation function
            elif self.batch_norm == 1:

                if self.regularization == 0:
                    # no regularization
                    self.model.add(keras.layers.Dense(self.nl[layer], kernel_initializer=self.w_init))

                elif self.regularization == 1:
                    # L1 regularization
                    self.model.add(keras.layers.Dense(self.nl[layer], kernel_initializer=self.w_init,
                                                      kernel_regularizer=keras.regularizers.l1()))

                elif self.regularization == 2:
                    # L2 regularization
                    self.model.add(keras.layers.Dense(self.nl[layer], kernel_initializer=self.w_init,
                                                      kernel_regularizer=keras.regularizers.l2()))

                elif self.regularization == 3:
                    # use dropout and max-norm
                    self.model.add(keras.layers.Dropout(self.dropout_prob[layer]))
                    self.model.add(keras.layers.Dense(self.nl[layer], kernel_initializer=self.w_init,
                                                      kernel_constraint=keras.constraints.max_norm(self.max_norm)))

                self.model.add(keras.layers.BatchNormalization())

                if self.activation == 'prelu':
                    # use advanced feature: parametric rectified linear unit defined as custom layer
                    self.model.add(keras.layers.PReLU(alpha_initializer='zeros'))
                else:
                    # use prescribed activation function
                    self.model.add(keras.layers.Activation(self.activation))

            # batch normalization after the activation function
            elif self.batch_norm == 2:

                if self.regularization == 0:
                    # no regularization
                    if self.activation == 'prelu':
                        # use advanced feature: parametric rectified linear unit defined as custom layer
                        self.model.add(keras.layers.Dense(self.nl[layer], kernel_initializer=self.w_init))
                        self.model.add(keras.layers.PReLU(alpha_initializer='zeros'))
                    else:
                        # use prescribed activation function
                        self.model.add(keras.layers.Dense(self.nl[layer], activation=self.activation,
                                                          kernel_initializer=self.w_init))

                elif self.regularization == 1:
                    # L1 regularization
                    if self.activation == 'prelu':
                        # use advanced feature: parametric rectified linear unit defined as custom layer
                        self.model.add(keras.layers.Dense(self.nl[layer], kernel_initializer=self.w_init,
                                                          kernel_regularizer=keras.regularizers.l1()))
                        self.model.add(keras.layers.PReLU(alpha_initializer='zeros'))
                    else:
                        # use prescribed activation function
                        self.model.add(keras.layers.Dense(self.nl[layer], activation=self.activation,
                                                          kernel_initializer=self.w_init,
                                                          kernel_regularizer=keras.regularizers.l1()))

                elif self.regularization == 2:
                    # L2 regularization
                    if self.activation == 'prelu':
                        # use advanced feature: parametric rectified linear unit defined as custom layer
                        self.model.add(keras.layers.Dense(self.nl[layer], kernel_initializer=self.w_init,
                                                          kernel_regularizer=keras.regularizers.l2()))
                        self.model.add(keras.layers.PReLU(alpha_initializer='zeros'))
                    else:
                        # use prescribed activation function
                        self.model.add(keras.layers.Dense(self.nl[layer], activation=self.activation,
                                                          kernel_initializer=self.w_init,
                                                          kernel_regularizer=keras.regularizers.l2()))

                elif self.regularization == 3:
                    # use dropout and max-norm
                    if self.activation == 'prelu':
                        # use advanced feature: parametric rectified linear unit defined as custom layer
                        self.model.add(keras.layers.Dropout(self.dropout_prob[layer]))
                        self.model.add(keras.layers.Dense(self.nl[layer], kernel_initializer=self.w_init,
                                                          kernel_constraint=keras.constraints.max_norm(self.max_norm)))
                        self.model.add(keras.layers.PReLU(alpha_initializer='zeros'))
                    else:
                        # use prescribed activation function
                        self.model.add(keras.layers.Dropout(self.dropout_prob[layer]))
                        self.model.add(keras.layers.Dense(self.nl[layer], activation=self.activation,
                                                          kernel_initializer=self.w_init,
                                                          kernel_constraint=keras.constraints.max_norm(self.max_norm)))

                self.model.add(keras.layers.BatchNormalization())

            layer += 1

        # output layer
        self.model.add(keras.layers.Dense(self.Y_train.shape[1], activation='linear'))

    def train_model(self, loss="mean_squared_error", verbose=2):
        """
        Train the artificial neural network
        :param loss: loss function used to train the ann
        :param verbose: level of verbosity used during training
        """
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(self.alpha, decay_steps=self.decay_steps,
                                                                  decay_rate=self.lr_decay, staircase=self.staircase)
        opt = keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=1e-8, amsgrad=False)
        self.model.compile(optimizer=opt, loss=loss, metrics=["mape"])
        self.history = self.model.fit(self.X_train, self.Y_train, epochs=self.n_epochs, batch_size=self.batch_size,
                                      verbose=verbose, validation_data=(self.X_dev, self.Y_dev), shuffle=True)

    def plot_training_history(self, plot_dir):
        """
        Plot training history
        :param plot_dir: directory to save plots
        """
        n_plots = 2
        new_colors = [plt.get_cmap('viridis')(1. * i / n_plots) for i in range(n_plots)]

        fig, ax = plt.subplots()
        ax.semilogy(self.history.history['loss'], color=new_colors[0], label='Train set loss')
        ax.semilogy(self.history.history['val_loss'], color=new_colors[1], label='Dev set loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.grid(1)
        plt.legend(loc='upper right')
        fig.savefig(os.path.join(plot_dir, 'training_history.jpeg'), dpi=400)
        plt.close(fig)


class HyperSearch:
    """
    Class implementing the optimization of a Multi-Layer Perceptron Neural Network.
    By default, dropout is not applied.

    Optimization variables (L + 3):
        - Number of neurons in each layer: nl (array of L int)
        - Size of mini-batches used in the optimization procedure: batch_size (int)
        - Initial learning rate: alpha (float)
        - Activation function: activation (string)

    Objectives (2):
        - Loss evaluated on the dev set
        - Evaluation time
    """

    def __init__(self, X_train, X_dev, Y_train, Y_dev, L, n_epochs, batch_norm=0, regularization=0,
                 w_init='he_uniform', lr_decay=1.0, decay_steps=100000, staircase=False, max_norm=4, dropout_prob=[]):
        """
        :param X_train: array of input features used for the training set
        :param X_dev: array of input features used for the dev set
        :param Y_train: array of labels used for the training set
        :param Y_dev: array of labels used for the dev set
        :param L: number of layers
        :param n_epochs: number of epochs used to train the neural network
        :param batch_norm:
            if 0, don't batch normalization;
            if 1, apply batch normalization to each hidden layer (before the activation function);
            if 2, apply batch normalization to each hidden layer (after the activation function)
        :param regularization:
            if 0, don't apply regularization;
            if 1, apply L1 regularization to each layer;
            if 2, apply L2 regularization to each layer;
            if 3, apply dropout to each layer
        :param w_init: method used to initialize the weights of the ann
        :param lr_decay: learning rate decay used to define the exponential decay schedule; if 1, there is no decay
        :param decay_steps: the learning rate drops significantly every decay_steps
        :param staircase: if True, the learning rate drops following a staircase instead of a continuous function
        :param max_norm: value of max-norm weight constraint used in each layer, when resorting to inverted dropout
        :param dropout_prob: list of dropout probabilities used in each layer, except for the output layer
        """
        self.count = 0
        self.X_train = X_train
        self.X_dev = X_dev
        self.Y_train = Y_train
        self.Y_dev = Y_dev
        self.L = L
        self.n_epochs = n_epochs
        self.batch_norm = batch_norm
        self.reg = regularization
        self.w_init = w_init
        self.lr_decay = lr_decay
        self.decay_steps = decay_steps
        self.staircase = staircase
        self.max_norm = max_norm
        self.dropout_prob = dropout_prob

    def evaluate(self, x):
        """
        Create a MLP with the prescribed set of hyper-parameters, train it, and compute
        its accuracy and its evaluation time
        :param x: array of design variables
        """
        self.count += 1

        # retrieve hyper-parameters for the current sample
        nl = [int(x[i]) for i in range(self.L)]
        batch_size = int(x[self.L])
        alpha = x[self.L + 1]

        if x[self.L + 2] == 0:
            activation = 'sigmoid'
        elif x[self.L + 2] == 1:
            activation = 'tanh'
        elif x[self.L + 2] == 2:
            activation = 'relu'
        elif x[self.L + 2] == 3:
            activation = 'selu'
        elif x[self.L + 2] == 4:
            activation = 'gelu'
        elif x[self.L + 2] == 5:
            activation = 'prelu'
        elif x[self.L + 2] == 6:
            activation = 'swish'

        # define the MLP
        model = MLP(self.X_train, self.X_dev, self.Y_train, self.Y_dev, self.L, nl, self.n_epochs,
                    activation=activation, w_init=self.w_init, alpha=alpha, lr_decay=self.lr_decay,
                    decay_steps=self.decay_steps, staircase=self.staircase, batch_size=batch_size,
                    batch_norm=self.batch_norm, regularization=self.reg, max_norm=self.max_norm,
                    dropout_prob=self.dropout_prob)

        print('\n# ------------------------------- MLP NO. %d ------------------------------- #' % self.count)
        print("Number of neurons per layer : ", str(model.nl))
        print("Batch size                  : %d" % model.batch_size)
        print("Learning rate               : %10.7f" % model.alpha)
        print("Activation function         : ", activation)

        # train the MLP
        model.set_model_architecture()
        model.train_model(verbose=2)

        # evaluate the accuracy and the computational cost at evaluation time
        start = time.time()
        model.model.predict(self.X_train)
        end = time.time()
        comp_cost = (end - start) / self.X_train.shape[0]
        accuracy = model.history.history['val_loss'][-1]

        print("Train set loss             : %10.8f" % model.history.history['loss'][-1])
        print("Dev set loss               : %10.8f" % model.history.history['val_loss'][-1])
        print("Evaluation time            : %10.8f s" % comp_cost)

        return accuracy, comp_cost

    def doe(self, samples, model_dir):

        # define space of hyper-parameters to be investigated
        if self.L == 1:
            xtypes = [INT, INT, FLOAT, (ENUM, 7)]
            bounds = [[2, 100], [4, 10], [1.0, 5.0],
                      ["sigmoid", "tanh", "relu", "selu", "gelu", "prelu", "swish"]]

        elif self.L == 2:
            xtypes = [INT, INT, INT, FLOAT, (ENUM, 7)]
            bounds = [[2, 100], [2, 100], [4, 10], [1.0, 5.0],
                      ["sigmoid", "tanh", "relu", "selu", "gelu", "prelu", "swish"]]

        elif self.L == 3:
            xtypes = [INT, INT, INT, INT, FLOAT, (ENUM, 7)]
            bounds = [[2, 100], [2, 100], [2, 100], [4, 10], [1.0, 5.0],
                      ["sigmoid", "tanh", "relu", "selu", "gelu", "prelu", "swish"]]

        elif self.L == 4:
            xtypes = [INT, INT, INT, INT, INT, FLOAT, (ENUM, 7)]
            bounds = [[2, 100], [2, 100], [2, 100], [2, 100], [4, 10], [1.0, 5.0],
                      ["sigmoid", "tanh", "relu", "selu", "gelu", "prelu", "swish"]]

        else:
            raise ValueError("Hyper-parameters search with more than 4 hidden layers is not implemented yet")

        # define sampling method and grid of samples
        sampling = MixedIntegerSamplingMethod(xtypes, bounds, LHS, criterion="m")
        doe = sampling(samples)

        # initialize array used to store results
        res = np.hstack((doe, np.zeros((samples, 2))))

        # evaluate each MLP architecture within the sampling space and save results
        for ii in range(samples):
            accuracy, comp_cost = self.evaluate(doe[ii, :])
            res[ii, -2] = accuracy
            res[ii, -1] = comp_cost

            # overwrite result file at each iteration
            pickle.dump(res, open(os.path.join(model_dir, 'doe.pkl'), 'wb'))

