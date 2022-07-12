#! /usr/bin/env python
# -*- coding: utf-8 -*-

########################################################################################################################
# Deep: data-driven equation of state application
# Authors: ir. A. Giuffre', ir. E. Bunschoten, Dr. ir. A. Cappiello, ir. M. Majer, Dr. ir. M. Pini
# Content: Optimize the set of hyper-parameters for the MLP architecture
# 2022 - TU Delft - All rights reserved
########################################################################################################################

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


class Plot:
    def __init__(self, folder, settings='normal', cmap='viridis'):
        script_dir = os.path.dirname(__file__)
        self.plot_dir = os.path.join(script_dir, folder)
        self.cmap = cmap
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        if settings == 'large':
            config = {  # setup matplotlib to use latex for output
                "pgf.texsystem": "pdflatex",
                "text.usetex": True,
                "font.family": "DejaVu Sans",
                "axes.titlesize": 36,
                "axes.labelsize": 36,
                "font.size": 36,
                "legend.fontsize": 24,
                "legend.frameon": True,
                "xtick.labelsize": 24,
                "ytick.labelsize": 24,
                "xtick.major.pad": 12,
                "ytick.major.pad": 12,
                "figure.autolayout": True,
                "figure.figsize": (9.6, 7.2)}
        else:
            config = {  # setup matplotlib to use latex for output
                "pgf.texsystem": "pdflatex",
                "text.usetex": True,
                "font.family": "DejaVu Sans",
                "axes.titlesize": 24,
                "axes.labelsize": 24,
                "font.size": 24,
                "legend.fontsize": 18,
                "legend.frameon": True,
                "xtick.labelsize": 18,
                "ytick.labelsize": 18,
                "xtick.major.pad": 8,
                "ytick.major.pad": 8,
                "figure.autolayout": True,
                "figure.figsize": (9.6, 7.2)}

        mpl.rcParams.update(config)

    def plot_objectives(self, accuracy, comp_cost, accuracy_hat, comp_cost_hat):

        fig, ax = plt.subplots()
        ax.scatter(accuracy, comp_cost, color='black', s=20)
        ax.scatter(accuracy_hat, comp_cost_hat, color='red', alpha=0.3, s=30)
        ax.set_xlabel(r'$L(\Theta)$')
        ax.set_ylabel(r'$C(\Theta)$')
        fig.savefig(self.plot_dir + '/objectives_doe.jpeg', dpi=400)
        plt.close(fig)

