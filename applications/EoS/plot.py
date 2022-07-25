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
from matplotlib.patches import Rectangle


class Plot:
    def __init__(self, folder, rho=None, e=None, rho_sat=None, e_sat=None, rho_c=None, e_c=None,
                 settings='normal', cmap='viridis'):
        """
        :param folder: directory where the plots will be saved
        :param rho: 2D array of density used to generate the data
        :param e: 2D array of internal energy used to generate the data
        :param rho_sat: array of density values identifying the saturation curve
        :param e_sat: array of internal energy values identifying the saturation curve
        :param rho_c: critical density
        :param e_c: critical internal energy
        :param settings: string identifying the pre-defined set of settings used for each plot
        :param cmap: string identifying the colormap used for each plot
        """
        script_dir = os.path.dirname(__file__)
        self.rho = rho
        self.e = e
        self.rho_sat = rho_sat
        self.e_sat = e_sat
        self.rho_c = rho_c
        self.e_c = e_c
        self.cmap = cmap
        self.plot_dir = os.path.join(script_dir, folder)

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

    def plot_dataset(self, s, ds_de, ds_drho, d2s_de_drho, d2s_de2, d2s_drho2):
        """
        Create contour plots of s, its 1st and 2nd derivatives over the prescribed range of 
        density and internal energy values.
        :param ds_de: 2D array of 1st derivative of entropy wrt internal energy
        :param ds_drho: 2D array of 1st derivative of entropy wrt density
        :param d2s_de_drho: 2D array of mixed derivative of entropy wrt internal energy and density
        :param d2s_de2: 2D array of 2nd derivative of entropy wrt internal energy
        :param d2s_drho2: 2D array of 2nd derivative of entropy wrt density
        """
        fig, ax = plt.subplots(2, 3, figsize=(14, 10), sharex=True, sharey=True)

        cs00 = ax[0, 0].contourf(self.rho, self.e / 1e3, s, cmap=self.cmap)
        ax[0, 0].contour(cs00, alpha=0.2, colors='black', linestyles='solid')
        ax[0, 0].semilogx(self.rho_sat, self.e_sat / 1e3, 'white')
        ax[0, 0].semilogx(self.rho_c, self.e_c / 1e3, 'o', color='white', ms=10)
        ax[0, 0].set_xlim(np.min(self.rho), np.max(self.rho))
        ax[0, 0].set_ylim(np.min(self.e / 1e3), np.max(self.e / 1e3))
        ax[0, 0].set_title('s')
        ax[0, 0].set_xlabel(r'$\rho \, \mathrm{[kg/m^3]}$')
        ax[0, 0].set_ylabel(r'$e \, \mathrm{[kJ/kg]}$')

        cs01 = ax[0, 1].contourf(self.rho, self.e / 1e3, ds_de, cmap=self.cmap)
        ax[0, 1].contour(cs01, alpha=0.2, colors='black', linestyles='solid')
        ax[0, 1].semilogx(self.rho_sat, self.e_sat / 1e3, 'white')
        ax[0, 1].semilogx(self.rho_c, self.e_c / 1e3, 'o', color='white', ms=10)
        ax[0, 1].set_xlim(np.min(self.rho), np.max(self.rho))
        ax[0, 1].set_ylim(np.min(self.e / 1e3), np.max(self.e / 1e3))
        ax[0, 1].set_title(r'$\partial s / \partial e$')
        ax[0, 1].set_xlabel(r'$\rho \, \mathrm{[kg/m^3]}$')
        ax[0, 1].set_ylabel(r'$e \, \mathrm{[kJ/kg]}$')

        cs02 = ax[0, 2].contourf(self.rho, self.e / 1e3, ds_drho, cmap=self.cmap)
        ax[0, 2].contour(cs02, alpha=0.2, colors='black', linestyles='solid')
        ax[0, 2].semilogx(self.rho_sat, self.e_sat / 1e3, 'white')
        ax[0, 2].semilogx(self.rho_c, self.e_c / 1e3, 'o', color='white', ms=10)
        ax[0, 2].set_xlim(np.min(self.rho), np.max(self.rho))
        ax[0, 2].set_ylim(np.min(self.e / 1e3), np.max(self.e / 1e3))
        ax[0, 2].set_title(r'$\partial s / \partial \rho$')
        ax[0, 2].set_xlabel(r'$\rho \, \mathrm{[kg/m^3]}$')
        ax[0, 2].set_ylabel(r'$e \, \mathrm{[kJ/kg]}$')

        cs10 = ax[1, 0].contourf(self.rho, self.e / 1e3, d2s_de_drho, cmap=self.cmap)
        ax[1, 0].contour(cs10, alpha=0.2, colors='black', linestyles='solid')
        ax[1, 0].semilogx(self.rho_sat, self.e_sat / 1e3, 'white')
        ax[1, 0].semilogx(self.rho_c, self.e_c / 1e3, 'o', color='white', ms=10)
        ax[1, 0].set_xlim(np.min(self.rho), np.max(self.rho))
        ax[1, 0].set_ylim(np.min(self.e / 1e3), np.max(self.e / 1e3))
        ax[1, 0].set_title(r'$\partial^2 s / (\partial e \partial \rho)$')
        ax[1, 0].set_xlabel(r'$\rho \, \mathrm{[kg/m^3]}$')
        ax[1, 0].set_ylabel(r'$e \, \mathrm{[kJ/kg]}$')

        cs11 = ax[1, 1].contourf(self.rho, self.e / 1e3, d2s_de2, cmap=self.cmap)
        ax[1, 1].contour(cs11, alpha=0.2, colors='black', linestyles='solid')
        ax[1, 1].semilogx(self.rho_sat, self.e_sat / 1e3, 'white')
        ax[1, 1].semilogx(self.rho_c, self.e_c / 1e3, 'o', color='white', ms=10)
        ax[1, 1].set_xlim(np.min(self.rho), np.max(self.rho))
        ax[1, 1].set_ylim(np.min(self.e / 1e3), np.max(self.e / 1e3))
        ax[1, 1].set_title(r'$\partial^2 s / \partial e^2$')
        ax[1, 1].set_xlabel(r'$\rho \, \mathrm{[kg/m^3]}$')
        ax[1, 1].set_ylabel(r'$e \, \mathrm{[kJ/kg]}$')

        cs12 = ax[1, 2].contourf(self.rho, self.e / 1e3, d2s_drho2, cmap=self.cmap)
        ax[1, 2].contour(cs12, alpha=0.2, colors='black', linestyles='solid')
        ax[1, 2].semilogx(self.rho_sat, self.e_sat / 1e3, 'white')
        ax[1, 2].semilogx(self.rho_c, self.e_c / 1e3, 'o', color='white', ms=10)
        ax[1, 2].set_xlim(np.min(self.rho), np.max(self.rho))
        ax[1, 2].set_ylim(np.min(self.e / 1e3), np.max(self.e / 1e3))
        ax[1, 2].set_title(r'$\partial^2 s / \partial \rho^2$')
        ax[1, 2].set_xlabel(r'$\rho \, \mathrm{[kg/m^3]}$')
        ax[1, 2].set_ylabel(r'$e \, \mathrm{[kJ/kg]}$')

        fig.savefig(self.plot_dir + '/thermodynamic_potential.jpeg', dpi=400)
        plt.close(fig)

    def plot_primary_props(self, P, T, h, c):
        """
        Create contour plots of the primary thermodynamic properties over the prescribed range of
        density and internal energy values.
        :param P: 2D array of pressure
        :param T: 2D array of temperature
        :param h: 2D array of specific enthalpy
        :param c: 2D array of sound speed
        """
        fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

        cs00 = ax[0, 0].contourf(self.rho, self.e / 1e3, P, cmap=self.cmap)
        ax[0, 0].contour(cs00, alpha=0.2, colors='black', linestyles='solid')
        ax[0, 0].semilogx(self.rho_sat, self.e_sat / 1e3, 'black')
        ax[0, 0].semilogx(self.rho_c, self.e_c / 1e3, 'o', color='black', ms=10)
        ax[0, 0].set_xlim(np.min(self.rho), np.max(self.rho))
        ax[0, 0].set_ylim(np.min(self.e / 1e3), np.max(self.e / 1e3))
        ax[0, 0].set_title('$P$')
        ax[0, 0].set_xlabel(r'$\rho \, \mathrm{[kg/m^3]}$')
        ax[0, 0].set_ylabel(r'$e \, \mathrm{[kJ/kg]}$')

        cs01 = ax[0, 1].contourf(self.rho, self.e / 1e3, T, cmap=self.cmap)
        ax[0, 1].contour(cs01, alpha=0.2, colors='black', linestyles='solid')
        ax[0, 1].semilogx(self.rho_sat, self.e_sat / 1e3, 'black')
        ax[0, 1].semilogx(self.rho_c, self.e_c / 1e3, 'o', color='black', ms=10)
        ax[0, 1].set_xlim(np.min(self.rho), np.max(self.rho))
        ax[0, 1].set_ylim(np.min(self.e / 1e3), np.max(self.e / 1e3))
        ax[0, 1].set_title('$T$')
        ax[0, 1].set_xlabel(r'$\rho \, \mathrm{[kg/m^3]}$')
        ax[0, 1].set_ylabel(r'$e \, \mathrm{[kJ/kg]}$')

        cs10 = ax[1, 0].contourf(self.rho, self.e / 1e3, h, cmap=self.cmap)
        ax[1, 0].contour(cs10, alpha=0.2, colors='black', linestyles='solid')
        ax[1, 0].semilogx(self.rho_sat, self.e_sat / 1e3, 'black')
        ax[1, 0].semilogx(self.rho_c, self.e_c / 1e3, 'o', color='black', ms=10)
        ax[1, 0].set_xlim(np.min(self.rho), np.max(self.rho))
        ax[1, 0].set_ylim(np.min(self.e / 1e3), np.max(self.e / 1e3))
        ax[1, 0].set_title('$h$')
        ax[1, 0].set_xlabel(r'$\rho \, \mathrm{[kg/m^3]}$')
        ax[1, 0].set_ylabel(r'$e \, \mathrm{[kJ/kg]}$')

        cs11 = ax[1, 1].contourf(self.rho, self.e / 1e3, np.log(c), cmap=self.cmap)
        ax[1, 1].contour(cs11, alpha=0.2, colors='black', linestyles='solid')
        ax[1, 1].semilogx(self.rho_sat, self.e_sat / 1e3, 'black')
        ax[1, 1].semilogx(self.rho_c, self.e_c / 1e3, 'o', color='black', ms=10)
        ax[1, 1].set_xlim(np.min(self.rho), np.max(self.rho))
        ax[1, 1].set_ylim(np.min(self.e / 1e3), np.max(self.e / 1e3))
        ax[1, 1].set_title('$log(c)$')
        ax[1, 1].set_xlabel(r'$\rho \, \mathrm{[kg/m^3]}$')
        ax[1, 1].set_ylabel(r'$e \, \mathrm{[kJ/kg]}$')

        fig.savefig(self.plot_dir + '/primary_properties.jpeg', dpi=400)
        plt.close(fig)
        
    def rho_e_chart(self, rho_is, e_is, rho_vec, e_vec):
        """
        :param rho_is: array of density values identifying the target isentropic transformation
        :param e_is: array of internal energy values identifying the target isentropic transformation
        :param rho_vec: array of density values used to generate the dataset
        :param e_vec: array of internal energy values used to generate the dataset
        """
        fig, ax = plt.subplots()
        ax.plot(self.rho_sat, self.e_sat / 1e3, color='black', label='saturation curve')
        ax.plot(rho_is, e_is / 1e3, color='blue', label='isentropic transformation')
        ax.plot(rho_is[0], e_is[0] / 1e3, 'o', color='blue')
        ax.add_patch(Rectangle((rho_vec[0], e_vec[0] / 1e3), (rho_vec[-1] - rho_vec[0]),
                               (e_vec[-1] / 1e3 - e_vec[0] / 1e3),
                               alpha=0.2, label='database range'))
        ax.plot(self.rho_c, self.e_c / 1e3, 'o', color='black', label='critical point')
        ax.set_xlabel('rho [kg/m3]')
        ax.set_ylabel('e [kJ/kg]')
        ax.legend()
        
        fig.savefig(self.plot_dir + '/rho_e_chart.jpeg', dpi=400)
        plt.close(fig)

    def T_s_chart(self, T_sat, s_sat, T_c, s_c, T_1phase, T_2phase, s_1phase, s_2phase, T_is, s_is):
        """
        :param T_sat: array of temperature values identifying the saturation curve
        :param s_sat: array of entropy values identifying the saturation curve
        :param T_c: critical temperature
        :param s_c: critical entropy
        :param T_1phase: 2D array of temperature over the single phase region
        :param T_2phase: 2D array of temperature over the two-phase region
        :param s_1phase: 2D array of entropy over the single phase region
        :param s_2phase: 2D array of entropy over the two-phase region
        :param T_is: array of temperature values identifying the target isentropic transformation
        :param s_is: array of entropy values identifying the target isentropic transformation
        """
        fig, ax = plt.subplots()
        ax.scatter(s_1phase, T_1phase, marker='.', s=1, c="black", alpha=0.3, label='database, single-phase')
        ax.scatter(s_2phase, T_2phase, marker='.', s=1, c="green", alpha=0.3, label='database, two-phase')
        ax.plot(s_sat, T_sat, color='black', label='saturation curve')
        ax.plot(s_c, T_c, 'o', color='black', label='critical point')
        ax.plot(s_is, T_is, color='blue', label='isentropic transformation')
        ax.plot(s_is[0], T_is[0], 'o', color='blue')
        ax.set_xlabel('s [J/kg.K]')
        ax.set_ylabel('T [K]')
        ax.legend()

        fig.savefig(self.plot_dir + '/T_s_chart.jpeg', dpi=400)
        plt.close(fig)

    def plot_hyper_search(self, hyperparameters, accuracy, comp_cost):
        """
        :param hyperparameters:
        :param accuracy:
        :param comp_cost:
        """
        fig, ax = plt.subplots()

        ax.scatter(accuracy, comp_cost, color='black', s=20, alpha=0.5)

        # retrieve the selected design point
        selected_point = np.asarray(plt.ginput(1, timeout=-1)).flatten()
        pos = self.find_nearest_2D(accuracy, comp_cost, selected_point[0], selected_point[1])

        print('\n# ------------------------------- SELECTED MLP ARCHITECTURE ------------------------------- #')
        print("Hyper-parameters  : " + str(hyperparameters[pos, :]))
        print("Dev set loss      : " + str(accuracy[pos]))
        print("Computational cost: " + str(comp_cost[pos]))

        ax.scatter(accuracy[pos], comp_cost[pos], s=20, color='red', label='Selected MLP architecture')

        plt.legend()
        ax.set_xlabel(r'$L(\Theta)$')
        ax.set_ylabel(r'$C(\Theta)$')
        fig.savefig(self.plot_dir + '/objectives_doe.jpeg', dpi=400)
        plt.close(fig)

    @staticmethod
    def find_nearest_2D(x_vec, y_vec, x, y):
        """
        Return the position of the elements of the two arrays minimizing the distance
        to the given couple of scalar values.
        :param x_vec: 1D array of x coordinates
        :param y_vec: 1D array of y coordinates
        :param x: x scalar value
        :param y: y scalar value
        """
        dist = np.sqrt((x - x_vec) ** 2 + (y - y_vec) ** 2)
        idx = np.argmin(dist)

        return idx

