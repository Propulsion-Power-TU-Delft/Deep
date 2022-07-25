#! /usr/bin/env python
# -*- coding: utf-8 -*-

########################################################################################################################
# Deep: data-driven equation of state application
# Authors: ir. A. Giuffre', ir. E. Bunschoten, Dr. ir. A. Cappiello, ir. M. Majer, Dr. ir. M. Pini
# Content: Create the dataset
# 2022 - TU Delft - All rights reserved
########################################################################################################################

import time
import pickle
import CoolProp
from plot import *
import numpy as np
import pandas as pd
from tqdm import tqdm


# user-defined input
data_folder = 'MLP_MM_250k_rho_above_10'        # name of the folder collecting the dataset
fluid = 'MM'                                    # working fluid
library = 'REFPROP'                             # thermodynamic library
samples_sat = 500                               # number of samples used to compute the saturation curve
samples = 500                                   # number of samples used over each direction of the thermodynamic grid
T_r_bounds = [0.9, 1.1]                         # min-max bounds of reduced temperature to compute the saturation curve
s_r_bounds = [1.05, 1.2]                        # min-max bounds of reduced entropy to compute the saturation curve
rho_r_bounds = [0.00373, 1.1]                    # min-max bounds of reduced density to create the thermodynamic grid
e_r_bounds = [0.95, 1.3]                        # min-max bounds of reduced int energy to create the thermodynamic grid
Pt_in = 1842300                                 # total inlet pressure of the target transformation
Tt_in = 530.275                                 # total inlet temperature of the target transformation
P_out = 1000000                                 # static outlet pressure of the target transformation
save_flag = False                                # if True, save the dataset
plot_flag = True                                # if True, plot the computed thermodynamic properties

# ------------------------------------------------------------------------------------------------------------------- #

# Thermodynamic Calculations

Tr_vec = np.linspace(T_r_bounds[0], T_r_bounds[1], samples_sat)
sr_vec = np.linspace(s_r_bounds[0], s_r_bounds[1], samples_sat)
rhor_vec = np.geomspace(rho_r_bounds[0], rho_r_bounds[1], samples)
er_vec = np.linspace(e_r_bounds[0], e_r_bounds[1], samples)

EoS = CoolProp.AbstractState(library, fluid)
Tc = EoS.T_critical()
Pc = EoS.p_critical()
EoS.update(CoolProp.PT_INPUTS, Pc, Tc)
sc = EoS.smass()
rho_c = EoS.rhomass()
e_c = EoS.umass()

# compute saturation curve in T-s
s_sat_v = []
s_sat_l = []
T_vec = Tr_vec * Tc
s_vec = sr_vec * sc

for ii in range(samples_sat):
    if T_vec[ii] <= Tc:

        # saturated vapor curve
        try:
            EoS.update(CoolProp.QT_INPUTS, 1.0, T_vec[ii])
            s_sat_v.append(EoS.smass())
        except ValueError:
            s_sat_v.append(np.nan)

        # saturated liquid curve
        try:
            EoS.update(CoolProp.QT_INPUTS, 0.0, T_vec[ii])
            s_sat_l.append(EoS.smass())
        except ValueError:
            s_sat_l.append(np.nan)

s_sat_v = np.asarray(s_sat_v)
s_sat_l = np.asarray(s_sat_l)
s_sat = np.concatenate((s_sat_l, s_sat_v[::-1]))
T_sat = np.linspace(min(T_vec), Tc, len(s_sat_v))
T_sat = np.concatenate((T_sat, T_sat[::-1]))

# compute saturation curve in rho-e
rho_sat = np.zeros(len(T_sat))
e_sat = np.zeros(len(T_sat))

for ii, T in enumerate(T_sat):
    EoS.update(CoolProp.SmassT_INPUTS, s_sat[ii], T)
    rho_sat[ii] = EoS.rhomass()
    e_sat[ii] = EoS.umass()

# compute thermodynamic states along the isentropic transformation for plotting purposes
rho_is = np.zeros(samples)
e_is = np.zeros(samples)
T_is = np.zeros(samples)
P_vec = np.linspace(Pt_in, P_out, samples)
EoS.update(CoolProp.PT_INPUTS, Pt_in, Tt_in)
s_in = EoS.smass()

for ii in range(samples):
    EoS.update(CoolProp.PSmass_INPUTS, P_vec[ii], s_in)
    rho_is[ii] = EoS.rhomass()
    e_is[ii] = EoS.umass()
    T_is[ii] = EoS.T()

# compute thermodynamic properties in the prescribed region using rho, e as independent variables
rho_vec = rhor_vec * rho_c
e_vec = er_vec * e_c
rho_matrix = np.zeros((samples, samples))
e_matrix = np.zeros((samples, samples))
s_matrix = np.zeros((samples, samples))
T_matrix = np.zeros((samples, samples))
P_matrix = np.zeros((samples, samples))
h_matrix = np.zeros((samples, samples))
c_matrix = np.zeros((samples, samples))
Q_matrix = np.zeros((samples, samples))
ds_de_matrix = np.zeros((samples, samples))
ds_drho_matrix = np.zeros((samples, samples))
d2s_dedrho_matrix = np.zeros((samples, samples))
d2s_de2_matrix = np.zeros((samples, samples))
d2s_drho2_matrix = np.zeros((samples, samples))

print('\ncomputing thermodynamic properties ...')
start = time.time()
for ii, e in tqdm(enumerate(e_vec)):
    for jj, rho in enumerate(rho_vec):

        try:
            EoS.update(CoolProp.DmassUmass_INPUTS, rho, e)
            rho_matrix[ii, jj] = rho
            e_matrix[ii, jj] = e
            T_matrix[ii, jj] = EoS.T()
            EoS.update(CoolProp.TUmass_INPUTS, T_matrix[ii, jj], e_matrix[ii, jj])
            s_matrix[ii, jj] = EoS.smass()
            P_matrix[ii, jj] = EoS.p()
            h_matrix[ii, jj] = EoS.hmass()
            c_matrix[ii, jj] = EoS.speed_sound()
            Q_matrix[ii, jj] = EoS.Q()
            ds_de_matrix[ii, jj] = EoS.first_partial_deriv(CoolProp.iSmass, CoolProp.iUmass, CoolProp.iDmass)
            ds_drho_matrix[ii, jj] = EoS.first_partial_deriv(CoolProp.iSmass, CoolProp.iDmass, CoolProp.iUmass)
            d2s_dedrho_matrix[ii, jj] = EoS.second_partial_deriv(CoolProp.iSmass, CoolProp.iUmass, CoolProp.iDmass,
                                                                 CoolProp.iDmass, CoolProp.iUmass)
            d2s_de2_matrix[ii, jj] = EoS.second_partial_deriv(CoolProp.iSmass, CoolProp.iUmass, CoolProp.iDmass,
                                                              CoolProp.iUmass, CoolProp.iDmass)
            d2s_drho2_matrix[ii, jj] = EoS.second_partial_deriv(CoolProp.iSmass, CoolProp.iDmass, CoolProp.iUmass,
                                                                CoolProp.iDmass, CoolProp.iUmass)

        except ValueError:
            rho_matrix[ii, jj] = np.nan
            e_matrix[ii, jj] = np.nan
            s_matrix[ii, jj] = np.nan
            T_matrix[ii, jj] = np.nan
            P_matrix[ii, jj] = np.nan
            h_matrix[ii, jj] = np.nan
            c_matrix[ii, jj] = np.nan
            Q_matrix[ii, jj] = np.nan
            ds_de_matrix[ii, jj] = np.nan
            ds_drho_matrix[ii, jj] = np.nan
            d2s_dedrho_matrix[ii, jj] = np.nan
            d2s_de2_matrix[ii, jj] = np.nan
            d2s_drho2_matrix[ii, jj] = np.nan

end = time.time()
print('\nComputational cost [s]')
print('Global     : %10.6f' % (end - start))
print('Single call: %10.6f' % ((end - start) / (samples ** 2)))

# cluster the points based on their thermodynamic phase (single or two-phase)
mask = Q_matrix >= 1
e_1phase = e_matrix[mask].flatten()
e_2phase = e_matrix[np.logical_not(mask)].flatten()
rho_1phase = rho_matrix[mask].flatten()
rho_2phase = rho_matrix[np.logical_not(mask)].flatten()
s_1phase = s_matrix[mask].flatten()
s_2phase = s_matrix[np.logical_not(mask)].flatten()
T_1phase = T_matrix[mask].flatten()
T_2phase = T_matrix[np.logical_not(mask)].flatten()
P_1phase = P_matrix[mask].flatten()
P_2phase = P_matrix[np.logical_not(mask)].flatten()
h_1phase = h_matrix[mask].flatten()
h_2phase = h_matrix[np.logical_not(mask)].flatten()
c_1phase = c_matrix[mask].flatten()
c_2phase = c_matrix[np.logical_not(mask)].flatten()
ds_de_1phase = ds_de_matrix[mask].flatten()
ds_de_2phase = ds_de_matrix[np.logical_not(mask)].flatten()
ds_drho_1phase = ds_drho_matrix[mask].flatten()
ds_drho_2phase = ds_drho_matrix[np.logical_not(mask)].flatten()
d2s_dedrho_1phase = d2s_dedrho_matrix[mask].flatten()
d2s_dedrho_2phase = d2s_dedrho_matrix[np.logical_not(mask)].flatten()
d2s_de2_1phase = d2s_de2_matrix[mask].flatten()
d2s_de2_2phase = d2s_de2_matrix[np.logical_not(mask)].flatten()
d2s_drho2_1phase = d2s_drho2_matrix[mask].flatten()
d2s_drho2_2phase = d2s_drho2_matrix[np.logical_not(mask)].flatten()

# create final numpy arrays for plotting and storage
X = np.vstack((rho_matrix.flatten(), e_matrix.flatten())).T
Y = np.vstack((s_matrix.flatten(), ds_de_matrix.flatten(), ds_drho_matrix.flatten(),
               d2s_dedrho_matrix.flatten(), d2s_de2_matrix.flatten(), d2s_drho2_matrix.flatten())).T
primary_props = np.vstack((rho_matrix.flatten(), e_matrix.flatten(), Q_matrix.flatten(),
                           P_matrix.flatten(), T_matrix.flatten(), h_matrix.flatten(), c_matrix.flatten())).T
X_1phase = np.vstack((rho_1phase, e_1phase)).T
Y_1phase = np.vstack((s_1phase, ds_de_1phase, ds_drho_1phase, d2s_dedrho_1phase, d2s_de2_1phase, d2s_drho2_1phase)).T
X_2phase = np.vstack((rho_2phase, e_2phase)).T
Y_2phase = np.vstack((s_2phase, ds_de_2phase, ds_drho_2phase, d2s_dedrho_2phase, d2s_de2_2phase, d2s_drho2_2phase)).T

# ------------------------------------------------------------------------------------------------------------------- #

# Data Storage

if save_flag:
    data_dir = os.path.join('data', data_folder)
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    pickle.dump(X, open(os.path.join(data_dir, 'X_full.pkl'), 'wb'))
    pickle.dump(Y, open(os.path.join(data_dir, 'Y_full.pkl'), 'wb'))
    pickle.dump(X_1phase, open(os.path.join(data_dir, 'X_1phase.pkl'), 'wb'))
    pickle.dump(Y_1phase, open(os.path.join(data_dir, 'Y_1phase.pkl'), 'wb'))
    pickle.dump(X_2phase, open(os.path.join(data_dir, 'X_2phase.pkl'), 'wb'))
    pickle.dump(Y_2phase, open(os.path.join(data_dir, 'Y_2phase.pkl'), 'wb'))
    pickle.dump(rho_sat, open(os.path.join(data_dir, 'rho_sat.pkl'), 'wb'))
    pickle.dump(e_sat, open(os.path.join(data_dir, 'e_sat.pkl'), 'wb'))
    pickle.dump(primary_props, open(os.path.join(data_dir, 'primary_props.pkl'), 'wb'))

    # Save csv file for look-up table generation
    df = pd.concat([pd.DataFrame(X), pd.DataFrame(Y)], axis=1)
    df.to_csv(os.path.join(data_dir, 'LuT.csv'), header=None, index=False)

# ------------------------------------------------------------------------------------------------------------------- #

# Plotting

if plot_flag:
    print('\nsaving figures ...')
    plot = Plot(os.path.join('plots', 'MLP', data_folder), rho_matrix, e_matrix, rho_sat, e_sat, rho_c, e_c)
    plot.plot_dataset(s_matrix, ds_de_matrix, ds_drho_matrix, d2s_dedrho_matrix, d2s_de2_matrix, d2s_drho2_matrix)
    plot.plot_primary_props(P_matrix, T_matrix, h_matrix, c_matrix)
    plot.rho_e_chart(rho_is, e_is, rho_vec, e_vec)
    plot.T_s_chart(T_sat, s_sat, Tc, sc, T_1phase, T_2phase, s_1phase, s_2phase, T_is, s_in * np.ones(samples))

# ------------------------------------------------------------------------------------------------------------------- #
