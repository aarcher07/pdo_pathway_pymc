import matplotlib as mpl
import aesara
import aesara.tensor as at
import arviz as az
import matplotlib.pyplot as plt
import os
import seaborn as sns

import pymc as pm
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
from prior_constants import NORM_PRIOR_STD_RT_SINGLE_EXP,NORM_PRIOR_MEAN_SINGLE_EXP, NORM_PRIOR_STD_RT_ALL_EXP, \
    NORM_PRIOR_MEAN_ALL_EXP, LOG_UNIF_PRIOR_ALL_EXP, DATA_LOG_UNIF_PARAMETER_RANGES, NORM_PRIOR_PARAMETER_ALL_EXP_DICT, \
    LOG_UNIF_G_EXT_INIT_PRIOR_PARAMETERS
from constants import PERMEABILITY_PARAMETERS, KINETIC_PARAMETERS, ENZYME_CONCENTRATIONS, \
    GLYCEROL_EXTERNAL_EXPERIMENTAL, ALL_PARAMETERS, PARAMETER_LIST, TIME_SAMPLES_EXPANDED, VARIABLE_NAMES, HRS_TO_SECS, \
    TIME_SPACING, DATA_INDEX, N_MODEL_PARAMETERS
import time
from os.path import dirname, abspath
import sys
from pathlib import Path
import numpy as np
from datetime import datetime
from scipy.stats import multivariate_normal
import pickle
from likelihood_funcs_adj import likelihood_adj, likelihood_derivative_adj
from os.path import dirname, abspath
from exp_data import *
from rhs_funcs import RHS, lib, problem, solver
import pandas as pd
from formatting_constants import VARS_ALL_EXP_TO_TEX
ROOT_PATH = dirname(abspath(__file__))
from pandas.plotting import scatter_matrix
from plot_nuts_results_funcs import plot_loglik_individual, plot_loglik_overlay, plot_corr_scatter, plot_corr, \
    plot_time_series_distribution, joint_Keq_distribution
nsamples = 1000
burn_in = 2000
nchains = 2
acc_rate = 0.7
atol = 1e-10
rtol = 1e-10
mxsteps = 1e5


# save samples
PARAMETER_SAMP_PATH = ROOT_PATH + '/samples'
directory_name = 'nsamples_' + str(nsamples) + '_burn_in_' + str(burn_in) + '_acc_rate_' + str(acc_rate) + \
                 '_nchains_' + str(nchains) + '_atol_' + str(atol) + '_rtol_' + str(rtol) + '_mxsteps_' + str(int(mxsteps))
directory_name = directory_name.replace('.','_').replace('-','_')
file_name = '2022_06_23_09_24_09_054903.nc'
data_file_location = os.path.join(PARAMETER_SAMP_PATH, directory_name, file_name)
samples = az.from_netcdf(data_file_location)

PLOT_SAMP_PATH = ROOT_PATH + '/plot_analysis'
plot_file_location = os.path.join(PLOT_SAMP_PATH, directory_name, file_name[:-3])
Path(plot_file_location).mkdir(parents=True, exist_ok=True)

# plot_loglik_individual(samples.sample_stats.lp, plot_file_location, nchains)
# plot_loglik_overlay(samples.sample_stats.lp, plot_file_location, nchains)
# plot_time_series_distribution(samples, plot_file_location, nchains, atol, rtol, mxsteps)
# plot_corr(samples, plot_file_location, nchains)
# plot_corr_scatter(samples, plot_file_location, nchains)
KeqDhaB = np.power(10,samples.posterior.k1DhaB)*np.power(10,samples.posterior.k3DhaB)/np.power(10,samples.posterior.k2DhaB)*np.power(10,samples.posterior.k4DhaB)
KeqDhaT = np.power(10,samples.posterior.k1DhaT)*np.power(10,samples.posterior.k3DhaT)/np.power(10,samples.posterior.k2DhaT)*np.power(10,samples.posterior.k4DhaT)
joint_Keq_distribution(KeqDhaB, KeqDhaT, plot_file_location, nchains)
