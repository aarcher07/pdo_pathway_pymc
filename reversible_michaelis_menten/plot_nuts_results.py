import matplotlib as mpl
import aesara
import aesara.tensor as at
import arviz as az
import matplotlib.pyplot as plt
import os
import pymc as pm
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
from prior_constants import NORM_PRIOR_STD_RT_SINGLE_EXP,NORM_PRIOR_MEAN_SINGLE_EXP, NORM_PRIOR_STD_RT_ALL_EXP, \
    NORM_PRIOR_MEAN_ALL_EXP, LOG_UNIF_PRIOR_ALL_EXP, DATA_LOG_UNIF_PARAMETER_RANGES, NORM_PRIOR_PARAMETER_ALL_EXP_DICT
from constants import PERMEABILITY_PARAMETERS, KINETIC_PARAMETERS, ENZYME_CONCENTRATIONS, GLYCEROL_EXTERNAL_EXPERIMENTAL, ALL_PARAMETERS
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
ROOT_PATH = dirname(abspath(__file__))

nsamples = 5000
burn_in = 1000
nchains = 2
acc_rate = 0.7
tol = 1e-8
mxsteps = 3e4


# save samples
PARAMETER_SAMP_PATH = ROOT_PATH + '/samples'
directory_name = 'nsamples_' + str(nsamples) + '_burn_in_' + str(burn_in) + '_acc_rate_' + str(acc_rate) + \
                 '_nchains_' + str(nchains)
directory_name = directory_name.replace('.', '_')
file_name = '2022_06_19_21_56_30_959087.nc'
data_file_location = os.path.join(PARAMETER_SAMP_PATH, directory_name, file_name)

samples = az.from_netcdf(data_file_location)

for i in range(nchains):
    plt.hist(samples.sample_stats.lp[i],alpha=0.5)
plt.show()
