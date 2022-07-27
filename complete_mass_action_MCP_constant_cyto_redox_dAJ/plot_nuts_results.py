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

from pathlib import Path
import numpy as np
from datetime import datetime
from scipy.stats import multivariate_normal
import pickle
from likelihood_funcs_adj import likelihood_adj, likelihood_derivative_adj
from os.path import dirname, abspath
ROOT_PATH = dirname(abspath(__file__))
from pandas.plotting import scatter_matrix
from plot_nuts_results_funcs import plot_loglik_individual, plot_loglik_overlay, plot_corr_scatter, plot_corr, \
    plot_time_series_distribution, joint_Keq_distribution, plot_trace

nsamples = int(1e3)
burn_in = int(2e3)
nchains = 2
acc_rate = 0.6
fwd_atol = 1e-8
fwd_rtol = 1e-8
bck_atol = 1e-4
bck_rtol = 1e-4
mxsteps = 1e5
init = 'adapt_diag'

# save samples
PARAMETER_SAMP_PATH = ROOT_PATH + '/samples' #TODO: remove _3HPA
directory_name = 'nsamples_' + str(nsamples) + '_burn_in_' + str(burn_in) + '_acc_rate_' + str(acc_rate) + \
                 '_nchains_' + str(nchains) + '_fwd_rtol_' + str(fwd_rtol) + '_fwd_atol_' + str(fwd_atol) + \
                 '_bck_rtol_' + str(bck_rtol) + '_bck_atol_' + str(bck_atol) + '_mxsteps_' + str(int(mxsteps)) \
                 + '_initialization_' + init
directory_name = directory_name.replace('.','_').replace('-','_').replace('+','_')
file_name = '2022_07_24_15_51_14_238436.nc'
data_file_location = os.path.join(PARAMETER_SAMP_PATH, directory_name, file_name)
samples = az.from_netcdf(data_file_location)


PLOT_SAMP_PATH = ROOT_PATH + '/plot_analysis'
plot_file_location = os.path.join(PLOT_SAMP_PATH, directory_name, file_name[:-3])
Path(plot_file_location).mkdir(parents=True, exist_ok=True)
# dataarray = samples.posterior.to_dataframe().loc[[0]]
# print(likelihood_adj(dataarray.iloc[-1,:].to_numpy()))
# print(dataarray.iloc[-1,:].to_dict())
# print(dataarray.iloc[-1,:].to_numpy())

# df = az.summary(samples)
# df.to_csv(os.path.join(plot_file_location,'summary_stats.csv'),sep = ' ')
# plot_trace(samples, plot_file_location)
# plot_loglik_individual(samples.sample_stats.lp, plot_file_location, nchains)
# plot_loglik_overlay(samples.sample_stats.lp, plot_file_location, nchains)
plot_time_series_distribution(samples, plot_file_location, nchains, fwd_atol, fwd_rtol, mxsteps)
# plot_corr(samples, plot_file_location, nchains)
# plot_corr_scatter(samples, plot_file_location, nchains)
# KeqDhaB = np.power(10,samples.posterior.KeqDhaB)
# KeqDhaT = np.power(10,samples.posterior.KeqDhaT)
# joint_Keq_distribution(KeqDhaB, KeqDhaT, plot_file_location, nchains)
