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

PLOT_SAMP_PATH = ROOT_PATH + '/plot_analysis'
plot_file_location = os.path.join(PLOT_SAMP_PATH, directory_name, file_name[:-3])
Path(plot_file_location).mkdir(parents=True, exist_ok=True)

# fig, ax = plt.subplots(2,nchains)
#
# for i in range(min(5,nchains)):
#     ax[i,0].hist(samples.sample_stats.lp[i],alpha=0.5)
#     ax[i, 0].set_title('Histogram of Log-Likelihood')
#     ax[i, 1].plot(list(range(len(samples.sample_stats.lp[i]))), samples.sample_stats.lp[i],alpha=0.5)
#     ax[i, 1].set_title('Trajectory of Log-Likelihood')
# fig.tight_layout()
# plt.savefig(os.path.join(plot_file_location, 'loglik_plot_individual.png'))
# plt.close()
#
# fig, ax = plt.subplots(1,2)
# for i in range(nchains):
#     ax[0].hist(samples.sample_stats.lp[i],alpha=0.1)
#     ax[0].set_title('Histogram of Log-Likelihood')
#     ax[1].plot(list(range(len(samples.sample_stats.lp[i]))), samples.sample_stats.lp[i],alpha=0.1)
#     ax[1].set_title('Trajectory of Log-Likelihood')
#
# ax[0].legend(['chain ' + str(i) for i in range(nchains)])
# ax[1].legend(['chain ' + str(i) for i in range(nchains)])
# plt.savefig(os.path.join(plot_file_location, 'loglik_plot_overlay.png'))
# plt.close()
# for exp_ind, gly_cond in enumerate([50, 60, 70, 80]):
#     fig, ax = plt.subplots(5, min(5, nchains))
#     for chain_ind in range(nchains):
#         dataarray = samples.posterior.to_dataframe().loc[[chain_ind]]
#         dataarray = dataarray[ALL_PARAMETERS]
#         lower, upper = LOG_UNIF_G_EXT_INIT_PRIOR_PARAMETERS["G_EXT_INIT_" + str(gly_cond)]
#         for jj in range(3):
#             ax[jj,chain_ind].scatter(TIME_SAMPLES_EXPANDED[gly_cond][::TIME_SPACING], DATA_SAMPLES[gly_cond][:, jj])
#
#         hpa_max = []
#         for j in range(0,nsamples,100):
#             param = dataarray.iloc[j,:].to_numpy()
#             param_sample = NORM_PRIOR_MEAN_SINGLE_EXP[gly_cond]
#             param_sample[:N_MODEL_PARAMETERS] = param[:N_MODEL_PARAMETERS]
#             g_ext_val = param[N_MODEL_PARAMETERS + exp_ind]
#             g_ext_val = lower + (upper - lower) / (1 + np.exp(-g_ext_val))
#             param_sample[N_MODEL_PARAMETERS] = g_ext_val
#
#             tvals = TIME_SAMPLES_EXPANDED[gly_cond] * HRS_TO_SECS
#
#             y0 = np.zeros((), dtype=problem.state_dtype)
#
#             y0['G_CYTO'] = 10 ** param_sample[PARAMETER_LIST.index('G_EXT_INIT')]
#             y0['H_CYTO'] = 0
#             y0['P_CYTO'] = INIT_CONDS_GLY_PDO_DCW[gly_cond][1]
#             y0['DHAB'] = 10 ** param_sample[PARAMETER_LIST.index('DHAB_INIT')]
#             y0['DHAB_C'] = 0
#             y0['DHAT'] = 10 ** param_sample[PARAMETER_LIST.index('DHAT_INIT')]
#             y0['DHAT_C'] = 0
#             y0['G_EXT'] = 10 ** param_sample[PARAMETER_LIST.index('G_EXT_INIT')]
#             y0['H_EXT'] = 0
#             y0['P_EXT'] = INIT_CONDS_GLY_PDO_DCW[gly_cond][1]
#             y0['dcw'] = 10 ** param_sample[PARAMETER_LIST.index('A')]
#
#             params_dict = {param_name: param_val for param_val, param_name in zip(param_sample, PARAMETER_LIST)}
#             # # We can also specify the parameters by name:
#             solver.set_params_dict(params_dict)
#             yout, grad_out, lambda_out = solver.make_output_buffers(tvals)
#
#             solver.solve_forward(t0=0, tvals=tvals, y0=y0, y_out=yout)
#             jj=0
#             for i, var in enumerate(VARIABLE_NAMES):
#                 if i in DATA_INDEX:
#                     ax[jj,chain_ind].plot(tvals / HRS_TO_SECS, yout.view(problem.state_dtype)[var], 'r', alpha=0.1)
#                     jj += 1
#                 elif var == 'H_CYTO':
#                     ax[3,chain_ind].plot(tvals / HRS_TO_SECS, yout.view(problem.state_dtype)[var], 'r', alpha=0.1)
#                     hpa_max.append(np.max(yout.view(problem.state_dtype)[var]))
#         ax[4, chain_ind].hist(hpa_max)
#         ax[0, chain_ind].set_title('Glycerol Time Series')
#         ax[1, chain_ind].set_title('1,3-PD Distribution')
#         ax[2, chain_ind].set_title('DCW Distribution')
#         ax[3, chain_ind].set_title('Cytosolic 3-HPA Time Series')
#         ax[4, chain_ind].set_title('Max 3-HPA Distribution')
#     plt.suptitle('Initial Glycerol ' + str(gly_cond) + ' g/L')
#     fig.tight_layout()
#     plt.savefig(os.path.join(plot_file_location, 'time_series_results_' + str(gly_cond) + '.png'))
#     plt.close()

az.plot_pair(samples,divergences=True,textsize=18)
plt.show()