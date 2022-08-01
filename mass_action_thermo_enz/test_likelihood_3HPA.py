import pickle
from exp_data import *
from prior_constants import NORM_PRIOR_STD_RT_SINGLE_EXP,NORM_PRIOR_MEAN_SINGLE_EXP, NORM_PRIOR_STD_RT_ALL_EXP, \
    NORM_PRIOR_MEAN_ALL_EXP, LOG_UNIF_PRIOR_ALL_EXP, LOG_UNIF_G_EXT_INIT_PRIOR_PARAMETERS
import pdo_model_sympy.prior_constants as pdo_pr_constants
import pickle
import numpy as np
import time
from likelihood_funcs_fwd_3HPA import likelihood_fwd, likelihood_derivative_fwd
from likelihood_funcs_adj_3HPA import likelihood_adj, likelihood_derivative_adj
from constants import *
from os.path import dirname, abspath
import arviz as az
import matplotlib.pyplot as plt
import os

ROOT_PATH = dirname(dirname(abspath(__file__)))

# nsamples = int(3e3)
# burn_in = int(3e3)
# nchains = 2
# acc_rate = 0.8
# atol = 1e-8
# rtol = 1e-8
# mxsteps = 1e5
# init = 'jitter+adapt_diag'

# save samples
# PARAMETER_SAMP_PATH = ROOT_PATH + '/mass_action_thermo/samples' #TODO: remove _3HPA
# directory_name = 'nsamples_' + str(nsamples) + '_burn_in_' + str(burn_in) + '_acc_rate_' + str(acc_rate) + \
#                  '_nchains_' + str(nchains) + '_atol_' + str(atol) + '_rtol_' + str(rtol) + '_mxsteps_' +\
#                  str(int(mxsteps))  + '_initialization_' + init
# directory_name = directory_name.replace('.','_').replace('-','_').replace('+','_')
# file_name = '2022_07_04_15_24_56_643933.nc'
# data_file_location = os.path.join(PARAMETER_SAMP_PATH, directory_name, file_name)
# samples = az.from_netcdf(data_file_location)
# dataarray = samples.posterior.to_dataframe().loc[[0]]
# param_sample = dataarray.iloc[-1,:].to_numpy()
param_sample = NORM_PRIOR_MEAN_ALL_EXP.copy()
param_sample_copy = np.zeros(N_MODEL_PARAMETERS + 4*3)
param_sample_copy[:N_MODEL_PARAMETERS] = param_sample[:N_MODEL_PARAMETERS]
param_sample_copy[PARAMETER_LIST.index('kcatfMetab')] = param_sample[PARAMETER_LIST.index('kcatfMetab')] - 1
param_sample_copy[N_MODEL_PARAMETERS + 4*INIT_CONSTANTS.index('DHAB_INIT') : (N_MODEL_PARAMETERS + 4*INIT_CONSTANTS.index('DHAB_INIT') + 4)] = param_sample[N_MODEL_PARAMETERS + INIT_CONSTANTS.index('DHAB_INIT')]
param_sample_copy[N_MODEL_PARAMETERS + 4*INIT_CONSTANTS.index('DHAT_INIT') : (N_MODEL_PARAMETERS + 4*INIT_CONSTANTS.index('DHAT_INIT') + 4)] = param_sample[N_MODEL_PARAMETERS + INIT_CONSTANTS.index('DHAT_INIT')]
param_sample_copy[N_MODEL_PARAMETERS + 4*INIT_CONSTANTS.index('E0_Metab') : (N_MODEL_PARAMETERS + 4*INIT_CONSTANTS.index('E0_Metab') + 4)] = 1
print(param_sample_copy)
# param_sample = NORM_PRIOR_MEAN_ALL_EXP.copy()[:(N_MODEL_PARAMETERS)]
# param_sample_copy = param_sample.copy()

fwd_rtol = 1e-8
fwd_atol = 1e-8
bck_rtol = 1e-4
bck_atol = 1e-4
fwd_mxsteps = int(1e5)
bck_mxsteps = int(1e5)

likelihood_fwd(param_sample_copy, rtol=fwd_rtol, atol=fwd_atol, mxsteps=fwd_mxsteps)

time_start = time.time()
print(likelihood_fwd(param_sample_copy, rtol=fwd_rtol, atol=fwd_atol, mxsteps=fwd_mxsteps))
time_end = time.time()
print('fwd : '+ str((time_end - time_start)/60))

likelihood_adj(param_sample_copy, fwd_rtol=fwd_rtol, fwd_atol=fwd_atol, fwd_mxsteps=fwd_mxsteps)
time_start = time.time()
print(likelihood_adj(param_sample_copy, fwd_rtol=fwd_rtol, fwd_atol=fwd_atol, fwd_mxsteps=fwd_mxsteps))
time_end = time.time()
print('adj : '+ str((time_end - time_start)/60))

time_start = time.time()
lik_fwd = likelihood_derivative_fwd(param_sample_copy, rtol = fwd_rtol, atol=fwd_atol, mxsteps=fwd_mxsteps)
time_end = time.time()
print('fwd : '+ str((time_end - time_start)/60))

time_start = time.time()
lik_adj = likelihood_derivative_adj(param_sample_copy, fwd_rtol=fwd_rtol, fwd_atol=fwd_atol,
                                    bck_rtol=bck_rtol, bck_atol=bck_rtol, bck_mxsteps=bck_mxsteps,
                                    fwd_mxsteps=fwd_mxsteps)
time_end = time.time()
print('adj : '+ str((time_end - time_start)/60))
print('fwd : ' + str(lik_fwd))
print('adj : ' + str(lik_adj))
lik_diff = lik_fwd - lik_adj
lik_rel_diff = np.abs(lik_diff)/np.abs(lik_fwd)
print(lik_diff)
print(lik_rel_diff)

indices_sorted = np.argsort(lik_rel_diff)[::-1]
topk =  5
print('abs diff: ' + str(lik_diff[indices_sorted[:topk]]))
print('rel diff: ' + str(lik_rel_diff[indices_sorted[:topk]]))
print('fwd dev: ' + str(lik_fwd[indices_sorted[:topk]]))
print('adj dev: ' + str(lik_adj[indices_sorted[:topk]]))
# print('param: ' + str(np.array(DEV_PARAMETERS_LIST)[indices_sorted[:topk]]))


