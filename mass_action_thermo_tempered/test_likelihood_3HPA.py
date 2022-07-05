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

nsamples = int(3e3)
burn_in = int(3e3)
nchains = 2
acc_rate = 0.8
atol = 1e-8
rtol = 1e-8
mxsteps = 1e5
init = 'jitter+adapt_diag'

# save samples
PARAMETER_SAMP_PATH = ROOT_PATH + '/mass_action_thermo/samples' #TODO: remove _3HPA
directory_name = 'nsamples_' + str(nsamples) + '_burn_in_' + str(burn_in) + '_acc_rate_' + str(acc_rate) + \
                 '_nchains_' + str(nchains) + '_atol_' + str(atol) + '_rtol_' + str(rtol) + '_mxsteps_' +\
                 str(int(mxsteps))  + '_initialization_' + init
directory_name = directory_name.replace('.','_').replace('-','_').replace('+','_')
file_name = '2022_07_04_15_24_56_643933.nc'
data_file_location = os.path.join(PARAMETER_SAMP_PATH, directory_name, file_name)
samples = az.from_netcdf(data_file_location)
dataarray = samples.posterior.to_dataframe().loc[[0]]
param_sample_copy = dataarray.iloc[-1,:].to_numpy()

# param_sample = NORM_PRIOR_MEAN_ALL_EXP.copy()[:(N_MODEL_PARAMETERS)]
# param_sample_copy = param_sample.copy()

# gly_init_val = param_sample[N_MODEL_PARAMETERS:(N_MODEL_PARAMETERS + 4)]
# for i, ((lower, upper), gly_init) in enumerate(zip(LOG_UNIF_G_EXT_INIT_PRIOR_PARAMETERS.values(), gly_init_val)):
#     param_sample_copy[N_MODEL_PARAMETERS + i] = np.log((gly_init-lower)/(upper - gly_init))

atol = 1e-10
rtol = 1e-10
mxsteps = int(1e5)
print('fwd')
time_start = time.time()
print(likelihood_fwd(param_sample_copy[:(N_MODEL_PARAMETERS)],atol = atol, rtol=rtol, mxsteps=mxsteps))
time_end = time.time()
print((time_end - time_start)/60)

print('adj')
time_start = time.time()
print(likelihood_adj(param_sample_copy[:(N_MODEL_PARAMETERS)],atol = atol, rtol=rtol, mxsteps=mxsteps))
time_end = time.time()
print((time_end - time_start)/60)

time_start = time.time()
lik_fwd = likelihood_derivative_fwd(param_sample_copy[:(N_MODEL_PARAMETERS)],atol = atol, rtol=rtol, mxsteps=mxsteps)
time_end = time.time()
print("fwd: " + str((time_end - time_start)/60))

time_start = time.time()
lik_adj = likelihood_derivative_adj(param_sample_copy[:(N_MODEL_PARAMETERS)],atol = atol, rtol=rtol, mxsteps=mxsteps)
time_end = time.time()
print("adj: " + str((time_end - time_start)/60))

print(lik_fwd)
print(lik_adj)
lik_diff = lik_fwd - lik_adj
lik_rel_diff = np.abs(lik_diff)/np.abs(lik_fwd)
print(lik_diff)
print(lik_rel_diff)


