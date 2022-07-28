import pickle
from exp_data import *
from prior_constants import NORM_PRIOR_STD_RT_SINGLE_EXP,NORM_PRIOR_MEAN_SINGLE_EXP, NORM_PRIOR_STD_RT_ALL_EXP, \
    NORM_PRIOR_MEAN_ALL_EXP, LOG_UNIF_PRIOR_ALL_EXP, LOG_UNIF_G_EXT_INIT_PRIOR_PARAMETERS
import pdo_model_sympy.prior_constants as pdo_pr_constants
import pickle
import numpy as np
import time
from likelihood_funcs_fwd import likelihood_fwd, likelihood_derivative_fwd
from likelihood_funcs_adj import likelihood_adj, likelihood_derivative_adj
from constants import *

param_sample = NORM_PRIOR_MEAN_ALL_EXP.copy()[:(N_MODEL_PARAMETERS+4)]
param_sample_copy = param_sample.copy()
lik_dev_params = np.zeros((N_MODEL_PARAMETERS + 4,))

gly_init_val = param_sample[N_MODEL_PARAMETERS:(N_MODEL_PARAMETERS + 4)]
for i, ((lower, upper), gly_init) in enumerate(zip(LOG_UNIF_G_EXT_INIT_PRIOR_PARAMETERS.values(), gly_init_val)):
    param_sample_copy[N_MODEL_PARAMETERS + i] = np.log((gly_init-lower)/(upper - gly_init))

atol = 1e-8
rtol = 1e-8
mxsteps = int(1e5)
time_start = time.time()
print(likelihood_fwd(param_sample_copy[:(N_MODEL_PARAMETERS)],atol = atol, rtol=rtol, mxsteps=mxsteps))
time_end = time.time()
print('fwd : '+ str((time_end - time_start)/60))

time_start = time.time()
print(likelihood_adj(param_sample_copy[:(N_MODEL_PARAMETERS)],atol = atol, rtol=rtol, mxsteps=mxsteps))
time_end = time.time()
print('adj : '+ str((time_end - time_start)/60))

time_start = time.time()
lik_fwd = likelihood_derivative_fwd(param_sample_copy[:(N_MODEL_PARAMETERS)],atol = atol, rtol=rtol, mxsteps=mxsteps)
time_end = time.time()
print('fwd : '+ str((time_end - time_start)/60))

time_start = time.time()
lik_adj = likelihood_derivative_adj(param_sample_copy[:(N_MODEL_PARAMETERS)],atol = atol, rtol=rtol, mxsteps=mxsteps)
time_end = time.time()
print('adj : '+ str((time_end - time_start)/60))

print('fwd : ' + str(lik_fwd))
print('adj : ' + str(lik_adj))
lik_diff = lik_fwd - lik_adj
lik_rel_diff = np.abs(lik_diff)/np.abs(lik_fwd)
print(lik_diff)
print(lik_rel_diff)


