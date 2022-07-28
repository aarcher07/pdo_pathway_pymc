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

param_sample = NORM_PRIOR_MEAN_ALL_EXP.copy()[:(N_MODEL_PARAMETERS)]
param_sample_copy = param_sample.copy()

fwd_rtol = 1e-8
fwd_atol = 1e-8
bck_rtol = 1e-8
bck_atol = 1e-8
fwd_mxsteps = int(1e5)
bck_mxsteps = int(1e5)

likelihood_fwd(param_sample_copy[:(N_MODEL_PARAMETERS)], rtol=fwd_rtol, atol=fwd_atol, mxsteps=fwd_mxsteps)

time_start = time.time()
print(likelihood_fwd(param_sample_copy[:(N_MODEL_PARAMETERS)], rtol=fwd_rtol, atol=fwd_atol, mxsteps=fwd_mxsteps))
time_end = time.time()
print('fwd : '+ str((time_end - time_start)/60))

likelihood_adj(param_sample_copy[:(N_MODEL_PARAMETERS)], fwd_rtol=fwd_rtol, fwd_atol=fwd_atol, fwd_mxsteps=fwd_mxsteps)
time_start = time.time()
print(likelihood_adj(param_sample_copy[:(N_MODEL_PARAMETERS)], fwd_rtol=fwd_rtol, fwd_atol=fwd_atol, fwd_mxsteps=fwd_mxsteps))
time_end = time.time()
print('adj : '+ str((time_end - time_start)/60))

time_start = time.time()
lik_fwd = likelihood_derivative_fwd(param_sample_copy[:(N_MODEL_PARAMETERS)], rtol = fwd_rtol, atol=fwd_atol, mxsteps=fwd_mxsteps)
time_end = time.time()
print('fwd : '+ str((time_end - time_start)/60))

time_start = time.time()
lik_adj = likelihood_derivative_adj(param_sample_copy[:(N_MODEL_PARAMETERS)], fwd_rtol=fwd_rtol, fwd_atol=fwd_atol,
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