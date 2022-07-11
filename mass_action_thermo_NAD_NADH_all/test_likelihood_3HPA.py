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

param_sample = NORM_PRIOR_MEAN_ALL_EXP.copy()[:(N_MODEL_PARAMETERS)]
param_sample_copy = param_sample.copy()
param_sample_copy[:N_MODEL_PARAMETERS] = np.array([-3.41, -3.84, -4.9, -4.3,
                                                   2.56e-1, -5.22, 2.93, 7.30,
                                                   2.77, 8.27e-1, 1.16, 8.23e-1,
                                                   1.40, 7.28e-1, 1.48, 3.22,
                                                   1.65, 1.21, -3.14e-1, 1.88,
                                                   -2.15, 5.22e-1, -5.26e-1, -4.25,
                                                   -5.98e-1, 5.41e-1, -6.85e-1, 2.88,
                                                   5.48e-3, -2.49, -1.98, 8.34e-2,
                                                   -8.34e-2, 4.82e-1, 6.73e-1, -9e-2])
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

print(lik_fwd)
print(lik_adj)
lik_diff = lik_fwd - lik_adj
lik_rel_diff = np.abs(lik_diff)/np.abs(lik_fwd)
print(lik_diff)
print(lik_rel_diff)


