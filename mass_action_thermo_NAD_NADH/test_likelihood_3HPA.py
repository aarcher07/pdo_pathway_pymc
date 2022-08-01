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


# param_sample = [-4.24731044, -4.55408872, -3.88537434,  2.6493212 ,  1.45793309  ,3.00148201,
#   7.57637432, -1.90825775 , 0.78338796,  0.51753279,  2.78512648, -1.7019693,
#   3.02025902 , 1.53522127 , 3.97437867 ,-2.54850616, -2.2058919 , -1.9171252,
#  -1.21556906 , 1.06626333 , 0.75083318, -1.01458699 ,-0.45768735, -0.38396804,
#  -0.09160343, -1.04471706]

param_sample = NORM_PRIOR_MEAN_ALL_EXP.copy()[:(N_MODEL_PARAMETERS)]
param_sample_copy = param_sample.copy()

# gly_init_val = param_sample[N_MODEL_PARAMETERS:(N_MODEL_PARAMETERS + 4)]
# for i, ((lower, upper), gly_init) in enumerate(zip(LOG_UNIF_G_EXT_INIT_PRIOR_PARAMETERS.values(), gly_init_val)):
#     param_sample_copy[N_MODEL_PARAMETERS + i] = np.log((gly_init-lower)/(upper - gly_init))

fwd_rtol = 1e-8
fwd_atol = 1e-8
bck_rtol = 1e-4
bck_atol = 1e-4
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

indices_sorted = np.argsort(lik_rel_diff)[::-1]
topk =  5
print('abs diff: ' + str(lik_diff[indices_sorted[:topk]]))
print('rel diff: ' + str(lik_rel_diff[indices_sorted[:topk]]))
print('fwd dev: ' + str(lik_fwd[indices_sorted[:topk]]))
print('adj dev: ' + str(lik_adj[indices_sorted[:topk]]))
print('param: ' + str(np.array(DEV_PARAMETERS_LIST)[indices_sorted[:topk]]))
