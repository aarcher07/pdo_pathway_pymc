import pickle
from exp_data import *
from prior_constants import NORM_PRIOR_STD_RT_SINGLE_EXP,NORM_PRIOR_MEAN_SINGLE_EXP, NORM_PRIOR_STD_RT_ALL_EXP, \
    NORM_PRIOR_MEAN_ALL_EXP, LOG_UNIF_PRIOR_ALL_EXP, LOG_UNIF_G_EXT_INIT_PRIOR_PARAMETERS
import pickle
import numpy as np
from likelihood_funcs_fwd import likelihood_fwd, likelihood_derivative_fwd
from likelihood_funcs_adj import likelihood_adj, likelihood_derivative_adj
from constants import *

param_sample = NORM_PRIOR_MEAN_ALL_EXP.copy()[:(N_MODEL_PARAMETERS+4)]
param_sample_copy = param_sample.copy()
gly_init_val = param_sample[N_MODEL_PARAMETERS:(N_MODEL_PARAMETERS + 4)]
for i, ((lower, upper), gly_init) in enumerate(zip(LOG_UNIF_G_EXT_INIT_PRIOR_PARAMETERS.values(), gly_init_val)):
    param_sample_copy[N_MODEL_PARAMETERS + i] = np.log((gly_init-lower)/(upper - gly_init))

gly_init_val = param_sample[N_MODEL_PARAMETERS:(N_MODEL_PARAMETERS + 4)]
for i, ((lower, upper), gly_init) in enumerate(zip(LOG_UNIF_G_EXT_INIT_PRIOR_PARAMETERS.values(), gly_init_val)):
    param_sample_copy[N_MODEL_PARAMETERS + i] = np.log((gly_init-lower)/(upper - gly_init))

print(likelihood_fwd(param_sample_copy[:(N_MODEL_PARAMETERS+4)]))
print(likelihood_adj(param_sample_copy[:(N_MODEL_PARAMETERS+4)]))

lik_fwd = likelihood_derivative_fwd(param_sample_copy[:(N_MODEL_PARAMETERS + 4)])
lik_adj = likelihood_derivative_adj(param_sample_copy[:(N_MODEL_PARAMETERS + 4)])
print(lik_fwd)
print(lik_adj)
lik_diff = lik_fwd - lik_adj
lik_rel_diff = np.abs(lik_diff)/np.abs(lik_fwd)
print(lik_rel_diff)


