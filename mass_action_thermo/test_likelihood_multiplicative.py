import pickle
from exp_data import *
from prior_constants import NORM_PRIOR_STD_RT_SINGLE_EXP,NORM_PRIOR_MEAN_SINGLE_EXP, NORM_PRIOR_STD_RT_ALL_EXP, \
    NORM_PRIOR_MEAN_ALL_EXP, LOG_UNIF_PRIOR_ALL_EXP, LOG_UNIF_G_EXT_INIT_PRIOR_PARAMETERS
import pdo_model_sympy.prior_constants as pdo_pr_constants
import pickle
import numpy as np
from likelihood_funcs_fwd import likelihood_fwd, likelihood_derivative_fwd
from likelihood_funcs_adj import likelihood_adj, likelihood_derivative_adj
from constants import *

PARAMETER_SAMP_PATH = '/home/aarcher/research/pdo-pathway-model/MCMC/output'
FILE_NAME = '/MCMC_results_data/mass_action/adaptive/preset_std/lambda_0,05_beta_0,1_burn_in_n_cov_2000/nsamples_300000/date_2022_03_19_15_02_31_500660_rank_0.pkl'

param_sample = NORM_PRIOR_MEAN_ALL_EXP.copy()
with open(PARAMETER_SAMP_PATH + FILE_NAME, 'rb') as f:
    postdraws = pickle.load(f)
    samples = postdraws['samples']
    burn_in_subset_samples = samples[int(2e4):]
    data_subset = burn_in_subset_samples[::600,:]
    param_mean = data_subset.mean(axis=0)
    param_mean_trans = np.matmul(pdo_pr_constants.NORM_PRIOR_STD_RT_ALL_EXP[:len(param_mean), :len(param_mean)].T, param_mean) + pdo_pr_constants.NORM_PRIOR_MEAN_ALL_EXP[
                                                                                                                :len(param_mean)]
param_sample[:(N_MODEL_PARAMETERS+4)] = param_mean_trans
param_sample[N_MODEL_PARAMETERS:(N_MODEL_PARAMETERS+4)] = np.log10(param_sample[N_MODEL_PARAMETERS:(N_MODEL_PARAMETERS+4)])
param_sample_copy = param_sample.copy()

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


