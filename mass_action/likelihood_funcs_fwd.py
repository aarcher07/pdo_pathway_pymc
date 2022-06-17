import sunode
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #for \text command
import numpy as np
from constants import *
import pickle
from exp_data import *
from prior_constants import NORM_PRIOR_STD_RT_SINGLE_EXP,NORM_PRIOR_MEAN_SINGLE_EXP, NORM_PRIOR_STD_RT_ALL_EXP, \
    NORM_PRIOR_MEAN_ALL_EXP, LOG_UNIF_PRIOR_ALL_EXP, LOG_UNIF_G_EXT_INIT_PRIOR_PARAMETERS
from formatting_constants import VARS_ALL_EXP_TO_TEX
import time
from rhs_funcs import RHS, lib, problem
import pdo_model_sympy.prior_constants as pdo_pr_constants

solver = sunode.solver.Solver(problem, solver='BDF', sens_mode='simultaneous')

def likelihood(param_vals):
    # set solver parameters
    lib.CVodeSStolerances(solver._ode, 1e-8, 1e-8)
    lib.CVodeSetMaxNumSteps(solver._ode, 10000)

    # initialize
    loglik = 0
    param_vals_copy = param_vals.copy()
    param_sample = np.zeros(N_UNKNOWN_PARAMETERS)

    gly_init_val = param_vals[N_MODEL_PARAMETERS:(N_MODEL_PARAMETERS+4)]
    for i,((lower,upper),gly_init) in enumerate(zip(LOG_UNIF_G_EXT_INIT_PRIOR_PARAMETERS.values(),gly_init_val)):
        param_vals_copy[N_MODEL_PARAMETERS + i] = lower + (upper - lower)/(1+np.exp(-gly_init))

    for exp_ind, gly_cond in enumerate([50,60,70,80]):
        param_sample[:N_MODEL_PARAMETERS] = param_vals_copy[:N_MODEL_PARAMETERS]
        param_sample[N_MODEL_PARAMETERS+0] = param_vals_copy[N_MODEL_PARAMETERS + exp_ind]
        param_sample[N_MODEL_PARAMETERS+1] = param_vals_copy[N_MODEL_PARAMETERS + 4 + exp_ind*N_DCW_PARAMETERS + 0]
        param_sample[N_MODEL_PARAMETERS+2] = param_vals_copy[N_MODEL_PARAMETERS + 4 + exp_ind*N_DCW_PARAMETERS + 1]
        param_sample[N_MODEL_PARAMETERS+3] = param_vals_copy[N_MODEL_PARAMETERS + 4 + exp_ind*N_DCW_PARAMETERS + 2]

        tvals = TIME_SAMPLES[gly_cond]*HRS_TO_SECS
        y0 = np.zeros((), dtype=problem.state_dtype)
        y0['G_CYTO'] = 10**param_sample[PARAMETER_LIST.index('G_EXT_INIT')]
        y0['H_CYTO'] = 0
        y0['P_CYTO'] = INIT_CONDS_GLY_PDO_DCW[gly_cond][1]
        y0['DHAB'] = 10**param_sample[PARAMETER_LIST.index('DHAB_INIT')]
        y0['DHAB_C'] = 0
        y0['DHAT'] = 10**param_sample[PARAMETER_LIST.index('DHAT_INIT')]
        y0['DHAT_C'] = 0
        y0['G_EXT'] = 10**param_sample[PARAMETER_LIST.index('G_EXT_INIT')]
        y0['H_EXT'] = 0
        y0['P_EXT'] = INIT_CONDS_GLY_PDO_DCW[gly_cond][1]
        y0['dcw'] =  10**param_sample[PARAMETER_LIST.index('A')]

        params_dict = { param_name:param_val for param_val,param_name in zip(param_sample, PARAMETER_LIST)}
        # # We can also specify the parameters by name:
        solver.set_params_dict(params_dict)

        yout, grad_out, lambda_out = solver.make_output_buffers(tvals)
        sens0 = np.zeros((19,11))
        sens0[PARAMETER_LIST.index('G_EXT_INIT'), VARIABLE_NAMES.index('G_CYTO')] = np.log(10)*(10**param_sample[PARAMETER_LIST.index('G_EXT_INIT')])
        sens0[PARAMETER_LIST.index('G_EXT_INIT'), VARIABLE_NAMES.index('G_EXT')] = np.log(10)*(10**param_sample[PARAMETER_LIST.index('G_EXT_INIT')])
        sens0[PARAMETER_LIST.index('DHAB_INIT'), VARIABLE_NAMES.index('DHAB')] = np.log(10)*(10**param_sample[PARAMETER_LIST.index('DHAB_INIT')])
        sens0[PARAMETER_LIST.index('DHAT_INIT'), VARIABLE_NAMES.index('DHAT')] = np.log(10)*(10**param_sample[PARAMETER_LIST.index('DHAT_INIT')])
        sens0[PARAMETER_LIST.index('A'), VARIABLE_NAMES.index('dcw')] = np.log(10)*(10**param_sample[PARAMETER_LIST.index('A')])

        solver.solve_forward(t0=0, tvals=tvals, y0=y0, y_out=yout)
        # jj=0
        # for i,var in enumerate(VARIABLE_NAMES):
        #     if i in DATA_INDEX:
        #         plt.plot(tvals / HRS_TO_SECS, yout.view(problem.state_dtype)[var])
        #         plt.scatter(tvals/HRS_TO_SECS, DATA_SAMPLES[gly_cond][:,jj])
        #         jj+=1
        #     plt.show()

        loglik += -0.5*(((DATA_SAMPLES[gly_cond]-yout[:,[7,9,10]])/np.array([15,15,0.1]))**2).sum()
    return loglik


def likelihood_derivative_adj(param_vals, tol=1e-8):
    # set solver parameters
    lib.CVodeSStolerances(solver._ode, tol, tol)
    lib.CVodeSetMaxNumSteps(solver._ode, 10000)

    # initialize
    lik_dev_params = np.zeros(N_MODEL_PARAMETERS + 4)
    param_vals_copy = param_vals.copy()
    param_sample = np.zeros(N_UNKNOWN_PARAMETERS)

    gly_init_val = param_vals[N_MODEL_PARAMETERS:(N_MODEL_PARAMETERS+4)]
    for i,((lower,upper),gly_init) in enumerate(zip(LOG_UNIF_G_EXT_INIT_PRIOR_PARAMETERS.values(),gly_init_val)):
        param_vals_copy[N_MODEL_PARAMETERS + i] = lower + (upper - lower)/(1+np.exp(-gly_init))
    time_tot = 0
    for exp_ind, gly_cond in enumerate([50,60,70,80]):
        param_sample[:N_MODEL_PARAMETERS] = param_vals_copy[:N_MODEL_PARAMETERS]
        param_sample[N_MODEL_PARAMETERS+0] = param_vals_copy[N_MODEL_PARAMETERS + exp_ind]
        param_sample[N_MODEL_PARAMETERS+1] = param_vals_copy[N_MODEL_PARAMETERS + 4 + exp_ind*N_DCW_PARAMETERS + 0]
        param_sample[N_MODEL_PARAMETERS+2] = param_vals_copy[N_MODEL_PARAMETERS + 4 + exp_ind*N_DCW_PARAMETERS + 1]
        param_sample[N_MODEL_PARAMETERS+3] = param_vals_copy[N_MODEL_PARAMETERS + 4 + exp_ind*N_DCW_PARAMETERS + 2]

        tvals = TIME_SAMPLES[gly_cond] * HRS_TO_SECS

        y0 = np.zeros((), dtype=problem.state_dtype)

        y0['G_CYTO'] = 10 ** param_sample[PARAMETER_LIST.index('G_EXT_INIT')]
        y0['H_CYTO'] = 0
        y0['P_CYTO'] = INIT_CONDS_GLY_PDO_DCW[gly_cond][1]
        y0['DHAB'] = 10 ** param_sample[PARAMETER_LIST.index('DHAB_INIT')]
        y0['DHAB_C'] = 0
        y0['DHAT'] = 10 ** param_sample[PARAMETER_LIST.index('DHAT_INIT')]
        y0['DHAT_C'] = 0
        y0['G_EXT'] = 10 ** param_sample[PARAMETER_LIST.index('G_EXT_INIT')]
        y0['H_EXT'] = 0
        y0['P_EXT'] = INIT_CONDS_GLY_PDO_DCW[gly_cond][1]
        y0['dcw'] = 10 ** param_sample[PARAMETER_LIST.index('A')]

        params_dict = {param_name: param_val for param_val, param_name in zip(param_sample, PARAMETER_LIST)}
        # # We can also specify the parameters by name:
        solver.set_params_dict(params_dict)
        yout, sens_out = solver.make_output_buffers(tvals)
        # initial sensitivities
        sens0 = np.zeros((len(DEV_PARAMETERS_LIST), len(VARIABLE_NAMES)))
        print(sens0.shape)
        sens0[PARAMETER_LIST.index('G_EXT_INIT'), VARIABLE_NAMES.index('G_CYTO')] = np.log(10) * (
                    10 ** param_sample[PARAMETER_LIST.index('G_EXT_INIT')])
        sens0[PARAMETER_LIST.index('G_EXT_INIT'), VARIABLE_NAMES.index('G_EXT')] = np.log(10) * (
                    10 ** param_sample[PARAMETER_LIST.index('G_EXT_INIT')])
        sens0[PARAMETER_LIST.index('DHAB_INIT'), VARIABLE_NAMES.index('DHAB')] = np.log(10) * (
                    10 ** param_sample[PARAMETER_LIST.index('DHAB_INIT')])
        sens0[PARAMETER_LIST.index('DHAT_INIT'), VARIABLE_NAMES.index('DHAT')] = np.log(10) * (
                    10 ** param_sample[PARAMETER_LIST.index('DHAT_INIT')])
        # sens0[PARAMETER_LIST.index('A'), VARIABLE_NAMES.index('dcw')] = np.log(10) * (
        #             10 ** param_sample[PARAMETER_LIST.index('A')])

        time_start = time.time()
        solver.solve(t0=0, tvals=tvals, y0=y0, y_out=yout, sens0=sens0, sens_out=sens_out)
        time_end = time.time()
        time_tot += (time_end - time_start) / 60

        # We can convert the solution to an xarray Dataset
        lik_dev = (DATA_SAMPLES[gly_cond] - yout[:, DATA_INDEX]) / (np.array([15, 15, 0.1]) ** 2)

        lik_dev_zeros = np.zeros_like(sens_out[:, 0, :])
        lik_dev_zeros[:, DATA_INDEX] = lik_dev

        jj=0
        for i,var in enumerate(VARIABLE_NAMES):
            if i in DATA_INDEX:
                plt.plot(tvals / HRS_TO_SECS, yout.view(problem.state_dtype)[var])
                plt.scatter(tvals/HRS_TO_SECS, DATA_SAMPLES[gly_cond][:,jj])
                jj+=1
            plt.show()
        for j, param in enumerate(DEV_PARAMETERS_LIST):
            lik_dev_param = (lik_dev_zeros * sens_out[:, j, :]).sum()
            if param == 'G_EXT_INIT':
                param_gly_name = param + '_' + str(gly_cond)
                lower, upper = LOG_UNIF_G_EXT_INIT_PRIOR_PARAMETERS[param_gly_name]
                gly_val = param_sample[N_MODEL_PARAMETERS]
                dGdtildeG = np.exp(-gly_val)*(lower - upper)/(1 + np.exp(-gly_val))**2
                lik_dev_params[N_MODEL_PARAMETERS + exp_ind] += lik_dev_param*dGdtildeG
            elif param in ['L', 'k', 'A']:

                jj = ['L', 'k', 'A'].index(param)
                lik_dev_params[N_MODEL_PARAMETERS + 4 + exp_ind * N_DCW_PARAMETERS + jj] += lik_dev_param
            else:
                lik_dev_params[j] += lik_dev_param

    return lik_dev_params

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
print(param_sample_copy)
print(likelihood_derivative_adj(param_sample_copy))
