import sunode
import matplotlib.pyplot as plt
from exp_data import INIT_CONDS_GLY_PDO_DCW, DATA_SAMPLES
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #for \text command
import numpy as np
from constants import *
from prior_constants import NORM_PRIOR_STD_RT_SINGLE_EXP,NORM_PRIOR_MEAN_SINGLE_EXP, NORM_PRIOR_STD_RT_ALL_EXP, \
    NORM_PRIOR_MEAN_ALL_EXP, LOG_UNIF_PRIOR_ALL_EXP, LOG_UNIF_G_EXT_INIT_PRIOR_PARAMETERS
from formatting_constants import VARS_ALL_EXP_TO_TEX
import time
from rhs_funcs import RHS, lib, problem

solver_no_sens = sunode.solver.Solver(problem, solver='BDF', sens_mode=None)
solver = sunode.solver.Solver(problem, solver='BDF', sens_mode='simultaneous')

def likelihood_fwd(param_vals, atol=1e-8, rtol=1e-8, mxsteps=int(1e4)):

    # set solver parameters
    lib.CVodeSStolerances(solver_no_sens._ode, atol, rtol)
    lib.CVodeSetMaxNumSteps(solver_no_sens._ode, mxsteps)

    # initialize
    loglik = 0
    param_vals_copy = param_vals.copy()

    # gly_init_val = param_vals[N_MODEL_PARAMETERS:(N_MODEL_PARAMETERS+4)]
    # for i,((lower,upper),gly_init) in enumerate(zip(LOG_UNIF_G_EXT_INIT_PRIOR_PARAMETERS.values(),gly_init_val)):
    #     param_vals_copy[N_MODEL_PARAMETERS + i] = lower + (upper - lower)/(1+np.exp(-gly_init))

    for exp_ind, gly_cond in enumerate([50,60,70,80]):
        param_sample = NORM_PRIOR_MEAN_SINGLE_EXP[gly_cond].copy()
        param_sample[:N_MODEL_PARAMETERS] = param_vals_copy[:N_MODEL_PARAMETERS]
        param_sample[PARAMETER_LIST.index('DHAB_INIT')] = param_vals_copy[N_MODEL_PARAMETERS +
                                                                          4 * INIT_CONSTANTS.index('DHAB_INIT')
                                                                          + exp_ind]
        param_sample[PARAMETER_LIST.index('DHAT_INIT')] = param_vals_copy[N_MODEL_PARAMETERS +
                                                                          4 * INIT_CONSTANTS.index('DHAT_INIT')
                                                                          + exp_ind]
        param_sample[PARAMETER_LIST.index('E0_Metab')] = param_vals_copy[N_MODEL_PARAMETERS +
                                                                         4 * INIT_CONSTANTS.index('E0_Metab')
                                                                         + exp_ind]
        # param_sample[N_MODEL_PARAMETERS+0] = param_vals_copy[N_MODEL_PARAMETERS + exp_ind]
        # param_sample[N_MODEL_PARAMETERS+1] = param_vals_copy[N_MODEL_PARAMETERS + 4 + exp_ind*N_DCW_PARAMETERS + 0]
        # param_sample[N_MODEL_PARAMETERS+2] = param_vals_copy[N_MODEL_PARAMETERS + 4 + exp_ind*N_DCW_PARAMETERS + 1]
        # param_sample[N_MODEL_PARAMETERS+3] = param_vals_copy[N_MODEL_PARAMETERS + 4 + exp_ind*N_DCW_PARAMETERS + 2]

        tvals = TIME_SAMPLES_EXPANDED_HPA[gly_cond]*HRS_TO_SECS
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
        solver_no_sens.set_params_dict(params_dict)

        yout = solver_no_sens.make_output_buffers(tvals)

        try:
            solver_no_sens.solve(t0=0, tvals=tvals, y0=y0, y_out=yout)
            # jj=0
            # for i,var in enumerate(VARIABLE_NAMES):
            #     if i in DATA_INDEX:
            #         plt.plot(tvals / HRS_TO_SECS, yout.view(problem.state_dtype)[var])
            #         plt.scatter(tvals/HRS_TO_SECS, DATA_SAMPLES[gly_cond][:,jj])
            #         jj+=1
            #     plt.show()
            cyto_hpa_max = np.max(yout[:, VARIABLE_NAMES.index('H_CYTO')])
            loglik += -0.5*(((DATA_SAMPLES[gly_cond]-yout[::TIME_SPACING_HPA,DATA_INDEX])/np.array([15,15,0.1]))**2).sum()\
                      - 0.5*cyto_hpa_max**2
        except sunode.solver.SolverError:
            loglik += np.nan
    return loglik


def likelihood_derivative_fwd(param_vals, atol=1e-8, rtol=1e-8, mxsteps=int(1e4)):

    # set solver parameters
    lib.CVodeSStolerances(solver._ode, atol, rtol)
    lib.CVodeSetMaxNumSteps(solver._ode, mxsteps)

    # initialize
    lik_dev_params = np.zeros_like(param_vals)
    param_vals_copy = param_vals.copy()

    # gly_init_val = param_vals[N_MODEL_PARAMETERS:(N_MODEL_PARAMETERS+4)]
    # for i,((lower,upper),gly_init) in enumerate(zip(LOG_UNIF_G_EXT_INIT_PRIOR_PARAMETERS.values(),gly_init_val)):
    #     param_vals_copy[N_MODEL_PARAMETERS + i] = lower + (upper - lower)/(1+np.exp(-gly_init))
    time_tot = 0
    for exp_ind, gly_cond in enumerate([50,60,70,80]):
        param_sample = NORM_PRIOR_MEAN_SINGLE_EXP[gly_cond].copy()
        param_sample[:N_MODEL_PARAMETERS] = param_vals_copy[:N_MODEL_PARAMETERS]
        param_sample[PARAMETER_LIST.index('DHAB_INIT')] = param_vals_copy[N_MODEL_PARAMETERS +
                                                                          4 * INIT_CONSTANTS.index('DHAB_INIT')
                                                                          + exp_ind]
        param_sample[PARAMETER_LIST.index('DHAT_INIT')] = param_vals_copy[N_MODEL_PARAMETERS +
                                                                          4 * INIT_CONSTANTS.index('DHAT_INIT')
                                                                          + exp_ind]
        param_sample[PARAMETER_LIST.index('E0_Metab')] = param_vals_copy[N_MODEL_PARAMETERS +
                                                                         4 * INIT_CONSTANTS.index('E0_Metab')
                                                                         + exp_ind]
        # param_sample[N_MODEL_PARAMETERS+1] = param_vals_copy[N_MODEL_PARAMETERS + 4 + exp_ind*N_DCW_PARAMETERS + 0]
        # param_sample[N_MODEL_PARAMETERS+2] = param_vals_copy[N_MODEL_PARAMETERS + 4 + exp_ind*N_DCW_PARAMETERS + 1]
        # param_sample[N_MODEL_PARAMETERS+3] = param_vals_copy[N_MODEL_PARAMETERS + 4 + exp_ind*N_DCW_PARAMETERS + 2]

        tvals = TIME_SAMPLES_EXPANDED_HPA[gly_cond] * HRS_TO_SECS

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
        # print(sens0.shape)
        # sens0[PARAMETER_LIST.index('G_EXT_INIT'), VARIABLE_NAMES.index('G_CYTO')] = np.log(10) * (
        #             10 ** param_sample[PARAMETER_LIST.index('G_EXT_INIT')])
        # sens0[PARAMETER_LIST.index('G_EXT_INIT'), VARIABLE_NAMES.index('G_EXT')] = np.log(10) * (
        #             10 ** param_sample[PARAMETER_LIST.index('G_EXT_INIT')])
        sens0[PARAMETER_LIST.index('DHAB_INIT'), VARIABLE_NAMES.index('DHAB')] = np.log(10) * (
                    10 ** param_sample[PARAMETER_LIST.index('DHAB_INIT')])
        sens0[PARAMETER_LIST.index('DHAT_INIT'), VARIABLE_NAMES.index('DHAT')] = np.log(10) * (
                    10 ** param_sample[PARAMETER_LIST.index('DHAT_INIT')])
        # sens0[PARAMETER_LIST.index('A'), VARIABLE_NAMES.index('dcw')] = np.log(10) * (
        #             10 ** param_sample[PARAMETER_LIST.index('A')])

        try:
            solver.solve(t0=0, tvals=tvals, y0=y0, y_out=yout, sens0=sens0, sens_out=sens_out)
            # jj=0
            # for i,var in enumerate(VARIABLE_NAMES):
            #     if i in DATA_INDEX:
            #         plt.plot(tvals / HRS_TO_SECS, yout.view(problem.state_dtype)[var])
            #         plt.scatter(tvals/HRS_TO_SECS, DATA_SAMPLES[gly_cond][:,jj])
            #         jj+=1
            #     plt.show()
        except sunode.solver.SolverError:
            sens_out[:] += np.nan
        # We can convert the solution to an xarray Dataset
        lik_dev = (DATA_SAMPLES[gly_cond] - yout[::TIME_SPACING_HPA, DATA_INDEX]) / (np.array([15, 15, 0.1]) ** 2)

        lik_dev_zeros = np.zeros_like(sens_out[:, 0, :])
        lik_dev_zeros[::TIME_SPACING_HPA, DATA_INDEX] = lik_dev
        cyto_hpa_arg_max = np.argmax(yout[:, VARIABLE_NAMES.index('H_CYTO')])
        lik_dev_zeros[cyto_hpa_arg_max, VARIABLE_NAMES.index('H_CYTO')] = -np.max(
            yout[:, VARIABLE_NAMES.index('H_CYTO')])

        # compute gradient
        for j, param in enumerate(DEV_PARAMETERS_LIST):
            lik_dev_param = (lik_dev_zeros * sens_out[:, j, :]).sum()
            if param in INIT_CONSTANTS[:-1]:
                lik_dev_params[N_MODEL_PARAMETERS + 4*INIT_CONSTANTS.index(param) + exp_ind] += lik_dev_param
            if param == 'G_EXT_INIT':
                param_gly_name = param + '_' + str(gly_cond)
                lower, upper = LOG_UNIF_G_EXT_INIT_PRIOR_PARAMETERS[param_gly_name]
                gly_val = param_sample[N_MODEL_PARAMETERS]
                dGdtildeG = np.exp(-gly_val)*(lower - upper)/(1 + np.exp(-gly_val))**2
                lik_dev_params[N_MODEL_PARAMETERS + 4*INIT_CONSTANTS.index(param) + exp_ind] += lik_dev_param*dGdtildeG
            elif param in ['L', 'k', 'A']:

                jj = ['L', 'k', 'A'].index(param)
                lik_dev_params[N_MODEL_PARAMETERS + 4*len(INIT_CONSTANTS) + exp_ind * N_DCW_PARAMETERS + jj] += lik_dev_param
            else:
                lik_dev_params[j] += lik_dev_param

    return lik_dev_params

