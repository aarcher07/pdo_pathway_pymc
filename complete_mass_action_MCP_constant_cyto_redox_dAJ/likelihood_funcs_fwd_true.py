import numpy as np
from constants import *
import pickle
from exp_data_13pd import *
from prior_constants import *
import time
from rhs_dAJ import RHS_dAJ, problem_dAJ
from scipy.constants import Avogadro
import sunode
import matplotlib.pyplot as plt

lib = sunode._cvodes.lib

solver_dAJ_no_sens = sunode.solver.Solver(problem_dAJ, solver='BDF', sens_mode=None)
# solver_WT_no_sens  = sunode.solver.Solver(problem_WT, solver='BDF', sens_mode=None)
# solver_dD_no_sens  = sunode.solver.Solver(problem_dD, solver='BDF', sens_mode=None)

solver_dAJ = sunode.solver.Solver(problem_dAJ, solver='BDF', sens_mode='simultaneous')
# solver_WT = sunode.solver.Solver(problem_WT, solver='BDF', sens_mode='simultaneous')
# solver_dD = sunode.solver.Solver(problem_dD, solver='BDF', sens_mode='simultaneous')

def likelihood_fwd_true(params, fwd_rtol = 1e-8, fwd_atol=1e-8, fwd_mxsteps=int(1e4)):
    tvals = TIME_SAMPLES*HRS_TO_SECS
    # non_enz_model_params = params[:-4]
    # cofactor_params = params[-4:]
    non_enz_model_params = params[:-8]
    enz_params_WT = params[-8:-4]
    enz_params_dAJ = params[-4:]

    lik = 0
    exp_cond = 'dAJ-L'#['WT-L', 'dAJ-L', 'dD-L', 'dP-L']:
    # set solver
    solver = solver_dAJ_no_sens
    problem = problem_dAJ
    y0 = np.zeros((), dtype=problem.state_dtype)
    lib.CVodeSStolerances(solver._ode, fwd_rtol, fwd_atol)
    lib.CVodeSetMaxNumSteps(solver._ode, fwd_mxsteps)

    for var in VARIABLE_NAMES:
        y0[var] = 0

    # set initial conditions
    y0['G_EXT'] = TIME_SERIES_MEAN[exp_cond]['glycerol'][0]
    y0['G_CYTO'] = TIME_SERIES_MEAN[exp_cond]['glycerol'][0]
    y0['G_MCP'] = TIME_SERIES_MEAN[exp_cond]['glycerol'][0]
    y0['H_EXT'] = TIME_SERIES_MEAN[exp_cond]['3-HPA'][0]
    y0['H_CYTO'] = TIME_SERIES_MEAN[exp_cond]['3-HPA'][0]
    y0['H_MCP'] = TIME_SERIES_MEAN[exp_cond]['3-HPA'][0]
    y0['P_EXT'] = TIME_SERIES_MEAN[exp_cond]['13PD'][0]
    y0['P_CYTO'] = TIME_SERIES_MEAN[exp_cond]['13PD'][0]
    y0['P_MCP'] = TIME_SERIES_MEAN[exp_cond]['13PD'][0]

    param_samples_copy = np.array([*non_enz_model_params,*enz_params_dAJ,
                                   *list(OD_PRIOR_PARAMETER_MEAN[exp_cond].values())])
    POLAR_VOLUME = (4./3.)*np.pi*((10**param_samples_copy[PARAMETER_LIST.index('AJ_radius')])**3)
    y0['NADH_MCP'] = (10**(param_samples_copy[PARAMETER_LIST.index('NADH_NAD_TOTAL_CYTO')]
                           + param_samples_copy[PARAMETER_LIST.index('NADH_NAD_RATIO_CYTO')]))/(10**param_samples_copy[PARAMETER_LIST.index('NADH_NAD_RATIO_CYTO')] + 1)
    y0['NAD_MCP'] = 10**param_samples_copy[PARAMETER_LIST.index('NADH_NAD_TOTAL_CYTO')]/(10**param_samples_copy[PARAMETER_LIST.index('NADH_NAD_RATIO_CYTO')] + 1)
    y0['PduCDE'] = (10**(param_samples_copy[PARAMETER_LIST.index('nPduCDE')]  + param_samples_copy[PARAMETER_LIST.index('nMCPs')]))/(Avogadro * POLAR_VOLUME)
    y0['PduP'] = (10**(param_samples_copy[PARAMETER_LIST.index('nPduP')] + param_samples_copy[PARAMETER_LIST.index('nMCPs')]))/(Avogadro * POLAR_VOLUME)
    y0['PduQ'] = (10**(param_samples_copy[PARAMETER_LIST.index('nPduQ')] + param_samples_copy[PARAMETER_LIST.index('nMCPs')]))/(Avogadro * POLAR_VOLUME)
    y0['PduL'] = (10**(param_samples_copy[PARAMETER_LIST.index('nPduL')]  + param_samples_copy[PARAMETER_LIST.index('nMCPs')]))/(Avogadro * POLAR_VOLUME)
    y0['OD'] = 10**param_samples_copy[PARAMETER_LIST.index('A')]

    params_dict = { param_name : param_val for param_val,param_name in zip(param_samples_copy, PARAMETER_LIST)}

    # We can also specify the parameters by name:
    solver.set_params_dict(params_dict)
    yout = solver.make_output_buffers(tvals)
    try:
        solver.solve(t0=0, tvals=tvals, y0=y0, y_out=yout) # first run always takes longer
        # jj = 0
        # for i,var in enumerate(VARIABLE_NAMES):
        #     if i in DATA_INDEX:
        #         plt.plot(TIME_SAMPLES, yout.view(problem_WT.state_dtype)[var])
        #         plt.scatter(TIME_SAMPLES, TIME_SERIES_MEAN[exp_cond].iloc[:,jj])
        #         plt.title(var)
        #         plt.show()
        #         jj+=1
        # print(((TIME_SERIES_MEAN[exp_cond] - yout[:, DATA_INDEX])/TIME_SERIES_STD[exp_cond])**2)
        lik -= 0.5*(((TIME_SERIES_MEAN[exp_cond] - yout[:, DATA_INDEX])/TIME_SERIES_STD[exp_cond])**2).to_numpy().sum()#TIME_SERIES_STD[exp_cond])**2).to_numpy().sum()
    except sunode.solver.SolverError:
        lik += np.nan
    return lik



def likelihood_derivative_fwd_true(params, fwd_rtol = 1e-8, fwd_atol=1e-8, mxsteps=int(1e4)):

    tvals = TIME_SAMPLES*HRS_TO_SECS
    # non_enz_model_params = params[:-4]
    # cofactor_params = params[-4:]

    non_enz_model_params = params[:-8]
    enz_params_WT = params[-8:-4]
    enz_params_dAJ = params[-4:]

    lik_dev_params_adj = np.zeros(len(DEV_PARAMETER_LIST) + 4)

    exp_cond = 'dAJ-L'#['WT-L', 'dAJ-L', 'dD-L', 'dP-L']:

    solver = solver_dAJ
    problem = problem_dAJ
    y0 = np.zeros((), dtype=problem.state_dtype)

    lib.CVodeSStolerances(solver._ode, fwd_rtol, fwd_atol)
    lib.CVodeSetMaxNumSteps(solver._ode, mxsteps)

    # initialize initial conditions
    for var in VARIABLE_NAMES:
        y0[var] = 0
    sens0 = np.zeros((len(DEV_PARAMETER_LIST),len(VARIABLE_NAMES)))

    # set y0 and sens0
    y0['G_EXT'] = TIME_SERIES_MEAN[exp_cond]['glycerol'][0]
    y0['G_CYTO'] = TIME_SERIES_MEAN[exp_cond]['glycerol'][0]
    y0['G_MCP'] = TIME_SERIES_MEAN[exp_cond]['glycerol'][0]
    y0['H_EXT'] = TIME_SERIES_MEAN[exp_cond]['3-HPA'][0]
    y0['H_CYTO'] = TIME_SERIES_MEAN[exp_cond]['3-HPA'][0]
    y0['H_MCP'] = TIME_SERIES_MEAN[exp_cond]['3-HPA'][0]
    y0['P_EXT'] = TIME_SERIES_MEAN[exp_cond]['13PD'][0]
    y0['P_CYTO'] = TIME_SERIES_MEAN[exp_cond]['13PD'][0]
    y0['P_MCP'] = TIME_SERIES_MEAN[exp_cond]['13PD'][0]

    param_samples_copy = np.array([*non_enz_model_params,*enz_params_dAJ,
                                         # *GEOMETRY_PARAMETER_MEAN.values(),
                                         # np.log10((MCP_RADIUS * (10 ** (GEOMETRY_PARAMETER_MEAN['nMCPs'] / 3.))
                                         #           + MCP_RADIUS * (10 ** (
                                         #                 GEOMETRY_PARAMETER_MEAN['nMCPs'] / 2.))) / 2),
                                         # *cofactor_params,
                                         # *dPDU_AJ_ENZ_NUMBER_PARAMETER_MEAN.values(),
                                         *list(OD_PRIOR_PARAMETER_MEAN[exp_cond].values())])
    POLAR_VOLUME = (4./3.)*np.pi*((10**param_samples_copy[PARAMETER_LIST.index('AJ_radius')])**3)
    y0['NADH_MCP'] = (10**(param_samples_copy[PARAMETER_LIST.index('NADH_NAD_TOTAL_CYTO')]
                           + param_samples_copy[PARAMETER_LIST.index('NADH_NAD_RATIO_CYTO')]))/(10**param_samples_copy[PARAMETER_LIST.index('NADH_NAD_RATIO_CYTO')] + 1)
    y0['NAD_MCP'] = 10**param_samples_copy[PARAMETER_LIST.index('NADH_NAD_TOTAL_CYTO')]/(10**param_samples_copy[PARAMETER_LIST.index('NADH_NAD_RATIO_CYTO')] + 1)
    y0['PduCDE'] = (10**(param_samples_copy[PARAMETER_LIST.index('nPduCDE')]+param_samples_copy[PARAMETER_LIST.index('nMCPs')]))/(Avogadro * POLAR_VOLUME)
    y0['PduP'] = 10**(param_samples_copy[PARAMETER_LIST.index('nPduP')] + param_samples_copy[PARAMETER_LIST.index('nMCPs')])/(Avogadro * POLAR_VOLUME)
    y0['PduQ'] = 10**(param_samples_copy[PARAMETER_LIST.index('nPduQ')] + param_samples_copy[PARAMETER_LIST.index('nMCPs')])/(Avogadro * POLAR_VOLUME)
    y0['PduL'] = 10**(param_samples_copy[PARAMETER_LIST.index('nPduL')] + param_samples_copy[PARAMETER_LIST.index('nMCPs')])/(Avogadro * POLAR_VOLUME)
    y0['OD'] = 10**param_samples_copy[PARAMETER_LIST.index('A')]

    sens0[PARAMETER_LIST.index('nPduCDE'), VARIABLE_NAMES.index('PduCDE')] = np.log(10)*(10**(param_samples_copy[PARAMETER_LIST.index('nPduCDE')] + param_samples_copy[PARAMETER_LIST.index('nMCPs')]))/(Avogadro * POLAR_VOLUME)
    sens0[PARAMETER_LIST.index('nPduP'), VARIABLE_NAMES.index('PduP')] = np.log(10)*(10**(param_samples_copy[PARAMETER_LIST.index('nPduP')]+param_samples_copy[PARAMETER_LIST.index('nMCPs')]))/(Avogadro * POLAR_VOLUME)
    sens0[PARAMETER_LIST.index('nPduQ'), VARIABLE_NAMES.index('PduQ')] = np.log(10)*(10**(param_samples_copy[PARAMETER_LIST.index('nPduQ')]+param_samples_copy[PARAMETER_LIST.index('nMCPs')]))/(Avogadro * POLAR_VOLUME)
    sens0[PARAMETER_LIST.index('nPduL'), VARIABLE_NAMES.index('PduL')] = np.log(10)*(10**(param_samples_copy[PARAMETER_LIST.index('nPduL')]+param_samples_copy[PARAMETER_LIST.index('nMCPs')]))/(Avogadro * POLAR_VOLUME)
    # sens0[PARAMETER_LIST.index('nPduW'), VARIABLE_NAMES.index('PduW')] = np.log(10)*(10**param_sample[PARAMETER_LIST.index('nPduW')])/(Avogadro * POLAR_VOLUME)

    sens0[PARAMETER_LIST.index('AJ_radius'), VARIABLE_NAMES.index('PduCDE')] = -3*np.log(10)*(10**(param_samples_copy[PARAMETER_LIST.index('nPduCDE')] + param_samples_copy[PARAMETER_LIST.index('nMCPs')]))/(Avogadro * POLAR_VOLUME)
    sens0[PARAMETER_LIST.index('AJ_radius'), VARIABLE_NAMES.index('PduP')] = -3*np.log(10)*(10**(param_samples_copy[PARAMETER_LIST.index('nPduP')] + param_samples_copy[PARAMETER_LIST.index('nMCPs')]))/(Avogadro * POLAR_VOLUME)
    sens0[PARAMETER_LIST.index('AJ_radius'), VARIABLE_NAMES.index('PduQ')] = -3*np.log(10)*(10**(param_samples_copy[PARAMETER_LIST.index('nPduQ')] + param_samples_copy[PARAMETER_LIST.index('nMCPs')]))/(Avogadro * POLAR_VOLUME)
    sens0[PARAMETER_LIST.index('AJ_radius'), VARIABLE_NAMES.index('PduL')] = -3*np.log(10)*(10**(param_samples_copy[PARAMETER_LIST.index('nPduL')] + param_samples_copy[PARAMETER_LIST.index('nMCPs')]))/(Avogadro * POLAR_VOLUME)

    sens0[PARAMETER_LIST.index('nMCPs'), VARIABLE_NAMES.index('PduCDE')] = np.log(10)*(10**(param_samples_copy[PARAMETER_LIST.index('nPduCDE')] + param_samples_copy[PARAMETER_LIST.index('nMCPs')]))/(Avogadro * POLAR_VOLUME)
    sens0[PARAMETER_LIST.index('nMCPs'), VARIABLE_NAMES.index('PduP')] = np.log(10)*(10**(param_samples_copy[PARAMETER_LIST.index('nPduP')] + param_samples_copy[PARAMETER_LIST.index('nMCPs')]))/(Avogadro * POLAR_VOLUME)
    sens0[PARAMETER_LIST.index('nMCPs'), VARIABLE_NAMES.index('PduQ')] = np.log(10)*(10**(param_samples_copy[PARAMETER_LIST.index('nPduQ')] + param_samples_copy[PARAMETER_LIST.index('nMCPs')]))/(Avogadro * POLAR_VOLUME)
    sens0[PARAMETER_LIST.index('nMCPs'), VARIABLE_NAMES.index('PduL')] = np.log(10)*(10**(param_samples_copy[PARAMETER_LIST.index('nPduL')] + param_samples_copy[PARAMETER_LIST.index('nMCPs')]))/(Avogadro * POLAR_VOLUME)

    sens0[PARAMETER_LIST.index('NADH_NAD_TOTAL_CYTO'), VARIABLE_NAMES.index('NADH_MCP')] = np.log(10)*(10**(param_samples_copy[PARAMETER_LIST.index('NADH_NAD_TOTAL_CYTO')] + param_samples_copy[PARAMETER_LIST.index('NADH_NAD_RATIO_CYTO')]))/(10**param_samples_copy[PARAMETER_LIST.index('NADH_NAD_RATIO_CYTO')] + 1)
    sens0[PARAMETER_LIST.index('NADH_NAD_RATIO_CYTO'), VARIABLE_NAMES.index('NADH_MCP')] = np.log(10)*(10**(param_samples_copy[PARAMETER_LIST.index('NADH_NAD_TOTAL_CYTO')] + param_samples_copy[PARAMETER_LIST.index('NADH_NAD_RATIO_CYTO')]))/(10**param_samples_copy[PARAMETER_LIST.index('NADH_NAD_RATIO_CYTO')] + 1)**2
    sens0[PARAMETER_LIST.index('NADH_NAD_TOTAL_CYTO'), VARIABLE_NAMES.index('NAD_MCP')] = np.log(10)*(10**param_samples_copy[PARAMETER_LIST.index('NADH_NAD_TOTAL_CYTO')])/(10**param_samples_copy[PARAMETER_LIST.index('NADH_NAD_RATIO_CYTO')] + 1)
    sens0[PARAMETER_LIST.index('NADH_NAD_RATIO_CYTO'), VARIABLE_NAMES.index('NAD_MCP')] = -np.log(10)*(10**(param_samples_copy[PARAMETER_LIST.index('NADH_NAD_TOTAL_CYTO')] + param_samples_copy[PARAMETER_LIST.index('NADH_NAD_RATIO_CYTO')]))/(10**param_samples_copy[PARAMETER_LIST.index('NADH_NAD_RATIO_CYTO')] + 1)**2
    params_dict = { param_name : param_val for param_val,param_name in zip(param_samples_copy, PARAMETER_LIST)}
    solver.set_params_dict(params_dict)
    yout, sens_out = solver.make_output_buffers(tvals)

    try:
        solver.solve(t0=0, tvals=tvals, y0=y0, y_out=yout, sens0 = sens0, sens_out=sens_out)
        grads = np.zeros_like(sens_out[:,0,:])
        grads[:,DATA_INDEX] = (TIME_SERIES_MEAN[exp_cond] - yout[:, DATA_INDEX])/(TIME_SERIES_STD[exp_cond])**2#(TIME_SERIES_STD[exp_cond])**2
        grad_out = [(grads*sens_out[:,j,:]).sum() for j in range(len(DEV_PARAMETER_LIST))]
        lik_dev_params_adj[:-8] += grad_out[:-4]
        lik_dev_params_adj[-4:] += grad_out[-4:]
    except sunode.solver.SolverError:
        lik_dev_params_adj += np.nan
    return lik_dev_params_adj









