import numpy as np
from constants import *
import pickle
from exp_data_13pd import *
from prior_constants import *
import time
from rhs_WT import RHS_WT, problem_WT
from rhs_delta_AJ import RHS_delta_AJ, problem_delta_AJ
from scipy.constants import Avogadro
import sunode
import matplotlib.pyplot as plt

lib = sunode._cvodes.lib
# The solver generates uses numba and sympy to generate optimized C functions
solver_delta_AJ = sunode.solver.AdjointSolver(problem_delta_AJ, solver='BDF')
solver_WT = sunode.solver.AdjointSolver(problem_WT, solver='BDF')

def likelihood_adj(params, fwd_rtol = 1e-8, fwd_atol=1e-8, mxsteps=int(1e4)):
    lib.CVodeSStolerances(solver_delta_AJ._ode, fwd_rtol, fwd_atol)
    lib.CVodeSetMaxNumSteps(solver_delta_AJ._ode, mxsteps)
    lib.CVodeSStolerances(solver_WT._ode, fwd_rtol, fwd_atol)
    lib.CVodeSetMaxNumSteps(solver_WT._ode, mxsteps)

    tvals = TIME_SAMPLES_EXPANDED*HRS_TO_SECS

    non_enz_model_params = params[:-8]
    enz_params_WT = params[-8:-4]
    enz_params_dAJ = params[-4:]

    lik = 0
    for exp_cond in ['WT-L', 'dAJ-L', 'dD-L', 'dP-L']:
        # set solver
        if exp_cond in ['WT-L', 'dD-L', 'dP-L']:
            solver = solver_WT
            y0 = np.zeros((), dtype=problem_WT.state_dtype)
        elif exp_cond == 'dAJ-L':
            solver = solver_delta_AJ
            y0 = np.zeros((), dtype=problem_delta_AJ.state_dtype)

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

        if exp_cond in ['WT-L', 'dD-L', 'dP-L']:
            param_samples_copy = np.concatenate((non_enz_model_params,enz_params_WT,list(OD_PRIOR_PARAMETER_MEAN[exp_cond].values())))

            y0['NADH_MCP'] = (10**(param_samples_copy[PARAMETER_LIST.index('NADH_NAD_TOTAL_MCP')]
                                   + param_samples_copy[PARAMETER_LIST.index('NADH_NAD_RATIO_MCP')]))/(10**param_samples_copy[PARAMETER_LIST.index('NADH_NAD_RATIO_MCP')] + 1)
            y0['NAD_MCP'] = 10**param_samples_copy[PARAMETER_LIST.index('NADH_NAD_TOTAL_MCP')]/(10**param_samples_copy[PARAMETER_LIST.index('NADH_NAD_RATIO_MCP')] + 1)
            y0['PduCDE'] = 10**param_samples_copy[PARAMETER_LIST.index('nPduCDE')]/(Avogadro * MCP_VOLUME)
            y0['PduP'] = 10**param_samples_copy[PARAMETER_LIST.index('nPduP')]/(Avogadro * MCP_VOLUME)
            y0['PduQ'] = 10**param_samples_copy[PARAMETER_LIST.index('nPduQ')]/(Avogadro * MCP_VOLUME)
            y0['PduL'] = 10**param_samples_copy[PARAMETER_LIST.index('nPduL')]/(Avogadro * MCP_VOLUME)
            y0['OD'] = 10**param_samples_copy[PARAMETER_LIST.index('A')]
        elif exp_cond == 'dAJ-L':
            param_samples_copy = np.concatenate((non_enz_model_params,
                                                 enz_params_dAJ,
                                                 list(OD_PRIOR_PARAMETER_MEAN[exp_cond].values())))
            POLAR_VOLUME = (4./3.)*np.pi*((10**param_samples_copy[PARAMETER_LIST.index('AJ_radius')])**3)
            y0['NADH_MCP'] = (10**(param_samples_copy[PARAMETER_LIST.index('NADH_NAD_TOTAL_CYTO')]
                                        + param_samples_copy[PARAMETER_LIST.index('NADH_NAD_RATIO_CYTO')]))/(10**param_samples_copy[PARAMETER_LIST.index('NADH_NAD_RATIO_CYTO')] + 1)
            y0['NAD_MCP'] = 10**param_samples_copy[PARAMETER_LIST.index('NADH_NAD_TOTAL_CYTO')]/(10**param_samples_copy[PARAMETER_LIST.index('NADH_NAD_RATIO_CYTO')] + 1)
            y0['PduCDE'] = param_samples_copy[PARAMETER_LIST.index('nMCPs')]*(10**param_samples_copy[PARAMETER_LIST.index('nPduCDE')])/(Avogadro * POLAR_VOLUME)
            y0['PduP'] = param_samples_copy[PARAMETER_LIST.index('nMCPs')]*(10**param_samples_copy[PARAMETER_LIST.index('nPduP')])/(Avogadro * POLAR_VOLUME)
            y0['PduQ'] = param_samples_copy[PARAMETER_LIST.index('nMCPs')]*(10**param_samples_copy[PARAMETER_LIST.index('nPduQ')])/(Avogadro * POLAR_VOLUME)
            y0['PduL'] = param_samples_copy[PARAMETER_LIST.index('nMCPs')]*(10**param_samples_copy[PARAMETER_LIST.index('nPduL')])/(Avogadro * POLAR_VOLUME)
            y0['OD'] = 10**param_samples_copy[PARAMETER_LIST.index('A')]

        params_dict = { param_name : param_val for param_val,param_name in zip(param_samples_copy, PARAMETER_LIST)}
        if exp_cond == 'dD-L':
            y0['PduCDE'] = 0
            params_dict['nPduCDE'] = 0
        elif exp_cond == 'dP-L':
            y0['PduP'] = 0
            params_dict['nPduP'] = 0

        # We can also specify the parameters by name:
        solver.set_params_dict(params_dict)
        yout, _, _ = solver.make_output_buffers(tvals)
        try:
            solver.solve_forward(t0=0, tvals=tvals, y0=y0, y_out=yout) # first run always takes longer
            # jj = 0
            # for i,var in enumerate(VARIABLE_NAMES):
            #     if i in DATA_INDEX:
            #         plt.plot(TIME_SAMPLES_EXPANDED, yout.view(problem_WT.state_dtype)[var])
            #         plt.scatter(TIME_SAMPLES_EXPANDED[::TIME_SPACING], TIME_SERIES_MEAN[exp_cond].iloc[:,jj])
            #         plt.title(var)
            #         plt.show()
            #         jj+=1

            lik += (((TIME_SERIES_MEAN[exp_cond] - yout[::TIME_SPACING, DATA_INDEX])/TIME_SERIES_STD[exp_cond])**2).to_numpy().sum()
        except sunode.solver.SolverError:
            lik += np.nan
    print(lik)
    return lik



def likelihood_derivative_adj(params, fwd_rtol = 1e-8, fwd_atol=1e-8,
                              bck_rtol = 1e-6, bck_atol = 1e-6, mxsteps=int(1e4)):
    lib.CVodeSStolerances(solver_delta_AJ._ode, fwd_rtol, fwd_atol)
    lib.CVodeSStolerancesB(solver_delta_AJ._ode, solver_delta_AJ._odeB, bck_rtol, bck_atol)
    lib.CVodeQuadSStolerancesB(solver_delta_AJ._ode, solver_delta_AJ._odeB, bck_rtol, bck_atol)
    lib.CVodeSetMaxNumSteps(solver_delta_AJ._ode, mxsteps)

    lib.CVodeSStolerances(solver_WT._ode, fwd_rtol, fwd_atol)
    lib.CVodeSStolerancesB(solver_WT._ode, solver_WT._odeB, bck_rtol, bck_atol)
    lib.CVodeQuadSStolerancesB(solver_WT._ode, solver_WT._odeB, bck_rtol, bck_atol)
    lib.CVodeSetMaxNumSteps(solver_WT._ode, mxsteps)

    tvals = TIME_SAMPLES_EXPANDED*HRS_TO_SECS

    non_enz_model_params = params[:-8]
    enz_params_WT = params[-8:-4]
    enz_params_dAJ = params[-4:]

    lik = 0
    lik_dev_params_adj = np.zeros(len(DEV_PARAMETER_LIST) + 4)

    for exp_cond in ['WT-L', 'dAJ-L', 'dD-L', 'dP-L']:

        # set solver
        if exp_cond in ['WT-L', 'dD-L', 'dP-L']:
            solver = solver_WT
            y0 = np.zeros((), dtype=problem_WT.state_dtype)
        elif exp_cond == 'dAJ-L':
            solver = solver_delta_AJ
            y0 = np.zeros((), dtype=problem_delta_AJ.state_dtype)


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


        if exp_cond in ['WT-L', 'dD-L', 'dP-L']:
            param_samples_copy = np.concatenate((non_enz_model_params,enz_params_WT,list(OD_PRIOR_PARAMETER_MEAN[exp_cond].values())))

            y0['NADH_MCP'] = (10**(param_samples_copy[PARAMETER_LIST.index('NADH_NAD_TOTAL_MCP')]
                                        + param_samples_copy[PARAMETER_LIST.index('NADH_NAD_RATIO_MCP')]))/(10**param_samples_copy[PARAMETER_LIST.index('NADH_NAD_RATIO_MCP')] + 1)
            y0['NAD_MCP'] = 10**param_samples_copy[PARAMETER_LIST.index('NADH_NAD_TOTAL_MCP')]/(10**param_samples_copy[PARAMETER_LIST.index('NADH_NAD_RATIO_MCP')] + 1)
            y0['PduCDE'] = 10**param_samples_copy[PARAMETER_LIST.index('nPduCDE')]/(Avogadro * MCP_VOLUME)
            y0['PduP'] = 10**param_samples_copy[PARAMETER_LIST.index('nPduP')]/(Avogadro * MCP_VOLUME)
            y0['PduQ'] = 10**param_samples_copy[PARAMETER_LIST.index('nPduQ')]/(Avogadro * MCP_VOLUME)
            y0['PduL'] = 10**param_samples_copy[PARAMETER_LIST.index('nPduL')]/(Avogadro * MCP_VOLUME)
            y0['OD'] = 10**param_samples_copy[PARAMETER_LIST.index('A')]


            sens0[PARAMETER_LIST.index('nPduCDE'), VARIABLE_NAMES.index('PduCDE')] = np.log(10)*(10**param_samples_copy[PARAMETER_LIST.index('nPduCDE')])/(Avogadro * MCP_VOLUME)
            sens0[PARAMETER_LIST.index('nPduP'), VARIABLE_NAMES.index('PduP')] = np.log(10)*(10**param_samples_copy[PARAMETER_LIST.index('nPduP')])/(Avogadro * MCP_VOLUME)
            sens0[PARAMETER_LIST.index('nPduQ'), VARIABLE_NAMES.index('PduQ')] = np.log(10)*(10**param_samples_copy[PARAMETER_LIST.index('nPduQ')])/(Avogadro * MCP_VOLUME)
            sens0[PARAMETER_LIST.index('nPduL'), VARIABLE_NAMES.index('PduL')] = np.log(10)*(10**param_samples_copy[PARAMETER_LIST.index('nPduL')])/(Avogadro * MCP_VOLUME)
            # sens0[LOCAL_PARAMETER_LIST.index('nPduW'), VARIABLE_NAMES.index('PduW')] = np.log(10)*(10**param_sample[LOCAL_PARAMETER_LIST.index('nPduW')])/(Avogadro * MCP_VOLUME)
            sens0[PARAMETER_LIST.index('NADH_NAD_TOTAL_MCP'), VARIABLE_NAMES.index('NADH_MCP')] = np.log(10)*(10**(param_samples_copy[PARAMETER_LIST.index('NADH_NAD_TOTAL_MCP')] + param_samples_copy[PARAMETER_LIST.index('NADH_NAD_RATIO_MCP')]))/(10**param_samples_copy[PARAMETER_LIST.index('NADH_NAD_RATIO_MCP')] + 1)
            sens0[PARAMETER_LIST.index('NADH_NAD_RATIO_MCP'), VARIABLE_NAMES.index('NADH_MCP')] = np.log(10)*(10**(param_samples_copy[PARAMETER_LIST.index('NADH_NAD_TOTAL_MCP')] + param_samples_copy[PARAMETER_LIST.index('NADH_NAD_RATIO_MCP')]))/(10**param_samples_copy[PARAMETER_LIST.index('NADH_NAD_RATIO_MCP')] + 1)**2
            sens0[PARAMETER_LIST.index('NADH_NAD_TOTAL_MCP'), VARIABLE_NAMES.index('NAD_MCP')] = np.log(10)*(10**param_samples_copy[PARAMETER_LIST.index('NADH_NAD_TOTAL_MCP')])/(10**param_samples_copy[PARAMETER_LIST.index('NADH_NAD_RATIO_MCP')] + 1)
            sens0[PARAMETER_LIST.index('NADH_NAD_RATIO_MCP'), VARIABLE_NAMES.index('NAD_MCP')] = -np.log(10)*(10**(param_samples_copy[PARAMETER_LIST.index('NADH_NAD_TOTAL_MCP')] + param_samples_copy[PARAMETER_LIST.index('NADH_NAD_RATIO_MCP')]))/(10**param_samples_copy[PARAMETER_LIST.index('NADH_NAD_RATIO_MCP')] + 1)**2
        elif exp_cond == 'dAJ-L':
            param_samples_copy = np.concatenate((non_enz_model_params,
                                                 enz_params_dAJ,
                                                 list(OD_PRIOR_PARAMETER_MEAN[exp_cond].values())))
            POLAR_VOLUME = (4./3.)*np.pi*((10**param_samples_copy[PARAMETER_LIST.index('AJ_radius')])**3)
            y0['NADH_MCP'] = (10**(param_samples_copy[PARAMETER_LIST.index('NADH_NAD_TOTAL_CYTO')]
                                        + param_samples_copy[PARAMETER_LIST.index('NADH_NAD_RATIO_CYTO')]))/(10**param_samples_copy[PARAMETER_LIST.index('NADH_NAD_RATIO_CYTO')] + 1)
            y0['NAD_MCP'] = 10**param_samples_copy[PARAMETER_LIST.index('NADH_NAD_TOTAL_CYTO')]/(10**param_samples_copy[PARAMETER_LIST.index('NADH_NAD_RATIO_CYTO')] + 1)
            y0['PduCDE'] = param_samples_copy[PARAMETER_LIST.index('nMCPs')]*(10**param_samples_copy[PARAMETER_LIST.index('nPduCDE')])/(Avogadro * POLAR_VOLUME)
            y0['PduP'] = param_samples_copy[PARAMETER_LIST.index('nMCPs')]*(10**param_samples_copy[PARAMETER_LIST.index('nPduP')])/(Avogadro * POLAR_VOLUME)
            y0['PduQ'] = param_samples_copy[PARAMETER_LIST.index('nMCPs')]*(10**param_samples_copy[PARAMETER_LIST.index('nPduQ')])/(Avogadro * POLAR_VOLUME)
            y0['PduL'] = param_samples_copy[PARAMETER_LIST.index('nMCPs')]*(10**param_samples_copy[PARAMETER_LIST.index('nPduL')])/(Avogadro * POLAR_VOLUME)
            y0['OD'] = 10**param_samples_copy[PARAMETER_LIST.index('A')]

            sens0[PARAMETER_LIST.index('nPduCDE'), VARIABLE_NAMES.index('PduCDE')] = param_samples_copy[PARAMETER_LIST.index('nMCPs')]*np.log(10)*(10**param_samples_copy[PARAMETER_LIST.index('nPduCDE')])/(Avogadro * POLAR_VOLUME)
            sens0[PARAMETER_LIST.index('nPduP'), VARIABLE_NAMES.index('PduP')] = param_samples_copy[PARAMETER_LIST.index('nMCPs')]*np.log(10)*(10**param_samples_copy[PARAMETER_LIST.index('nPduP')])/(Avogadro * POLAR_VOLUME)
            sens0[PARAMETER_LIST.index('nPduQ'), VARIABLE_NAMES.index('PduQ')] = param_samples_copy[PARAMETER_LIST.index('nMCPs')]*np.log(10)*(10**param_samples_copy[PARAMETER_LIST.index('nPduQ')])/(Avogadro * POLAR_VOLUME)
            sens0[PARAMETER_LIST.index('nPduL'), VARIABLE_NAMES.index('PduL')] = param_samples_copy[PARAMETER_LIST.index('nMCPs')]*np.log(10)*(10**param_samples_copy[PARAMETER_LIST.index('nPduL')])/(Avogadro * POLAR_VOLUME)
            # sens0[LOCAL_PARAMETER_LIST.index('nPduW'), VARIABLE_NAMES.index('PduW')] = np.log(10)*(10**param_sample[LOCAL_PARAMETER_LIST.index('nPduW')])/(Avogadro * POLAR_VOLUME)

            sens0[PARAMETER_LIST.index('AJ_radius'), VARIABLE_NAMES.index('PduCDE')] = -3*np.log(10)*param_samples_copy[PARAMETER_LIST.index('nMCPs')]*(10**param_samples_copy[PARAMETER_LIST.index('nPduCDE')])/(Avogadro * POLAR_VOLUME)
            sens0[PARAMETER_LIST.index('AJ_radius'), VARIABLE_NAMES.index('PduP')] = -3*np.log(10)*param_samples_copy[PARAMETER_LIST.index('nMCPs')]*(10**param_samples_copy[PARAMETER_LIST.index('nPduP')])/(Avogadro * POLAR_VOLUME)
            sens0[PARAMETER_LIST.index('AJ_radius'), VARIABLE_NAMES.index('PduQ')] = -3*np.log(10)*param_samples_copy[PARAMETER_LIST.index('nMCPs')]*(10**param_samples_copy[PARAMETER_LIST.index('nPduQ')])/(Avogadro * POLAR_VOLUME)
            sens0[PARAMETER_LIST.index('AJ_radius'), VARIABLE_NAMES.index('PduL')] = -3*np.log(10)*param_samples_copy[PARAMETER_LIST.index('nMCPs')]*(10**param_samples_copy[PARAMETER_LIST.index('nPduL')])/(Avogadro * POLAR_VOLUME)

            sens0[PARAMETER_LIST.index('nMCPs'), VARIABLE_NAMES.index('PduCDE')] = (10**param_samples_copy[PARAMETER_LIST.index('nPduCDE')])/(Avogadro * POLAR_VOLUME)
            sens0[PARAMETER_LIST.index('nMCPs'), VARIABLE_NAMES.index('PduP')] = (10**param_samples_copy[PARAMETER_LIST.index('nPduP')])/(Avogadro * POLAR_VOLUME)
            sens0[PARAMETER_LIST.index('nMCPs'), VARIABLE_NAMES.index('PduQ')] = (10**param_samples_copy[PARAMETER_LIST.index('nPduQ')])/(Avogadro * POLAR_VOLUME)
            sens0[PARAMETER_LIST.index('nMCPs'), VARIABLE_NAMES.index('PduL')] = (10**param_samples_copy[PARAMETER_LIST.index('nPduL')])/(Avogadro * POLAR_VOLUME)

            sens0[PARAMETER_LIST.index('NADH_NAD_TOTAL_MCP'), VARIABLE_NAMES.index('NADH_MCP')] = np.log(10)*(10**(param_samples_copy[PARAMETER_LIST.index('NADH_NAD_TOTAL_MCP')] + param_samples_copy[PARAMETER_LIST.index('NADH_NAD_RATIO_MCP')]))/(10**param_samples_copy[PARAMETER_LIST.index('NADH_NAD_RATIO_MCP')] + 1)
            sens0[PARAMETER_LIST.index('NADH_NAD_RATIO_MCP'), VARIABLE_NAMES.index('NADH_MCP')] = np.log(10)*(10**(param_samples_copy[PARAMETER_LIST.index('NADH_NAD_TOTAL_MCP')] + param_samples_copy[PARAMETER_LIST.index('NADH_NAD_RATIO_MCP')]))/(10**param_samples_copy[PARAMETER_LIST.index('NADH_NAD_RATIO_MCP')] + 1)**2
            sens0[PARAMETER_LIST.index('NADH_NAD_TOTAL_MCP'), VARIABLE_NAMES.index('NAD_MCP')] = np.log(10)*(10**param_samples_copy[PARAMETER_LIST.index('NADH_NAD_TOTAL_MCP')])/(10**param_samples_copy[PARAMETER_LIST.index('NADH_NAD_RATIO_MCP')] + 1)
            sens0[PARAMETER_LIST.index('NADH_NAD_RATIO_MCP'), VARIABLE_NAMES.index('NAD_MCP')] = -np.log(10)*(10**(param_samples_copy[PARAMETER_LIST.index('NADH_NAD_TOTAL_MCP')] + param_samples_copy[PARAMETER_LIST.index('NADH_NAD_RATIO_MCP')]))/(10**param_samples_copy[PARAMETER_LIST.index('NADH_NAD_RATIO_MCP')] + 1)**2
        params_dict = { param_name : param_val for param_val,param_name in zip(param_samples_copy, PARAMETER_LIST)}
        if exp_cond == 'dD-L':
            y0['PduCDE'] = 0
            params_dict['nPduCDE'] = 0
            sens0[PARAMETER_LIST.index('nPduCDE'), VARIABLE_NAMES.index('PduCDE')] = 0
        elif exp_cond == 'dP-L':
            y0['PduP'] = 0
            params_dict['nPduP'] = 0
            sens0[PARAMETER_LIST.index('nPduP'), VARIABLE_NAMES.index('PduP')] = 0
            # We can also specify the parameters by name:
        solver.set_params_dict(params_dict)
        yout, grad_out, lambda_out = solver.make_output_buffers(tvals)

        try:
            solver.solve_forward(t0=0, tvals=tvals, y0=y0, y_out=yout)
            # jj = 0
            # for i,var in enumerate(VARIABLE_NAMES):
            #     if i in DATA_INDEX:
            #         plt.plot(TIME_SAMPLES_EXPANDED, yout.view(problem_WT.state_dtype)[var])
            #         plt.scatter(TIME_SAMPLES_EXPANDED[::TIME_SPACING], TIME_SERIES_MEAN[exp_cond].iloc[:,jj])
            #         plt.title(var)
            #         plt.show()
            #         jj+=1
            grads = np.zeros_like(yout)
            lik_dev = (TIME_SERIES_MEAN[exp_cond] - yout[::TIME_SPACING, DATA_INDEX])/(TIME_SERIES_STD[exp_cond])**2
            grads[::TIME_SPACING, DATA_INDEX] = lik_dev
            solver.solve_backward(t0=tvals[-1], tend= tvals[0],tvals=tvals[1:-1],
                                  grads=grads, grad_out=grad_out, lamda_out=lambda_out)
            grad_out = -np.matmul(sens0, lambda_out-grads[0,:]) + grad_out
            lik_dev_params_adj[:-8] += grad_out[:-4]
            if exp_cond in ['WT-L', 'dD-L', 'dP-L']:
                lik_dev_params_adj[-8:-4] += grad_out[-4:]
            elif exp_cond in 'dAJ-L':
                lik_dev_params_adj[-4:] += grad_out[-4:]
        except sunode.solver.SolverError:
            lik_dev_params_adj += np.nan
    return lik_dev_params_adj









