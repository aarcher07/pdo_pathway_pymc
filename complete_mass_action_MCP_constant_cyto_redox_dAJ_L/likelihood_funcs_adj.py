import numpy as np
from constants import *
import pickle
from exp_data_13pd import *
from prior_constants import *
import time
# from rhs_WT import RHS_WT, problem_WT
from rhs_dAJ_L import RHS_dAJ_L, problem_dAJ_L
# from rhs_dD import RHS_dD, problem_dD

from scipy.constants import Avogadro
import sunode
import matplotlib.pyplot as plt

lib = sunode._cvodes.lib
# The solver generates uses numba and sympy to generate optimized C functions
solver_dAJ = sunode.solver.AdjointSolver(problem_dAJ_L, solver='BDF')
# solver_WT = sunode.solver.AdjointSolver(problem_WT, solver='BDF')
# solver_dD = sunode.solver.AdjointSolver(problem_dD, solver='BDF')

def likelihood_adj(params, fwd_rtol = 1e-8, fwd_atol=1e-8, fwd_mxsteps=int(1e4)):
    lib.CVodeSStolerances(solver_dAJ._ode, fwd_rtol, fwd_atol)
    lib.CVodeSetMaxNumSteps(solver_dAJ._ode, fwd_mxsteps)
    # lib.CVodeSStolerances(solver_WT._ode, post_fwd_rtol, post_fwd_atol)
    # lib.CVodeSetMaxNumSteps(solver_WT._ode, mxsteps)
    # lib.CVodeSStolerances(solver_dD._ode, post_fwd_rtol, post_fwd_atol)
    # lib.CVodeSetMaxNumSteps(solver_dD._ode, mxsteps)

    tvals = TIME_SAMPLES*HRS_TO_SECS
    exp_cond = 'dAJ_L' #['WT_L', 'dAJ_L', 'dD_L', 'dP_L']:
    param_vals = [*params[GLOBAL_PERMEABILITY_CELL_PARAMETERS_INDICES],
                   *params[GLOBAL_PERMEABILITY_MCP_PARAMETERS_INDICES],
                   *params[GLOBAL_KINETIC_PARAMETERS_INDICES],
                   *params[GLOBAL_GlpK_PARAMETERS_INDICES_DICT[exp_cond]],
                  *params[GLOBAL_MCP_GEOMETRY_PARAMETERS_INDICES],
                  *params[GLOBAL_COFACTOR_PARAMETERS_INDICES_DICT[exp_cond]],
                   *params[GLOBAL_ENZYME_CONCENTRATIONS_INDICES_DICT[exp_cond]],
                   *list(OD_PRIOR_PARAMETER_MEAN[exp_cond].values())
                   ]

    POLAR_VOLUME = (4./3.)*np.pi*((10 ** param_vals[LOCAL_PARAMETER_LIST.index('dAJ_radius')]) ** 3)

    lik = 0
    solver = solver_dAJ
    y0 = np.zeros((), dtype=problem_dAJ_L.state_dtype)

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
    y0['NADH_MCP'] = (10 ** (param_vals[LOCAL_PARAMETER_LIST.index('NADH_NAD_TOTAL_CYTO')]
                             + param_vals[LOCAL_PARAMETER_LIST.index('NADH_NAD_RATIO_CYTO')])) / (10 ** param_vals[LOCAL_PARAMETER_LIST.index('NADH_NAD_RATIO_CYTO')] + 1)
    y0['NAD_MCP'] = 10 ** param_vals[LOCAL_PARAMETER_LIST.index('NADH_NAD_TOTAL_CYTO')] / (10 ** param_vals[LOCAL_PARAMETER_LIST.index('NADH_NAD_RATIO_CYTO')] + 1)
    y0['PduCDE'] = (10 ** (param_vals[LOCAL_PARAMETER_LIST.index('nPduCDE')] + param_vals[LOCAL_PARAMETER_LIST.index('nMCPs')])) / (Avogadro * POLAR_VOLUME)
    y0['PduP'] = (10 ** (param_vals[LOCAL_PARAMETER_LIST.index('nPduP')] + param_vals[LOCAL_PARAMETER_LIST.index('nMCPs')])) / (Avogadro * POLAR_VOLUME)
    y0['PduQ'] = (10 ** (param_vals[LOCAL_PARAMETER_LIST.index('nPduQ')] + param_vals[LOCAL_PARAMETER_LIST.index('nMCPs')])) / (Avogadro * POLAR_VOLUME)
    y0['PduL'] = (10 ** (param_vals[LOCAL_PARAMETER_LIST.index('nPduL')] + param_vals[LOCAL_PARAMETER_LIST.index('nMCPs')])) / (Avogadro * POLAR_VOLUME)
    y0['OD'] = 10 ** param_vals[LOCAL_PARAMETER_LIST.index('A')]

    params_dict = {param_name : param_val for param_val,param_name in zip(param_vals, LOCAL_PARAMETER_LIST)}

    # We can also specify the parameters by name:
    solver.set_params_dict(params_dict)
    yout, _, _ = solver.make_output_buffers(tvals)
    try:
        solver.solve_forward(t0=0, tvals=tvals, y0=y0, y_out=yout) # first run always takes longer
        # jj = 0
        # for i,var in enumerate(VARIABLE_NAMES):
        #     if i in DATA_INDEX:
        #         plt.plot(TIME_SAMPLES, yout.view(problem_dAJ_L.state_dtype)[var])
        #         plt.scatter(TIME_SAMPLES_EXPANDED[::TIME_SPACING], TIME_SERIES_MEAN[exp_cond].iloc[:,jj])
        #         plt.title(var)
        #         jj+=1
        #     plt.show()
        # print(TIME_SERIES_MEAN[exp_cond])
        # print(yout[::TIME_SPACING, DATA_INDEX])
        # print(TIME_SERIES_MEAN[exp_cond] - yout[::TIME_SPACING, DATA_INDEX])
        # print(TIME_SERIES_STD[exp_cond])
        # print( yout[::TIME_SPACING, DATA_INDEX])
        # print((TIME_SERIES_MEAN[exp_cond] - yout[::TIME_SPACING, DATA_INDEX]))
        # print(((TIME_SERIES_MEAN[exp_cond] - yout[::TIME_SPACING, DATA_INDEX]) / TIME_SERIES_STD[exp_cond]) ** 2)
        lik_curr = -0.5*(((TIME_SERIES_MEAN[exp_cond] - yout[:, DATA_INDEX])/np.array([1,0.01,1,0.1]))**2)
        # print(lik_curr)
        # print(lik_curr)
        lik += lik_curr.to_numpy().sum()
    except sunode.solver.SolverError:
        lik += np.nan
    print(lik)
    return lik



def likelihood_derivative_adj(params, fwd_rtol = 1e-8, fwd_atol=1e-8,
                              bck_rtol = 1e-4, bck_atol = 1e-4, fwd_mxsteps=int(1e4),
                                   bck_mxsteps=int(1e4)):
    lib.CVodeSStolerances(solver_dAJ._ode, fwd_rtol, fwd_atol)
    lib.CVodeSStolerancesB(solver_dAJ._ode, solver_dAJ._odeB, bck_rtol, bck_atol)
    lib.CVodeQuadSStolerancesB(solver_dAJ._ode, solver_dAJ._odeB, bck_rtol, bck_atol)
    lib.CVodeSetMaxNumSteps(solver_dAJ._ode, fwd_mxsteps)
    lib.CVodeSetMaxNumStepsB(solver_dAJ._ode, solver_dAJ._odeB, bck_mxsteps)

    tvals = TIME_SAMPLES_EXPANDED*HRS_TO_SECS
    exp_cond = 'dAJ_L' #['WT_L', 'dAJ_L', 'dD_L', 'dP_L']:

    param_vals = [*params[GLOBAL_PERMEABILITY_CELL_PARAMETERS_INDICES],
                   *params[GLOBAL_PERMEABILITY_MCP_PARAMETERS_INDICES],
                   *params[GLOBAL_KINETIC_PARAMETERS_INDICES],
                   *params[GLOBAL_GlpK_PARAMETERS_INDICES_DICT[exp_cond]],
                  *params[GLOBAL_MCP_GEOMETRY_PARAMETERS_INDICES],
                  *params[GLOBAL_COFACTOR_PARAMETERS_INDICES_DICT[exp_cond]],
                   *params[GLOBAL_ENZYME_CONCENTRATIONS_INDICES_DICT[exp_cond]],
                   *list(OD_PRIOR_PARAMETER_MEAN[exp_cond].values())
                   ]
    POLAR_VOLUME = (4./3.)*np.pi*((10 ** param_vals[LOCAL_PARAMETER_LIST.index('dAJ_radius')]) ** 3)

    solver = solver_dAJ
    y0 = np.zeros((), dtype=problem_dAJ_L.state_dtype)
    lik_dev_params_adj = np.zeros(len(GLOBAL_DEV_PARAMETERS))
    # initialize initial conditions
    for var in VARIABLE_NAMES:
        y0[var] = 0
    sens0 = np.zeros((len(LOCAL_DEV_PARAMETER_LIST), len(VARIABLE_NAMES)))

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
    y0['NADH_MCP'] = (10 ** (param_vals[LOCAL_PARAMETER_LIST.index('NADH_NAD_TOTAL_CYTO')]
                             + param_vals[LOCAL_PARAMETER_LIST.index('NADH_NAD_RATIO_CYTO')])) / (10 ** param_vals[LOCAL_PARAMETER_LIST.index('NADH_NAD_RATIO_CYTO')] + 1)
    y0['NAD_MCP'] = 10 ** param_vals[LOCAL_PARAMETER_LIST.index('NADH_NAD_TOTAL_CYTO')] / (10 ** param_vals[LOCAL_PARAMETER_LIST.index('NADH_NAD_RATIO_CYTO')] + 1)
    y0['PduCDE'] = (10 ** (param_vals[LOCAL_PARAMETER_LIST.index('nPduCDE')] + param_vals[LOCAL_PARAMETER_LIST.index('nMCPs')])) / (Avogadro * POLAR_VOLUME)
    y0['PduP'] = 10 ** (param_vals[LOCAL_PARAMETER_LIST.index('nPduP')] + param_vals[LOCAL_PARAMETER_LIST.index('nMCPs')]) / (Avogadro * POLAR_VOLUME)
    y0['PduQ'] = 10 ** (param_vals[LOCAL_PARAMETER_LIST.index('nPduQ')] + param_vals[LOCAL_PARAMETER_LIST.index('nMCPs')]) / (Avogadro * POLAR_VOLUME)
    y0['PduL'] = 10 ** (param_vals[LOCAL_PARAMETER_LIST.index('nPduL')] + param_vals[LOCAL_PARAMETER_LIST.index('nMCPs')]) / (Avogadro * POLAR_VOLUME)
    y0['OD'] = 10 ** param_vals[LOCAL_PARAMETER_LIST.index('A')]

    sens0[LOCAL_DEV_PARAMETER_LIST.index('nPduCDE'), VARIABLE_NAMES.index('PduCDE')] = np.log(10) * (10 ** (param_vals[LOCAL_PARAMETER_LIST.index('nPduCDE')] + param_vals[LOCAL_PARAMETER_LIST.index('nMCPs')])) / (Avogadro * POLAR_VOLUME)
    sens0[LOCAL_DEV_PARAMETER_LIST.index('nPduP'), VARIABLE_NAMES.index('PduP')] = np.log(10) * (10 ** (param_vals[LOCAL_PARAMETER_LIST.index('nPduP')] + param_vals[LOCAL_PARAMETER_LIST.index('nMCPs')])) / (Avogadro * POLAR_VOLUME)
    sens0[LOCAL_DEV_PARAMETER_LIST.index('nPduQ'), VARIABLE_NAMES.index('PduQ')] = np.log(10) * (10 ** (param_vals[LOCAL_PARAMETER_LIST.index('nPduQ')] + param_vals[LOCAL_PARAMETER_LIST.index('nMCPs')])) / (Avogadro * POLAR_VOLUME)
    sens0[LOCAL_DEV_PARAMETER_LIST.index('nPduL'), VARIABLE_NAMES.index('PduL')] = np.log(10) * (10 ** (param_vals[LOCAL_PARAMETER_LIST.index('nPduL')] + param_vals[LOCAL_PARAMETER_LIST.index('nMCPs')])) / (Avogadro * POLAR_VOLUME)

    sens0[LOCAL_DEV_PARAMETER_LIST.index('dAJ_radius'), VARIABLE_NAMES.index('PduCDE')] = -3 * np.log(10) * (10 ** (param_vals[LOCAL_PARAMETER_LIST.index('nPduCDE')] + param_vals[LOCAL_PARAMETER_LIST.index('nMCPs')])) / (Avogadro * POLAR_VOLUME)
    sens0[LOCAL_DEV_PARAMETER_LIST.index('dAJ_radius'), VARIABLE_NAMES.index('PduP')] = -3 * np.log(10) * (10 ** (param_vals[LOCAL_PARAMETER_LIST.index('nPduP')] + param_vals[LOCAL_PARAMETER_LIST.index('nMCPs')])) / (Avogadro * POLAR_VOLUME)
    sens0[LOCAL_DEV_PARAMETER_LIST.index('dAJ_radius'), VARIABLE_NAMES.index('PduQ')] = -3 * np.log(10) * (10 ** (param_vals[LOCAL_PARAMETER_LIST.index('nPduQ')] + param_vals[LOCAL_PARAMETER_LIST.index('nMCPs')])) / (Avogadro * POLAR_VOLUME)
    sens0[LOCAL_DEV_PARAMETER_LIST.index('dAJ_radius'), VARIABLE_NAMES.index('PduL')] = -3 * np.log(10) * (10 ** (param_vals[LOCAL_PARAMETER_LIST.index('nPduL')] + param_vals[LOCAL_PARAMETER_LIST.index('nMCPs')])) / (Avogadro * POLAR_VOLUME)

    sens0[LOCAL_DEV_PARAMETER_LIST.index('nMCPs'), VARIABLE_NAMES.index('PduCDE')] = np.log(10) * (10 ** (param_vals[LOCAL_PARAMETER_LIST.index('nPduCDE')] + param_vals[LOCAL_PARAMETER_LIST.index('nMCPs')])) / (Avogadro * POLAR_VOLUME)
    sens0[LOCAL_DEV_PARAMETER_LIST.index('nMCPs'), VARIABLE_NAMES.index('PduP')] = np.log(10) * (10 ** (param_vals[LOCAL_PARAMETER_LIST.index('nPduP')] + param_vals[LOCAL_PARAMETER_LIST.index('nMCPs')])) / (Avogadro * POLAR_VOLUME)
    sens0[LOCAL_DEV_PARAMETER_LIST.index('nMCPs'), VARIABLE_NAMES.index('PduQ')] = np.log(10) * (10 ** (param_vals[LOCAL_PARAMETER_LIST.index('nPduQ')] + param_vals[LOCAL_PARAMETER_LIST.index('nMCPs')])) / (Avogadro * POLAR_VOLUME)
    sens0[LOCAL_DEV_PARAMETER_LIST.index('nMCPs'), VARIABLE_NAMES.index('PduL')] = np.log(10) * (10 ** (param_vals[LOCAL_PARAMETER_LIST.index('nPduL')] + param_vals[LOCAL_PARAMETER_LIST.index('nMCPs')])) / (Avogadro * POLAR_VOLUME)

    sens0[LOCAL_DEV_PARAMETER_LIST.index('NADH_NAD_TOTAL_CYTO'), VARIABLE_NAMES.index('NADH_MCP')] = np.log(10) * (10 ** (param_vals[LOCAL_PARAMETER_LIST.index('NADH_NAD_TOTAL_CYTO')] + param_vals[LOCAL_PARAMETER_LIST.index('NADH_NAD_RATIO_CYTO')])) / (10 ** param_vals[LOCAL_PARAMETER_LIST.index('NADH_NAD_RATIO_CYTO')] + 1)
    sens0[LOCAL_DEV_PARAMETER_LIST.index('NADH_NAD_RATIO_CYTO'), VARIABLE_NAMES.index('NADH_MCP')] = np.log(10) * (10 ** (param_vals[LOCAL_PARAMETER_LIST.index('NADH_NAD_TOTAL_CYTO')] + param_vals[LOCAL_PARAMETER_LIST.index('NADH_NAD_RATIO_CYTO')])) / (10 ** param_vals[LOCAL_PARAMETER_LIST.index('NADH_NAD_RATIO_CYTO')] + 1) ** 2
    sens0[LOCAL_DEV_PARAMETER_LIST.index('NADH_NAD_TOTAL_CYTO'), VARIABLE_NAMES.index('NAD_MCP')] = np.log(10) * (10 ** param_vals[LOCAL_PARAMETER_LIST.index('NADH_NAD_TOTAL_CYTO')]) / (10 ** param_vals[LOCAL_PARAMETER_LIST.index('NADH_NAD_RATIO_CYTO')] + 1)
    sens0[LOCAL_DEV_PARAMETER_LIST.index('NADH_NAD_RATIO_CYTO'), VARIABLE_NAMES.index('NAD_MCP')] = -np.log(10) * (10 ** (param_vals[LOCAL_PARAMETER_LIST.index('NADH_NAD_TOTAL_CYTO')] + param_vals[LOCAL_PARAMETER_LIST.index('NADH_NAD_RATIO_CYTO')])) / (10 ** param_vals[LOCAL_PARAMETER_LIST.index('NADH_NAD_RATIO_CYTO')] + 1) ** 2
    params_dict = {param_name : param_val for param_val,param_name in zip(param_vals, LOCAL_PARAMETER_LIST)}

    solver.set_params_dict(params_dict)
    yout, grad_out, lambda_out = solver.make_output_buffers(tvals)

    try:
        solver.solve_forward(t0=0, tvals=tvals, y0=y0, y_out=yout)

        grads = np.zeros_like(yout)
        lik_dev = (TIME_SERIES_MEAN[exp_cond] - yout[::TIME_SPACING, DATA_INDEX])/(np.array([1,0.01,1,0.1]))**2
        grads[::TIME_SPACING, DATA_INDEX] = lik_dev
        solver.solve_backward(t0=tvals[-1], tend= tvals[0],tvals=tvals[1:-1],
                              grads=grads, grad_out=grad_out, lamda_out=lambda_out)
        grad_out = -np.matmul(sens0, lambda_out-grads[0,:]) + grad_out
    except sunode.solver.SolverError:
        grad_out[:] = np.nan
    lik_dev_params_adj[GLOBAL_PERMEABILITY_CELL_PARAMETERS_INDICES] = grad_out[LOCAL_PERMEABILITY_CELL_PARAMETERS_INDICES]
    lik_dev_params_adj[GLOBAL_PERMEABILITY_MCP_PARAMETERS_INDICES] = grad_out[LOCAL_PERMEABILITY_MCP_PARAMETERS_INDICES]
    lik_dev_params_adj[GLOBAL_KINETIC_PARAMETERS_INDICES] = grad_out[LOCAL_KINETIC_PARAMETERS_INDICES]
    lik_dev_params_adj[GLOBAL_GlpK_PARAMETERS_INDICES_DICT[exp_cond]] = grad_out[LOCAL_GlpK_PARAMETERS_INDICES]
    lik_dev_params_adj[GLOBAL_MCP_GEOMETRY_PARAMETERS_INDICES] = grad_out[LOCAL_MCP_GEOMETRY_PARAMETERS_INDICES]
    lik_dev_params_adj[GLOBAL_COFACTOR_PARAMETERS_INDICES_DICT[exp_cond]] = grad_out[LOCAL_COFACTOR_PARAMETERS_INDICES]
    lik_dev_params_adj[GLOBAL_ENZYME_CONCENTRATIONS_INDICES_DICT[exp_cond]] = grad_out[
        LOCAL_ENZYME_CONCENTRATIONS_INDICES]
    return lik_dev_params_adj









