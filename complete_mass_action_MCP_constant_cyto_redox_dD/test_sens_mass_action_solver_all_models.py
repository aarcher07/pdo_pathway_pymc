import sunode
import matplotlib.pyplot as plt
import numpy as np
from constants import *
import pickle
from exp_data_13pd import *
from prior_constants import *
import time
from rhs_WT import RHS_WT, lib, problem
from scipy.constants import Avogadro
# The solver generates uses numba and sympy to generate optimized C functions



########################################################################################################################
#################################################### ADJOINT SOLVER ####################################################
########################################################################################################################

solver = sunode.solver.AdjointSolver(problem, solver='BDF')
lib.CVodeSStolerances(solver._ode, 1e-8, 1e-8)
lib.CVodeSStolerancesB(solver._ode, solver._odeB, 1e-4, 1e-4)
lib.CVodeQuadSStolerancesB(solver._ode, solver._odeB, 1e-4, 1e-4)
lib.CVodeSetMaxNumSteps(solver._ode, 1000000)

tvals = TIME_SAMPLES_EXPANDED*HRS_TO_SECS
param_sample = np.array([*CELL_PERMEABILITY_MEAN.values(),
                         *MCP_PERMEABILITY_MEAN.values(),
                         *KINETIC_PARAMETER_MEAN.values(),
                         *GEOMETRY_PARAMETER_MEAN.values(),
                         np.log10((MCP_RADIUS*(10**(GEOMETRY_PARAMETER_MEAN['nMCPs']/3.))
                                   + MCP_RADIUS*(10**(GEOMETRY_PARAMETER_MEAN['nMCPs']/2.)))/2),
                         *COFACTOR_NUMBER_PARAMETER_MEAN.values(),
                         *PDU_WT_ENZ_NUMBERS_PARAMETER_MEAN.values(),
                         ])
lik_dev_params_adj = np.zeros(len(DEV_PARAMETER_LIST))
time_tot = 0
y0 = np.zeros((), dtype=problem.state_dtype)
for var in VARIABLE_NAMES:
    y0[var] = 0

y0['NADH_MCP'] = (10**(param_sample[PARAMETER_LIST.index('NADH_NAD_TOTAL_MCP')] + param_sample[PARAMETER_LIST.index('NADH_NAD_RATIO_MCP')]))/(10**param_sample[PARAMETER_LIST.index('NADH_NAD_RATIO_MCP')] + 1)
y0['NAD_MCP'] = 10**param_sample[PARAMETER_LIST.index('NADH_NAD_TOTAL_MCP')]/(10**param_sample[PARAMETER_LIST.index('NADH_NAD_RATIO_MCP')] + 1)
y0['PduCDE'] = 10**param_sample[PARAMETER_LIST.index('nPduCDE')]/(Avogadro * MCP_VOLUME)
y0['PduP'] = 10**param_sample[PARAMETER_LIST.index('nPduP')]/(Avogadro * MCP_VOLUME)
y0['PduQ'] = 10**param_sample[PARAMETER_LIST.index('nPduQ')]/(Avogadro * MCP_VOLUME)
y0['PduL'] = 10**param_sample[PARAMETER_LIST.index('nPduL')]/(Avogadro * MCP_VOLUME)
# y0['PduW'] = 10**param_sample[PARAMETER_LIST.index('nPduW')]/(Avogadro * MCP_VOLUME)

sens0 = np.zeros((len(DEV_PARAMETER_LIST),len(VARIABLE_NAMES)))
# initial sensitivities
sens0[PARAMETER_LIST.index('nPduCDE'), VARIABLE_NAMES.index('PduCDE')] = np.log(10)*(10**param_sample[PARAMETER_LIST.index('nPduCDE')])/(Avogadro * MCP_VOLUME)
sens0[PARAMETER_LIST.index('nPduP'), VARIABLE_NAMES.index('PduP')] = np.log(10)*(10**param_sample[PARAMETER_LIST.index('nPduP')])/(Avogadro * MCP_VOLUME)
sens0[PARAMETER_LIST.index('nPduQ'), VARIABLE_NAMES.index('PduQ')] = np.log(10)*(10**param_sample[PARAMETER_LIST.index('nPduQ')])/(Avogadro * MCP_VOLUME)
sens0[PARAMETER_LIST.index('nPduL'), VARIABLE_NAMES.index('PduL')] = np.log(10)*(10**param_sample[PARAMETER_LIST.index('nPduL')])/(Avogadro * MCP_VOLUME)
# sens0[PARAMETER_LIST.index('nPduW'), VARIABLE_NAMES.index('PduW')] = np.log(10)*(10**param_sample[PARAMETER_LIST.index('nPduW')])/(Avogadro * MCP_VOLUME)
sens0[PARAMETER_LIST.index('NADH_NAD_TOTAL_MCP'), VARIABLE_NAMES.index('NADH_MCP')] = np.log(10)*(10**(param_sample[PARAMETER_LIST.index('NADH_NAD_TOTAL_MCP')] + param_sample[PARAMETER_LIST.index('NADH_NAD_RATIO_MCP')]))/(10**param_sample[PARAMETER_LIST.index('NADH_NAD_RATIO_MCP')] + 1)
sens0[PARAMETER_LIST.index('NADH_NAD_RATIO_MCP'), VARIABLE_NAMES.index('NADH_MCP')] = np.log(10)*(10**(param_sample[PARAMETER_LIST.index('NADH_NAD_TOTAL_MCP')] + param_sample[PARAMETER_LIST.index('NADH_NAD_RATIO_MCP')]))/(10**param_sample[PARAMETER_LIST.index('NADH_NAD_RATIO_MCP')] + 1)**2
sens0[PARAMETER_LIST.index('NADH_NAD_TOTAL_MCP'), VARIABLE_NAMES.index('NAD_MCP')] = np.log(10)*(10**param_sample[PARAMETER_LIST.index('NADH_NAD_TOTAL_MCP')])/(10**param_sample[PARAMETER_LIST.index('NADH_NAD_RATIO_MCP')] + 1)
sens0[PARAMETER_LIST.index('NADH_NAD_RATIO_MCP'), VARIABLE_NAMES.index('NAD_MCP')] = -np.log(10)*(10**(param_sample[PARAMETER_LIST.index('NADH_NAD_TOTAL_MCP')] + param_sample[PARAMETER_LIST.index('NADH_NAD_RATIO_MCP')]))/(10**param_sample[PARAMETER_LIST.index('NADH_NAD_RATIO_MCP')] + 1)**2


for exp_cond in ['WT-L', 'dD-L', 'dP-L']:
    param_samples_copy = np.concatenate((param_sample,list(OD_PRIOR_PARAMETER_MEAN[exp_cond].values())))
    params_dict = { param_name : param_val for param_val,param_name in zip(param_samples_copy, PARAMETER_LIST)}
    y0_copy = y0.copy()
    y0_copy['G_EXT'] = TIME_SERIES_MEAN[exp_cond]['glycerol'][0]
    y0_copy['G_CYTO'] = TIME_SERIES_MEAN[exp_cond]['glycerol'][0]
    y0_copy['G_MCP'] = TIME_SERIES_MEAN[exp_cond]['glycerol'][0]
    y0_copy['H_EXT'] = TIME_SERIES_MEAN[exp_cond]['3-HPA'][0]
    y0_copy['H_CYTO'] = TIME_SERIES_MEAN[exp_cond]['3-HPA'][0]
    y0_copy['H_MCP'] = TIME_SERIES_MEAN[exp_cond]['3-HPA'][0]
    y0_copy['P_EXT'] = TIME_SERIES_MEAN[exp_cond]['13PD'][0]
    y0_copy['P_CYTO'] = TIME_SERIES_MEAN[exp_cond]['13PD'][0]
    y0_copy['P_MCP'] = TIME_SERIES_MEAN[exp_cond]['13PD'][0]

    params_dict_copy = params_dict.copy()
    sens0_copy = sens0.copy()
    y0_copy['OD'] = 10**param_samples_copy[PARAMETER_LIST.index('A')]

    if exp_cond == 'dD-L':
        y0_copy['PduCDE'] = 0
        params_dict_copy['nPduCDE'] = 0
        sens0_copy[PARAMETER_LIST.index('nPduCDE'), VARIABLE_NAMES.index('PduCDE')] = 0

    elif exp_cond == 'dP-L':
        y0_copy['PduP'] = 0
        params_dict_copy['nPduP'] = 0
        sens0_copy[PARAMETER_LIST.index('nPduP'), VARIABLE_NAMES.index('PduP')] = 0

    # We can also specify the parameters by name:
    solver.set_params_dict(params_dict_copy)
    yout, grad_out, lambda_out = solver.make_output_buffers(tvals)

    solver.solve_forward(t0=0, tvals=tvals, y0=y0_copy, y_out=yout) # first run always takes longer
    time_total = 0
    time_start = time.time()
    solver.solve_forward(t0=0, tvals=tvals, y0=y0_copy, y_out=yout)
    time_end = time.time()
    time_total += (time_end-time_start)/60

    jj = 0
    for i,var in enumerate(VARIABLE_NAMES):
        if i in DATA_INDEX:
            plt.plot(TIME_SAMPLES_EXPANDED, yout.view(problem.state_dtype)[var])
            plt.scatter(TIME_SAMPLES_EXPANDED[::TIME_SPACING], TIME_SERIES_MEAN[exp_cond].iloc[:,jj])
            plt.title(var)
            plt.show()
            jj+=1
    grads = np.zeros_like(yout)
    lik_dev = (TIME_SERIES_MEAN[exp_cond] - yout[::TIME_SPACING, DATA_INDEX])/(TIME_SERIES_STD[exp_cond])**2
    grads[::TIME_SPACING, DATA_INDEX] = lik_dev

    time_start = time.time()
    solver.solve_backward(t0=tvals[-1], tend= tvals[0],tvals=tvals[1:-1],
                          grads=grads, grad_out=grad_out, lamda_out=lambda_out)
    time_end = time.time()
    time_tot += (time_end-time_start)/60

    grad_out = -np.matmul(sens0_copy,lambda_out-grads[0,:]) + grad_out
    lik_dev_params_adj += grad_out
print(time_tot)

########################################################################################################################
#################################################### FWD SOLVER ########################################################
########################################################################################################################
solver = sunode.solver.Solver(problem, solver='BDF', sens_mode='simultaneous')
lib.CVodeSStolerances(solver._ode, 1e-8, 1e-8)
lib.CVodeSetMaxNumSteps(solver._ode, 10000)
tvals = TIME_SAMPLES*HRS_TO_SECS
lik_dev_params_fwd = np.zeros(len(DEV_PARAMETER_LIST))

for exp_cond in ['WT-L', 'dD-L', 'dP-L']:
    param_samples_copy = np.concatenate((param_sample,list(OD_PRIOR_PARAMETER_MEAN[exp_cond].values())))
    params_dict = { param_name : param_val for param_val,param_name in zip(param_samples_copy, PARAMETER_LIST)}
    y0_copy = y0.copy()
    y0_copy['G_EXT'] = TIME_SERIES_MEAN[exp_cond]['glycerol'][0]
    y0_copy['G_CYTO'] = TIME_SERIES_MEAN[exp_cond]['glycerol'][0]
    y0_copy['G_MCP'] = TIME_SERIES_MEAN[exp_cond]['glycerol'][0]
    y0_copy['H_EXT'] = TIME_SERIES_MEAN[exp_cond]['3-HPA'][0]
    y0_copy['H_CYTO'] = TIME_SERIES_MEAN[exp_cond]['3-HPA'][0]
    y0_copy['H_MCP'] = TIME_SERIES_MEAN[exp_cond]['3-HPA'][0]
    y0_copy['P_EXT'] = TIME_SERIES_MEAN[exp_cond]['13PD'][0]
    y0_copy['P_CYTO'] = TIME_SERIES_MEAN[exp_cond]['13PD'][0]
    y0_copy['P_MCP'] = TIME_SERIES_MEAN[exp_cond]['13PD'][0]
    params_dict_copy = params_dict.copy()
    sens0_copy = sens0.copy()
    y0_copy['OD'] = 10**param_samples_copy[PARAMETER_LIST.index('A')]
    if exp_cond == 'dD-L':
        y0_copy['PduCDE'] = 0
        params_dict_copy['nPduCDE'] = 0
        sens0_copy[PARAMETER_LIST.index('nPduCDE'), VARIABLE_NAMES.index('PduCDE')] = 0
    elif exp_cond == 'dP-L':
        y0_copy['PduP'] = 0
        params_dict_copy['nPduP'] = 0
        sens0_copy[PARAMETER_LIST.index('nPduP'), VARIABLE_NAMES.index('PduP')] = 0

    solver.set_params_dict(params_dict_copy)
    yout, sens_out = solver.make_output_buffers(tvals)
    solver.solve(t0=0, tvals=tvals, y0=y0_copy, y_out=yout, sens0 = sens0_copy, sens_out=sens_out)

    time_tot = 0
    time_start = time.time()
    solver.solve(t0=0, tvals=tvals, y0=y0_copy, y_out=yout, sens0 = sens0_copy, sens_out=sens_out)
    time_end = time.time()
    time_tot += (time_end-time_start)/60
    lik_dev = (TIME_SERIES_MEAN[exp_cond] - yout[:, DATA_INDEX])/(TIME_SERIES_STD[exp_cond])**2
    lik_dev_zeros = np.zeros_like(sens_out[:,0,:])
    lik_dev_zeros[:,DATA_INDEX] = lik_dev
    for j,param in enumerate(DEV_PARAMETER_LIST):
        lik_dev_param = (lik_dev_zeros*sens_out[:,j,:]).sum()
        lik_dev_params_fwd[j] += lik_dev_param
print(time_tot)

########################################################################################################################
#################################################### COMPARISON ########################################################
########################################################################################################################
grad_diff = lik_dev_params_adj - lik_dev_params_fwd
print(lik_dev_params_adj)
print(lik_dev_params_fwd)
print(grad_diff)
print(grad_diff/np.abs(lik_dev_params_adj))
# mass balance check
# print( EXTERNAL_VOLUME*(yout.view(problem.state_dtype)['G_EXT'] + yout.view(problem.state_dtype)['H_EXT']
#                         + yout.view(problem.state_dtype)['P_EXT'] +
#                         + yout.view(problem.state_dtype)['HCoA_EXT'] + yout.view(problem.state_dtype)['HPhosph_EXT']
#                         + yout.view(problem.state_dtype)['Hate_EXT'])
#
#        + OD_TO_CELL_COUNT*(yout.view(problem.state_dtype)['G_CYTO'] + yout.view(problem.state_dtype)['H_CYTO']
#                             + yout.view(problem.state_dtype)['P_CYTO'] +
#                             + yout.view(problem.state_dtype)['HCoA_CYTO'] + yout.view(problem.state_dtype)['HPhosph_CYTO']
#                             + yout.view(problem.state_dtype)['Hate_CYTO']   + yout.view(problem.state_dtype)['PduW_C'])
#
#        + (10**param_sample[PARAMETER_LIST.index('nMCPs')]) * OD_TO_CELL_COUNT *(yout.view(problem.state_dtype)['G_MCP']
#                                                                                 + yout.view(problem.state_dtype)['H_MCP']
#                                                                                 + yout.view(problem.state_dtype)['P_MCP']
#                                                                                 + yout.view(problem.state_dtype)['HCoA_MCP']
#                                                                                 + yout.view(problem.state_dtype)['HPhosph_MCP']
#                                                                                 + yout.view(problem.state_dtype)['PduCDE_C']
#                                                                                 + yout.view(problem.state_dtype)[
#                                                                                     'PduL_C']
#                                                                                 + yout.view(problem.state_dtype)[
#                                                                                     'PduQ_NADH_HPA']
#                                                                                 + yout.view(problem.state_dtype)['PduP_NAD_HPA']))

