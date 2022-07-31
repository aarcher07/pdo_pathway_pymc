import sunode
import matplotlib.pyplot as plt
import numpy as np
from constants import *
import pickle
from exp_data_13pd import *
from prior_constants import *
import time
from rhs_dAJ_L import RHS_dAJ_L, problem_dAJ_L
from scipy.constants import Avogadro
# The solver generates uses numba and sympy to generate optimized C functions


########################################################################################################################
#################################################### ADJOINT SOLVER ####################################################
########################################################################################################################
lib = sunode._cvodes.lib
problem=problem_dAJ_L
solver = sunode.solver.AdjointSolver(problem_dAJ_L, solver='BDF')
lib.CVodeSStolerances(solver._ode, 1e-10, 1e-10)
lib.CVodeSStolerancesB(solver._ode, solver._odeB, 1e-4, 1e-4)
lib.CVodeQuadSStolerancesB(solver._ode, solver._odeB, 1e-4, 1e-4)
lib.CVodeSetMaxNumSteps(solver._ode, 10000)

tvals = TIME_SAMPLES_EXPANDED*HRS_TO_SECS
param_sample = np.array([*CELL_PERMEABILITY_MEAN.values(),
                         *MCP_PERMEABILITY_MEAN.values(),
                         *KINETIC_PARAMETER_MEAN.values(),
                         *GEOMETRY_PARAMETER_MEAN.values(),
                         np.log10((MCP_RADIUS*(10**(GEOMETRY_PARAMETER_MEAN['nMCPs']/3.))
                                   + MCP_RADIUS*(10**(GEOMETRY_PARAMETER_MEAN['nMCPs']/2.)))/2),
                         *COFACTOR_NUMBER_PARAMETER_MEAN_dAJ_L.values(),
                         *PDU_ENZ_NUMBER_PARAMETER_MEAN_dAJ_L.values(),
                         *list(OD_PRIOR_PARAMETER_MEAN['dAJ_L'].values())
                         ])

param_sample[LOCAL_PARAMETER_LIST.index('NADH_NAD_RATIO_CYTO')] = 1 / 20
param_sample[LOCAL_PARAMETER_LIST.index('NADH_NAD_RATIO_MCP')] = 1 / 20

POLAR_VOLUME = (4./3.)*np.pi*((10 ** param_sample[LOCAL_PARAMETER_LIST.index('dAJ_radius')]) ** 3)
lik_dev_params = np.zeros(len(LOCAL_DEV_PARAMETER_LIST))
time_tot = 0
y0 = np.zeros((), dtype=problem.state_dtype)
for var in VARIABLE_NAMES:
    y0[var] = 0
y0['G_EXT'] = TIME_SERIES_MEAN['dAJ_L']['glycerol'][0]
y0['G_CYTO'] = TIME_SERIES_MEAN['dAJ_L']['glycerol'][0]
y0['G_MCP'] = TIME_SERIES_MEAN['dAJ_L']['glycerol'][0]
y0['H_EXT'] = TIME_SERIES_MEAN['dAJ_L']['3-HPA'][0]
y0['H_CYTO'] = TIME_SERIES_MEAN['dAJ_L']['3-HPA'][0]
y0['H_MCP'] = TIME_SERIES_MEAN['dAJ_L']['3-HPA'][0]
y0['P_EXT'] = TIME_SERIES_MEAN['dAJ_L']['13PD'][0]
y0['P_CYTO'] = TIME_SERIES_MEAN['dAJ_L']['13PD'][0]
y0['P_MCP'] = TIME_SERIES_MEAN['dAJ_L']['13PD'][0]
y0['NADH_MCP'] = (10 ** (param_sample[LOCAL_PARAMETER_LIST.index('NADH_NAD_TOTAL_CYTO')] + param_sample[LOCAL_PARAMETER_LIST.index('NADH_NAD_RATIO_CYTO')])) / (10 ** param_sample[LOCAL_PARAMETER_LIST.index('NADH_NAD_RATIO_CYTO')] + 1)
y0['NAD_MCP'] = 10 ** param_sample[LOCAL_PARAMETER_LIST.index('NADH_NAD_TOTAL_CYTO')] / (10 ** param_sample[LOCAL_PARAMETER_LIST.index('NADH_NAD_RATIO_CYTO')] + 1)
y0['PduCDE'] = (10 ** (param_sample[LOCAL_PARAMETER_LIST.index('nPduCDE')] + param_sample[LOCAL_PARAMETER_LIST.index('nMCPs')])) / (Avogadro * POLAR_VOLUME)
y0['PduP'] = (10 ** (param_sample[LOCAL_PARAMETER_LIST.index('nPduP')] + param_sample[LOCAL_PARAMETER_LIST.index('nMCPs')])) / (Avogadro * POLAR_VOLUME)
y0['PduQ'] = (10 ** (param_sample[LOCAL_PARAMETER_LIST.index('nPduQ')] + param_sample[LOCAL_PARAMETER_LIST.index('nMCPs')])) / (Avogadro * POLAR_VOLUME)
y0['PduL'] = (10 ** (param_sample[LOCAL_PARAMETER_LIST.index('nPduL')] + param_sample[LOCAL_PARAMETER_LIST.index('nMCPs')])) / (Avogadro * POLAR_VOLUME)
# y0['PduW'] = 10**param_sample[LOCAL_PARAMETER_LIST.index('nPduW')]/(Avogadro * POLAR_VOLUME)
y0['OD'] = 10**param_sample[LOCAL_PARAMETER_LIST.index('A')]
params_dict = {param_name : param_val for param_val,param_name in zip(param_sample, LOCAL_PARAMETER_LIST)}
# # We can also specify the parameters by name:
solver.set_params_dict(params_dict)
yout, grad_out, lambda_out = solver.make_output_buffers(tvals)

# # initial sensitivities
sens0 = np.zeros((len(LOCAL_DEV_PARAMETER_LIST), len(VARIABLE_NAMES)))
sens0[LOCAL_PARAMETER_LIST.index('nPduCDE'), VARIABLE_NAMES.index('PduCDE')] = np.log(10) * (10 ** (param_sample[LOCAL_PARAMETER_LIST.index('nPduCDE')] + param_sample[LOCAL_PARAMETER_LIST.index('nMCPs')])) / (Avogadro * POLAR_VOLUME)
sens0[LOCAL_PARAMETER_LIST.index('nPduP'), VARIABLE_NAMES.index('PduP')] = np.log(10) * (10 ** (param_sample[LOCAL_PARAMETER_LIST.index('nPduP')] + param_sample[LOCAL_PARAMETER_LIST.index('nMCPs')])) / (Avogadro * POLAR_VOLUME)
sens0[LOCAL_PARAMETER_LIST.index('nPduQ'), VARIABLE_NAMES.index('PduQ')] = np.log(10) * (10 ** (param_sample[LOCAL_PARAMETER_LIST.index('nPduQ')] + param_sample[LOCAL_PARAMETER_LIST.index('nMCPs')])) / (Avogadro * POLAR_VOLUME)
sens0[LOCAL_PARAMETER_LIST.index('nPduL'), VARIABLE_NAMES.index('PduL')] = np.log(10) * (10 ** (param_sample[LOCAL_PARAMETER_LIST.index('nPduL')] + param_sample[LOCAL_PARAMETER_LIST.index('nMCPs')])) / (Avogadro * POLAR_VOLUME)
# sens0[LOCAL_PARAMETER_LIST.index('nPduW'), VARIABLE_NAMES.index('PduW')] = np.log(10)*(10**param_sample[LOCAL_PARAMETER_LIST.index('nPduW')])/(Avogadro * POLAR_VOLUME)

sens0[LOCAL_PARAMETER_LIST.index('dAJ_radius'), VARIABLE_NAMES.index('PduCDE')] = -3 * np.log(10) * (10 ** (param_sample[LOCAL_PARAMETER_LIST.index('nPduCDE')] + param_sample[LOCAL_PARAMETER_LIST.index('nMCPs')])) / (Avogadro * POLAR_VOLUME)
sens0[LOCAL_PARAMETER_LIST.index('dAJ_radius'), VARIABLE_NAMES.index('PduP')] = -3 * np.log(10) * (10 ** (param_sample[LOCAL_PARAMETER_LIST.index('nPduP')] + param_sample[LOCAL_PARAMETER_LIST.index('nMCPs')])) / (Avogadro * POLAR_VOLUME)
sens0[LOCAL_PARAMETER_LIST.index('dAJ_radius'), VARIABLE_NAMES.index('PduQ')] = -3 * np.log(10) * (10 ** (param_sample[LOCAL_PARAMETER_LIST.index('nPduQ')] + param_sample[LOCAL_PARAMETER_LIST.index('nMCPs')])) / (Avogadro * POLAR_VOLUME)
sens0[LOCAL_PARAMETER_LIST.index('dAJ_radius'), VARIABLE_NAMES.index('PduL')] = -3 * np.log(10) * (10 ** (param_sample[LOCAL_PARAMETER_LIST.index('nPduL')] + param_sample[LOCAL_PARAMETER_LIST.index('nMCPs')])) / (Avogadro * POLAR_VOLUME)

sens0[LOCAL_PARAMETER_LIST.index('nMCPs'), VARIABLE_NAMES.index('PduCDE')] = np.log(10) * (10 ** (param_sample[LOCAL_PARAMETER_LIST.index('nPduCDE')] + param_sample[LOCAL_PARAMETER_LIST.index('nMCPs')])) / (Avogadro * POLAR_VOLUME)
sens0[LOCAL_PARAMETER_LIST.index('nMCPs'), VARIABLE_NAMES.index('PduP')] = np.log(10) * (10 ** (param_sample[LOCAL_PARAMETER_LIST.index('nPduP')] + param_sample[LOCAL_PARAMETER_LIST.index('nMCPs')])) / (Avogadro * POLAR_VOLUME)
sens0[LOCAL_PARAMETER_LIST.index('nMCPs'), VARIABLE_NAMES.index('PduQ')] = np.log(10) * (10 ** (param_sample[LOCAL_PARAMETER_LIST.index('nPduQ')] + param_sample[LOCAL_PARAMETER_LIST.index('nMCPs')])) / (Avogadro * POLAR_VOLUME)
sens0[LOCAL_PARAMETER_LIST.index('nMCPs'), VARIABLE_NAMES.index('PduL')] = np.log(10) * (10 ** (param_sample[LOCAL_PARAMETER_LIST.index('nPduL')] + param_sample[LOCAL_PARAMETER_LIST.index('nMCPs')])) / (Avogadro * POLAR_VOLUME)

sens0[LOCAL_PARAMETER_LIST.index('NADH_NAD_TOTAL_CYTO'), VARIABLE_NAMES.index('NADH_MCP')] = np.log(10) * (10 ** (param_sample[LOCAL_PARAMETER_LIST.index('NADH_NAD_TOTAL_CYTO')] + param_sample[LOCAL_PARAMETER_LIST.index('NADH_NAD_RATIO_CYTO')])) / (10 ** param_sample[LOCAL_PARAMETER_LIST.index('NADH_NAD_RATIO_CYTO')] + 1)
sens0[LOCAL_PARAMETER_LIST.index('NADH_NAD_RATIO_CYTO'), VARIABLE_NAMES.index('NADH_MCP')] = np.log(10) * (10 ** (param_sample[LOCAL_PARAMETER_LIST.index('NADH_NAD_TOTAL_CYTO')] + param_sample[LOCAL_PARAMETER_LIST.index('NADH_NAD_RATIO_CYTO')])) / (10 ** param_sample[LOCAL_PARAMETER_LIST.index('NADH_NAD_RATIO_CYTO')] + 1) ** 2
sens0[LOCAL_PARAMETER_LIST.index('NADH_NAD_TOTAL_CYTO'), VARIABLE_NAMES.index('NAD_MCP')] = np.log(10) * (10 ** param_sample[LOCAL_PARAMETER_LIST.index('NADH_NAD_TOTAL_CYTO')]) / (10 ** param_sample[LOCAL_PARAMETER_LIST.index('NADH_NAD_RATIO_CYTO')] + 1)
sens0[LOCAL_PARAMETER_LIST.index('NADH_NAD_RATIO_CYTO'), VARIABLE_NAMES.index('NAD_MCP')] = -np.log(10) * (10 ** (param_sample[LOCAL_PARAMETER_LIST.index('NADH_NAD_TOTAL_CYTO')] + param_sample[LOCAL_PARAMETER_LIST.index('NADH_NAD_RATIO_CYTO')])) / (10 ** param_sample[LOCAL_PARAMETER_LIST.index('NADH_NAD_RATIO_CYTO')] + 1) ** 2
solver.solve_forward(t0=0, tvals=tvals, y0=y0, y_out=yout) # first run always takes longer
time_total = 0
time_start = time.time()
solver.solve_forward(t0=0, tvals=tvals, y0=y0, y_out=yout)
time_end = time.time()
time_total += (time_end-time_start)/60

jj = 0
for i,var in enumerate(VARIABLE_NAMES):
    plt.plot(TIME_SAMPLES_EXPANDED, yout.view(problem_dAJ_L.state_dtype)[var])
    if i in DATA_INDEX:
        plt.scatter(TIME_SAMPLES_EXPANDED[::TIME_SPACING], TIME_SERIES_MEAN['dAJ_L'].iloc[:,jj])
        jj += 1
    plt.title(var)
    plt.show()

print(yout.view(problem.state_dtype)['PduCDE'] + yout.view(problem.state_dtype)['PduCDE_C'])
print(yout.view(problem.state_dtype)['PduQ'] + yout.view(problem.state_dtype)['PduQ_NADH']
      + yout.view(problem.state_dtype)['PduQ_NADH_HPA'] + yout.view(problem.state_dtype)['PduQ_NAD'])
print(yout.view(problem.state_dtype)['PduP'] + yout.view(problem.state_dtype)['PduP_NAD']
      + yout.view(problem.state_dtype)['PduP_NAD_HPA'] + yout.view(problem.state_dtype)['PduP_NADH'])
print(yout.view(problem.state_dtype)['PduL'] + yout.view(problem.state_dtype)['PduL_C'])

print(EXTERNAL_VOLUME * (yout.view(problem.state_dtype)['G_EXT'] + yout.view(problem.state_dtype)['H_EXT']
                        + yout.view(problem.state_dtype)['P_EXT'] +
                        + yout.view(problem.state_dtype)['HCoA_EXT'] + yout.view(problem.state_dtype)['HPhosph_EXT'])

      + OD_TO_CELL_COUNT * (yout.view(problem.state_dtype)['G_CYTO'] + yout.view(problem.state_dtype)['H_CYTO']
                            + yout.view(problem.state_dtype)['P_CYTO'] +
                            + yout.view(problem.state_dtype)['HCoA_CYTO'] + yout.view(problem.state_dtype)['HPhosph_CYTO'])

      + (10 ** param_sample[LOCAL_PARAMETER_LIST.index('nMCPs')]) * OD_TO_CELL_COUNT * (yout.view(problem.state_dtype)['G_MCP']
                                                                                        + yout.view(problem.state_dtype)['H_MCP']
                                                                                        + yout.view(problem.state_dtype)['P_MCP']
                                                                                        + yout.view(problem.state_dtype)['HCoA_MCP']
                                                                                        + yout.view(problem.state_dtype)['HPhosph_MCP']
                                                                                        + yout.view(problem.state_dtype)['PduCDE_C']
                                                                                        + yout.view(problem.state_dtype)[
                                                                                    'PduL_C']
                                                                                        + yout.view(problem.state_dtype)[
                                                                                    'PduQ_NADH_HPA']
                                                                                        + yout.view(problem.state_dtype)['PduP_NAD_HPA']))

grads = np.zeros_like(yout)
lik_dev = (TIME_SERIES_MEAN['dAJ_L'] - yout[::TIME_SPACING, DATA_INDEX])/(TIME_SERIES_STD['dAJ_L'])**2
grads[::TIME_SPACING, DATA_INDEX] = lik_dev

# backsolve
time_start = time.time()
solver.solve_backward(t0=tvals[-1], tend= tvals[0],tvals=tvals[1:-1],
                      grads=grads, grad_out=grad_out, lamda_out=lambda_out)
time_end = time.time()
time_tot += (time_end-time_start)/60

print(time_tot)
grad_out = -np.matmul(sens0,lambda_out-grads[0,:]) + grad_out

########################################################################################################################
#################################################### FWD SOLVER ########################################################
########################################################################################################################

solver = sunode.solver.Solver(problem, solver='BDF', sens_mode='simultaneous')
lib.CVodeSStolerances(solver._ode, 1e-8, 1e-8)
lib.CVodeSetMaxNumSteps(solver._ode, 10000)
tvals = TIME_SAMPLES*HRS_TO_SECS
solver.set_params_dict(params_dict)
yout, sens_out = solver.make_output_buffers(tvals)

solver.solve(t0=0, tvals=tvals, y0=y0, y_out=yout, sens0 = sens0, sens_out=sens_out)

time_tot = 0
time_start = time.time()
solver.solve(t0=0, tvals=tvals, y0=y0, y_out=yout, sens0 = sens0, sens_out=sens_out)
time_end = time.time()
time_tot += (time_end-time_start)/60
print(time_tot)

lik_dev = (TIME_SERIES_MEAN['dAJ_L'] - yout[:, DATA_INDEX])/(TIME_SERIES_STD['dAJ_L'])**2
lik_dev_zeros = np.zeros_like(sens_out[:,0,:])
lik_dev_zeros[:,DATA_INDEX] = lik_dev
for j,param in enumerate(LOCAL_DEV_PARAMETER_LIST):
    lik_dev_param = (lik_dev_zeros*sens_out[:,j,:]).sum()
    lik_dev_params[j] += lik_dev_param

########################################################################################################################
#################################################### COMPARISON ########################################################
########################################################################################################################
grad_diff = grad_out - lik_dev_params
print(grad_out)
print(lik_dev_params)
print(grad_diff)
print(grad_diff/np.abs(grad_out))
