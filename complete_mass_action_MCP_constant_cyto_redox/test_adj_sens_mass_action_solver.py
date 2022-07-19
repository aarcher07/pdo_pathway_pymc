import sunode
import matplotlib.pyplot as plt
import numpy as np
from constants import *
import pickle
from exp_data_13pd import *
from prior_constants import *
import time
from rhs_WT import RHS_WT, lib, problem

# The solver generates uses numba and sympy to generate optimized C functions
solver = sunode.solver.AdjointSolver(problem, solver='BDF')
lib.CVodeSStolerances(solver._ode, 1e-8, 1e-8)
lib.CVodeSStolerancesB(solver._ode, solver._odeB, 1e-8, 1e-8)
lib.CVodeQuadSStolerancesB(solver._ode, solver._odeB, 1e-8, 1e-8)
lib.CVodeSetMaxNumSteps(solver._ode, 100000)

param_sample = np.array([*CELL_PERMEABILITY_MEAN.values(),
                         *MCP_PERMEABILITY_MEAN.values(),
                         *KINETIC_PARAMETER_MEAN.values(),
                         *PDU_WT_ENZ_NUMBERS_PARAMETER_MEAN.values(),
                         *COFACTOR_NUMBER_PARAMETER_MEAN.values(),
                         *list(OD_PRIOR_PARAMETERS_MEAN['WT-L'].values())
                         ])
lik_dev_params = np.zeros(len(DEV_PARAMETER_LIST))
time_tot = 0
print(param_sample)
# for exp_ind, gly_cond in enumerate([50,60,70,80]):
#     param_sample = NORM_PRIOR_MEAN_SINGLE_EXP[gly_cond]
#     param_sample[:(N_MODEL_PARAMETERS+1)] = [*param_sample_copy[:N_MODEL_PARAMETERS], param_sample_copy[N_MODEL_PARAMETERS + exp_ind]]
#     tvals = TIME_SAMPLES_EXPANDED[gly_cond]*HRS_TO_SECS
#
#     y0 = np.zeros((), dtype=problem.state_dtype)
#     for var in VARIABLE_NAMES:
#         y0[var] = 0
#     y0['G_CYTO'] = 10**param_sample[PARAMETER_LIST.index('G_EXT_INIT')]
#     y0['P_CYTO'] = INIT_CONDS_GLY_PDO_DCW[gly_cond][1]
#     y0['NADH'] = (10**(param_sample[PARAMETER_LIST.index('NADH_NAD_TOTAL_INIT')] + param_sample[PARAMETER_LIST.index('NADH_NAD_RATIO_INIT')]))/(10**param_sample[PARAMETER_LIST.index('NADH_NAD_RATIO_INIT')] + 1)
#     y0['NAD'] = 10**param_sample[PARAMETER_LIST.index('NADH_NAD_TOTAL_INIT')]/(10**param_sample[PARAMETER_LIST.index('NADH_NAD_RATIO_INIT')] + 1)
#     y0['DHAB'] = 10**param_sample[PARAMETER_LIST.index('DHAB_INIT')]
#     y0['DHAT'] = 10**param_sample[PARAMETER_LIST.index('DHAT_INIT')]
#     y0['DHAD'] = 10**param_sample[PARAMETER_LIST.index('DHAD_INIT')]
#     y0['E0'] = 10**param_sample[PARAMETER_LIST.index('E0_INIT')]
#     y0['G_EXT'] = 10**param_sample[PARAMETER_LIST.index('G_EXT_INIT')]
#     y0['P_EXT'] = INIT_CONDS_GLY_PDO_DCW[gly_cond][1]
#     y0['dcw'] =  10**param_sample[PARAMETER_LIST.index('A')]
#
#     params_dict = { param_name : param_val for param_val,param_name in zip(param_sample, PARAMETER_LIST)}
#     # # We can also specify the parameters by name:
#     solver.set_params_dict(params_dict)
#     yout, grad_out, lambda_out = solver.make_output_buffers(tvals)
#
#     # initial sensitivities
#     sens0 = np.zeros((len(DEV_PARAMETERS_LIST),len(VARIABLE_NAMES)))
#     # sens0[PARAMETER_LIST.index('G_EXT_INIT'), VARIABLE_NAMES.index('G_CYTO')] = np.log(10)*(10**param_sample[PARAMETER_LIST.index('G_EXT_INIT')])
#     # sens0[PARAMETER_LIST.index('G_EXT_INIT'), VARIABLE_NAMES.index('G_EXT')] = np.log(10)*(10**param_sample[PARAMETER_LIST.index('G_EXT_INIT')])
#     sens0[PARAMETER_LIST.index('DHAB_INIT'), VARIABLE_NAMES.index('DHAB')] = np.log(10)*(10**param_sample[PARAMETER_LIST.index('DHAB_INIT')])
#     sens0[PARAMETER_LIST.index('DHAD_INIT'), VARIABLE_NAMES.index('DHAD')] = np.log(10)*(10**param_sample[PARAMETER_LIST.index('DHAD_INIT')])
#     sens0[PARAMETER_LIST.index('E0_INIT'), VARIABLE_NAMES.index('E0')] = np.log(10)*(10**param_sample[PARAMETER_LIST.index('E0_INIT')])
#
#     sens0[PARAMETER_LIST.index('DHAT_INIT'), VARIABLE_NAMES.index('DHAT')] = np.log(10)*(10**param_sample[PARAMETER_LIST.index('DHAT_INIT')])
#     sens0[PARAMETER_LIST.index('NADH_NAD_TOTAL_INIT'), VARIABLE_NAMES.index('NADH')] = np.log(10)*(10**(param_sample[PARAMETER_LIST.index('NADH_NAD_TOTAL_INIT')] + param_sample[PARAMETER_LIST.index('NADH_NAD_RATIO_INIT')]))/(10**param_sample[PARAMETER_LIST.index('NADH_NAD_RATIO_INIT')] + 1)
#     sens0[PARAMETER_LIST.index('NADH_NAD_RATIO_INIT'), VARIABLE_NAMES.index('NADH')] = np.log(10)*(10**(param_sample[PARAMETER_LIST.index('NADH_NAD_TOTAL_INIT')] + param_sample[PARAMETER_LIST.index('NADH_NAD_RATIO_INIT')]))/(10**param_sample[PARAMETER_LIST.index('NADH_NAD_RATIO_INIT')] + 1)**2
#     sens0[PARAMETER_LIST.index('NADH_NAD_TOTAL_INIT'), VARIABLE_NAMES.index('NAD')] = np.log(10)*(10**param_sample[PARAMETER_LIST.index('NADH_NAD_TOTAL_INIT')])/(10**param_sample[PARAMETER_LIST.index('NADH_NAD_RATIO_INIT')] + 1)
#     sens0[PARAMETER_LIST.index('NADH_NAD_RATIO_INIT'), VARIABLE_NAMES.index('NAD')] = -np.log(10)*(10**(param_sample[PARAMETER_LIST.index('NADH_NAD_TOTAL_INIT')] + param_sample[PARAMETER_LIST.index('NADH_NAD_RATIO_INIT')]))/(10**param_sample[PARAMETER_LIST.index('NADH_NAD_RATIO_INIT')] + 1)**2
#     # sens0[PARAMETER_LIST.index('A'), VARIABLE_NAMES.index('dcw')] = np.log(10)*(10**param_sample[PARAMETER_LIST.index('A')])
#
#     time_start = time.time()
#     solver.solve_forward(t0=0, tvals=tvals, y0=y0, y_out=yout)
#     jj =0
#     for i,var in enumerate(VARIABLE_NAMES):
#         if i in DATA_INDEX:
#             plt.plot(tvals / HRS_TO_SECS, yout.view(problem.state_dtype)[var])
#             plt.scatter(tvals[::TIME_SPACING]/HRS_TO_SECS, DATA_SAMPLES[gly_cond][:,jj])
#             plt.title(var)
#             plt.show()
#             jj+=1
#         if var in ['H_CYTO']:
#             plt.plot(tvals / HRS_TO_SECS, yout.view(problem.state_dtype)[var])
#             plt.title(var)
#             plt.show()
#     # print('DHAB INIT ' + str(y0['DHAB'] + y0['DHAB_C']))
#     # print('DHAB FINAL ' + str(yout[-1,VARIABLE_NAMES.index('DHAB')] + yout[-1,VARIABLE_NAMES.index('DHAB_C')]))
#     #
#     # print('E0 INIT ' + str(y0['E0'] + y0['E0_C']))
#     # print('E0 FINAL ' + str(yout[-1,VARIABLE_NAMES.index('E0')] + yout[-1,VARIABLE_NAMES.index('E0_C')]))
#     #
#     # print('DHAT INIT ' + str(y0['DHAT']))
#     # print('DHAT FINAL ' + str(yout[-1,VARIABLE_NAMES.index('DHAT')] + yout[-1,VARIABLE_NAMES.index('DHAT_NAD')]
#     #                           + yout[-1,VARIABLE_NAMES.index('DHAT_NADH_HPA')]
#     #                           + yout[-1,VARIABLE_NAMES.index('DHAT_NADH')]))
#     #
#     # print('DHAD INIT ' + str(y0['DHAD']))
#     # print('DHAD FINAL ' + str(yout[-1,VARIABLE_NAMES.index('DHAD')] + yout[-1,VARIABLE_NAMES.index('DHAD_NAD')]
#     #                           + yout[-1,VARIABLE_NAMES.index('DHAD_NAD_GLY')]
#     #                           + yout[-1,VARIABLE_NAMES.index('DHAD_NADH')]))
#     # print('NAD/NADH INIT ' + str(y0['NADH'] + y0['NAD']))
#     #
#     # print('NAD/NADH FINAL ' + str(yout[-1,VARIABLE_NAMES.index('NAD')] + yout[-1,VARIABLE_NAMES.index('NADH')]
#     #
#     #                               + yout[-1,VARIABLE_NAMES.index('DHAD_NAD')]
#     #                               + yout[-1,VARIABLE_NAMES.index('DHAD_NAD_GLY')]
#     #                               + yout[-1,VARIABLE_NAMES.index('DHAD_NADH')]
#     #
#     #                               + yout[-1,VARIABLE_NAMES.index('DHAT_NAD')]
#     #                           + yout[-1,VARIABLE_NAMES.index('DHAT_NADH_HPA')]
#     #                           + yout[-1,VARIABLE_NAMES.index('DHAT_NADH')] + yout[-1,VARIABLE_NAMES.index('E0_C')]))
#     time_end = time.time()
#     time_tot += (time_end-time_start)/60
#
#     grads = np.zeros_like(yout)
#     lik_dev = (DATA_SAMPLES[gly_cond] - yout[::TIME_SPACING, DATA_INDEX])/np.array([15,15,0.1])**2
#     grads[::TIME_SPACING, DATA_INDEX] = lik_dev
#
#     # backsolve
#     time_start = time.time()
#     solver.solve_backward(t0=tvals[-1], tend= tvals[0],tvals=tvals[1:-1],
#                           grads=grads, grad_out=grad_out, lamda_out=lambda_out)
#     time_end = time.time()
#     time_tot += (time_end-time_start)/60
#
#     grad_out = -np.matmul(sens0,lambda_out-grads[0,:]) + grad_out
#     for j,param in enumerate(DEV_PARAMETERS_LIST):
#         if param == 'G_EXT_INIT':
#             lik_dev_params[N_MODEL_PARAMETERS + exp_ind] += grad_out[j]
#         elif param in ['L','k','A']:
#             jj = ['L','k','A'].index(param)
#             lik_dev_params[N_MODEL_PARAMETERS + 4 + exp_ind*N_DCW_PARAMETERS + jj ] += grad_out[j]
#         else:
#             lik_dev_params[j] += grad_out[j]
#
# print(lik_dev_params[:N_MODEL_PARAMETERS])
# print(lik_dev_params[N_MODEL_PARAMETERS:(N_MODEL_PARAMETERS+4)])
# print(lik_dev_params[(N_MODEL_PARAMETERS+4):])
# print(time_tot)
