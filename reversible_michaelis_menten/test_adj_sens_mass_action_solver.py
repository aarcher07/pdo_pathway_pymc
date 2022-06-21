import sunode
import matplotlib.pyplot as plt
import numpy as np
from constants import *
import pickle
from exp_data import *
from prior_constants import NORM_PRIOR_STD_RT_SINGLE_EXP,NORM_PRIOR_MEAN_SINGLE_EXP, NORM_PRIOR_STD_RT_ALL_EXP, NORM_PRIOR_MEAN_ALL_EXP
import time
from rhs_funcs import RHS, lib, problem

# The solver generates uses numba and sympy to generate optimized C functions
solver = sunode.solver.AdjointSolver(problem, solver='BDF')
lib.CVodeSStolerances(solver._ode, 1e-8, 1e-8)
lib.CVodeSStolerancesB(solver._ode, solver._odeB, 1e-8, 1e-8)
lib.CVodeQuadSStolerancesB(solver._ode, solver._odeB, 1e-8, 1e-8)
lib.CVodeSetMaxNumSteps(solver._ode, 10000)

time_tot = 0
lik_dev_params = np.zeros((N_MODEL_PARAMETERS + 4,))

for exp_ind, gly_cond in enumerate([50,60,70,80]):
    param_sample = NORM_PRIOR_MEAN_SINGLE_EXP[gly_cond]
    print(param_sample)
    tvals = TIME_SAMPLES_EXPANDED[gly_cond]*HRS_TO_SECS

    y0 = np.zeros((), dtype=problem.state_dtype)

    y0['G_CYTO'] = 10**param_sample[PARAMETER_LIST.index('G_EXT_INIT')]
    y0['H_CYTO'] = 0
    y0['P_CYTO'] = INIT_CONDS_GLY_PDO_DCW[gly_cond][1]
    y0['G_EXT'] = 10**param_sample[PARAMETER_LIST.index('G_EXT_INIT')]
    y0['H_EXT'] = 0
    y0['P_EXT'] = INIT_CONDS_GLY_PDO_DCW[gly_cond][1]
    y0['dcw'] =  10**param_sample[PARAMETER_LIST.index('A')]

    params_dict = { param_name : param_val for param_val,param_name in zip(param_sample, PARAMETER_LIST)}

    # # We can also specify the parameters by name:
    solver.set_params_dict(params_dict)
    yout, grad_out, lambda_out = solver.make_output_buffers(tvals)

    # initial sensitivities
    sens0 = np.zeros((len(DEV_PARAMETERS_LIST),len(VARIABLE_NAMES)))
    sens0[PARAMETER_LIST.index('G_EXT_INIT'), VARIABLE_NAMES.index('G_CYTO')] = np.log(10)*(10**param_sample[PARAMETER_LIST.index('G_EXT_INIT')])
    sens0[PARAMETER_LIST.index('G_EXT_INIT'), VARIABLE_NAMES.index('G_EXT')] = np.log(10)*(10**param_sample[PARAMETER_LIST.index('G_EXT_INIT')])
    # sens0[PARAMETER_LIST.index('A'), VARIABLE_NAMES.index('dcw')] = np.log(10)*(10**param_sample[PARAMETER_LIST.index('A')])

    time_start = time.time()
    solver.solve_forward(t0=0, tvals=tvals, y0=y0, y_out=yout)
    time_end = time.time()
    time_tot += (time_end-time_start)/60
    jj = 0
    for i, var in enumerate(VARIABLE_NAMES):
        if i in DATA_INDEX:
            plt.plot(tvals / HRS_TO_SECS, yout.view(problem.state_dtype)[var])
            plt.scatter(tvals[::TIME_SPACING] / HRS_TO_SECS, DATA_SAMPLES[gly_cond][:, jj])
            jj += 1
        plt.show()

    grads = np.zeros_like(yout)
    lik_dev = (DATA_SAMPLES[gly_cond] - yout[::TIME_SPACING, DATA_INDEX])/np.array([15,15,0.1])**2
    grads[::TIME_SPACING, DATA_INDEX] = lik_dev

    # backsolve
    time_start = time.time()
    solver.solve_backward(t0=tvals[-1], tend= tvals[0],tvals=tvals[1:-1],
                          grads=grads, grad_out=grad_out, lamda_out=lambda_out)
    time_end = time.time()
    time_tot += (time_end-time_start)/60

    grad_out = -np.matmul(sens0,lambda_out-grads[0,:]) + grad_out
    print(grad_out)
    for j,param in enumerate(DEV_PARAMETERS_LIST):
        if param == 'G_EXT_INIT':
            lik_dev_params[N_MODEL_PARAMETERS + exp_ind] += grad_out[j]
        elif param in ['L','k','A']:
            jj = ['L','k','A'].index(param)
            lik_dev_params[N_MODEL_PARAMETERS + 4 + exp_ind*N_DCW_PARAMETERS + jj ] += grad_out[j]
        else:
            lik_dev_params[j] += grad_out[j]

print(yout[0,:3].sum()*yout[0,-1]*DCW_TO_CELL_COUNT*CELL_VOLUME + yout[0,3:].sum()*EXTERNAL_VOLUME)
print(yout[-1,:3].sum()*yout[-1,-1]*DCW_TO_CELL_COUNT*CELL_VOLUME + yout[-1,3:].sum()*EXTERNAL_VOLUME)

print(lik_dev_params[:N_MODEL_PARAMETERS])
print(lik_dev_params[N_MODEL_PARAMETERS:(N_MODEL_PARAMETERS+4)])
print(lik_dev_params[(N_MODEL_PARAMETERS+4):])
print(time_tot)
