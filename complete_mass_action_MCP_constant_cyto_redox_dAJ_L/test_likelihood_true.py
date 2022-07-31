from prior_constants import *
import numpy as np
from likelihood_funcs_adj_true import likelihood_adj_true, likelihood_derivative_adj_true
from likelihood_funcs_fwd_true import likelihood_fwd_true, likelihood_derivative_fwd_true
import time

from constants import *
param_sample = np.array([*CELL_PERMEABILITY_MEAN.values(),
                         *MCP_PERMEABILITY_MEAN.values(),
                         *KINETIC_PARAMETER_MEAN.values(),
                         *GEOMETRY_PARAMETER_MEAN.values(),
                         np.log10((MCP_RADIUS*(10**(GEOMETRY_PARAMETER_MEAN['nMCPs']/3.))
                                   + MCP_RADIUS*(10**(GEOMETRY_PARAMETER_MEAN['nMCPs']/2.)))/2),
                         *COFACTOR_NUMBER_PARAMETER_MEAN.values(),
                         *PDU_ENZ_NUMBERS_PARAMETER_MEAN.values(),
                         ])
likelihood_adj_true(param_sample)
likelihood_fwd_true(param_sample)

time_start_adj = time.time()
lik_adj = likelihood_adj_true(param_sample)
time_end_adj = time.time()

time_start_fwd = time.time()
lik_fwd = likelihood_fwd_true(param_sample)
time_end_fwd = time.time()

print('lik adj: ' + str(lik_adj))
print('time adj: ' +  str((time_end_adj - time_start_adj)/60))
print('lik fwd: ' + str(lik_fwd))
print('time fwd: ' +  str((time_end_fwd - time_start_fwd)/60))

time_start_adj = time.time()
dev_adj = likelihood_derivative_adj_true(param_sample, fwd_rtol=1e-6, bck_atol=1e-6, bck_mxsteps=int(1e5))
time_end_adj = time.time()

time_start_fwd = time.time()
dev_fwd = likelihood_derivative_fwd_true(param_sample)
time_end_fwd = time.time()

print(dev_adj)
print('time adj: ' +  str((time_end_adj - time_start_adj)/60))
print('time fwd: ' +  str((time_end_fwd - time_start_fwd)/60))

diff = dev_adj - dev_fwd
rel_diff = abs(diff/dev_adj)
print('abs diff ' + str(diff))
print('rel diff' + str(rel_diff))
