from prior_constants import *
import numpy as np
from likelihood_funcs_adj import likelihood_adj, likelihood_derivative_adj
from likelihood_funcs_fwd import likelihood_fwd, likelihood_derivative_fwd
import time

from constants import *
param_sample = np.array([*CELL_PERMEABILITY_MEAN.values(),
                         *MCP_PERMEABILITY_MEAN.values(),
                         *KINETIC_PARAMETER_MEAN.values(),
                         *GEOMETRY_PARAMETER_MEAN.values(),
                         np.log10((MCP_RADIUS*(10**(GEOMETRY_PARAMETER_MEAN['nMCPs']/3.))
                                   + MCP_RADIUS*(10**(GEOMETRY_PARAMETER_MEAN['nMCPs']/2.)))/2),
                         *COFACTOR_NUMBER_PARAMETER_MEAN.values(),
                         *PDU_WT_ENZ_NUMBERS_PARAMETER_MEAN.values(),
                         *dPDU_AJ_ENZ_NUMBER_PARAMETER_MEAN.values(),
                         ])
likelihood_adj(param_sample)
likelihood_fwd(param_sample)

time_start_adj = time.time()
lik_adj = likelihood_adj(param_sample)
time_end_adj = time.time()

time_start_fwd = time.time()
lik_fwd = likelihood_fwd(param_sample)
time_end_fwd = time.time()

print('lik adj: ' + str(lik_adj))
print('time adj: ' +  str((time_end_adj - time_start_adj)/60))
print('lik fwd: ' + str(lik_fwd))
print('time fwd: ' +  str((time_end_fwd - time_start_fwd)/60))

time_start_adj = time.time()
dev_adj = likelihood_derivative_adj(param_sample, bck_rtol=1e-4, bck_atol=1e-4, fwd_mxsteps=int(1e5), bck_mxsteps=int(1e5))
time_end_adj = time.time()

time_start_fwd = time.time()
dev_fwd = likelihood_derivative_fwd(param_sample)
time_end_fwd = time.time()

print(dev_adj)
print('time adj: ' +  str((time_end_adj - time_start_adj)/60))
print('time fwd: ' +  str((time_end_fwd - time_start_fwd)/60))

diff = dev_adj - dev_fwd
rel_diff = abs(diff/dev_adj)
print('abs diff ' + str(diff))
print('rel diff' + str(rel_diff))
