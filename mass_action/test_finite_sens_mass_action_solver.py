import sunode
import matplotlib.pyplot as plt
import numpy as np
from constants import *
import pickle
from exp_data import *
from prior_constants import NORM_PRIOR_STD_RT_SINGLE_EXP,NORM_PRIOR_MEAN_SINGLE_EXP, NORM_PRIOR_STD_RT_ALL_EXP, NORM_PRIOR_MEAN_ALL_EXP
import time
PARAMETER_SAMP_PATH = '/Volumes/Wario/PycharmProjects/pdo_pathway_model/MCMC/output'
FILE_NAME = '/MCMC_results_data/mass_action/adaptive/preset_std/lambda_0,05_beta_0,01_burn_in_n_cov_2000/nsamples_100000/date_2022_03_04_02_11_52_142790_rank_0.pkl'
NN = np.sum([val.shape[1]*val.shape[0] for val in DATA_SAMPLES.values()])

def RHS(t, x, params):
    """
    Computes the spatial derivative of the system at time point, t
    :param_mean t: time
    :param_mean x: state variables
    :param_mean params: parameter list
    """

    ################################################################################################################
    ################################################# Initialization ###############################################
    ################################################################################################################

    # differential equation parameters
    d = {} # convert to list to allow use of symbolic derivatives

    # cell growth
    # differential equation parameters
    ncells = x.dcw * DCW_TO_CELL_COUNT

    ################################################################################################################
    #################################################### cytosol reactions #########################################
    ################################################################################################################

    PermCellGlycerol = 10**vars(params)['PermCellGlycerol'] # todo to all variables
    PermCell3HPA = 10**params.PermCell3HPA
    PermCellPDO = 10**params.PermCellPDO

    k1DhaB = 10**params.k1DhaB
    k2DhaB = 10**params.k2DhaB
    k3DhaB = 10**params.k3DhaB
    k4DhaB = 10**params.k4DhaB
    k1DhaT = 10**params.k1DhaT
    k2DhaT = 10**params.k2DhaT
    k3DhaT = 10**params.k3DhaT
    k4DhaT = 10**params.k4DhaT

    VmaxfMetab = 10**params.VmaxfMetab
    KmMetabG = 10**params.KmMetabG


    L = 10**params.L
    k = 10**params.k

    cell_area_cell_volume = CELL_SURFACE_AREA / CELL_VOLUME
    cell_area_external_volume = CELL_SURFACE_AREA / EXTERNAL_VOLUME
    R_Metab = VmaxfMetab * x.G_CYTO / (KmMetabG + x.G_CYTO)

    d['G_CYTO'] = cell_area_cell_volume * PermCellGlycerol * (x.G_EXT - x.G_CYTO) - k1DhaB * x.G_CYTO * x.DHAB + k2DhaB * x.DHAB_C - R_Metab  # microcompartment equation for G
    d['H_CYTO'] = cell_area_cell_volume * PermCell3HPA * (x.H_EXT - x.H_CYTO) + k3DhaB * x.DHAB_C - k4DhaB * \
           x.H_CYTO * x.DHAB - k1DhaT * x.H_CYTO * x.DHAT + k2DhaT *x. DHAT_C  # microcompartment equation for H
    d['P_CYTO'] = cell_area_cell_volume * PermCellPDO * (x.P_EXT - x.P_CYTO) - k4DhaT * x.P_CYTO * x.DHAT + k3DhaT * x.DHAT_C \
        # microcompartment equation for P

    d['DHAB'] = - k1DhaB * x.G_CYTO * x.DHAB + k2DhaB * x.DHAB_C + k3DhaB * x.DHAB_C \
           - k4DhaB * x.H_CYTO * x.DHAB

    d['DHAB_C'] = k1DhaB * x.G_CYTO * x.DHAB - k2DhaB * x.DHAB_C - k3DhaB * x.DHAB_C \
           + k4DhaB * x.H_CYTO * x.DHAB

    d['DHAT'] = - k1DhaT * x.H_CYTO * x.DHAT + k2DhaT * x.DHAT_C + k3DhaT * x.DHAT_C \
           - k4DhaT * x.P_CYTO * x.DHAT

    d['DHAT_C'] = k1DhaT * x.H_CYTO * x.DHAT - k2DhaT * x.DHAT_C - k3DhaT * x.DHAT_C \
           + k4DhaT * x.P_CYTO * x.DHAT

    ################################################################################################################
    ######################################### external volume equations ############################################
    ################################################################################################################
    d['G_EXT'] = ncells * cell_area_external_volume * PermCellGlycerol * (x.G_CYTO - x.G_EXT)
    d['H_EXT'] = ncells * cell_area_external_volume * PermCell3HPA * (x.H_CYTO - x.H_EXT)
    d['P_EXT'] = ncells * cell_area_external_volume * PermCellPDO * (x.P_CYTO - x.P_EXT)
    d['dcw'] = k*(1-x.dcw/L)*x.dcw
    return d


problem = sunode.symode.SympyProblem(
    params={ param: () for param in PARAMETER_LIST},

    states={ var: () for var in VARIABLE_NAMES},

    rhs_sympy=RHS,

    derivative_params=[ (param,)  for param in PARAMETER_LIST]
)


#
# The solver generates uses numba and sympy to generate optimized C functions
solver = sunode.solver.Solver(problem, solver='BDF')

#
#
# # We can use numpy structured arrays as input, so that we don't need
# # to think about how the different variables are stored in the array.
# # This does not introduce any runtime overhead during solving.

exp_ind = 1
N_MODEL_PARAMETERS = 15
N_DCW_PARAMETERS = 3
N_UNKNOWN_PARAMETERS = 19
N_TOTAL_PARAMETERS = 15 + 4 + 12
with open(PARAMETER_SAMP_PATH + FILE_NAME, 'rb') as f:
    postdraws = pickle.load(f)
    samples = postdraws['samples']
    burn_in_subset_samples = samples[int(2e4):]
    data_subset = burn_in_subset_samples[::600,:]
    param_mean = data_subset.mean(axis=0)
    param_mean_trans = np.matmul(NORM_PRIOR_STD_RT_ALL_EXP[:len(param_mean), :len(param_mean)].T, param_mean) + NORM_PRIOR_MEAN_ALL_EXP[
                                                                                                                :len(param_mean)]
param_vals = np.zeros((N_TOTAL_PARAMETERS,))
param_vals[:(N_MODEL_PARAMETERS + 4)] = param_mean_trans
for exp_ind, gly_cond in enumerate([50,60,70,80]):
    param_sample = NORM_PRIOR_MEAN_SINGLE_EXP[gly_cond].copy()
    param_vals[N_MODEL_PARAMETERS + exp_ind] = np.log10(param_vals[N_MODEL_PARAMETERS + exp_ind])
    param_vals[N_MODEL_PARAMETERS + 4 + exp_ind*N_DCW_PARAMETERS + 0] = np.log10(param_sample[PARAMETER_LIST.index('L')])
    param_vals[N_MODEL_PARAMETERS + 4 + exp_ind*N_DCW_PARAMETERS + 1] = np.log10(param_sample[PARAMETER_LIST.index('k')]/HRS_TO_SECS)
    param_vals[N_MODEL_PARAMETERS + 4 + exp_ind*N_DCW_PARAMETERS + 2] = np.log10(param_sample[PARAMETER_LIST.index('A')])

def likelihood(param_vals):
    lik_dev = 0
    param_sample = np.zeros(N_UNKNOWN_PARAMETERS)
    for exp_ind, gly_cond in enumerate([50,60,70,80]):
        param_sample[:N_MODEL_PARAMETERS] = param_vals[:N_MODEL_PARAMETERS]
        param_sample[N_MODEL_PARAMETERS+0] = param_vals[N_MODEL_PARAMETERS + exp_ind]
        param_sample[N_MODEL_PARAMETERS+1] = param_vals[N_MODEL_PARAMETERS + 4 + exp_ind*N_DCW_PARAMETERS + 0]
        param_sample[N_MODEL_PARAMETERS+2] = param_vals[N_MODEL_PARAMETERS + 4 + exp_ind*N_DCW_PARAMETERS + 1]
        param_sample[N_MODEL_PARAMETERS+3] = param_vals[N_MODEL_PARAMETERS + 4 + exp_ind*N_DCW_PARAMETERS + 2]

        tvals = TIME_SAMPLES[gly_cond]*HRS_TO_SECS
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
        solver.set_params_dict(params_dict)
        yout = solver.make_output_buffers(tvals)

        solver.solve(t0=0, tvals=tvals, y0=y0, y_out=yout)
        jj = 0
        # for i,var in enumerate(VARIABLE_NAMES):
        #     plt.plot(tvals/HRS_TO_SECS,yout.view(problem.state_dtype)[var])
        #     if i in [7,9,10]:
        #         plt.scatter(tvals/HRS_TO_SECS, DATA_SAMPLES[gly_cond][:,jj])
        #         jj+=1
        #     plt.show()
        # We can convert the solution to an xarray Dataset
        lik_dev += (((DATA_SAMPLES[gly_cond]-yout[:,[7,9,10]])/np.array([15,15,0.1]))**2).sum()/NN
    return lik_dev

lik = likelihood(param_vals)
lik_dev_params = np.zeros_like(param_vals)
eps = 1e-5
for i in range(lik_dev_params.shape[0]):
     shift = np.zeros_like(param_vals)
     shift[i] = eps
     lik_dev_params[i] = (likelihood(param_vals+shift) - lik)/eps
print(lik_dev_params[N_MODEL_PARAMETERS:(N_MODEL_PARAMETERS+4)])
print(lik_dev_params[(N_MODEL_PARAMETERS+4):])
