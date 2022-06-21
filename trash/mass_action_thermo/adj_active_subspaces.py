import sunode
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #for \text command
import numpy as np
from constants import *
import pickle
from exp_data import *
from prior_constants import NORM_PRIOR_STD_RT_SINGLE_EXP,NORM_PRIOR_MEAN_SINGLE_EXP, NORM_PRIOR_STD_RT_ALL_EXP, \
    NORM_PRIOR_MEAN_ALL_EXP, LOG_UNIF_PRIOR_ALL_EXP
from formatting_constants import VARS_ALL_EXP_TO_TEX
import time


lib = sunode._cvodes.lib
NN = np.sum([val.shape[1]*val.shape[0] for val in DATA_SAMPLES.values()])
PARAMETER_SAMP_PATH = '/Volumes/Wario/PycharmProjects/pdo_pathway_model/MCMC/output'
FILE_NAME = '/MCMC_results_data/mass_action/adaptive/preset_std/lambda_0,05_beta_0,01_burn_in_n_cov_2000/nsamples_100000/date_2022_03_04_02_11_52_142790_rank_0.pkl'

exp_ind = 1
N_MODEL_PARAMETERS = 15
N_DCW_PARAMETERS = 3
N_UNKNOWN_PARAMETERS = 19
N_TOTAL_PARAMETERS = 15 + 4 + 12

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
    k4DhaB = 10**(-params.DeltaGDhaB - params.k2DhaB + params.k3DhaB + params.k1DhaB)
    k1DhaT = 10**params.k1DhaT
    k2DhaT = 10**params.k2DhaT
    k3DhaT = 10**params.k3DhaT
    k4DhaT = 10**(-params.DeltaGDhaT - params.k2DhaT + params.k3DhaT + params.k1DhaT)
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
solver = sunode.solver.AdjointSolver(problem, solver='BDF')

def likelihood_derivative(param_vals):
    lik_dev_params = np.zeros((N_MODEL_PARAMETERS + 4 + 4*N_DCW_PARAMETERS,))
    param_sample = np.zeros(N_UNKNOWN_PARAMETERS)
    time_tot = 0
    for exp_ind, gly_cond in enumerate([50,60,70,80]):
        param_sample[:N_MODEL_PARAMETERS] = param_vals[:N_MODEL_PARAMETERS]
        param_sample[N_MODEL_PARAMETERS+0] = param_vals[N_MODEL_PARAMETERS + exp_ind]
        param_sample[N_MODEL_PARAMETERS+1] = param_vals[N_MODEL_PARAMETERS + 4 + exp_ind*N_DCW_PARAMETERS + 0]
        param_sample[N_MODEL_PARAMETERS+2] = param_vals[N_MODEL_PARAMETERS + 4 + exp_ind*N_DCW_PARAMETERS + 1] - np.log10(HRS_TO_SECS)
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
        lib.CVodeSStolerances(solver._ode, 1e-8, 1e-8)
        lib.CVodeSStolerancesB(solver._ode, solver._odeB, 1e-8, 1e-8)
        lib.CVodeQuadSStolerancesB(solver._ode, solver._odeB, 1e-8, 1e-8)
        lib.CVodeSetMaxNumSteps(solver._ode, 5000)
        lib.CVodeSetMaxNumStepsB(solver._ode, solver._odeB, 5000)
        yout, grad_out, lambda_out = solver.make_output_buffers(tvals)
        sens0 = np.zeros((19,11))
        sens0[PARAMETER_LIST.index('G_EXT_INIT'), VARIABLE_NAMES.index('G_CYTO')] = np.log(10)*(10**param_sample[PARAMETER_LIST.index('G_EXT_INIT')])
        sens0[PARAMETER_LIST.index('G_EXT_INIT'), VARIABLE_NAMES.index('G_EXT')] = np.log(10)*(10**param_sample[PARAMETER_LIST.index('G_EXT_INIT')])
        sens0[PARAMETER_LIST.index('DHAB_INIT'), VARIABLE_NAMES.index('DHAB')] = np.log(10)*(10**param_sample[PARAMETER_LIST.index('DHAB_INIT')])
        sens0[PARAMETER_LIST.index('DHAT_INIT'), VARIABLE_NAMES.index('DHAT')] = np.log(10)*(10**param_sample[PARAMETER_LIST.index('DHAT_INIT')])
        sens0[PARAMETER_LIST.index('A'), VARIABLE_NAMES.index('dcw')] = np.log(10)*(10**param_sample[PARAMETER_LIST.index('A')])

        time_start = time.time()
        solver.solve_forward(t0=0, tvals=tvals, y0=y0, y_out=yout)
        time_end = time.time()
        time_tot += (time_end-time_start)/60

        # jj=0
        # for i,var in enumerate(VARIABLE_NAMES):
        #     plt.plot(tvals/HRS_TO_SECS,yout.view(problem.state_dtype)[var])
        #     if i in [7,9,10]:
        #         plt.scatter(tvals/HRS_TO_SECS, DATA_SAMPLES[gly_cond][:,jj])
        #         jj+=1
        #     plt.show()


        # We can convert the solution to an xarray Dataset
        grads = np.zeros_like(yout)
        lik_dev = 0.5*(DATA_SAMPLES[gly_cond]-yout[:,[7,9,10]])/(NN*np.array([15,15,0.1])**2)
        grads[:,[7,9,10]] = lik_dev

        # backsolve
        solver.solve_backward(t0=tvals[-1], tend= tvals[0],tvals=tvals[1:-1][::-1],
                              grads=grads, grad_out=grad_out, lamda_out=lambda_out)
        grad_out = -np.matmul(sens0,lambda_out) + grad_out

        for j,param in enumerate(PARAMETER_LIST):
            if param == 'G_EXT_INIT':
                lik_dev_params[N_MODEL_PARAMETERS + exp_ind] += grad_out[j]
            elif param in ['L','k','A']:
                jj = ['L','k','A'].index(param)
                lik_dev_params[N_MODEL_PARAMETERS + 4 + exp_ind*N_DCW_PARAMETERS + jj ] += grad_out[j]
            else:
                lik_dev_params[j] += grad_out[j]
    return lik_dev_params

def sample_gradient(N):
    rng = np.random.default_rng(1)
    samples = rng.uniform(-1,1,size=(N,N_TOTAL_PARAMETERS))
    param_samples = samples*(LOG_UNIF_PRIOR_ALL_EXP[:,1] - LOG_UNIF_PRIOR_ALL_EXP[:,0])/2 + \
                    (LOG_UNIF_PRIOR_ALL_EXP[:,1] + LOG_UNIF_PRIOR_ALL_EXP[:,0])/2
    lik_dev = []
    for param_val in param_samples:
        try:
            lik_dev.append(likelihood_derivative(param_val))
        except sunode.solver.SolverError:
            continue

    lik_dev_array = np.array(lik_dev)
    lik_dev_array_trans = np.array(lik_dev_array * (2/(LOG_UNIF_PRIOR_ALL_EXP[:,1] - LOG_UNIF_PRIOR_ALL_EXP[:,0])))
    return np.matmul(lik_dev_array_trans.T,lik_dev_array_trans)/N

N = 10000
time_start = time.time()
outer_grad = np.array(sample_gradient(N))
time_end = time.time()
print((time_end-time_start)/N)
w, v = np.linalg.eigh(outer_grad)
w = np.flip(w)
v = np.flip(v,axis=1)

plt.bar(range(len(w)), np.log10(w))
x_axis_labels = [r"$\lambda_{" + str(i + 1) + "}$" for i in range(len(w))]
plt.yticks(fontsize=20)
plt.ylabel(r'$\log_{10}(\lambda_i)$', fontsize=20)
plt.xticks(list(range(len(w))), x_axis_labels, fontsize=7.5)
plt.savefig('active_subspaces_plots/adj_eigenvalues_mass_action_Nsamples_' +  str(N), bbox_inches='tight')
plt.close()

activity_scores = np.log10((v**2).dot(w))
sort_indices = np.argsort(activity_scores)
plt.bar(range(len(w)), activity_scores[sort_indices[::-1]])
plt.yticks(fontsize=20)
plt.ylabel(r'$\log_{10}(\mathrm{activity}\,\, \mathrm{scores})$', fontsize=10)
plt.xticks(list(range(len(w))), np.array(list(VARS_ALL_EXP_TO_TEX.values()))[sort_indices[::-1]], fontsize=7.5,
           rotation = 45)
plt.savefig('active_subspaces_plots/adj_activity_scores_mass_action_Nsamples_' +  str(N), bbox_inches='tight')
plt.close()

plt.imshow(v, cmap="Greys")
cbar = plt.colorbar()
plt.yticks(list(range(N_TOTAL_PARAMETERS)), list(VARS_ALL_EXP_TO_TEX.values()), fontsize=10)
cbar.ax.tick_params(labelsize=10)
plt.savefig('active_subspaces_plots/adj_eigenvectors_mass_action_Nsamples_' +  str(N), bbox_inches='tight')
plt.close()
