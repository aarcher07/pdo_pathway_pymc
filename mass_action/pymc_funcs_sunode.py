import matplotlib as mpl
import aesara
import aesara.tensor as at
import arviz as az
import matplotlib.pyplot as plt
import os
import pymc as pm
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
from prior_constants import NORM_PRIOR_STD_RT_SINGLE_EXP,NORM_PRIOR_MEAN_SINGLE_EXP, NORM_PRIOR_STD_RT_ALL_EXP, \
    NORM_PRIOR_MEAN_ALL_EXP, LOG_UNIF_PRIOR_ALL_EXP, DATA_LOG_UNIF_PARAMETER_RANGES, NORM_PRIOR_PARAMETER_ALL_EXP_DICT
from constants import PERMEABILITY_PARAMETERS, KINETIC_PARAMETERS, ENZYME_CONCENTRATIONS, GLYCEROL_EXTERNAL_EXPERIMENTAL,\
    ALL_PARAMETERS, VARIABLE_NAMES, DEV_PARAMETERS_LIST, TIME_SAMPLES_EXPANDED, HRS_TO_SECS, TIME_SPACING
from exp_data import INIT_CONDS_GLY_PDO_DCW, NORM_DCW_MEAN_PRIOR_TRANS_PARAMETERS, DATA_SAMPLES
from rhs_funcs import RHS
import time
from os.path import dirname, abspath
import sys
from pathlib import Path
import numpy as np
from datetime import datetime
from likelihood_funcs_adj import likelihood_adj, likelihood_derivative_adj  #TODO : change to _3HPA
from os.path import dirname, abspath
import sunode.wrappers.as_aesara

ROOT_PATH = dirname(abspath(__file__))
lib = sunode._cvodes.lib



with pm.Model() as model:
    permeability_params = [pm.TruncatedNormal(param_name, mu=NORM_PRIOR_PARAMETER_ALL_EXP_DICT[param_name][0],
                                              sigma= NORM_PRIOR_PARAMETER_ALL_EXP_DICT[param_name][1],
                                              lower=DATA_LOG_UNIF_PARAMETER_RANGES[param_name][0],
                                              upper=DATA_LOG_UNIF_PARAMETER_RANGES[param_name][1])
                           for param_name in PERMEABILITY_PARAMETERS]

    kinetic_params = [pm.TruncatedNormal(param_name, mu = NORM_PRIOR_PARAMETER_ALL_EXP_DICT[param_name][0],
                                sigma = NORM_PRIOR_PARAMETER_ALL_EXP_DICT[param_name][1], lower = -7, upper = 7)
                      for param_name in KINETIC_PARAMETERS]

    enzyme_init = [pm.TruncatedNormal(param_name, mu = NORM_PRIOR_PARAMETER_ALL_EXP_DICT[param_name][0],
                                      sigma = NORM_PRIOR_PARAMETER_ALL_EXP_DICT[param_name][1], lower = -4, upper = 2)
                   for param_name in ENZYME_CONCENTRATIONS]

    prior_variables = [*permeability_params, *kinetic_params, *enzyme_init]#, *gly_init]
    prior_variable_dict = {param:(prior_var, ()) for param, prior_var in zip(DEV_PARAMETERS_LIST, prior_variables)}
    y0 = {var: (np.float64(0), ()) for var in VARIABLE_NAMES}
    y0['DHAB'] = (pm.Deterministic('10_DHAB', 10 ** prior_variables[DEV_PARAMETERS_LIST.index('DHAB_INIT')]),
                  ())  # TODO: might be an issue
    y0['DHAT'] = (pm.Deterministic('10_DHAT', 10 ** prior_variables[DEV_PARAMETERS_LIST.index('DHAT_INIT')]),
                  ())  # TODO: might be an issue
    for exp_cond in [50, 60, 70, 80]:
        y0['G_CYTO'] = (INIT_CONDS_GLY_PDO_DCW[exp_cond][0], ())
        y0['P_CYTO'] = (INIT_CONDS_GLY_PDO_DCW[exp_cond][1], ())
        y0['G_EXT'] = (INIT_CONDS_GLY_PDO_DCW[exp_cond][0], ())
        y0['P_EXT'] = (INIT_CONDS_GLY_PDO_DCW[exp_cond][1], ())
        y0['dcw'] = (10**NORM_DCW_MEAN_PRIOR_TRANS_PARAMETERS.loc[exp_cond, 'mean A'], ())

        y_hat, _, problem, solver, _, _ = sunode.wrappers.as_aesara.solve_ivp(
            y0=y0,
            params={**prior_variable_dict,
                'L': 10**NORM_DCW_MEAN_PRIOR_TRANS_PARAMETERS.loc[exp_cond, 'mean L'],
                'k': 10 ** NORM_DCW_MEAN_PRIOR_TRANS_PARAMETERS.loc[exp_cond, 'mean L'],
                },
            # A functions that computes the right-hand-side of the ode using
            # sympy variables.
            rhs=RHS,
            # The time points where we want to access the solution
            tvals=TIME_SAMPLES_EXPANDED[exp_cond] * HRS_TO_SECS,
            t0=TIME_SAMPLES_EXPANDED[exp_cond][0] * HRS_TO_SECS,
        )
        lib.CVodeSStolerances(solver._ode, 1e-8, 1e-8)
        lib.CVodeSStolerancesB(solver._ode, solver._odeB, 1e-8, 1e-8)
        lib.CVodeQuadSStolerancesB(solver._ode, solver._odeB, 1e-8, 1e-8)
        lib.CVodeSetMaxNumSteps(solver._ode, int(1e5))
        lib.CVodeSetMaxNumStepsB(solver._ode, solver._odeB, int(1e5))

        pm.Deterministic('G_EXT_' + str(exp_cond), y_hat['G_EXT'])
        pm.Deterministic('P_EXT_' + str(exp_cond), y_hat['P_EXT'])
        pm.Deterministic('H_EXT_' + str(exp_cond), y_hat['H_CYTO'])
        pm.LogNormal('lik_G_EXT_' + str(exp_cond), mu=y_hat['G_EXT'][::TIME_SPACING], sigma=0.05, observed=DATA_SAMPLES[exp_cond][:,0])
        pm.LogNormal('lik_P_EXT_' + str(exp_cond), mu=y_hat['P_EXT'][::TIME_SPACING], sigma=0.05, observed=DATA_SAMPLES[exp_cond][:,1])

with model:
    idata = pm.sample(tune=50, draws=50, chains=6, cores=6)
    print(idata.posterior)