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
import pymc as pm
import sunode.wrappers.as_theano

lib = sunode._cvodes.lib

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


with pm.Model() as model:
    hares_start = pm.HalfNormal('hares_start', sd=50)
    lynx_start = pm.HalfNormal('lynx_start', sd=50)

    ratio = pm.Beta('ratio', alpha=0.5, beta=0.5)

    fixed_hares = pm.HalfNormal('fixed_hares', sd=50)
    fixed_lynx = pm.Deterministic('fixed_lynx', ratio * fixed_hares)

    period = pm.Gamma('period', mu=10, sd=1)
    freq = pm.Deterministic('freq', 2 * np.pi / period)

    log_speed_ratio = pm.Normal('log_speed_ratio', mu=0, sd=0.1)
    speed_ratio = np.exp(log_speed_ratio)

    # Compute the parameters of the ode based on our prior parameters
    alpha = pm.Deterministic('alpha', freq * speed_ratio * ratio)
    beta = pm.Deterministic('beta', freq * speed_ratio / fixed_hares)
    gamma = pm.Deterministic('gamma', freq / speed_ratio / ratio)
    delta = pm.Deterministic('delta', freq / speed_ratio / fixed_hares / ratio)

    y_hat, _, problem, solver, _, _ = sunode.wrappers.as_theano.solve_ivp(
        y0={
            # The initial conditions of the ode. Each variable
            # needs to specify a theano or numpy variable and a shape.
            # This dict can be nested.
            'hares': (hares_start, ()),
            'lynx': (lynx_start, ()),
        },
        params={
            # Each parameter of the ode. sunode will only compute derivatives
            # with respect to theano variables. The shape needs to be specified
            # as well. It it infered automatically for numpy variables.
            # This dict can be nested.
            'alpha': (alpha, ()),
            'beta': (beta, ()),
            'gamma': (gamma, ()),
            'delta': (delta, ()),
            'extra': np.zeros(1),
        },
        # A functions that computes the right-hand-side of the ode using
        # sympy variables.
        rhs=RHS,
        # The time points where we want to access the solution
        tvals=times,
        t0=times[0],
    )

    lib.CVodeSStolerances(solver._ode, 1e-10, 1e-10)
    lib.CVodeSStolerancesB(solver._ode, solver._odeB, 1e-8, 1e-8)
    lib.CVodeQuadSStolerancesB(solver._ode, solver._odeB, 1e-8, 1e-8)
    lib.CVodeSetMaxNumSteps(solver._ode, 5000)
    lib.CVodeSetMaxNumStepsB(solver._ode, solver._odeB, 5000)

    # We can access the individual variables of the solution using the
    # variable names.
    pm.Deterministic('hares_mu', y_hat['hares'])
    pm.Deterministic('lynx_mu', y_hat['lynx'])

    sd = pm.HalfNormal('sd')
    pm.Lognormal('hares', mu=y_hat['hares'], sd=sd, observed=hare_data)
    pm.Lognormal('lynx', mu=y_hat['lynx'], sd=sd, observed=lynx_data)
