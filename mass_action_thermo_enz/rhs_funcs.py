import sunode
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #for \text command
import numpy as np
from constants import *


lib = sunode._cvodes.lib

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
    k4DhaB = 10**(params.k1DhaB + params.k3DhaB - params.KeqDhaB - params.k2DhaB)
    k1DhaT = 10**params.k1DhaT
    k2DhaT = 10**params.k2DhaT
    k3DhaT = 10**params.k3DhaT
    k4DhaT = 10**(params.k1DhaT + params.k3DhaT - params.KeqDhaT - params.k2DhaT)
    VmaxfMetab = 10**(params.kcatfMetab + params.E0_Metab)
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

    derivative_params=[ (param,)  for param in DEV_PARAMETERS_LIST]
)

#
# The solver generates uses numba and sympy to generate optimized C functions
solver = sunode.solver.AdjointSolver(problem, solver='BDF')