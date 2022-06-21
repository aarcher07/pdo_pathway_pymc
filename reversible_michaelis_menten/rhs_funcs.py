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

    VmaxfDhaB = 10**params.VmaxfDhaB
    KmGlycerolDhaB = 10**params.KmGlycerolDhaB
    VmaxrDhaB = 10**params.VmaxrDhaB
    KmHPADhaB = 10**params.KmHPADhaB
    VmaxfDhaT = 10**params.VmaxfDhaT
    KmHPADhaT = 10**params.KmHPADhaT
    VmaxrDhaT = 10**params.VmaxrDhaT
    KmPDODhaT = 10**params.KmPDODhaT
    VmaxfMetab = 10**params.VmaxfMetab
    KmMetabG = 10**params.KmMetabG


    L = 10**params.L
    k = 10**params.k

    cell_area_cell_volume = CELL_SURFACE_AREA / CELL_VOLUME
    cell_area_external_volume = CELL_SURFACE_AREA / EXTERNAL_VOLUME
    R_DhaBf = VmaxfDhaB * x.G_CYTO / (KmGlycerolDhaB + x.G_CYTO)
    R_DhaBr = VmaxrDhaB * x.G_CYTO / (KmHPADhaB + x.G_CYTO)

    R_DhaTf = VmaxfDhaT * x.G_CYTO / (KmHPADhaT + x.G_CYTO)
    R_DhaTr = VmaxrDhaT * x.G_CYTO / (KmPDODhaT + x.G_CYTO)

    R_Metab = VmaxfMetab * x.G_CYTO / (KmMetabG + x.G_CYTO)

    d['G_CYTO'] = cell_area_cell_volume * PermCellGlycerol * (x.G_EXT - x.G_CYTO) - R_DhaBf + R_DhaBr - R_Metab  # microcompartment equation for G
    d['H_CYTO'] = cell_area_cell_volume * PermCell3HPA * (x.H_EXT - x.H_CYTO) + R_DhaBf - R_DhaBr - R_DhaTf + R_DhaTr # microcompartment equation for H
    d['P_CYTO'] = cell_area_cell_volume * PermCellPDO * (x.P_EXT - x.P_CYTO) + R_DhaTf - R_DhaTr
        # microcompartment equation for P

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