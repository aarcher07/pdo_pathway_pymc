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
    # PermCellDHA = 10**params.PermCellDHA

    k1DhaB = 10**params.k1DhaB
    k2DhaB = 10**params.k2DhaB
    k3DhaB = 10**params.k3DhaB
    k4DhaB = 10**(params.k1DhaB + params.k3DhaB - params.KeqDhaB - params.k2DhaB)

    k1DhaT = 10**params.k1DhaT
    k2DhaT = 10**params.k2DhaT
    k3DhaT =  10**params.k3DhaT
    k4DhaT = 10**params.k4DhaT
    k5DhaT = 10**params.k5DhaT
    k6DhaT = 10**params.k6DhaT
    k7DhaT = 10**params.k7DhaT
    k8DhaT = 10**(params.k1DhaT + params.k3DhaT + params.k5DhaT + params.k7DhaT - params.KeqDhaT - params.k2DhaT
                  - params.k4DhaT - params.k6DhaT)

    # k1DhaD = 10**params.k1DhaD
    # k2DhaD = 10**params.k2DhaD
    # k3DhaD = 10**params.k3DhaD
    # k4DhaD = 10**params.k4DhaD
    # k5DhaD = 10**params.k5DhaD
    # k6DhaD = 10**params.k6DhaD
    # k7DhaD = 10**params.k7DhaD
    # k8DhaD = 10**(params.k1DhaD + params.k3DhaD + params.k5DhaD + params.k7DhaD - params.KeqDhaD
    #               - params.k2DhaD - params.k4DhaD - params.k6DhaD)

    k1E0 = 10**params.k1E0
    k2E0 = 10**params.k2E0
    k3E0 = 10**params.k3E0
    k4E0 = 10**params.k4E0

    VmaxfDhaK = 10**params.VmaxfDhaK # TODO: CHANGE
    KmDhaK = 10**params.KmDhaK


    L = 10**params.L
    k = 10**params.k

    cell_area_cell_volume = CELL_SURFACE_AREA / CELL_VOLUME
    cell_area_external_volume = CELL_SURFACE_AREA / EXTERNAL_VOLUME
    R_DhaK = VmaxfDhaK * x.G_CYTO / (KmDhaK + x.G_CYTO)

    d['G_CYTO'] = cell_area_cell_volume * PermCellGlycerol * (x.G_EXT - x.G_CYTO) \
                  - k1DhaB * x.G_CYTO * x.DHAB + k2DhaB * x.DHAB_C  \
                  - R_DhaK #- k3DhaD * x.G_CYTO * x.DHAD_NAD + k4DhaD * x.DHAD_NAD_GLY

    d['H_CYTO'] = cell_area_cell_volume * PermCell3HPA * (x.H_EXT - x.H_CYTO) \
                  + k3DhaB * x.DHAB_C - k4DhaB * x.H_CYTO * x.DHAB \
                  - k3DhaT * x.H_CYTO * x.DHAT_NADH + k4DhaT * x.DHAT_NADH_HPA

    d['P_CYTO'] = cell_area_cell_volume * PermCellPDO * (x.P_EXT - x.P_CYTO) \
                  - k6DhaT * x.P_CYTO * x.DHAT_NAD + k5DhaT * x.DHAT_NADH_HPA

    # d['DHA_CYTO'] = cell_area_cell_volume * PermCellDHA * (x.DHA_EXT - x.DHA_CYTO) \
    #            - k6DhaD * x.DHA_CYTO * x.DHAD_NADH + k5DhaD * x.DHAD_NAD_GLY - R_DhaK

    d['NAD'] = k7DhaT * x.DHAT_NAD - k8DhaT * x.DHAT * x.NAD\
               - k4E0 * x.NAD * x.E0 + k3E0 * x.E0_C\
             # - k1DhaD * x.NAD * x.DHAD + k2DhaD * x.DHAD_NAD\

    d['NADH'] = - k1DhaT * x.NADH * x.DHAT + k2DhaT * x.DHAT_NADH \
                - k1E0 * x.NADH * x.E0 + k2E0 * x.E0_C \
                #+ k7DhaD * x.DHAD_NADH - k8DhaD * x.NADH * x.DHAD \

    d['DHAB'] = - k1DhaB * x.G_CYTO * x.DHAB + k2DhaB * x.DHAB_C + k3DhaB * x.DHAB_C \
        - k4DhaB * x.H_CYTO * x.DHAB

    d['DHAB_C'] = k1DhaB * x.G_CYTO * x.DHAB - k2DhaB * x.DHAB_C - k3DhaB * x.DHAB_C \
                  + k4DhaB * x.H_CYTO * x.DHAB

    d['DHAT'] = - k1DhaT * x.NADH * x.DHAT + k2DhaT * x.DHAT_NADH + k7DhaT * x.DHAT_NAD \
                - k8DhaT * x.DHAT * x.NAD

    d['DHAT_NADH'] = k1DhaT * x.NADH * x.DHAT - k2DhaT * x.DHAT_NADH \
                      - k3DhaT * x.DHAT_NADH * x.H_CYTO + k4DhaT * x.DHAT_NADH_HPA

    d['DHAT_NADH_HPA'] = -k4DhaT * x.DHAT_NADH_HPA + k3DhaT * x.DHAT_NADH * x.H_CYTO \
                          - k5DhaT * x.DHAT_NADH_HPA + k6DhaT * x.DHAT_NAD * x.P_CYTO

    d['DHAT_NAD'] = k5DhaT * x.DHAT_NADH_HPA - k6DhaT * x.DHAT_NAD * x.P_CYTO \
                    - k7DhaT * x.DHAT_NAD + k8DhaT * x.DHAT * x.NAD

    # d['DHAD'] = - k1DhaD * x.NAD * x.DHAD + k2DhaD * x.DHAD_NAD + k7DhaD * x.DHAD_NADH \
    #             - k8DhaD * x.NADH * x.DHAD
    #
    # d['DHAD_NAD'] =  k1DhaD * x.NAD * x.DHAD - k2DhaD * x.DHAD_NAD \
    #                  - k3DhaD * x.DHAD_NAD * x.G_CYTO + k4DhaD * x.DHAD_NAD_GLY
    #
    # d['DHAD_NAD_GLY'] = -k4DhaD * x.DHAD_NAD_GLY + k3DhaD * x.DHAD_NAD * x.G_CYTO \
    #                     - k5DhaD * x.DHAD_NAD_GLY + k6DhaD * x.DHAD_NADH * x.DHA_CYTO
    #
    # d['DHAD_NADH'] = k5DhaD * x.DHAD_NAD_GLY - k6DhaD * x.DHAD_NADH * x.DHA_CYTO \
    #                 - k7DhaD * x.DHAD_NADH + k8DhaD * x.DHAD * x.NADH

    d['E0'] = - k1E0 * x.NADH * x.E0 + k2E0 * x.E0_C + k3E0 * x.E0_C \
              - k4E0 * x.NAD * x.E0

    d['E0_C'] = k1E0 * x.NADH * x.E0 - k2E0 * x.E0_C - k3E0 * x.E0_C \
                + k4E0 * x.NAD * x.E0

    ################################################################################################################
    ######################################### external volume equations ############################################
    ################################################################################################################

    d['G_EXT'] = ncells * cell_area_external_volume * PermCellGlycerol * (x.G_CYTO - x.G_EXT)
    d['H_EXT'] = ncells * cell_area_external_volume * PermCell3HPA * (x.H_CYTO - x.H_EXT)
    d['P_EXT'] = ncells * cell_area_external_volume * PermCellPDO * (x.P_CYTO - x.P_EXT)
    # d['DHA_EXT'] = ncells * cell_area_external_volume * PermCellDHA * (x.DHA_CYTO - x.DHA_EXT)

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