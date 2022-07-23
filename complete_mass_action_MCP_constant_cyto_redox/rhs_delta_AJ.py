import sunode
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #for \text command
import numpy as np
from constants import *

def RHS_delta_AJ(t, x, params):
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
    ncells = x.OD * OD_TO_CELL_COUNT

    ################################################################################################################
    #################################################### cytosol reactions #########################################
    ################################################################################################################

    PermCellGlycerol = 10**vars(params)['PermCellGlycerol'] # todo to all variables
    PermCell3HPA = 10**params.PermCell3HPA
    PermCellPDO = 10**params.PermCellPDO
    PermCellHCoA = 10**params.PermCellHCoA
    PermCellHPhosph = 10**params.PermCellHPhosph
    # PermCellHate = 10**params.PermCellHate

    PermPolar = 0.5

    k1PduCDE = 10**params.k1PduCDE
    k2PduCDE = 10**params.k2PduCDE
    k3PduCDE = 10**params.k3PduCDE
    k4PduCDE = 10**(params.k1PduCDE + params.k3PduCDE - params.KeqPduCDE - params.k2PduCDE)

    k1PduQ = 10**params.k1PduQ
    k2PduQ =  10**params.k2PduQ
    k3PduQ =  10**params.k3PduQ
    k4PduQ = 10**params.k4PduQ
    k5PduQ = 10**params.k5PduQ
    k6PduQ = 10**params.k6PduQ
    k7PduQ = 10**params.k7PduQ
    k8PduQ = 10**(params.k1PduQ + params.k3PduQ + params.k5PduQ + params.k7PduQ - params.KeqPduQ - params.k2PduQ
                  - params.k4PduQ - params.k6PduQ)

    k1PduP = 10**params.k1PduP
    k2PduP = 10**params.k2PduP
    k3PduP = 10**params.k3PduP
    k4PduP = 10**params.k4PduP
    k5PduP = 10**params.k5PduP
    k6PduP = 10**params.k6PduP
    k7PduP = 10**params.k7PduP
    k8PduP = 10**(params.k1PduP + params.k3PduP + params.k5PduP + params.k7PduP - params.KeqPduP
                  - params.k2PduP - params.k4PduP - params.k6PduP)

    k1PduL = 10**params.k1PduL
    k2PduL = 10**params.k2PduL
    k3PduL = 10**params.k3PduL
    k4PduL = 10**(params.k1PduL + params.k3PduL - params.KeqPduL - params.k2PduL)

    # k1PduW = 10**params.k1PduW
    # k2PduW = 10**params.k2PduW
    # k3PduW = 10**params.k3PduW
    # k4PduW = 10**(params.k1PduW + params.k3PduW - (params.KeqPduLW - params.KeqPduL) - params.k2PduW)

    VmaxfGlpK = 10**params.VmaxfGlpK # TODO: CHANGE
    KmGlpK = 10**params.KmGlpK

    radius_AJ = 10**params.AJ_radius
    polar_volume = (4./3.)*np.pi*(radius_AJ**3)
    polar_surface_area = 4*np.pi*(radius_AJ**2)

    NADH_NAD_TOTAL_CYTO = 10 ** params.NADH_NAD_TOTAL_CYTO
    NADH_NAD_RATIO_CYTO = 10 ** params.NADH_NAD_RATIO_CYTO

    L = 10**params.L
    k = 10**params.k

    NADH_CYTO = NADH_NAD_TOTAL_CYTO * NADH_NAD_RATIO_CYTO / (1 + NADH_NAD_RATIO_CYTO)
    NAD_CYTO  =  NADH_NAD_TOTAL_CYTO * 1 / (1 + NADH_NAD_RATIO_CYTO)

    polar_surface_area_polar_volume = polar_surface_area / polar_volume
    polar_surface_area_cell_volume = polar_surface_area / CELL_VOLUME
    cell_area_cell_volume = CELL_SURFACE_AREA / CELL_VOLUME
    cell_area_external_volume = CELL_SURFACE_AREA / EXTERNAL_VOLUME
    R_GlpK = VmaxfGlpK * x.G_CYTO / (KmGlpK + x.G_CYTO)

    ################################################################################################################
    ################################################ MCP equations #################################################
    ################################################################################################################

    d['G_MCP'] = polar_surface_area_polar_volume * PermPolar * (x.G_CYTO - x.G_MCP) \
                 - k1PduCDE * x.G_MCP * x.PduCDE + k2PduCDE * x.PduCDE_C

    d['H_MCP'] = polar_surface_area_polar_volume * PermPolar * (x.H_CYTO - x.H_MCP) \
                 + k3PduCDE * x.PduCDE_C - k4PduCDE * x.H_MCP * x.PduCDE \
                 - k3PduQ * x.H_MCP * x.PduQ_NADH + k4PduQ * x.PduQ_NADH_HPA \
                 - k3PduP * x.H_MCP * x.PduP_NAD + k4PduP * x.PduP_NAD_HPA

    d['P_MCP'] = polar_surface_area_polar_volume * PermPolar * (x.P_CYTO - x.P_MCP) \
                 - k6PduQ * x.P_MCP * x.PduQ_NAD + k5PduQ * x.PduQ_NADH_HPA

    d['HCoA_MCP'] = polar_surface_area_polar_volume * PermPolar * (x.HCoA_CYTO - x.HCoA_MCP) \
                    - k6PduP * x.HCoA_MCP * x.PduP_NADH + k5PduP * x.PduP_NAD_HPA \
                    - k1PduL * x.HCoA_MCP * x.PduL + k2PduL * x.PduL_C

    d['HPhosph_MCP'] = polar_surface_area_polar_volume * PermPolar * (x.HPhosph_CYTO - x.HPhosph_MCP) \
                       + k3PduL * x.PduL_C - k4PduL * x.HPhosph_MCP * x.PduL

    d['NAD_MCP'] = polar_surface_area_polar_volume * PermPolar * (NAD_CYTO - x.NAD_MCP) \
                   + k7PduQ * x.PduQ_NAD - k8PduQ * x.PduQ * x.NAD_MCP \
                   - k1PduP * x.NAD_MCP * x.PduP + k2PduP * x.PduP_NAD

    d['NADH_MCP'] = polar_surface_area_polar_volume * PermPolar * (NADH_CYTO - x.NADH_MCP) \
                    - k1PduQ * x.NADH_MCP * x.PduQ + k2PduQ * x.PduQ_NADH \
                    + k7PduP * x.PduP_NADH - k8PduP * x.NADH_MCP * x.PduP

    d['PduCDE'] = - k1PduCDE * x.G_MCP * x.PduCDE + k2PduCDE * x.PduCDE_C + k3PduCDE * x.PduCDE_C \
                  - k4PduCDE * x.H_MCP * x.PduCDE

    d['PduCDE_C'] = -d['PduCDE']

    d['PduQ'] = - k1PduQ * x.NADH_MCP * x.PduQ + k2PduQ * x.PduQ_NADH + k7PduQ * x.PduQ_NAD \
                - k8PduQ * x.PduQ * x.NAD_MCP

    d['PduQ_NADH'] = k1PduQ * x.NADH_MCP * x.PduQ - k2PduQ * x.PduQ_NADH \
                     - k3PduQ * x.PduQ_NADH * x.H_MCP + k4PduQ * x.PduQ_NADH_HPA

    d['PduQ_NADH_HPA'] = - k4PduQ * x.PduQ_NADH_HPA + k3PduQ * x.PduQ_NADH * x.H_MCP \
                         - k5PduQ * x.PduQ_NADH_HPA + k6PduQ * x.PduQ_NAD * x.P_MCP

    d['PduQ_NAD'] = k5PduQ * x.PduQ_NADH_HPA - k6PduQ * x.PduQ_NAD * x.P_MCP \
                    - k7PduQ * x.PduQ_NAD + k8PduQ * x.PduQ * x.NAD_MCP

    d['PduP'] = - k1PduP * x.NAD_MCP * x.PduP + k2PduP * x.PduP_NAD + k7PduP * x.PduP_NADH \
                - k8PduP * x.NADH_MCP * x.PduP

    d['PduP_NAD'] = k1PduP * x.NAD_MCP * x.PduP - k2PduP * x.PduP_NAD \
                    - k3PduP * x.PduP_NAD * x.H_MCP + k4PduP * x.PduP_NAD_HPA

    d['PduP_NAD_HPA'] = -k4PduP * x.PduP_NAD_HPA + k3PduP * x.PduP_NAD * x.H_MCP \
                        - k5PduP * x.PduP_NAD_HPA + k6PduP * x.PduP_NADH * x.HCoA_MCP

    d['PduP_NADH'] = k5PduP * x.PduP_NAD_HPA - k6PduP * x.PduP_NADH * x.HCoA_MCP \
                     - k7PduP * x.PduP_NADH + k8PduP * x.PduP * x.NADH_MCP

    d['PduL'] = - k1PduL * x.HCoA_MCP * x.PduL + k2PduL * x.PduL_C + k3PduL * x.PduL_C \
                - k4PduL * x.HPhosph_MCP * x.PduL

    d['PduL_C'] = -d['PduL']

    ################################################################################################################
    ############################################### cytosol equations ##############################################
    ################################################################################################################

    d['G_CYTO'] = - cell_area_cell_volume * PermCellGlycerol * (x.G_CYTO - x.G_EXT) \
                  - polar_surface_area_cell_volume * PermPolar * (x.G_CYTO - x.G_MCP) \
                  - R_GlpK

    d['H_CYTO'] = - cell_area_cell_volume * PermCell3HPA * (x.H_CYTO - x.H_EXT) \
                  - polar_surface_area_cell_volume * PermPolar * (x.H_CYTO - x.H_MCP)

    d['P_CYTO'] = - cell_area_cell_volume * PermCellPDO * (x.P_CYTO - x.P_EXT) \
                  - polar_surface_area_cell_volume * PermPolar * (x.P_CYTO - x.P_MCP)

    d['HCoA_CYTO'] = - cell_area_cell_volume * PermCellHCoA * (x.HCoA_CYTO - x.HCoA_EXT) \
                     - polar_surface_area_cell_volume * PermPolar * (x.HCoA_CYTO - x.HCoA_MCP)

    d['HPhosph_CYTO'] = - cell_area_cell_volume * PermCellHPhosph * (x.HPhosph_CYTO - x.HPhosph_EXT) \
                        - polar_surface_area_cell_volume * PermPolar * (x.HPhosph_CYTO - x.HPhosph_MCP)
                        # - k1PduW * x.HPhosph_CYTO * x.PduW + k2PduW * x.PduW_C

    # d['Hate_CYTO'] = - cell_area_cell_volume * PermCellHate * (x.Hate_CYTO - x.Hate_EXT) \
    #                  + k3PduW * x.PduW_C - k4PduW * x.Hate_CYTO * x.PduW
    #
    # d['PduW'] = - k1PduW * x.HPhosph_CYTO * x.PduW + k2PduW * x.PduW_C + k3PduW * x.PduW_C \
    #             - k4PduW * x.Hate_CYTO * x.PduW
    #
    # d['PduW_C'] = -d['PduW']

    ###############################################################################################################
    ######################################### external volume equations ############################################
    ################################################################################################################

    d['G_EXT'] = ncells * cell_area_external_volume * PermCellGlycerol * (x.G_CYTO - x.G_EXT)
    d['H_EXT'] = ncells * cell_area_external_volume * PermCell3HPA * (x.H_CYTO - x.H_EXT)
    d['P_EXT'] = ncells * cell_area_external_volume * PermCellPDO * (x.P_CYTO - x.P_EXT)
    d['HCoA_EXT'] = ncells * cell_area_external_volume * PermCellHCoA * (x.HCoA_CYTO - x.HCoA_EXT)
    d['HPhosph_EXT'] = ncells * cell_area_external_volume * PermCellHPhosph * (x.HPhosph_CYTO - x.HPhosph_EXT)
    # d['Hate_EXT'] = ncells * cell_area_external_volume * PermCellHate * (x.Hate_CYTO - x.Hate_EXT)
    d['OD'] = k * (1 - x.OD / L) * x.OD

    return d


problem_delta_AJ = sunode.symode.SympyProblem(
    params={ param: () for param in PARAMETER_LIST},

    states={ var: () for var in VARIABLE_NAMES},

    rhs_sympy=RHS_delta_AJ,

    derivative_params=[ (param,)  for param in DEV_PARAMETER_LIST]
)

#
