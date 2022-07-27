import numpy as np
from exp_data_13pd import TIME_SAMPLES

#######################################################################################################################
############################################ CELL GEOMETRY CONSTANTS ##################################################
#######################################################################################################################

CELL_RADIUS = 0.375e-6
CELL_LENGTH = 2.47e-6
CELL_SURFACE_AREA = 2 * np.pi * CELL_RADIUS * CELL_LENGTH
CELL_VOLUME = 4 * np.pi / 3 * (CELL_RADIUS) ** 3 + np.pi * (CELL_LENGTH - 2 * CELL_RADIUS) * (CELL_RADIUS ** 2)

HRS_TO_SECS = 60 * 60
# DCW to CELL CONCENTRATION
OD_TO_DCW = 0.1  # grams of cell dry weight/L per 1 OD
OD_TO_CELL_CONCENTRATION = 1e15  # number of cell/m^3 per 1 OD
EXTERNAL_VOLUME = 5e-6  # external volume from experiment
OD_TO_CELL_COUNT = OD_TO_CELL_CONCENTRATION * EXTERNAL_VOLUME  # number of cells per OD

MCP_RADIUS = 7.e-8  # in metres
MCP_VOLUME = (4. / 3.) * np.pi * (MCP_RADIUS ** 3)
MCP_SURFACE_AREA = 4 * np.pi * (MCP_RADIUS ** 2)
#######################################################################################################################
############################################ PARAMETER NAMES ##########################################################
#######################################################################################################################

PERMEABILITY_CELL_PARAMETERS = ['PermCellGlycerol',
                                'PermCell3HPA',
                                'PermCellPDO',
                                'PermCellHCoA',
                                'PermCellHPhosph']
# 'PermCellHate']

PERMEABILITY_MCP_PARAMETERS = ['PermMCPGlycerol',
                               'PermMCP3HPA',
                               'PermMCPPDO',
                               'PermMCPHCoA',
                               'PermMCPHPhosph',
                               'PermMCPNADH',
                               'PermMCPNAD']

KINETIC_PARAMETERS = ['k1PduCDE', 'k2PduCDE', 'k3PduCDE', 'KeqPduCDE',
                      'k1PduQ', 'k2PduQ', 'k3PduQ', 'k4PduQ', 'k5PduQ', 'k6PduQ', 'k7PduQ', 'KeqPduQ',
                      'k1PduP', 'k2PduP', 'k3PduP', 'k4PduP', 'k5PduP', 'k6PduP', 'k7PduP', 'KeqPduP',
                      'k1PduL', 'k2PduL', 'k3PduL', 'KeqPduL',
                      # 'k1PduW', 'k2PduW', 'k3PduW', 'KeqPduLW',
                      'VmaxfGlpK', 'KmGlpK', 'VmaxfGlpK_dD', 'KmGlpK_dD']

MCP_PARAMETERS = ['nMCPs', 'AJ_radius']

THERMO_PARAMETERS = ['KeqPduCDE', 'KeqPduQ', 'KeqPduP', 'KeqPduL']  # 'KeqPduLW']

COFACTOR_PARAMETERS = ['NADH_NAD_TOTAL_CYTO', 'NADH_NAD_RATIO_CYTO',
                       'NADH_NAD_TOTAL_MCP', 'NADH_NAD_RATIO_MCP']

ENZYME_CONCENTRATIONS = ['nPduCDE',
                         'nPduQ',
                         'nPduP',
                         'nPduL']
# 'nPduW']

MODEL_PARAMETERS = [*PERMEABILITY_CELL_PARAMETERS,
                    *PERMEABILITY_MCP_PARAMETERS,
                    *KINETIC_PARAMETERS,
                    *MCP_PARAMETERS,
                    *COFACTOR_PARAMETERS,
                    *ENZYME_CONCENTRATIONS
                    ]

DCW_PARAMETERS = ['L', 'k', 'A']

INIT_PARAMETERS_LIST = [*ENZYME_CONCENTRATIONS, 'A']

PARAMETER_LIST = [*MODEL_PARAMETERS, *DCW_PARAMETERS]
DEV_PARAMETER_LIST = [*PERMEABILITY_CELL_PARAMETERS,
                    *PERMEABILITY_MCP_PARAMETERS,
                    *KINETIC_PARAMETERS,
                    *MCP_PARAMETERS,
                    *COFACTOR_PARAMETERS,
                    *ENZYME_CONCENTRATIONS
                      ]

VARIABLE_NAMES = ['G_MCP', 'H_MCP', 'P_MCP', 'HCoA_MCP', 'HPhosph_MCP',
                  'NADH_MCP', 'NAD_MCP',

                  'PduCDE', 'PduCDE_C',
                  'PduQ', 'PduQ_NADH', 'PduQ_NADH_HPA', 'PduQ_NAD',
                  'PduP', 'PduP_NAD', 'PduP_NAD_HPA', 'PduP_NADH',
                  'PduL', 'PduL_C',

                  'G_CYTO', 'H_CYTO', 'P_CYTO', 'HCoA_CYTO', 'HPhosph_CYTO',  # 'Hate_CYTO',
                  # 'PduW', 'PduW_C',

                  'G_EXT', 'H_EXT', 'P_EXT', 'HCoA_EXT', 'HPhosph_EXT', 'OD']  # 'Hate_EXT', 'OD']

DATA_INDEX = [VARIABLE_NAMES.index('G_EXT'), VARIABLE_NAMES.index('H_EXT'), VARIABLE_NAMES.index('P_EXT'),
              VARIABLE_NAMES.index('OD')]

TIME_SPACING = 100
TIME_SAMPLES_EXPANDED = [np.linspace(TIME_SAMPLES[i], TIME_SAMPLES[i + 1], num=TIME_SPACING, endpoint=False) for i in
                         range(len(TIME_SAMPLES) - 1)]
TIME_SAMPLES_EXPANDED = list(np.concatenate(TIME_SAMPLES_EXPANDED))
TIME_SAMPLES_EXPANDED.append(TIME_SAMPLES_EXPANDED[-1])
TIME_SAMPLES_EXPANDED = np.array(TIME_SAMPLES_EXPANDED)

N_MODEL_PARAMETERS = len(MODEL_PARAMETERS)
N_DCW_PARAMETERS = 3
N_UNKNOWN_PARAMETERS = N_MODEL_PARAMETERS + N_DCW_PARAMETERS

#######################################################################################################################
################################################# PLOT CONSTANTS ######################################################
#######################################################################################################################


MM_KINETIC_PARAMETERS = ['kcat_PduCDE_f', 'kcat_PduCDE_Glycerol', 'kcat_PduCDE_r', 'kcat_PduCDE_HPA', 'KeqPduCDE',
                         'KeqPduQ', 'KeqPduP', 'KeqPduL', 'VmaxfGlpK', 'KmGlpK', 'VmaxfGlpK_dD', 'KmGlpK_dD']

ENZYME_CONCENTRATIONS_WT = [enz_conc + '_WT' for enz_conc in ENZYME_CONCENTRATIONS]
ENZYME_CONCENTRATIONS_dAJ = [enz_conc + '_dAJ' for enz_conc in ENZYME_CONCENTRATIONS]

PLOT_PARAMETERS = [*PERMEABILITY_CELL_PARAMETERS,
                   *PERMEABILITY_MCP_PARAMETERS,
                   *MM_KINETIC_PARAMETERS,
                   *MCP_PARAMETERS,
                   *COFACTOR_PARAMETERS,
                   *ENZYME_CONCENTRATIONS_WT,
                   *ENZYME_CONCENTRATIONS_dAJ]

########################################################################################################################
########################################################################################################################

MODEL_PARAMS_TO_TEX = {'PermCellGlycerol': r'$P_{\mathrm{Cell, Gly}}$',
                       'PermCell3HPA': r'$P_{\mathrm{Cell, HPA}}$',
                       'PermCellPDO': r'$P_{\mathrm{Cell, PDO}}$',
                       'PermCellHCoA': r'$P_{\mathrm{Cell, HCoA}}$',
                       'PermCellHPhosph': r'$P_{\mathrm{Cell, HPhosph}}$',

                       'PermMCPGlycerol': r'$P_{\mathrm{MCP, Gly}}$',
                       'PermMCP3HPA': r'$P_{\mathrm{MCP, Gly}}$',
                       'PermMCPPDO': r'$P_{\mathrm{MCP, PDO}}$',
                       'PermMCPHCoA': r'$P_{\mathrm{MCP, HCoA}}$',
                       'PermMCPHPhosph': r'$P_{\mathrm{MCP, HPhosph}}$',
                       'PermMCPNADH': r'$P_{\mathrm{MCP, NADH}}$',
                       'PermMCPNAD': r'$P_{\mathrm{MCP, NAD}}$',

                       'k1PduCDE': r'$k_{1,\mathrm{PduCDE}}$',
                       'k2PduCDE': r'$k_{2,\mathrm{PduCDE}}$',
                       'k3PduCDE': r'$k_{3,\mathrm{PduCDE}}$',
                       'KeqPduCDE': r'$K_{eq}^{\mathrm{PduCDE}}$',

                       'k1PduQ': r'$k_{1,\mathrm{PduQ}}$',
                       'k2PduQ': r'$k_{2,\mathrm{PduQ}}$',
                       'k3PduQ': r'$k_{3,\mathrm{PduQ}}$',
                       'k4PduQ': r'$k_{4,\mathrm{PduQ}}$',
                       'k5PduQ': r'$k_{5,\mathrm{PduQ}}$',
                       'k6PduQ': r'$k_{6,\mathrm{PduQ}}$',
                       'k7PduQ': r'$k_{7,\mathrm{PduQ}}$',
                       'KeqPduQ': r'$K_{eq}^{\mathrm{PduQ}}$',

                       'k1PduP': r'$k_{1,\mathrm{PduP}}$',
                       'k2PduP': r'$k_{2,\mathrm{PduP}}$',
                       'k3PduP': r'$k_{3,\mathrm{PduP}}$',
                       'k4PduP': r'$k_{4,\mathrm{PduP}}$',
                       'k5PduP': r'$k_{5,\mathrm{PduP}}$',
                       'k6PduP': r'$k_{6,\mathrm{PduP}}$',
                       'k7PduP': r'$k_{7,\mathrm{PduP}}$',
                       'KeqPduP': r'$K_{eq}^{\mathrm{PduP}}$',

                       'k1PduL': r'$k_{1,\mathrm{PduL}}$',
                       'k2PduL': r'$k_{2,\mathrm{PduL}}$',
                       'k3PduL': r'$k_{3,\mathrm{PduL}}$',
                       'KeqPduL': r'$K_{eq}^{\mathrm{PduL}}$',

                       'VmaxfGlpK': r'$V_{\mathrm{max,GlpK}}^{f}$',
                       'KmGlpK': r'$K_{\mathrm{M,KmGlpK}}^{Glycerol}$',
                       'VmaxfGlpK_dD': r'$V_{\mathrm{max,GlpK}}^{f,dD}$',
                       'KmGlpK_dD': r'$K_{\mathrm{M,KmGlpK}}^{Glycerol, dD}$',

                       'kcat_PduCDE_f': r'$V_{\mathrm{max,PduCDE}}^{f}$',
                       'kcat_PduCDE_Glycerol': r'$K_{\mathrm{M,PduCDE}}^{Glycerol}$',
                       'kcat_PduCDE_r': r'$k_{\mathrm{cat,PduCDE}}^{f}$',
                       'kcat_PduCDE_HPA': r'$K_{\mathrm{M,PduCDE}}^{\mathrm{Glycerol}}$',

                       'nMCPs': 'nMCPs',
                       'AJ_radius': r'$r_{\mathrm{dAJ}}$',

                       'NADH_NAD_TOTAL_CYTO': r"$[\mathrm{NAD}]_C(0) + [\mathrm{NADH}]_C(0)$",
                       'NADH_NAD_RATIO_CYTO': r"$[\mathrm{NADH}_C:\mathrm{NAD}]_C(0)$",
                       'NADH_NAD_TOTAL_MCP': r"$[\mathrm{NAD}]_{\mathrm{MCP}}(0) + [\mathrm{NADH}]_{\mathrm{MCP}}(0)$",
                       'NADH_NAD_RATIO_MCP': r"$[\mathrm{NADH}_{\mathrm{MCP}}:\mathrm{NAD}]_{\mathrm{MCP}}(0)$",

                       'nPduCDE_WT': r'$nPduCDE^{\mathrm{WT}}$',
                       'nPduQ_WT': r'$nPduQ^{\mathrm{WT}}$',
                       'nPduP_WT': r'$nPduP^{\mathrm{WT}}$',
                       'nPduL_WT': r'$nPduL^{\mathrm{WT}}$',

                       'nPduCDE_dAJ': r'$nPduCDE^{\mathrm{dAJ}}$',
                       'nPduQ_dAJ': r'$nPduQ^{\mathrm{dAJ}}$',
                       'nPduP_dAJ': r'$nPduP^{\mathrm{dAJ}}$',
                       'nPduL_dAJ': r'$nPduL^{\mathrm{dAJ}}$'
                       }
