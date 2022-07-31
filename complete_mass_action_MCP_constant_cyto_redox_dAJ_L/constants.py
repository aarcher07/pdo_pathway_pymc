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
                      'k1PduL', 'k2PduL', 'k3PduL', 'KeqPduL']
                      # 'k1PduW', 'k2PduW', 'k3PduW', 'KeqPduLW'

GlpK_PARAMETERS = ['VmaxfGlpK', 'KmGlpK']

MCP_PARAMETERS = ['nMCPs', 'dAJ_radius']

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
                    *GlpK_PARAMETERS,
                    *MCP_PARAMETERS,
                    *COFACTOR_PARAMETERS,
                    *ENZYME_CONCENTRATIONS
                    ]

DCW_PARAMETERS = ['L', 'k', 'A']

INIT_PARAMETERS_LIST = [*ENZYME_CONCENTRATIONS, 'A']

LOCAL_PARAMETER_LIST = [*MODEL_PARAMETERS, *DCW_PARAMETERS]
LOCAL_DEV_PARAMETER_LIST = [*PERMEABILITY_CELL_PARAMETERS,
                            *PERMEABILITY_MCP_PARAMETERS,
                            *KINETIC_PARAMETERS,
                            *GlpK_PARAMETERS,
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

TIME_SPACING = 50
TIME_SAMPLES_EXPANDED = [np.linspace(TIME_SAMPLES[i], TIME_SAMPLES[i + 1], num=TIME_SPACING, endpoint=False) for i in
                         range(len(TIME_SAMPLES) - 1)]
TIME_SAMPLES_EXPANDED = list(np.concatenate(TIME_SAMPLES_EXPANDED))
TIME_SAMPLES_EXPANDED.append(TIME_SAMPLES_EXPANDED[-1])
TIME_SAMPLES_EXPANDED = np.array(TIME_SAMPLES_EXPANDED)

N_MODEL_PARAMETERS = len(MODEL_PARAMETERS)
N_DCW_PARAMETERS = 3
N_UNKNOWN_PARAMETERS = N_MODEL_PARAMETERS + N_DCW_PARAMETERS

#######################################################################################################################
################################################## ALL CONSTANTS ######################################################
#######################################################################################################################

GLOBAL_GlpK_PARAMETERS = [ param_name + '_' + exp_cond  for exp_cond in ['WT_L', 'dAJ_L', 'dP_L', 'dD_L'] for param_name in ['VmaxfGlpK', 'KmGlpK']]
GLOBAL_COFACTOR_PARAMETERS = [ param_name + '_' + exp_cond  for exp_cond in ['WT_L', 'dAJ_L', 'dP_L'] for param_name in COFACTOR_PARAMETERS]
GLOBAL_ENZYME_PARAMETERS = [ param_name + '_' + exp_cond  for exp_cond in ['WT_L', 'dAJ_L', 'dP_L'] for param_name in ENZYME_CONCENTRATIONS]

GLOBAL_DEV_PARAMETERS = [*PERMEABILITY_CELL_PARAMETERS,
                    *PERMEABILITY_MCP_PARAMETERS,
                    *KINETIC_PARAMETERS,
                    *GLOBAL_GlpK_PARAMETERS,
                         *MCP_PARAMETERS,
                    *GLOBAL_COFACTOR_PARAMETERS,
                    *GLOBAL_ENZYME_PARAMETERS
                      ]

#######################################################################################################################
################################################# PLOT CONSTANTS ######################################################
#######################################################################################################################


MM_KINETIC_PARAMETERS = ['kcat_PduCDE_f', 'Km_PduCDE_Glycerol', 'kcat_PduCDE_r', 'Km_PduCDE_HPA', 'KeqPduCDE',
                         'kcat_PduQ_f', 'Km_PduQ_NADH', 'Km_PduQ_HPA', 'kcat_PduQ_NAD', 'Km_PduQ_PDO',  'KeqPduQ',
                         'kcat_PduP_f', 'Km_PduP_NAD', 'Km_PduP_HPA', 'kcat_PduP_NADH', 'Km_PduP_HCoA', 'KeqPduP',
                         'kcat_PduL_f', 'Km_PduL_HCoA', 'kcat_PduL_r', 'Km_PduL_HPhosph', 'KeqPduL',
                         *GLOBAL_GlpK_PARAMETERS]

PLOT_PARAMETERS = [*PERMEABILITY_CELL_PARAMETERS,
                   *PERMEABILITY_MCP_PARAMETERS,
                   *MM_KINETIC_PARAMETERS,
                   *MCP_PARAMETERS,
                   *GLOBAL_COFACTOR_PARAMETERS,
                   *GLOBAL_ENZYME_PARAMETERS]

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

                       'VmaxfGlpK_WT_L': r'$V_{\mathrm{max,GlpK}}^{\mathrm{WT_L, f}}$',
                       'KmGlpK_WT_L': r'$K_{\mathrm{M,KmGlpK}}^{\mathrm{WT_L, Glycerol}}$',

                       'VmaxfGlpK_dAJ_L': r'$V_{\mathrm{max,GlpK}}^{\mathrm{f,dAJ_L}}$',
                       'KmGlpK_dAJ_L': r'$K_{\mathrm{M,KmGlpK}}^{\mathrm{Glycerol, dAJ_L}}$',

                       'VmaxfGlpK_dP_L': r'$V_{\mathrm{max,GlpK}}^{\mathrm{f,dP_L}}$',
                       'KmGlpK_dP_L': r'$K_{\mathrm{M,KmGlpK}}^{\mathrm{Glycerol, dP_L}}$',

                       'VmaxfGlpK_dD_L': r'$V_{\mathrm{max,GlpK}}^{{\mathrmf,dD_L}}$',
                       'KmGlpK_dD_L': r'$K_{\mathrm{M,KmGlpK}}^{\mathrm{Glycerol, dD_L}}$',

                       'kcat_PduCDE_f': r'$k_{\mathrm{cat,PduCDE}}^{\mathrm{f}}$',
                       'Km_PduCDE_Glycerol': r'$K_{\mathrm{M,PduCDE}}^{\mathrm{Glycerol}}$',
                       'kcat_PduCDE_r': r'$k_{\mathrm{cat,PduCDE}}^{\mathrm{r}}$',
                       'Km_PduCDE_HPA': r'$K_{\mathrm{M,PduCDE}}^{\mathrm{HPA}}$',

                       'kcat_PduQ_f': r'$k_{\mathrm{cat,PduQ}}^{\mathrm{f}}$',
                       'Km_PduQ_NADH': r'$K_{\mathrm{M,PduQ}}^{\mathrm{NADH}}$',
                       'Km_PduQ_HPA': r'$K_{\mathrm{M,PduQ}}^{\mathrm{HPA}}$',
                       'kcat_PduQ_r': r'$k_{\mathrm{cat,PduQ}}^{\mathrm{r}}$',
                       'Km_PduQ_PDO': r'$K_{\mathrm{M,PduQ}}^{\mathrm{PDO}}$',
                       'Km_PduQ_NAD': r'$K_{\mathrm{M,PduQ}}^{\mathrm{NAD}}$',

                       'kcat_PduP_f': r'$k_{\mathrm{cat,PduP}}^{\mathrm{f}}$',
                       'Km_PduP_NAD': r'$K_{\mathrm{M,PduP}}^{\mathrm{NAD}}$',
                       'Km_PduP_HPA': r'$K_{\mathrm{M,PduP}}^{\mathrm{HPA}}$',
                       'kcat_PduP_r': r'$k_{\mathrm{cat,PduP}}^{\mathrm{r}}$',
                       'Km_PduP_HCoA': r'$K_{\mathrm{M,PduP}}^{\mathrm{HCoA}}$',
                       'Km_PduP_NADH': r'$K_{\mathrm{M,PduP}}^{\mathrm{NADH}}$',

                       'kcat_PduL_f': r'$k_{\mathrm{cat,PduL}}^{\mathrm{f}}$',
                       'Km_PduL_HCoA': r'$K_{\mathrm{M,PduL}}^{\mathrm{HCoA}}$',
                       'kcat_PduL_r': r'$k_{\mathrm{cat,PduL}}^{\mathrm{r}}$',
                       'Km_PduL_HPhosph': r'$K_{\mathrm{M,PduL}}^{\mathrm{HPhosph}}$',

                       'nMCPs': '\mathrm{nMCPs}',
                       'dAJ_radius': r'$r_{\mathrm{dAJ_L}}$',

                       'NADH_NAD_TOTAL_CYTO_WT_L': r"$[\mathrm{Total initial cyto NAD/NADH}$",
                       'NADH_NAD_RATIO_CYTO_WT_L': r"$[\mathrm{NADH}_{\mathrm{C}}:\mathrm{NAD}]_{\mathrm{C}}(0)$",
                       'NADH_NAD_TOTAL_MCP_WT_L': r"$[\mathrm{Total initial MCP NAD/NADH}$",
                       'NADH_NAD_RATIO_MCP_WT_L': r"$[\mathrm{NADH}_{\mathrm{MCP}}:\mathrm{NAD}]_{\mathrm{MCP}}(0)$",

                       'NADH_NAD_TOTAL_CYTO_dAJ_L': r"$[\mathrm{Total initial cyto NAD/NADH}$",
                       'NADH_NAD_RATIO_CYTO_dAJ_L': r"$[\mathrm{NADH}_{\mathrm{C}}:\mathrm{NAD}]_{\mathrm{C}}(0)$",
                       'NADH_NAD_TOTAL_MCP_dAJ_L': r"$[\mathrm{Total initial MCP NAD/NADH}$",
                       'NADH_NAD_RATIO_MCP_dAJ_L': r"$[\mathrm{NADH}_{\mathrm{MCP}}:\mathrm{NAD}]_{\mathrm{MCP}}(0)$",

                       'NADH_NAD_TOTAL_CYTO_dP_L': r"$[\mathrm{Total initial cyto NAD/NADH}$",
                       'NADH_NAD_RATIO_CYTO_dP_L': r"$[\mathrm{NADH}_{\mathrm{C}}:\mathrm{NAD}]_{\mathrm{C}}(0)$",
                       'NADH_NAD_TOTAL_MCP_dP_L': r"$[\mathrm{Total initial MCP NAD/NADH}$",
                       'NADH_NAD_RATIO_MCP_dP_L': r"$[\mathrm{NADH}_{\mathrm{MCP}}:\mathrm{NAD}]_{\mathrm{MCP}}(0)$",

                       'nPduCDE_WT_L': r'$\mathrm{nPduCDE}^{\mathrm{WT_L}}$',
                       'nPduQ_WT_L': r'$\mathrm{nPduQ}^{\mathrm{WT_L}}$',
                       'nPduP_WT_L': r'$\mathrm{nPduP}^{\mathrm{WT_L}}$',
                       'nPduL_WT_L': r'$\mathrm{nPduL}^{\mathrm{WT_L}}$',

                       'nPduCDE_dP_L': r'$\mathrm{nPduCDE}^{\mathrm{dP_L}}$',
                       'nPduQ_dP_L': r'$\mathrm{nPduQ}^{\mathrm{dP_L}}$',
                       'nPduP_dP_L': r'$\mathrm{nPduP}^{\mathrm{dP_L}}$',
                       'nPduL_dP_L': r'$\mathrm{nPduL}^{\mathrm{dP_L}}$',

                       'nPduCDE_dAJ_L': r'$\mathrm{nPduCDE}^{\mathrm{dAJ_L}}$',
                       'nPduQ_dAJ_L': r'$\mathrm{nPduQ}^{\mathrm{dAJ_L}}$',
                       'nPduP_dAJ_L': r'$\mathrm{nPduP}^{\mathrm{dAJ_L}}$',
                       'nPduL_dAJ_L': r'$\mathrm{nPduL}^{\mathrm{dAJ_L}}$'
                       }


MODEL_PARAMS_TO_UNITS = {'PermCellGlycerol': '(m/s)',
                       'PermCell3HPA': '(m/s)',
                       'PermCellPDO': '(m/s)',
                       'PermCellHCoA': '(m/s)',
                       'PermCellHPhosph': '(m/s)',

                       'PermMCPGlycerol': '(m/s)',
                       'PermMCP3HPA': '(m/s)',
                       'PermMCPPDO': '(m/s)',
                       'PermMCPHCoA': '(m/s)',
                       'PermMCPHPhosph': '(m/s)',
                       'PermMCPNADH': '(m/s)',
                       'PermMCPNAD': '(m/s)',


                       'KeqPduCDE': '',
                       'KeqPduQ': '',
                       'KeqPduP': ' ',
                       'KeqPduL': ' ',

                       'VmaxfGlpK': '(mM/s)',
                       'KmGlpK': '(mM)',
                       'VmaxfGlpK_dD_L': '(mM/s)',
                       'KmGlpK_dD_L': '(mM)',

                       'kcat_PduCDE_f': '(/s)',
                       'Km_PduCDE_Glycerol': '(mM)',
                       'kcat_PduCDE_r': '(/s)',
                       'Km_PduCDE_HPA': '(mM)',

                       'kcat_PduQ_f': '(/s)',
                       'Km_PduQ_NADH': '(mM)',
                       'Km_PduQ_HPA': '(mM)',
                       'kcat_PduQ_r': '(/s)',
                       'Km_PduQ_PDO': '(mM)',
                       'Km_PduQ_NAD': '(mM)',

                       'kcat_PduP_f': '(/s)',
                       'Km_PduP_NAD': '(mM)',
                       'Km_PduP_HPA': '(mM)',
                       'kcat_PduP_r': '(/s)',
                       'Km_PduP_HCoA': '(mM)',
                       'Km_PduP_NADH': '(mM)',

                       'kcat_PduL_f': '(/s)',
                       'Km_PduL_HCoA': '(mM)',
                       'kcat_PduL_r': '(/s)',
                       'Km_PduL_HPhosph': '(mM)',

                       'nMCPs': 'nMCPs',
                       'dAJ_radius': 'm',

                       'NADH_NAD_TOTAL_CYTO_WT_L': "(mM)",
                       'NADH_NAD_RATIO_CYTO_WT_L': "(mM)",
                       'NADH_NAD_TOTAL_MCP_WT_L': "",
                       'NADH_NAD_RATIO_MCP_WT_L': "",

                       'NADH_NAD_TOTAL_CYTO_dAJ_L': "(mM)",
                       'NADH_NAD_RATIO_CYTO_dAJ_L': "(mM)",
                       'NADH_NAD_TOTAL_MCP_dAJ_L': "",
                       'NADH_NAD_RATIO_MCP_dAJ_L': "",

                         'NADH_NAD_TOTAL_CYTO_dP_L': "(mM)",
                         'NADH_NAD_RATIO_CYTO_dP_L': "(mM)",
                         'NADH_NAD_TOTAL_MCP_dP_L': "",
                         'NADH_NAD_RATIO_MCP_dP_L': "",

                       'nPduCDE_WT_L': '(number of enzymes)',
                       'nPduQ_WT_L': '(number of enzymes)',
                       'nPduP_WT_L': '(number of enzymes)',
                       'nPduL_WT_L': '(number of enzymes)',

                       'nPduCDE_dP_L': '(number of enzymes)',
                       'nPduQ_dP_L': '(number of enzymes)',
                       'nPduP_dP_L': '(number of enzymes)',
                       'nPduL_dP_L': '(number of enzymes)',

                       'nPduCDE_dAJ_L': '(number of enzymes)',
                       'nPduQ_dAJ_L': '(number of enzymes)',
                       'nPduP_dAJ_L': '(number of enzymes)',
                       'nPduL_dAJ_L': '(number of enzymes)'
                       }


########################################################################################################################
################################################## PARAMETER INDICES ###################################################
########################################################################################################################

LOCAL_PERMEABILITY_CELL_PARAMETERS_INDICES = [LOCAL_DEV_PARAMETER_LIST.index(param) for param in PERMEABILITY_CELL_PARAMETERS]
LOCAL_PERMEABILITY_MCP_PARAMETERS_INDICES = [LOCAL_DEV_PARAMETER_LIST.index(param) for param in PERMEABILITY_MCP_PARAMETERS]
LOCAL_KINETIC_PARAMETERS_INDICES = [LOCAL_DEV_PARAMETER_LIST.index(param) for param in KINETIC_PARAMETERS]
LOCAL_GlpK_PARAMETERS_INDICES = [LOCAL_DEV_PARAMETER_LIST.index(param) for param in GlpK_PARAMETERS]
LOCAL_MCP_GEOMETRY_PARAMETERS_INDICES = [LOCAL_DEV_PARAMETER_LIST.index(param) for param in MCP_PARAMETERS]
LOCAL_COFACTOR_PARAMETERS_INDICES = [LOCAL_DEV_PARAMETER_LIST.index(param) for param in COFACTOR_PARAMETERS]
LOCAL_ENZYME_CONCENTRATIONS_INDICES = [LOCAL_DEV_PARAMETER_LIST.index(param) for param in ENZYME_CONCENTRATIONS]

GLOBAL_PERMEABILITY_CELL_PARAMETERS_INDICES = [GLOBAL_DEV_PARAMETERS.index(param) for param in PERMEABILITY_CELL_PARAMETERS]
GLOBAL_PERMEABILITY_MCP_PARAMETERS_INDICES = [GLOBAL_DEV_PARAMETERS.index(param) for param in PERMEABILITY_MCP_PARAMETERS]
GLOBAL_KINETIC_PARAMETERS_INDICES = [GLOBAL_DEV_PARAMETERS.index(param) for param in KINETIC_PARAMETERS]
GLOBAL_GlpK_PARAMETERS_INDICES_DICT = {exp_cond:[GLOBAL_DEV_PARAMETERS.index(param + '_' + exp_cond) for param in GlpK_PARAMETERS] for exp_cond in ['WT_L', 'dAJ_L', 'dP_L']}
GLOBAL_MCP_GEOMETRY_PARAMETERS_INDICES = [GLOBAL_DEV_PARAMETERS.index(param) for param in MCP_PARAMETERS]
GLOBAL_COFACTOR_PARAMETERS_INDICES_DICT = {exp_cond:[GLOBAL_DEV_PARAMETERS.index(param + '_' + exp_cond) for param in COFACTOR_PARAMETERS] for exp_cond in ['WT_L', 'dAJ_L', 'dP_L']}
GLOBAL_ENZYME_CONCENTRATIONS_INDICES_DICT = {exp_cond:[GLOBAL_DEV_PARAMETERS.index(param + '_' + exp_cond) for param in ENZYME_CONCENTRATIONS] for exp_cond in ['WT_L', 'dAJ_L', 'dP_L']}
