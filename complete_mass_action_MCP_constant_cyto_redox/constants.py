import numpy as np
from exp_data import TIME_SAMPLES

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
EXTERNAL_VOLUME = 0.002  # external volume from experiment
OD_TO_CELL_COUNT = OD_TO_CELL_CONCENTRATION*EXTERNAL_VOLUME # number of cells per OD

MCP_RADIUS = 7.e-8 # in metres
MCP_VOLUME = (4./3.)*np.pi*(MCP_RADIUS**3)
MCP_SURFACE_AREA = 4*np.pi*(MCP_RADIUS**2)
#######################################################################################################################
############################################ PARAMETER NAMES ##########################################################
#######################################################################################################################

PERMEABILITY_CELL_PARAMETERS = ['PermCellGlycerol',
                                'PermCell3HPA',
                                'PermCellPDO',
                                'PermCellHate',
                                'PermCellHCoA']

PERMEABILITY_MCP_PARAMETERS = ['PermMCPGlycerol',
                               'PermMCP3HPA',
                               'PermMCPPDO',
                               'PermMCPHate',
                               'PermMCPHCoA',
                               'PermMCPNADH',
                               'PermMCPNAD']

KINETIC_PARAMETERS = ['k1PduCDE', 'k2PduCDE', 'k3PduCDE', 'KeqPduCDE',
                      'k1PduQ', 'k2PduQ', 'k3PduQ', 'k4PduQ', 'k5PduQ', 'k6DPduQ', 'k7PduQ', 'KeqPduQ',
                      'k1PduP', 'k2PduP', 'k3PduP', 'k4PduP', 'k5PduP', 'k6DPduP', 'k7PduP', 'KeqPduP',
                      'k1PduL', 'k2PduL', 'k3PduL', 'k4PduL', 'KeqPduL', 'VmaxfGlpK', 'KmGlpK']

THERMO_PARAMETERS = ['KeqPduCDE', 'KeqPduQ', 'KeqPduP', 'KeqPduL']

COFACTOR_PARAMETERS = ['NADH_NAD_TOTAL_CYTO', 'NADH_NAD_RATIO_CYTO',
                       'NADH_NAD_TOTAL_MCP', 'NADH_NAD_RATIO_MCP']

MCP_PARAMETERS = ['nMCPs', 'AJ_radius']

MODEL_PARAMETERS = [*PERMEABILITY_CELL_PARAMETERS,
                    *PERMEABILITY_MCP_PARAMETERS,
                    *KINETIC_PARAMETERS,
                    *COFACTOR_PARAMETERS
                    ]

ENZYME_CONCENTRATIONS = ['PduCDE_INIT',
                         'PduQ_INIT',
                         'PduP_INIT',
                         'PduL_INIT']

DCW_PARAMETERS = ['L', 'k', 'A']

INIT_PARAMETERS_LIST = [*ENZYME_CONCENTRATIONS, 'A']

WT_PARAMETER_LIST = [*MODEL_PARAMETERS, MCP_PARAMETERS[0], *ENZYME_CONCENTRATIONS, *DCW_PARAMETERS]
DELTA_AJ_PARAMETER_LIST = [*MODEL_PARAMETERS, MCP_PARAMETERS[1], *ENZYME_CONCENTRATIONS, *DCW_PARAMETERS]

VARIABLE_NAMES = ['G_MCP', 'H_MCP', 'P_MCP', 'Hate_MCP', 'HCoA_MCP',

                  'PduCDE', 'PduCDE_C',
                  'PduQ', 'PduQ_NADH', 'PduQ_NADH_HPA', 'PduQ_NAD',
                  'PduP', 'PduP_NAD', 'PduP_NAD_HPA', 'PduP_NADH',
                  'PduL', 'PduL_C',

                  'NADH_MCP', 'NAD_MCP',

                  'G_CYTO', 'H_CYTO', 'P_CYTO', 'Hate_CYTO', 'HCoA_CYTO',

                  'G_EXT', 'H_EXT', 'P_EXT', 'Hate_EXT', 'HCoA_EXT', 'OD']

DATA_INDEX = [VARIABLE_NAMES.index('G_EXT'), VARIABLE_NAMES.index('H_EXT'), VARIABLE_NAMES.index('P_EXT'),
              VARIABLE_NAMES.index('dcw')]

TIME_SPACING = 15  # TODO: CHANGE TO 15 for _HPA.py and 5 _HPA_2.py
TIME_SAMPLES_EXPANDED = [np.linspace(TIME_SAMPLES[i], TIME_SAMPLES[i + 1], num=TIME_SPACING, endpoint=False) for i in
                         range(len(TIME_SAMPLES) - 1)]
TIME_SAMPLES_EXPANDED = list(np.concatenate(TIME_SAMPLES_EXPANDED))
TIME_SAMPLES_EXPANDED.append(TIME_SAMPLES_EXPANDED[-1])
TIME_SAMPLES_EXPANDED = np.array(TIME_SAMPLES_EXPANDED)

N_MODEL_PARAMETERS = len(MODEL_PARAMETERS)
N_DCW_PARAMETERS = 3
N_UNKNOWN_PARAMETERS = N_MODEL_PARAMETERS + N_DCW_PARAMETERS
