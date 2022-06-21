import numpy as np
from exp_data import *

CELL_RADIUS = 0.375e-6
CELL_LENGTH = 2.47e-6
CELL_SURFACE_AREA = 2*np.pi*CELL_RADIUS*CELL_LENGTH
CELL_VOLUME = 4*np.pi/3*(CELL_RADIUS)**3 + np.pi*(CELL_LENGTH - 2*CELL_RADIUS)*(CELL_RADIUS**2)

HRS_TO_SECS = 60*60
# DCW to CELL CONCENTRATION
OD_TO_DCW = 0.1 #grams of cell dry weight/L per 1 OD
OD_TO_CELL_CONCENTRATION = 1e15 #number of cell/m^3 per 1 OD
DCW_TO_CELL_CONCENTRATION = OD_TO_CELL_CONCENTRATION/OD_TO_DCW #number of cell/m^3 per 1 g/L

EXTERNAL_VOLUME = 0.002 # external volume from experiment
DCW_TO_CELL_COUNT = DCW_TO_CELL_CONCENTRATION*EXTERNAL_VOLUME



MODEL_CONSTANTS = ['PermCellGlycerol','PermCellPDO','PermCell3HPA',
                  'VmaxfDhaB', 'KmGlycerolDhaB', 'VmaxrDhaB', 'KmHPADhaB',
                  'VmaxfDhaT', 'KmHPADhaT', 'VmaxrDhaT', 'KmPDODhaT',
                  'VmaxfMetab', 'KmMetabG',
                  ]

INIT_CONSTANTS = ['G_EXT_INIT']

DCW_CONSTANTS = ['L','k','A']

N_MODEL_PARAMETERS = len(MODEL_CONSTANTS) + len(INIT_CONSTANTS) -1
N_DCW_PARAMETERS = len(DCW_CONSTANTS)
N_UNKNOWN_PARAMETERS = len(MODEL_CONSTANTS) + len(INIT_CONSTANTS) + len(DCW_CONSTANTS)
N_TOTAL_PARAMETERS = len(MODEL_CONSTANTS) + len(INIT_CONSTANTS) -1 + 4 + 4*len(DCW_CONSTANTS)

INIT_PARAMETERS_LIST = [*INIT_CONSTANTS, 'A']

DEV_PARAMETERS_LIST = [*MODEL_CONSTANTS, *INIT_CONSTANTS]

PARAMETER_LIST = [*MODEL_CONSTANTS, *INIT_CONSTANTS, *DCW_CONSTANTS]

VARIABLE_NAMES = ['G_CYTO', 'H_CYTO','P_CYTO',
                   'G_EXT', 'H_EXT','P_EXT', 'dcw']

PERMEABILITY_PARAMETERS = ['PermCellGlycerol','PermCellPDO','PermCell3HPA']

KINETIC_PARAMETERS = ['VmaxfDhaB', 'KmGlycerolDhaB', 'VmaxrDhaB', 'KmHPADhaB',
                      'VmaxfDhaT', 'KmHPADhaT', 'VmaxrDhaT', 'KmPDODhaT',
                      'VmaxfMetab', 'KmMetabG']

GLYCEROL_EXTERNAL_EXPERIMENTAL = ['G_EXT_INIT_50', 'G_EXT_INIT_60', 'G_EXT_INIT_70', 'G_EXT_INIT_80']

DCW_PARAMETERS_EXPERIMENTAL = ['L_50','k_50','A_50',
                               'L_60','k_60','A_60',
                               'L_70','k_70','A_70',
                               'L_80','k_80','A_80']

ALL_PARAMETERS = [*PERMEABILITY_PARAMETERS, *KINETIC_PARAMETERS, *GLYCEROL_EXTERNAL_EXPERIMENTAL]

DATA_INDEX = [VARIABLE_NAMES.index('G_EXT'), VARIABLE_NAMES.index('P_EXT'), VARIABLE_NAMES.index('dcw')]
TIME_SAMPLES_EXPANDED = {}
TIME_SPACING = 100
for exp_cond, time_samps in TIME_SAMPLES.items():
    time_samps_expanded = [np.linspace(time_samps[i],time_samps[i+1],num=TIME_SPACING, endpoint=False) for i in range(len(time_samps)-1)]
    time_samps_expanded = list(np.concatenate(time_samps_expanded))
    time_samps_expanded.append(time_samps[-1])
    TIME_SAMPLES_EXPANDED[exp_cond] = np.array(time_samps_expanded)