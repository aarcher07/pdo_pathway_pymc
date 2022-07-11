import numpy as np
from exp_data import TIME_SAMPLES

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



MODEL_CONSTANTS = ['PermCellGlycerol','PermCellPDO','PermCell3HPA', 'PermCellDHA',
                   'k1DhaB', 'k2DhaB', 'k3DhaB', 'KeqDhaB',
                   'k1DhaT', 'k2DhaT', 'k3DhaT', 'k4DhaT', 'k5DhaT', 'k6DhaT', 'k7DhaT', 'KeqDhaT',
                   'k1DhaD', 'k2DhaD', 'k3DhaD', 'k4DhaD', 'k5DhaD', 'k6DhaD', 'k7DhaD', 'KeqDhaD',
                   'k1E0', 'k2E0', 'k3E0', 'k4E0',
                   'VmaxfDhaK', 'KmDhaK'
                  ]

INIT_CONSTANTS = ['DHAB_INIT',
                  'DHAT_INIT',
                  'DHAD_INIT',
                  'E0_INIT',
                  'NADH_NAD_TOTAL_INIT',
                  'NADH_NAD_RATIO_INIT',
                  'G_EXT_INIT']

DCW_CONSTANTS = ['L','k','A']


INIT_PARAMETERS_LIST = [*INIT_CONSTANTS, 'A']

DEV_PARAMETERS_LIST = [*MODEL_CONSTANTS, *INIT_CONSTANTS[:-1]]

PARAMETER_LIST = [*MODEL_CONSTANTS, *INIT_CONSTANTS, *DCW_CONSTANTS]
VARIABLE_NAMES = ['G_CYTO', 'H_CYTO','P_CYTO', 'DHA_CYTO',
                  'NADH', 'NAD',
                  'DHAB', 'DHAB_C',
                  'DHAT', 'DHAT_NADH', 'DHAT_NADH_HPA', 'DHAT_NAD',
                  'DHAD', 'DHAD_NAD', 'DHAD_NAD_GLY', 'DHAD_NADH',
                  'E0', 'E0_C',
                  'G_EXT', 'H_EXT','P_EXT', 'DHA_EXT', 'dcw']

PERMEABILITY_PARAMETERS = ['PermCellGlycerol','PermCellPDO','PermCell3HPA', 'PermCellDHA']

KINETIC_PARAMETERS = ['k1DhaB', 'k2DhaB', 'k3DhaB', 'KeqDhaB',
                      'k1DhaT', 'k2DhaT', 'k3DhaT', 'k4DhaT', 'k5DhaT', 'k6DhaT', 'k7DhaT', 'KeqDhaT',
                      'k1DhaD', 'k2DhaD', 'k3DhaD', 'k4DhaD', 'k5DhaD', 'k6DhaD', 'k7DhaD', 'KeqDhaD',
                      'k1E0', 'k2E0', 'k3E0', 'k4E0',
                      'VmaxfDhaK', 'KmDhaK']

THERMO_PARAMETERS = ['KeqDhaB', 'KeqDhaT', 'KeqDhaD']

ENZYME_CONCENTRATIONS = ['DHAB_INIT', 'DHAT_INIT', 'DHAD_INIT', 'E0_INIT']

COFACTOR_PARAMETERS = ['NADH_NAD_TOTAL_INIT', 'NADH_NAD_RATIO_INIT']

GLYCEROL_EXTERNAL_EXPERIMENTAL = ['G_EXT_INIT_50', 'G_EXT_INIT_60', 'G_EXT_INIT_70', 'G_EXT_INIT_80']

DCW_PARAMETERS_EXPERIMENTAL = ['L_50','k_50','A_50',
                               'L_60','k_60','A_60',
                               'L_70','k_70','A_70',
                               'L_80','k_80','A_80']

ALL_PARAMETERS = [*PERMEABILITY_PARAMETERS, *KINETIC_PARAMETERS, *ENZYME_CONCENTRATIONS, *COFACTOR_PARAMETERS]

DATA_INDEX = [VARIABLE_NAMES.index('G_EXT'), VARIABLE_NAMES.index('P_EXT'), VARIABLE_NAMES.index('dcw')]
TIME_SAMPLES_EXPANDED = {}
TIME_SPACING = 15 # TODO: CHANGE TO 15 for _HPA.py and 5 _HPA_2.py
for exp_cond, time_samps in TIME_SAMPLES.items():
    time_samps_expanded = [np.linspace(time_samps[i],time_samps[i+1],num=TIME_SPACING, endpoint=False) for i in range(len(time_samps)-1)]
    time_samps_expanded = list(np.concatenate(time_samps_expanded))
    time_samps_expanded.append(time_samps[-1])
    TIME_SAMPLES_EXPANDED[exp_cond] = np.array(time_samps_expanded)

N_MODEL_PARAMETERS = len(MODEL_CONSTANTS) + len(INIT_CONSTANTS) - 1
N_DCW_PARAMETERS = 3
N_UNKNOWN_PARAMETERS = N_MODEL_PARAMETERS + N_DCW_PARAMETERS + 1
N_TOTAL_PARAMETERS = N_MODEL_PARAMETERS + 4 + 12