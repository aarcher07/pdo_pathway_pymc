import numpy as np
from exp_data import TIME_SAMPLES, NORM_DCW_MEAN_PRIOR_TRANS_PARAMETERS

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

N_MODEL_PARAMETERS = 15
N_DCW_PARAMETERS = 3
N_UNKNOWN_PARAMETERS = 19
N_TOTAL_PARAMETERS = 15 + 4 + 12

MODEL_CONSTANTS = ['PermCellGlycerol','PermCellPDO','PermCell3HPA',
                  'k1DhaB', 'k2DhaB', 'k3DhaB', 'k4DhaB',
                  'k1DhaT', 'k2DhaT', 'k3DhaT', 'k4DhaT',
                  'VmaxfMetab', 'KmMetabG',
                  ]

INIT_CONSTANTS = ['DHAB_INIT',
                  'DHAT_INIT',
                  'G_EXT_INIT']

DCW_CONSTANTS = [key[5:] for key in NORM_DCW_MEAN_PRIOR_TRANS_PARAMETERS.columns]

INIT_PARAMETERS_LIST = [*INIT_CONSTANTS, 'A']

DEV_PARAMETERS_LIST = [*MODEL_CONSTANTS, *INIT_CONSTANTS[:2]]

PARAMETER_LIST = [*MODEL_CONSTANTS, *INIT_CONSTANTS, *DCW_CONSTANTS]

VARIABLE_NAMES = ['G_CYTO', 'H_CYTO','P_CYTO',
                   'DHAB', 'DHAB_C',
                   'DHAT', 'DHAT_C',
                   'G_EXT', 'H_EXT','P_EXT', 'dcw']

PERMEABILITY_PARAMETERS = ['PermCellGlycerol','PermCellPDO','PermCell3HPA']

KINETIC_PARAMETERS = ['k1DhaB', 'k2DhaB', 'k3DhaB', 'k4DhaB',
                      'k1DhaT', 'k2DhaT', 'k3DhaT', 'k4DhaT',
                      'VmaxfMetab', 'KmMetabG']

ENZYME_CONCENTRATIONS = ['DHAB_INIT', 'DHAT_INIT']

GLYCEROL_EXTERNAL_EXPERIMENTAL = ['G_EXT_INIT_50', 'G_EXT_INIT_60', 'G_EXT_INIT_70', 'G_EXT_INIT_80']

DCW_PARAMETERS_EXPERIMENTAL = ['L_50','k_50','A_50',
                               'L_60','k_60','A_60',
                               'L_70','k_70','A_70',
                               'L_80','k_80','A_80']

ALL_PARAMETERS = [*PERMEABILITY_PARAMETERS, *KINETIC_PARAMETERS, *ENZYME_CONCENTRATIONS]
PLOT_PARAMETERS = [*PERMEABILITY_PARAMETERS, 'kcatfDhaB', 'KmGlycerolDhaB', 'kcatrDhaB', 'KmHPADhaB',
                    'kcatfDhaT', 'KmHPADhaT', 'kcatrDhaT', 'KmPDODhaT', 'VmaxfMetab', 'KmMetabG',
                   *ENZYME_CONCENTRATIONS ]
DATA_INDEX = [VARIABLE_NAMES.index('G_EXT'), VARIABLE_NAMES.index('P_EXT'), VARIABLE_NAMES.index('dcw')]
TIME_SAMPLES_EXPANDED = {}
TIME_SPACING = 15 # TODO: CHANGE TO 15 for _HPA.py and 5 _HPA_2.py
for exp_cond, time_samps in TIME_SAMPLES.items():
    time_samps_expanded = [np.linspace(time_samps[i],time_samps[i+1],num=TIME_SPACING, endpoint=False) for i in range(len(time_samps)-1)]
    time_samps_expanded = list(np.concatenate(time_samps_expanded))
    time_samps_expanded.append(time_samps[-1])
    TIME_SAMPLES_EXPANDED[exp_cond] = np.array(time_samps_expanded)
