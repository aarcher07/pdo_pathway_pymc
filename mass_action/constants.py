import numpy as np

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
                  'k1DhaB', 'k2DhaB', 'k3DhaB', 'k4DhaB',
                  'k1DhaT', 'k2DhaT', 'k3DhaT', 'k4DhaT',
                  'VmaxfMetab', 'KmMetabG',
                  ]

INIT_CONSTANTS = ['DHAB_INIT',
                  'DHAT_INIT',
                  'G_EXT_INIT']

DCW_CONSTANTS = ['L','k','A']

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

GLYCEROL_EXTERNAL = ['G_EXT_INIT_50', 'G_EXT_INIT_60', 'G_EXT_INIT_70', 'G_EXT_INIT_80']
