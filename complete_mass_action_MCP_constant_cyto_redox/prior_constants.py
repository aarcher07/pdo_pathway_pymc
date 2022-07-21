"""
Constants parameters

Programme written by aarcher07
Editing History:
- 1/3/21
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from os.path import dirname, abspath
from exp_data_13pd import NORM_OD_MEAN_PRIOR_TRANS_PARAMETERS, NORM_OD_STD_PRIOR_TRANS_PARAMETERS

ROOT_PATH = dirname(dirname(dirname(dirname(abspath(__file__)))))

# Uniform distribution parameters

CELL_PERMEABILITY_PARAMETER_RANGES = {'PermCellGlycerol': np.log10([1e-6, 1e-2]),
                                      'PermCellPDO': np.log10([1e-6, 1e-2]),
                                      'PermCell3HPA': np.log10([1e-6, 1e-2]),
                                      'PermCellHCoA': np.log10([1e-6, 1e-2]),
                                      'PermCellHPhosph': np.log10([1e-6, 1e-2]),
                                      'PermCellHate': np.log10([1e-6, 1e-2])
                                      }

MCP_PERMEABILITY_PARAMETER_RANGES = {'PermMCPGlycerol': np.log10([1e-12, 1e-6]),
                                     'PermMCPPDO': np.log10([1e-12, 1e-6]),
                                     'PermMCP3HPA': np.log10([1e-12, 1e-6]),
                                     'PermMCPHCoA': np.log10([1e-12, 1e-6]),
                                     'PermMCPHPhosph': np.log10([1e-12, 1e-6]),
                                     'PermMCPNADH': np.log10([1e-12, 1e-6]),
                                     'PermMCPNAD': np.log10([1e-12, 1e-6])
                                     }

KINETIC_PARAMETER_RANGES = {'k1PduCDE': np.log10([1e0, 1e4]),
                            'k2PduCDE': np.log10([1e-2, 1e2]),
                            'k3PduCDE': np.log10([1e0, 1e4]),
                            'KeqPduCDE': np.log10([1e7, 1e8]),

                            'k1PduQ': np.log10([1e0, 1e4]),
                            'k2PduQ': np.log10([1e-2, 1e2]),
                            'k3PduQ': np.log10([1e0, 1e4]),
                            'k4PduQ': np.log10([1e-2, 1e2]),
                            'k5PduQ': np.log10([1e0, 1e4]),
                            'k6PduQ': np.log10([1e-2, 1e2]),
                            'k7PduQ': np.log10([1e0, 1e4]),
                            'KeqPduQ': np.log10([1e2, 1e5]),

                            'k1PduP': np.log10([1e0, 1e4]),
                            'k2PduP': np.log10([1e-2, 1e2]),
                            'k3PduP': np.log10([1e0, 1e4]),
                            'k4PduP': np.log10([1e-2, 1e2]),
                            'k5PduP': np.log10([1e0, 1e4]),
                            'k6PduP': np.log10([1e-2, 1e2]),
                            'k7PduP': np.log10([1e0, 1e4]),
                            'KeqPduP': np.log10([1e0, 5e3]),

                            'k1PduL': np.log10([1e-2, 1e0]),
                            'k2PduL': np.log10([1e0, 1e2]),
                            'k3PduL': np.log10([1e-2, 1e0]),
                            'KeqPduL': np.log10([1e-10, 1e0]),

                            'k1PduW': np.log10([1e0, 1e4]),
                            'k2PduW': np.log10([1e-2, 1e2]),
                            'k3PduW': np.log10([1e0, 1e4]),
                            'KeqPduLW': np.log10([1e0, 1e3]),

                            'VmaxfGlpK': np.log10([1e0 * 0.1, 1e1 * 10]),
                            'KmGlpK': np.log10([1e-3, 1e-1])
                            }

GEOMETRY_PARAMETER_RANGES = {'nMCPs': np.log10([3, 30])}


COFACTOR_NUMBER_PARAMETER_RANGES = {'NADH_NAD_TOTAL_CYTO': np.log10([0.1, 1e1]),
                                    'NADH_NAD_RATIO_CYTO': np.log10([1e-2, 1e-3]),
                                    'NADH_NAD_TOTAL_MCP': np.log10([0.1, 1e1]),
                                    'NADH_NAD_RATIO_MCP': np.log10([1e-2, 1e-3]),
                                    }

PDU_WT_ENZ_NUMBERS_PARAMETER_RANGES = {'nPduCDE': np.log10([4e2, 6e2]),
                                       'nPduQ': np.log10([1.5e2, 1.75e2]),
                                       'nPduP': np.log10([1.5e2, 2.5e2]),
                                       'nPduL': np.log10([2e1, 4e1]),
                                       'nPduW': np.log10([5e0, 2e1])}

dPDU_AJ_ENZ_NUMBER_PARAMETER_RANGES = {'nPduCDE': np.log10([1e2, 3e2]),
                                       'nPduQ': np.log10([1.5e2, 1.75e2]),
                                       'nPduP': np.log10([7.5e1, 1.5e2]),
                                       'nPduL': np.log10([2e1, 4e1]),
                                       'nPduW': np.log10([5e0, 2e1])}


OD_PRIOR_PARAMETERS_RANGES = {exp_cond: {param_name: [mean - 4 * std, mean + 4 * std] for param_name, mean, std in
                                         zip(NORM_OD_MEAN_PRIOR_TRANS_PARAMETERS.columns,
                                             NORM_OD_MEAN_PRIOR_TRANS_PARAMETERS.loc[exp_cond, :],
                                             NORM_OD_STD_PRIOR_TRANS_PARAMETERS.loc[exp_cond, :])}
                              for exp_cond in NORM_OD_MEAN_PRIOR_TRANS_PARAMETERS.index}


LOG_UNIF_PRIOR_ALL_EXP = np.array([*list(CELL_PERMEABILITY_PARAMETER_RANGES.values()),
                                   *list(MCP_PERMEABILITY_PARAMETER_RANGES.values()),
                                   *list(KINETIC_PARAMETER_RANGES.values()),
                                   *list(COFACTOR_NUMBER_PARAMETER_RANGES.values()),
                                   *list(PDU_WT_ENZ_NUMBERS_PARAMETER_RANGES.values()),
                                   *list(dPDU_AJ_ENZ_NUMBER_PARAMETER_RANGES.values()),
                                   *list(GEOMETRY_PARAMETER_RANGES.values()),
                                   *[ranges for pr_dict in OD_PRIOR_PARAMETERS_RANGES.values() for ranges in
                                     pr_dict.values()]
                                   ])

# Normal model distribution parameters
CELL_PERMEABILITY_MEAN = {key: np.mean(val) for (key, val) in CELL_PERMEABILITY_PARAMETER_RANGES.items()}
CELL_PERMEABILITY_STD = {param_name: (CELL_PERMEABILITY_PARAMETER_RANGES[param_name][1]
                                      - np.mean(CELL_PERMEABILITY_PARAMETER_RANGES[param_name])) / 2
                         for param_name in CELL_PERMEABILITY_PARAMETER_RANGES.keys()}
MCP_PERMEABILITY_MEAN = {key: np.mean(val) for (key, val) in MCP_PERMEABILITY_PARAMETER_RANGES.items()}
MCP_PERMEABILITY_STD = {param_name: (MCP_PERMEABILITY_PARAMETER_RANGES[param_name][1]
                                     - np.mean(MCP_PERMEABILITY_PARAMETER_RANGES[param_name])) / 2
                        for param_name in MCP_PERMEABILITY_PARAMETER_RANGES.keys()}

KINETIC_PARAMETER_MEAN = {key: np.mean(val) for (key, val) in KINETIC_PARAMETER_RANGES.items()}
KINETIC_PARAMETERS_STD = {param_name: (KINETIC_PARAMETER_RANGES[param_name][1]
                                       - np.mean(KINETIC_PARAMETER_RANGES[param_name])) / 2
                          for param_name in KINETIC_PARAMETER_RANGES.keys()}

PDU_WT_ENZ_NUMBERS_PARAMETER_MEAN = {key: np.mean(val) for (key, val) in PDU_WT_ENZ_NUMBERS_PARAMETER_RANGES.items()}
PDU_WT_ENZ_NUMBERS_PARAMETER_STD = {param_name: (PDU_WT_ENZ_NUMBERS_PARAMETER_RANGES[param_name][1]
                                                 - np.mean(PDU_WT_ENZ_NUMBERS_PARAMETER_RANGES[param_name])) / 2
                                    for param_name in PDU_WT_ENZ_NUMBERS_PARAMETER_RANGES.keys()}

dPDU_AJ_ENZ_NUMBER_PARAMETER_MEAN = {key: np.mean(val) for (key, val) in dPDU_AJ_ENZ_NUMBER_PARAMETER_RANGES.items()}
dPDU_AJ_ENZ_NUMBER_PARAMETER_STD = {param_name: (dPDU_AJ_ENZ_NUMBER_PARAMETER_RANGES[param_name][1]
                                                 - np.mean(dPDU_AJ_ENZ_NUMBER_PARAMETER_RANGES[param_name])) / 2
                                    for param_name in dPDU_AJ_ENZ_NUMBER_PARAMETER_RANGES.keys()}

COFACTOR_NUMBER_PARAMETER_MEAN = {key: np.mean(val) for (key, val) in COFACTOR_NUMBER_PARAMETER_RANGES.items()}
COFACTOR_NUMBER_PARAMETER_STD = {param_name: (COFACTOR_NUMBER_PARAMETER_RANGES[param_name][1]
                                              - np.mean(COFACTOR_NUMBER_PARAMETER_RANGES[param_name])) / 2
                                 for param_name in COFACTOR_NUMBER_PARAMETER_RANGES.keys()}


GEOMETRY_PARAMETER_MEAN = {key: np.mean(val) for (key, val) in GEOMETRY_PARAMETER_RANGES.items()}
GEOMETRY_PARAMETER_STD = {param_name: (GEOMETRY_PARAMETER_RANGES[param_name][1]
                                       - np.mean(GEOMETRY_PARAMETER_RANGES[param_name])) / 2
                          for param_name in GEOMETRY_PARAMETER_RANGES.keys()}

# DCW model distribution parameters
OD_PRIOR_PARAMETERS_MEAN = {exp_cond: {param_name: mean for param_name, mean in
                                       zip(NORM_OD_MEAN_PRIOR_TRANS_PARAMETERS.columns,
                                           NORM_OD_MEAN_PRIOR_TRANS_PARAMETERS.loc[exp_cond, :])}
                            for exp_cond in NORM_OD_MEAN_PRIOR_TRANS_PARAMETERS.index}

OD_PRIOR_PARAMETERS_STD = {exp_cond: {param_name: mean for param_name, mean in
                                      zip(NORM_OD_STD_PRIOR_TRANS_PARAMETERS.columns,
                                          NORM_OD_STD_PRIOR_TRANS_PARAMETERS.loc[exp_cond, :])}
                           for exp_cond in NORM_OD_STD_PRIOR_TRANS_PARAMETERS.index}
