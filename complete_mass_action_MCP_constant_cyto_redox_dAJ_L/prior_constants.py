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
                                      # 'PermCellHate': np.log10([1e-6, 1e-2])
                                      }

MCP_PERMEABILITY_PARAMETER_RANGES = {'PermMCPGlycerol': np.log10([1e-15, 1e-6]),
                                     'PermMCPPDO': np.log10([1e-15, 1e-6]),
                                     'PermMCP3HPA': np.log10([1e-15, 1e-6]),
                                     'PermMCPHCoA': np.log10([1e-15, 1e-6]),
                                     'PermMCPHPhosph': np.log10([1e-15, 1e-6]),
                                     'PermMCPNADH': np.log10([1e-15, 1e-6]),
                                     'PermMCPNAD': np.log10([1e-15, 1e-6])
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
                            'KeqPduL': np.log10([1e-5, 1e-3]),

                            # 'k1PduW': np.log10([1e0, 1e4]),
                            # 'k2PduW': np.log10([1e-2, 1e2]),
                            # 'k3PduW': np.log10([1e0, 1e4]),
                            # 'KeqPduLW': np.log10([1e0, 1e3]),

                            'VmaxfGlpK_WT_L': np.log10([1e3 * 0.01, 1e3 * 1]),
                            'KmGlpK_WT_L': np.log10([1e-3, 1e-1]),
                            'VmaxfGlpK_dAJ_L': np.log10([1e3 * 0.01, 1e3 * 1]),
                            'KmGlpK_dAJ_L': np.log10([1e-3, 1e-1]),
                            'VmaxfGlpK_dP_L': np.log10([1e3 * 0.01, 1e3 * 1]),
                            'KmGlpK_dP_L': np.log10([1e-3, 1e-1]),
                            'VmaxfGlpK_dD_L': np.log10([1e3 * 0.01, 1e3 * 1]),
                            'KmGlpK_dD_L': np.log10([1e-3, 1e-1])
                            }

GEOMETRY_PARAMETER_RANGES = {'nMCPs': np.log10([3, 30])}


COFACTOR_NUMBER_PARAMETER_RANGES_WT_L = {'NADH_NAD_TOTAL_CYTO_WT_L': np.log10([0.1, 1e1]),
                                    'NADH_NAD_RATIO_CYTO_WT_L': np.log10([1e-2,1e0]),
                                       'NADH_NAD_TOTAL_MCP_WT_L': np.log10([0.1, 1e1]),
                                       'NADH_NAD_RATIO_MCP_WT_L': np.log10([1e-2, 1e1])}

COFACTOR_NUMBER_PARAMETER_RANGES_dAJ_L = {'NADH_NAD_TOTAL_CYTO_dAJ_L': np.log10([0.1, 1e1]),
                                       'NADH_NAD_RATIO_CYTO_dAJ_L': np.log10([1e-2,1e0]),
                                        'NADH_NAD_TOTAL_MCP_dAJ_L': np.log10([0.1, 1e1]),
                                       'NADH_NAD_RATIO_MCP_dAJ_L': np.log10([1e-2, 1e1])
                                        }

COFACTOR_NUMBER_PARAMETER_RANGES_dP_L = {'NADH_NAD_TOTAL_CYTO_dP_L': np.log10([0.1, 1e1]),
                                    'NADH_NAD_RATIO_CYTO_dP_L': np.log10([1e-2, 1e0]),
                                    'NADH_NAD_TOTAL_MCP_dP_L': np.log10([0.1, 1e1]),
                                    'NADH_NAD_RATIO_MCP_dP_L': np.log10([1e-2, 1e1])
                                    }

COFACTOR_NUMBER_PARAMETER_RANGES = {**COFACTOR_NUMBER_PARAMETER_RANGES_WT_L,
                                    **COFACTOR_NUMBER_PARAMETER_RANGES_dAJ_L,
                                    **COFACTOR_NUMBER_PARAMETER_RANGES_dP_L
                                    }

PDU_ENZ_NUMBERS_PARAMETER_RANGES_WT_L = {'nPduCDE_WT_L': np.log10([4e2, 6e2]),
                                       'nPduQ_WT_L': np.log10([1.5e2, 1.75e2]),
                                       'nPduP_WT_L': np.log10([1.5e2, 2.5e2]),
                                       'nPduL_WT_L': np.log10([2e1, 4e1])}
                                       #'nPduW': np.log10([5e0, 2e1])}

PDU_ENZ_NUMBERS_PARAMETER_RANGES_dAJ_L = {'nPduCDE_dAJ_L': np.log10([1e2, 3e2]),
                                       'nPduQ_dAJ_L': np.log10([1.5e2, 1.75e2]),
                                       'nPduP_dAJ_L': np.log10([7.5e1, 1.5e2]),
                                       'nPduL_dAJ_L': np.log10([2e1, 4e1])}
                                       # 'nPduW': np.log10([5e0, 2e1])}

PDU_ENZ_NUMBERS_PARAMETER_RANGES_dP_L = {'nPduCDE_dP_L': np.log10([4e2, 6e2]),
                                       'nPduQ_dP_L': np.log10([1.5e2, 1.75e2]),
                                       'nPduP_dP_L': np.log10([1.5e2, 2.5e2]),
                                       'nPduL_dP_L': np.log10([2e1, 4e1])}


PDU_ENZ_NUMBERS_PARAMETER_RANGES = {**PDU_ENZ_NUMBERS_PARAMETER_RANGES_WT_L,
                                    **PDU_ENZ_NUMBERS_PARAMETER_RANGES_dAJ_L,
                                    **PDU_ENZ_NUMBERS_PARAMETER_RANGES_dP_L
                                    }


OD_PRIOR_PARAMETER_RANGES = {exp_cond: {param_name + '_' + exp_cond: [mean - 4 * std, mean + 4 * std] for param_name, mean, std in
                                        zip(NORM_OD_MEAN_PRIOR_TRANS_PARAMETERS.columns,
                                             NORM_OD_MEAN_PRIOR_TRANS_PARAMETERS.loc[exp_cond, :],
                                             NORM_OD_STD_PRIOR_TRANS_PARAMETERS.loc[exp_cond, :])}
                             for exp_cond in NORM_OD_MEAN_PRIOR_TRANS_PARAMETERS.index}


LOG_UNIF_PRIOR_ALL_EXP_LIST = np.array([*list(CELL_PERMEABILITY_PARAMETER_RANGES.values()),
                                        *list(MCP_PERMEABILITY_PARAMETER_RANGES.values()),
                                        *list(KINETIC_PARAMETER_RANGES.values()),
                                        *list(GEOMETRY_PARAMETER_RANGES.values()),
                                        *list(COFACTOR_NUMBER_PARAMETER_RANGES.values()),
                                        *list(PDU_ENZ_NUMBERS_PARAMETER_RANGES.values()),
                                        *[ranges for pr_dict in OD_PRIOR_PARAMETER_RANGES.values() for ranges in
                                     pr_dict.values()]
                                        ])

LOG_UNIF_PRIOR_ALL_EXP = {**CELL_PERMEABILITY_PARAMETER_RANGES,
                                   **MCP_PERMEABILITY_PARAMETER_RANGES,
                                   **KINETIC_PARAMETER_RANGES,
                               **GEOMETRY_PARAMETER_RANGES,

                               **COFACTOR_NUMBER_PARAMETER_RANGES,
                                   **PDU_ENZ_NUMBERS_PARAMETER_RANGES,
                                   **{key:val for pr_dict in OD_PRIOR_PARAMETER_RANGES.values() for key,val in pr_dict.items()}}

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
KINETIC_PARAMETER_STD = {param_name: (KINETIC_PARAMETER_RANGES[param_name][1]
                                       - np.mean(KINETIC_PARAMETER_RANGES[param_name])) / 2
                          for param_name in KINETIC_PARAMETER_RANGES.keys()}

GEOMETRY_PARAMETER_MEAN = {key: np.mean(val) for (key, val) in GEOMETRY_PARAMETER_RANGES.items()}
GEOMETRY_PARAMETER_STD = {param_name: (GEOMETRY_PARAMETER_RANGES[param_name][1]
                                       - np.mean(GEOMETRY_PARAMETER_RANGES[param_name])) / 2
                          for param_name in GEOMETRY_PARAMETER_RANGES.keys()}


COFACTOR_NUMBER_PARAMETER_MEAN_WT_L = {key: np.mean(val) for (key, val) in COFACTOR_NUMBER_PARAMETER_RANGES_WT_L.items()}
COFACTOR_NUMBER_PARAMETER_STD_WT_L = {param_name: (COFACTOR_NUMBER_PARAMETER_RANGES_WT_L[param_name][1]
                                              - np.mean(COFACTOR_NUMBER_PARAMETER_RANGES_WT_L[param_name])) / 2
                                 for param_name in COFACTOR_NUMBER_PARAMETER_RANGES_WT_L.keys()}


COFACTOR_NUMBER_PARAMETER_MEAN_dAJ_L = {key: np.mean(val) for (key, val) in COFACTOR_NUMBER_PARAMETER_RANGES_dAJ_L.items()}
COFACTOR_NUMBER_PARAMETER_STD_dAJ_L = {param_name: (COFACTOR_NUMBER_PARAMETER_RANGES_dAJ_L[param_name][1]
                                              - np.mean(COFACTOR_NUMBER_PARAMETER_RANGES_dAJ_L[param_name])) / 2
                                 for param_name in COFACTOR_NUMBER_PARAMETER_RANGES_dAJ_L.keys()}

COFACTOR_NUMBER_PARAMETER_MEAN_dP_L = {key: np.mean(val) for (key, val) in COFACTOR_NUMBER_PARAMETER_RANGES_dP_L.items()}
COFACTOR_NUMBER_PARAMETER_STD_dP_L = {param_name: (COFACTOR_NUMBER_PARAMETER_RANGES_dP_L[param_name][1]
                                              - np.mean(COFACTOR_NUMBER_PARAMETER_RANGES_dP_L[param_name])) / 2
                                 for param_name in COFACTOR_NUMBER_PARAMETER_RANGES_dP_L.keys()}

COFACTOR_NUMBER_PARAMETER_MEAN = {**COFACTOR_NUMBER_PARAMETER_MEAN_WT_L,
                                    **COFACTOR_NUMBER_PARAMETER_MEAN_dAJ_L,
                                    **COFACTOR_NUMBER_PARAMETER_MEAN_dP_L
                                    }

COFACTOR_NUMBER_PARAMETER_STD = {**COFACTOR_NUMBER_PARAMETER_STD_WT_L,
                                    **COFACTOR_NUMBER_PARAMETER_STD_dAJ_L,
                                    **COFACTOR_NUMBER_PARAMETER_STD_dP_L
                                    }

PDU_ENZ_NUMBERS_PARAMETER_MEAN_WT_L = {key: np.mean(val) for (key, val) in PDU_ENZ_NUMBERS_PARAMETER_RANGES_WT_L.items()}
PDU_ENZ_NUMBERS_PARAMETER_STD_WT_L = {param_name: (PDU_ENZ_NUMBERS_PARAMETER_RANGES_WT_L[param_name][1]
                                                 - np.mean(PDU_ENZ_NUMBERS_PARAMETER_RANGES_WT_L[param_name])) / 2
                                    for param_name in PDU_ENZ_NUMBERS_PARAMETER_RANGES_WT_L.keys()}

PDU_ENZ_NUMBER_PARAMETER_MEAN_dAJ_L = {key: np.mean(val) for (key, val) in PDU_ENZ_NUMBERS_PARAMETER_RANGES_dAJ_L.items()}
PDU_ENZ_NUMBER_PARAMETER_STD_dAJ_L = {param_name: (PDU_ENZ_NUMBERS_PARAMETER_RANGES_dAJ_L[param_name][1]
                                                 - np.mean(PDU_ENZ_NUMBERS_PARAMETER_RANGES_dAJ_L[param_name])) / 2
                                    for param_name in PDU_ENZ_NUMBERS_PARAMETER_RANGES_dAJ_L.keys()}

PDU_ENZ_NUMBER_PARAMETER_MEAN_dP_L = {key: np.mean(val) for (key, val) in PDU_ENZ_NUMBERS_PARAMETER_RANGES_dP_L.items()}
PDU_ENZ_NUMBER_PARAMETER_STD_dP_L = {param_name: (PDU_ENZ_NUMBERS_PARAMETER_RANGES_dP_L[param_name][1]
                                                 - np.mean(PDU_ENZ_NUMBERS_PARAMETER_RANGES_dP_L[param_name])) / 2
                                    for param_name in PDU_ENZ_NUMBERS_PARAMETER_RANGES_dP_L.keys()}

PDU_ENZ_NUMBERS_PARAMETER_MEAN = {**PDU_ENZ_NUMBERS_PARAMETER_MEAN_WT_L,
                                    **PDU_ENZ_NUMBER_PARAMETER_MEAN_dAJ_L,
                                    **PDU_ENZ_NUMBER_PARAMETER_MEAN_dP_L
                                    }

PDU_ENZ_NUMBERS_PARAMETER_STD = {**PDU_ENZ_NUMBERS_PARAMETER_STD_WT_L,
                                    **PDU_ENZ_NUMBER_PARAMETER_STD_dAJ_L,
                                    **PDU_ENZ_NUMBER_PARAMETER_STD_dP_L
                                    }
# DCW model distribution parameters
OD_PRIOR_PARAMETER_MEAN = {exp_cond: {param_name + '_' + exp_cond: mean for param_name, mean in
                                      zip(NORM_OD_MEAN_PRIOR_TRANS_PARAMETERS.columns,
                                           NORM_OD_MEAN_PRIOR_TRANS_PARAMETERS.loc[exp_cond, :])}
                           for exp_cond in NORM_OD_MEAN_PRIOR_TRANS_PARAMETERS.index}

OD_PRIOR_PARAMETER_STD = {exp_cond: {param_name + '_' + exp_cond: mean for param_name, mean in
                                     zip(NORM_OD_STD_PRIOR_TRANS_PARAMETERS.columns,
                                          NORM_OD_STD_PRIOR_TRANS_PARAMETERS.loc[exp_cond, :])}
                          for exp_cond in NORM_OD_STD_PRIOR_TRANS_PARAMETERS.index}

LOG_NORM_PRIOR_ALL_EXP_MEAN = {**CELL_PERMEABILITY_MEAN,
                                   **MCP_PERMEABILITY_MEAN,
                                   **KINETIC_PARAMETER_MEAN,
                               **GEOMETRY_PARAMETER_MEAN,

                               **COFACTOR_NUMBER_PARAMETER_MEAN,
                                   **PDU_ENZ_NUMBERS_PARAMETER_MEAN,
                                   **{key:val for pr_dict in OD_PRIOR_PARAMETER_MEAN.values() for key,val in pr_dict.items()}}
LOG_NORM_PRIOR_ALL_EXP_STD = {**CELL_PERMEABILITY_STD,
                                   **MCP_PERMEABILITY_STD,
                                   **KINETIC_PARAMETER_STD,
                               **GEOMETRY_PARAMETER_STD,

                               **COFACTOR_NUMBER_PARAMETER_STD,
                                   **PDU_ENZ_NUMBERS_PARAMETER_STD,
                                   **{key:val for pr_dict in OD_PRIOR_PARAMETER_STD.values() for key,val in pr_dict.items()}}
