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
from exp_data import INIT_CONDS_GLY_PDO_DCW, NORM_DCW_MEAN_PRIOR_TRANS_PARAMETERS, NORM_DCW_STD_PRIOR_TRANS_PARAMETERS
ROOT_PATH = dirname(dirname(dirname(dirname(abspath(__file__)))))


#Uniform distribution parameters

DATA_LOG_UNIF_PARAMETER_RANGES = {'PermCellGlycerol': np.log10([1e-6, 1e-2]),
                                  'PermCellPDO': np.log10([1e-6, 1e-2]),
                                  'PermCell3HPA':  np.array([1e-6,np.log10(1e-2)]),

                                  'k1DhaB': np.log10([1e2, 1e4]),
                                  'k2DhaB': np.log10([1e-1, 1e2]),
                                  'k3DhaB': np.log10([1e2, 1e4]),
                                  'k4DhaB': np.log10([1e-1, 1e2]),

                                  'k1DhaT': np.log10([1e2, 1e4]),
                                  'k2DhaT': np.log10([1e-1, 1e2]),
                                  'k3DhaT': np.log10([1e2, 1e4]),
                                  'k4DhaT': np.log10([1e-1, 1e2]),

                                  'VmaxfMetab': np.log10([1e3*0.1, 1e4*10]),
                                  'KmMetabG': np.log10([1e-3,1e-1]),

                                  'DHAB_INIT': np.log10([0.1,1e1]),
                                  'DHAT_INIT': np.log10([0.1, 1e1])}

LOG_UNIF_G_EXT_INIT_PRIOR_PARAMETERS = {'G_EXT_INIT_50': [np.log10(INIT_CONDS_GLY_PDO_DCW[50][0] - 4*15.),
                                                          np.log10(INIT_CONDS_GLY_PDO_DCW[50][0] + 4*15.)],
                                    'G_EXT_INIT_60': [np.log10(INIT_CONDS_GLY_PDO_DCW[60][0] - 4*15),
                                                      np.log10(INIT_CONDS_GLY_PDO_DCW[60][0] + 4*15.)],
                                    'G_EXT_INIT_70': [np.log10(INIT_CONDS_GLY_PDO_DCW[70][0] - 4*15.),
                                                      np.log10(INIT_CONDS_GLY_PDO_DCW[70][0] + 4*15.)],
                                    'G_EXT_INIT_80': [np.log10(INIT_CONDS_GLY_PDO_DCW[80][0] - 4*15.),
                                                      np.log10(INIT_CONDS_GLY_PDO_DCW[80][0] + 4*15.)]
                                    }

LOG_UNIF_DCW_PRIOR_PARAMETERS_50 = {param_name + '_50': [mean-4*std, mean+4*std] for param_name, mean, std in zip(NORM_DCW_MEAN_PRIOR_TRANS_PARAMETERS.columns,
                                                                                                                                      NORM_DCW_MEAN_PRIOR_TRANS_PARAMETERS.loc[50, :],
                                                                                                                                      NORM_DCW_STD_PRIOR_TRANS_PARAMETERS.loc[50, :])}

LOG_UNIF_DCW_PRIOR_PARAMETERS_60 = {param_name + '_60': [mean-4*std, mean+4*std] for param_name, mean, std in zip(NORM_DCW_MEAN_PRIOR_TRANS_PARAMETERS.columns,
                                                                                                                                      NORM_DCW_MEAN_PRIOR_TRANS_PARAMETERS.loc[60, :],
                                                                                                                                      NORM_DCW_STD_PRIOR_TRANS_PARAMETERS.loc[60, :])}

LOG_UNIF_DCW_PRIOR_PARAMETERS_70 = {param_name + '_70': [mean-4*std, mean+4*std] for param_name, mean, std in zip(NORM_DCW_MEAN_PRIOR_TRANS_PARAMETERS.columns,
                                                                                                                                      NORM_DCW_MEAN_PRIOR_TRANS_PARAMETERS.loc[70, :],
                                                                                                                                      NORM_DCW_STD_PRIOR_TRANS_PARAMETERS.loc[70, :])}

LOG_UNIF_DCW_PRIOR_PARAMETERS_80 = {param_name + '_80': [mean-4*std, mean+4*std] for param_name, mean, std in zip(NORM_DCW_MEAN_PRIOR_TRANS_PARAMETERS.columns,
                                                                                                                                      NORM_DCW_MEAN_PRIOR_TRANS_PARAMETERS.loc[80, :],
                                                                                                                                      NORM_DCW_STD_PRIOR_TRANS_PARAMETERS.loc[80, :])}

# prior parameters for single experiment
LOG_UNIF_PRIOR_SINGLE_EXP = {}
LOG_UNIF_PRIOR_SINGLE_EXP[50] = np.array([*list(DATA_LOG_UNIF_PARAMETER_RANGES.values()),
                                          LOG_UNIF_G_EXT_INIT_PRIOR_PARAMETERS['G_EXT_INIT_50'],
                                               *list(LOG_UNIF_DCW_PRIOR_PARAMETERS_50.values())])
LOG_UNIF_PRIOR_SINGLE_EXP[60] = np.array([*list(DATA_LOG_UNIF_PARAMETER_RANGES.values()),
                                               LOG_UNIF_G_EXT_INIT_PRIOR_PARAMETERS['G_EXT_INIT_60'],
                                               *list(LOG_UNIF_DCW_PRIOR_PARAMETERS_60.values())])
LOG_UNIF_PRIOR_SINGLE_EXP[70] = np.array([*list(DATA_LOG_UNIF_PARAMETER_RANGES.values()),
                                               LOG_UNIF_G_EXT_INIT_PRIOR_PARAMETERS['G_EXT_INIT_70'],
                                               *list(LOG_UNIF_DCW_PRIOR_PARAMETERS_70.values())])
LOG_UNIF_PRIOR_SINGLE_EXP[80] = np.array([*list(DATA_LOG_UNIF_PARAMETER_RANGES.values()),
                                               LOG_UNIF_G_EXT_INIT_PRIOR_PARAMETERS['G_EXT_INIT_80'],
                                               *list(LOG_UNIF_DCW_PRIOR_PARAMETERS_80.values())])

LOG_UNIF_PRIOR_ALL_EXP = np.array([*list(DATA_LOG_UNIF_PARAMETER_RANGES.values()),
                                   *LOG_UNIF_G_EXT_INIT_PRIOR_PARAMETERS.values(),
                                   *list(LOG_UNIF_DCW_PRIOR_PARAMETERS_50.values()),
                                   *list(LOG_UNIF_DCW_PRIOR_PARAMETERS_60.values()),
                                   *list(LOG_UNIF_DCW_PRIOR_PARAMETERS_70.values()),
                                   *list(LOG_UNIF_DCW_PRIOR_PARAMETERS_80.values())])



# Normal model distribution parameters
LOG_NORM_MODEL_PRIOR_MEAN = {key: np.mean(val) for (key,val) in DATA_LOG_UNIF_PARAMETER_RANGES.items()}
LOG_NORM_MODEL_PRIOR_MEAN['PermCell3HPA'] = -4
LOG_NORM_MODEL_PRIOR_STD = {param_name: (DATA_LOG_UNIF_PARAMETER_RANGES[param_name][1]
                                         -np.mean(DATA_LOG_UNIF_PARAMETER_RANGES[param_name]))/2
                            for param_name in DATA_LOG_UNIF_PARAMETER_RANGES.keys()}

LOG_NORM_MODEL_PRIOR_STD['PermCell3HPA'] = (DATA_LOG_UNIF_PARAMETER_RANGES['PermCell3HPA'][1]+4)/2

LOG_NORM_MODEL_PRIOR_PARAMETERS = {param_name: [LOG_NORM_MODEL_PRIOR_MEAN[param_name],
                                                LOG_NORM_MODEL_PRIOR_STD[param_name]]
                                   for param_name in DATA_LOG_UNIF_PARAMETER_RANGES.keys()}

# Glycerol model distribution parameters
NORM_G_EXT_INIT_PRIOR_PARAMETERS = {'G_EXT_INIT_50': [np.log10(INIT_CONDS_GLY_PDO_DCW[50][0]), np.log10(1 + 15./INIT_CONDS_GLY_PDO_DCW[50][0])],
                                    'G_EXT_INIT_60': [np.log10(INIT_CONDS_GLY_PDO_DCW[60][0]), np.log10(1 + 15./INIT_CONDS_GLY_PDO_DCW[50][0])],
                                    'G_EXT_INIT_70': [np.log10(INIT_CONDS_GLY_PDO_DCW[70][0]), np.log10(1 + 15./INIT_CONDS_GLY_PDO_DCW[50][0])],
                                    'G_EXT_INIT_80': [np.log10(INIT_CONDS_GLY_PDO_DCW[80][0]), np.log10(1 + 15./INIT_CONDS_GLY_PDO_DCW[50][0])]
                                    }

NORM_G_EXT_INIT_PRIOR_MEAN = {param_name: NORM_G_EXT_INIT_PRIOR_PARAMETERS[param_name][0]
                              for param_name in NORM_G_EXT_INIT_PRIOR_PARAMETERS.keys()}

NORM_G_EXT_INIT_PRIOR_STD = {param_name: NORM_G_EXT_INIT_PRIOR_PARAMETERS[param_name][1]
                             for param_name in NORM_G_EXT_INIT_PRIOR_PARAMETERS.keys()}

# DCW model distribution parameters

NORM_DCW_PRIOR_PARAMETERS_50 = {param_name + '_50': [mean, std] for param_name, mean, std in zip(NORM_DCW_MEAN_PRIOR_TRANS_PARAMETERS.columns,
                                                                                                 NORM_DCW_MEAN_PRIOR_TRANS_PARAMETERS.loc[50, :],
                                                                                                 NORM_DCW_STD_PRIOR_TRANS_PARAMETERS.loc[50, :])}

NORM_DCW_PRIOR_PARAMETERS_60 = {param_name + '_60': [mean, std] for param_name, mean, std in zip(NORM_DCW_MEAN_PRIOR_TRANS_PARAMETERS.columns,
                                                                                                 NORM_DCW_MEAN_PRIOR_TRANS_PARAMETERS.loc[60, :],
                                                                                                 NORM_DCW_STD_PRIOR_TRANS_PARAMETERS.loc[60, :])}

NORM_DCW_PRIOR_PARAMETERS_70 = {param_name + '_70': [mean, std] for param_name, mean, std in zip(NORM_DCW_MEAN_PRIOR_TRANS_PARAMETERS.columns,
                                                                                                 NORM_DCW_MEAN_PRIOR_TRANS_PARAMETERS.loc[70, :],
                                                                                                 NORM_DCW_STD_PRIOR_TRANS_PARAMETERS.loc[70, :])}

NORM_DCW_PRIOR_PARAMETERS_80 = {param_name + '_80': [mean, std] for param_name, mean, std in zip(NORM_DCW_MEAN_PRIOR_TRANS_PARAMETERS.columns,
                                                                                                 NORM_DCW_MEAN_PRIOR_TRANS_PARAMETERS.loc[80, :],
                                                                                                 NORM_DCW_STD_PRIOR_TRANS_PARAMETERS.loc[80, :])}

# prior parameters for single experiment
NORM_PRIOR_MEAN_SINGLE_EXP = {}
NORM_PRIOR_STD_RT_SINGLE_EXP = {}
for gly_cond in [50,60,70,80]:
    NORM_PRIOR_MEAN_SINGLE_EXP[gly_cond] = np.array([*list(LOG_NORM_MODEL_PRIOR_MEAN.values()),
                                                     NORM_G_EXT_INIT_PRIOR_MEAN['G_EXT_INIT_' + str(gly_cond)],
                                                     *NORM_DCW_MEAN_PRIOR_TRANS_PARAMETERS.loc[gly_cond, :].tolist()])
    NORM_PRIOR_STD_RT_SINGLE_EXP[gly_cond] = np.diag([*list(LOG_NORM_MODEL_PRIOR_STD.values()),
                                                      NORM_G_EXT_INIT_PRIOR_STD['G_EXT_INIT_' + str(gly_cond)],
                                                      *NORM_DCW_STD_PRIOR_TRANS_PARAMETERS.loc[gly_cond, :].tolist()])

# prior parameters for all experiment
NORM_PRIOR_PARAMETER_ALL_EXP_DICT = {**LOG_NORM_MODEL_PRIOR_PARAMETERS, **NORM_G_EXT_INIT_PRIOR_PARAMETERS,
                                     **NORM_DCW_PRIOR_PARAMETERS_50,  **NORM_DCW_PRIOR_PARAMETERS_60,
                                     **NORM_DCW_PRIOR_PARAMETERS_70, **NORM_DCW_PRIOR_PARAMETERS_80}

NORM_PRIOR_MEAN_ALL_EXP = np.array([*list(LOG_NORM_MODEL_PRIOR_MEAN.values()),
                                    *list(NORM_G_EXT_INIT_PRIOR_MEAN.values()),
                                    *NORM_DCW_MEAN_PRIOR_TRANS_PARAMETERS.loc[50, :].tolist(),
                                    *NORM_DCW_MEAN_PRIOR_TRANS_PARAMETERS.loc[60, :].tolist(),
                                    *NORM_DCW_MEAN_PRIOR_TRANS_PARAMETERS.loc[70, :].tolist(),
                                    *NORM_DCW_MEAN_PRIOR_TRANS_PARAMETERS.loc[80, :].tolist()])

NORM_PRIOR_STD_RT_ALL_EXP = np.diag([*list(LOG_NORM_MODEL_PRIOR_STD.values()),
                                     *list(NORM_G_EXT_INIT_PRIOR_STD.values()),
                                     *NORM_DCW_STD_PRIOR_TRANS_PARAMETERS.loc[50, :].tolist(),
                                     *NORM_DCW_STD_PRIOR_TRANS_PARAMETERS.loc[60, :].tolist(),
                                     *NORM_DCW_STD_PRIOR_TRANS_PARAMETERS.loc[70, :].tolist(),
                                     *NORM_DCW_STD_PRIOR_TRANS_PARAMETERS.loc[80, :].tolist()])


