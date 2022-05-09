# teX names for parameters
MODEL_PARAMS_TO_TEX = {'PermCellGlycerol':r'$P_G$',
                       'PermCellPDO':r'$P_P$',
                       'PermCell3HPA':r'$P_H$',
                       'k1DhaB': r'$k_{1,\text{DhaB}}$',
                       'k2DhaB': r'$k_{2,\text{DhaB}}$',
                       'k3DhaB': r'$k_{3,\text{DhaB}}$',
                       'k4DhaB': r'$k_{4,\text{DhaB}}$',
                       'k1DhaT': r'$k_{1,\text{DhaT}}$',
                       'k2DhaT': r'$k_{2,\text{DhaT}}$',
                       'k3DhaT': r'$k_{3,\text{DhaT}}$',
                       'k4DhaT': r'$k_{4,\text{DhaT}}$',
                       'VmaxfMetab': r'$V_{\text{max,Metab}}^{f}$',
                       'KmMetabG': r'$K_{\text{M,Metab}}^{G}$',
                       'DHAB_INIT': r'$[\text{DhaB}](0)$',
                       'DHAT_INIT' : r'$[\text{DhaB}](0)$'}

G_EXT_INIT_TO_TEX = {'G_EXT_INIT_50': "$G(0)$ for first experiment",
                     'G_EXT_INIT_60': "$G(0)$ for second experiment",
                     'G_EXT_INIT_70': "$G(0)$ for third experiment",
                     'G_EXT_INIT_80': "$G(0)$ for fourth experiment"
                     }

DCW_TO_TEX_50 = {param_name + "_50": "$"+param_name + "_{50}$" for param_name in NORM_DCW_MEAN_PRIOR_PARAMETERS.columns}
DCW_TO_TEX_60 = {param_name + "_60": "$"+param_name + "_{60}$" for param_name in NORM_DCW_MEAN_PRIOR_PARAMETERS.columns}
DCW_TO_TEX_70 = {param_name + "_70": "$"+param_name + "_{70}$" for param_name in NORM_DCW_MEAN_PRIOR_PARAMETERS.columns}
DCW_TO_TEX_80 = {param_name + "_80": "$"+param_name + "_{80}$" for param_name in NORM_DCW_MEAN_PRIOR_PARAMETERS.columns}

VARS_ALL_EXP_TO_TEX = {**MODEL_PARAMS_TO_TEX, **G_EXT_INIT_TO_TEX, **DCW_TO_TEX_50, **DCW_TO_TEX_60, **DCW_TO_TEX_70,
                       **DCW_TO_TEX_80}

MODEL_PARAMS_TO_UNITS =  {'PermCellGlycerol':'m/s',
                          'PermCellPDO':'m/s',
                          'PermCell3HPA':'m/s',
                          'k1DhaB': '/mM s',
                          'k2DhaB': '/s',
                          'k3DhaB': '/s',
                          'k4DhaB': '/mM s',
                          'k1DhaT': '/mM s',
                          'k2DhaT': '/s',
                          'k3DhaT': '/s',
                          'k4DhaT': '/mM s',
                          'VmaxfMetab': 'mM/s',
                          'KmMetabG': 'mM',
                          'DHAB_INIT': 'mM',
                          'DHAT_INIT': 'mM'}

G_EXT_INIT_TO_TEX = {'G_EXT_INIT_50': "mM",
                     'G_EXT_INIT_60': "mM",
                     'G_EXT_INIT_70': "mM",
                     'G_EXT_INIT_80': "mM"
                     }

DCW_TO_UNITS_50 = {param_name + "_50": "" for param_name in NORM_DCW_MEAN_PRIOR_PARAMETERS.columns}
DCW_TO_UNITS_60 = {param_name + "_60": "" for param_name in NORM_DCW_MEAN_PRIOR_PARAMETERS.columns}
DCW_TO_UNITS_70 = {param_name + "_70": "" for param_name in NORM_DCW_MEAN_PRIOR_PARAMETERS.columns}
DCW_TO_UNITS_80 = {param_name + "_80": "" for param_name in NORM_DCW_MEAN_PRIOR_PARAMETERS.columns}

VARS_ALL_EXP_TO_UNITS = {**MODEL_PARAMS_TO_UNITS, **G_EXT_INIT_TO_TEX, **DCW_TO_UNITS_50, **DCW_TO_UNITS_60, **DCW_TO_UNITS_70,
                         **DCW_TO_UNITS_80}
