from constants import DCW_CONSTANTS

# teX names for parameters
MODEL_PARAMS_TO_TEX = {'PermCellGlycerol':r'$P_G$',
                       'PermCellPDO':r'$P_P$',
                       'PermCell3HPA':r'$P_H$',
                       'k1DhaB': r'$k_{1,\mathrm{DhaB}}$',
                       'k2DhaB': r'$k_{2,\mathrm{DhaB}}$',
                       'k3DhaB': r'$k_{3,\mathrm{DhaB}}$',
                       'KeqDhaB': r'$K_{eq}^{\mathrm{DhaB}}$',
                       'k1DhaT': r'$k_{1,\mathrm{DhaT}}$',
                       'k2DhaT': r'$k_{2,\mathrm{DhaT}}$',
                       'k3DhaT': r'$k_{3,\mathrm{DhaT}}$',
                       'KeqDhaT': r'$K_{eq}^{\mathrm{DhaT}}$',
                       'VmaxfMetab': r'$V_{\mathrm{max,Metab}}^{f}$',
                       'KmMetabG': r'$K_{\mathrm{M,Metab}}^{G}$',
                       'DHAB_INIT': r'$[\mathrm{DhaB}](0)$',
                       'DHAT_INIT' : r'$[\mathrm{DhaB}](0)$',
                       'kcatfDhaB': r'$k_{\mathrm{cat,DhaB}}^{f}$',
                       'KmGlycerolDhaB': r'$K_{\mathrm{M,DhaB}}^{\mathrm{Glycerol}}$',
                       'kcatrDhaB': r'$k_{\mathrm{cat,DhaB}}^{r}$',
                       'KmHPADhaB': r'$K_{\mathrm{M,DhaB}}^{\mathrm{HPA}}$',
                       'kcatfDhaT': r'$k_{\mathrm{cat,DhaT}}^{f}$',
                       'KmHPADhaT': r'$K_{\mathrm{M,DhaT}}^{\mathrm{HPA}}$',
                       'kcatrDhaT': r'$k_{\mathrm{cat,DhaT}}^{r}$',
                       'KmPDODhaT': r'$K_{\mathrm{M,DhaT}}^{\mathrm{PDO}}$'}

G_EXT_INIT_TO_TEX = {'G_EXT_INIT_50': r"$G_1(0)$",
                     'G_EXT_INIT_60': r"$G_2(0)$",
                     'G_EXT_INIT_70': r"$G_3(0)$",
                     'G_EXT_INIT_80': r"$G_4(0)$"
                     }

DCW_TO_TEX_50 = {param_name + "_50": "$"+param_name + "_{1}$" for param_name in DCW_CONSTANTS}
DCW_TO_TEX_60 = {param_name + "_60": "$"+param_name + "_{2}$" for param_name in DCW_CONSTANTS}
DCW_TO_TEX_70 = {param_name + "_70": "$"+param_name + "_{3}$" for param_name in DCW_CONSTANTS}
DCW_TO_TEX_80 = {param_name + "_80": "$"+param_name + "_{4}$" for param_name in DCW_CONSTANTS}

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
DCW_UNITS = {'L':'g/L', 'k':'1/hr', 'A': 'g/L'}
DCW_TO_UNITS_50 = {param_name + "_50": param_unit for param_name, param_unit in DCW_UNITS.items()}
DCW_TO_UNITS_60 = {param_name + "_60": param_unit for param_name, param_unit in DCW_UNITS.items()}
DCW_TO_UNITS_70 = {param_name + "_70": param_unit for param_name, param_unit in DCW_UNITS.items()}
DCW_TO_UNITS_80 = {param_name + "_80": param_unit for param_name, param_unit in DCW_UNITS.items()}

VARS_ALL_EXP_TO_UNITS = {**MODEL_PARAMS_TO_UNITS, **G_EXT_INIT_TO_TEX, **DCW_TO_UNITS_50, **DCW_TO_UNITS_60, **DCW_TO_UNITS_70,
                         **DCW_TO_UNITS_80}
