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
                       'kcatfMetab': r'$V_{\mathrm{max,Metab}}^{f}$',
                       'KmMetabG': r'$K_{\mathrm{M,Metab}}^{G}$'}

ENZ_INIT_TO_TEX = {'DHAB_INIT_50': r"$[\mathrm{DhaB}]_1(0)$",
                     'DHAB_INIT_60': r"$[\mathrm{DhaB}]_2(0)$",
                     'DHAB_INIT_70': r"$[\mathrm{DhaB}]_3(0)$",
                     'DHAB_INIT_80': r"$[\mathrm{DhaB}]_4(0)$",
                     'DHAT_INIT_50': r"$[\mathrm{DhaT}]_1(0)$",
                     'DHAT_INIT_60': r"$[\mathrm{DhaT}]_2(0)$",
                     'DHAT_INIT_70': r"$[\mathrm{DhaT}]_3(0)$",
                     'DHAT_INIT_80': r"$[\mathrm{DhaT}]_4(0)$",
                     'E0_Metab_50': r"$[\mathrm{Metab}]_1$",
                     'E0_Metab_60': r"$[\mathrm{Metab}]_2$",
                     'E0_Metab_70': r"$[\mathrm{Metab}]_3$",
                     'E0_Metab_80': r"$[\mathrm{Metab}]_4$"
                     }

G_EXT_INIT_TO_TEX = {'G_EXT_INIT_50': r"$G_1(0)$",
                     'G_EXT_INIT_60': r"$G_2(0)$",
                     'G_EXT_INIT_70': r"$G_3(0)$",
                     'G_EXT_INIT_80': r"$G_4(0)$"
                     }

DCW_TO_TEX_50 = {param_name + "_50": "$"+param_name + "_{1}$" for param_name in DCW_CONSTANTS}
DCW_TO_TEX_60 = {param_name + "_60": "$"+param_name + "_{2}$" for param_name in DCW_CONSTANTS}
DCW_TO_TEX_70 = {param_name + "_70": "$"+param_name + "_{3}$" for param_name in DCW_CONSTANTS}
DCW_TO_TEX_80 = {param_name + "_80": "$"+param_name + "_{4}$" for param_name in DCW_CONSTANTS}

VARS_ALL_EXP_TO_TEX = {**MODEL_PARAMS_TO_TEX, **ENZ_INIT_TO_TEX, **G_EXT_INIT_TO_TEX, **DCW_TO_TEX_50, **DCW_TO_TEX_60, **DCW_TO_TEX_70,
                       **DCW_TO_TEX_80}

MODEL_PARAMS_TO_UNITS =  {'PermCellGlycerol':'m/s',
                          'PermCellPDO':'m/s',
                          'PermCell3HPA':'m/s',
                          'k1DhaB': '/mM s',
                          'k2DhaB': '/s',
                          'k3DhaB': '/s',
                          'KeqDhaB': '',
                          'k1DhaT': '/mM s',
                          'k2DhaT': '/s',
                          'k3DhaT': '/s',
                          'KeqDhaT': '',
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
