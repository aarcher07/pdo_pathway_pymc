import pandas as pd
from os.path import dirname, abspath
import numpy as np
ROOT_PATH = dirname(abspath(__file__))

type_ = {'Reactant': str,
        'Time': np.float64,
         'WT': np.float64,
         'WT-L': np.float64,
         'dD': np.float64,
         'dD-L': np.float64,
         'dAJ': np.float64,
         'dAJ-L': np.float64,
         'dP': np.float64,
         'dP-L': np.float64,
         }
TIME_SERIES_MEAN = pd.read_excel(ROOT_PATH + "/data_files/13_pdo_salmonella_typ_data.xlsx", sheet_name = "mean",
                                 index_col=[0,1], dtype=type_).unstack('Reactant')
TIME_SAMPLES = TIME_SERIES_MEAN.index.to_numpy()
new_cols = TIME_SERIES_MEAN.columns.reindex(['glycerol', '3-HPA','13PD', 'OD600' ], level=1)
TIME_SERIES_MEAN = TIME_SERIES_MEAN.reindex(columns=new_cols[0])
TIME_SERIES_STD = pd.read_excel(ROOT_PATH + "/data_files/13_pdo_salmonella_typ_data.xlsx", sheet_name = "std",
                                 index_col=[0,1], dtype=type_).unstack('Reactant')
TIME_SERIES_STD[TIME_SERIES_STD< 1e-4] = 1e-4
new_cols = TIME_SERIES_STD.columns.reindex(['glycerol', '3-HPA','13PD', 'OD600' ], level=1)
TIME_SERIES_STD = TIME_SERIES_STD.reindex(columns=new_cols[0])

NORM_OD_MEAN_PRIOR_TRANS_PARAMETERS = pd.read_csv(ROOT_PATH + '/data_files/OD_mean_params_trans', index_col = 0)
NORM_OD_STD_PRIOR_TRANS_PARAMETERS = pd.read_csv(ROOT_PATH + '/data_files/OD_std_params_trans', index_col = 0)
