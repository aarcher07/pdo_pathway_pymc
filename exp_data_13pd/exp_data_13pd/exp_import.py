import pandas as pd
from os.path import dirname, abspath

ROOT_PATH = dirname(abspath(__file__))

TIME_SERIES_MEAN = pd.read_excel(ROOT_PATH + "/data_files/13_pdo_salmonella_typ_data.xlsx", sheet_name = "mean",
                                 index_col=[0,1]).unstack('Reactant')
TIME_SAMPLES = TIME_SERIES_MEAN.index.to_numpy()
TIME_SERIES_STD = pd.read_excel(ROOT_PATH + "/data_files/13_pdo_salmonella_typ_data.xlsx", sheet_name = "std",
                                 index_col=[0,1]).unstack('Reactant')



