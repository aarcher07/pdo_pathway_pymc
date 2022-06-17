from os.path import dirname, abspath
import pandas as pd
import pickle
ROOT_PATH = dirname(dirname(abspath(__file__)))

# reformat experimental data as dictionary
experimental_time_series_df = pd.read_csv(ROOT_PATH + "/data_files/experimental_time_series_cleaned.csv")

TIME_SAMPLES = {} # dictionary of time samples for each initial glycerol concentration experiment
DATA_SAMPLES = {} # dictionary of Glycerol, PDO and DCW collected for each initial glycerol concentration experiment

for gly_cond in [50,60,70,80]:
    rows_bool = experimental_time_series_df.loc[:,"Glycerol Init (g/L)"] == gly_cond
    TIME_SAMPLES[gly_cond] = experimental_time_series_df.loc[rows_bool,"Time (hrs)"].to_numpy()
    DATA_SAMPLES[gly_cond] = experimental_time_series_df[["Glycerol (mM)","PDO (mM)","DCW (g/L)"]][rows_bool].to_numpy()

# initial conditions
INIT_CONDS_GLY_PDO_DCW = {gly_cond:DATA_SAMPLES[gly_cond][0,0:2] for gly_cond in [50,60,70,80]}

with open(ROOT_PATH + '/data_files/std_experimental_data.pkl', 'rb') as f:
    STD_EXPERIMENTAL_DATA = pickle.load(f)

NORM_DCW_MEAN_PRIOR_PARAMETERS = pd.read_csv(ROOT_PATH + '/data_files/dcw_fit_params', index_col = 0)
NORM_DCW_STD_PRIOR_PARAMETERS = pd.read_csv(ROOT_PATH + '/data_files/dcw_std_params', index_col = 0)

NORM_DCW_MEAN_PRIOR_TRANS_PARAMETERS = pd.read_csv(ROOT_PATH + '/data_files/dcw_mean_params_log_trans', index_col = 0)
NORM_DCW_STD_PRIOR_TRANS_PARAMETERS = pd.read_csv(ROOT_PATH + '/data_files/dcw_std_params_log_trans', index_col = 0)
