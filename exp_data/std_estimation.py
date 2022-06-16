"""
Computes the standard deviation of the experimental exp_data in mM for glycerol and PDO, and
g/L for DCW and stores a dictionary
"""

import numpy as np
import matplotlib.pyplot as plt
from os.path import dirname, abspath
import pandas as pd
from exp_data.cleaning_constants import *
import pickle
ROOT_PATH = dirname(dirname(abspath(__file__)))

time_series_df_cleaned_complete = pd.read_csv(ROOT_PATH + "/exp_data/data_files/experimental_time_series_rounded_times.csv")
sigma_all_exp= {}

for init_mass_conc_gly in [50,60,70,80]:
    rows_bool = time_series_df_cleaned_complete.loc[:,"Glycerol Init"] == init_mass_conc_gly
    time_samples = time_series_df_cleaned_complete.loc[rows_bool,"Time"]
    sigma_data_gly_pdo_dcw = []
    for reactant in ["Glycerol", "PDO", "DCW"]:
        sigma_sq_data = []
        data_list = []
        data_samples = time_series_df_cleaned_complete.loc[:, reactant][rows_bool]

        if reactant == "Glycerol" or reactant == "PDO":
            sigma_sq_val = 2
            mass_1 = 5
            mass_2 = 1e-2
        if reactant == "DCW":
            sigma_sq_val = 0.1**2
            mass_1 = 0.35
            mass_2 = 1e-2

        for i in data_samples.index:
            if data_samples.loc[i] > mass_1:
                sigma_sq = sigma_sq_val
            elif data_samples.loc[i] > mass_2:
                sigma_sq = (data_samples[i] / 3.5) ** 2
            else:
                sigma_sq = 1e-9
            sigma_sq_data.append(sigma_sq)
        # complete exp_data
        plt.errorbar(time_samples, data_samples, yerr=2*np.sqrt(np.array(sigma_sq_data)), fmt='none')
        plt.pause(5)
        plt.close()

        if reactant == "Glycerol":
            sigma_sq_data_millimolar = np.array(sigma_sq_data) * (moles_per_liter_2_millimolar / MW_GLYCEROL) ** 2
        elif reactant == "PDO":
            sigma_sq_data_millimolar = np.array(sigma_sq_data) * (moles_per_liter_2_millimolar / MW_13PDO) ** 2
        else:
            sigma_sq_data_millimolar = np.array(sigma_sq_data)

        sigma_data_gly_pdo_dcw.append(np.sqrt(sigma_sq_data_millimolar).tolist())
    df = pd.DataFrame(sigma_data_gly_pdo_dcw).transpose()
    df.columns=["Glycerol", "PDO", "DCW"]
    sigma_all_exp[init_mass_conc_gly] = df.to_numpy()

with open('data_files/std_experimental_data.pkl', 'wb') as f:
    pickle.dump(sigma_all_exp, f, pickle.HIGHEST_PROTOCOL)