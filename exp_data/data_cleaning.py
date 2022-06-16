from exp_data.cleaning_constants import *
from os.path import dirname, abspath
import pandas as pd
import matplotlib.pyplot as plt

ROOT_PATH = dirname(dirname(abspath(__file__)))

time_series_df = pd.read_csv(ROOT_PATH + "/exp_data/data_files/experimental_time_series.csv")
time_series_df_cleaned = pd.read_csv(ROOT_PATH + "/exp_data/data_files/experimental_time_series_rounded_times_subset.csv")
time_series_df_cleaned_complete = pd.read_csv(ROOT_PATH + "/exp_data/data_files/experimental_time_series_rounded_times.csv")

# convert Glycerol, g/L, to mM
time_series_df.loc[:,"Glycerol"] = time_series_df.loc[:,"Glycerol"]*moles_per_liter_2_millimolar/MW_GLYCEROL
time_series_df_cleaned.loc[:,"Glycerol"] = time_series_df_cleaned.loc[:,"Glycerol"]*moles_per_liter_2_millimolar/MW_GLYCEROL
time_series_df_cleaned_complete.loc[:, "Glycerol"] = time_series_df_cleaned_complete.loc[:, "Glycerol"] * moles_per_liter_2_millimolar / MW_GLYCEROL

# convert 1,3-PDO, g/L, to mM
time_series_df.loc[:,"PDO"] = time_series_df.loc[:,"PDO"]*moles_per_liter_2_millimolar/MW_13PDO
time_series_df_cleaned.loc[:,"PDO"] = time_series_df_cleaned.loc[:,"PDO"]*moles_per_liter_2_millimolar/MW_13PDO
time_series_df_cleaned_complete.loc[:, "PDO"] = time_series_df_cleaned_complete.loc[:, "PDO"] * moles_per_liter_2_millimolar / MW_13PDO

#change column names
rename_dict = {"Glycerol Init": "Glycerol Init (g/L)", "Time": "Time (hrs)", "Glycerol": "Glycerol (mM)", "PDO":"PDO (mM)", "DCW":"DCW (g/L)"}
time_series_df.rename(columns=rename_dict, errors="raise",inplace=True)
time_series_df_cleaned.rename(columns=rename_dict,inplace=True)
time_series_df_cleaned_complete.rename(columns=rename_dict, inplace=True)

time_series_df.to_csv('data_files/experimental_time_series_cleaned.csv',index=False)
time_series_df_cleaned.to_csv('data_files/experimental_time_series_rounded_times_subset_cleaned.csv',index=False)
time_series_df_cleaned_complete.to_csv('data_files/experimental_time_series_rounded_times_cleaned.csv', index=False)

for gly_cond in [50,60,70,80]:
    bools = time_series_df["Glycerol Init (g/L)"] == gly_cond
    plt.scatter(time_series_df.loc[bools, "Time (hrs)"], time_series_df.loc[bools, "Glycerol (mM)"])
    plt.ylabel("Glycerol (mM)")
    plt.xlabel("Time (hrs)")
    plt.title("Glycerol concentration given the initial condition, " + str(gly_cond) + " g/L")
    # plt.savefig('/home/aarcher/Downloads/Glycerol_mM_'+str(gly_cond))
    # plt.close()
    plt.show()

    plt.scatter(time_series_df.loc[bools, "Time (hrs)"], time_series_df.loc[bools, "PDO (mM)"])
    plt.ylabel("1,3-PDO (mM)")
    plt.xlabel("Time (hrs)")
    plt.title("1,3-PDO concentration given the initial condition, " + str(gly_cond) + " g/L")
    # plt.savefig('/home/aarcher/Downloads/PDO_mM_'+str(gly_cond))
    # plt.close()
    plt.show()
