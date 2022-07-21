import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from os.path import dirname, abspath
import pandas as pd
from exp_data_13pd import TIME_SAMPLES, TIME_SERIES_MEAN
ROOT_PATH = dirname(dirname(abspath(__file__)))

HRS_TO_SECS = 60*60
def sigmoid_trans(t, L, k, A):
    t0 = np.log(L/A - 1)/k
    y =  L / (1 + np.exp(-k*(t-t0)))
    return y

def sigmoid(t, L , k, t0):
    #t0 is the time at which it is L/2 -- controls location of inflection point
    #L maximum concnetration
    y = L / (1 + np.exp(-k*(t-t0)))
    return y

OD_fit_params = {}
for exp_cond in ['WT-L','dD-L','dAJ-L','dP-L']:

    #get exp_data and fit splines
    time_pts = TIME_SAMPLES
    #time_pts_fit = np.concatenate((time_pts, np.linspace(time_pts[-1] + 0.5,time_pts[-1] + 10,20)))
    OD_pts = TIME_SERIES_MEAN[exp_cond][['OD600']].T.to_numpy()[0]
#   DCW_log_pts_fit = np.concatenate((DCW_log_pts, [1.01*DCW_log_pts[-1]]*20))
#
    p0 = [max(OD_pts), 1, np.median(time_pts)]  # this is an mandatory initial guess
    popt, pcov = curve_fit(sigmoid, time_pts, OD_pts,p0, method='dogbox')
    popt_ = popt.copy()
    popt_[1] = popt[1]/HRS_TO_SECS
    popt_[-1] = popt[0]/(1+np.exp(popt[1]*popt[2]))
    OD_fit_params[exp_cond] = np.log10(popt_)

    # 10^ transformed plots
    sigmoid_t = lambda t: sigmoid_trans(t, *popt_)
    # plots
    t = np.linspace(0, TIME_SAMPLES[-1] + 3, num=int(10**3))
    plt.scatter(TIME_SAMPLES,  OD_pts, label='exp_data')
    plt.plot(t, sigmoid_t(t*HRS_TO_SECS), label='sigmoid')
    plt.legend(loc='upper right',fontsize=10)
    plt.title("Plot of sigmoid fit to OD data for \n condition, " + exp_cond, fontsize=20)
    plt.ylabel("OD",fontsize=20)
    plt.xlabel("time (hrs)",fontsize=20)
    plt.xticks(fontsize=20)
    plt.savefig('OD_plots/OD_sigmoid_plot_'+exp_cond.replace('-','_'), bbox_inches="tight")
    plt.close()
#
OD_fit_params_df = pd.DataFrame.from_dict(OD_fit_params,orient='index',columns=['L','k','A'])
#
OD_fit_params_df.to_csv(ROOT_PATH + '/exp_data_13pd/exp_data_13pd/data_files/OD_mean_params_trans', index_label='Exp Condition')
#
