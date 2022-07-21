import numpy as np
import matplotlib.pyplot as plt
from exp_data_13pd import TIME_SAMPLES, TIME_SERIES_MEAN
from os.path import dirname, abspath
import pandas as pd
ROOT_PATH = dirname(dirname(abspath(__file__)))
HRS_TO_SECS = 60 * 60
OD_fit_params_df= pd.read_csv(ROOT_PATH + '/exp_data_13pd/exp_data_13pd/data_files/OD_mean_params_trans', index_col=0)

def sigmoid(t, L, k, A):
    t0 = np.log(L/A - 1)/k
    y =  L / (1 + np.exp(-k*(t-t0)))
    return y

std_vec_dict = {'WT-L': [0.0075,0.005/HRS_TO_SECS,0.2],
                'dD-L': [0.01,0.005/HRS_TO_SECS,0.2],
                'dAJ-L': [0.01,0.025/HRS_TO_SECS,0.2],
                'dP-L': [0.01,0.025/HRS_TO_SECS,0.2]}

for exp_cond in  ['WT-L','dD-L','dAJ-L','dP-L']:
    OD_pts = TIME_SERIES_MEAN[exp_cond][['OD600']].T.to_numpy()[0]
    std_vec = std_vec_dict[exp_cond]
    # 10^ transformed plots
    sigmoid_fits = []
    while(len(sigmoid_fits) < 1000):
        popt = OD_fit_params_df.loc[exp_cond,:].values + np.array([np.random.normal(scale=std,size = 1)[0] for std in std_vec])
        sigmoid_t = lambda t: sigmoid(t, *np.power(10,popt))

        # plots
        t = np.linspace(0, TIME_SAMPLES[-1] + 3, num=int(10**3))
        sigmoid_fits.append(sigmoid_t(t*HRS_TO_SECS))

    sigmoid_fits = np.array(sigmoid_fits)
    sigmoid_fits_25 = np.quantile(sigmoid_fits,0.025, axis=0)
    sigmoid_fits_mean = np.mean(sigmoid_fits, axis=0)
    sigmoid_fits_975 = np.quantile(sigmoid_fits,0.975, axis=0)

    plt.scatter(TIME_SAMPLES,  OD_pts, label='exp. data')
    plt.plot(t, sigmoid_fits_25, 'y--')#, label='$2.5\%$')
    plt.plot(t, sigmoid_fits_mean, 'b', label='mean')
    plt.plot(t, sigmoid_fits_975, 'y--')#, label='$97.5\%$')

    plt.legend(loc='lower right',fontsize=10)
    plt.title("Plot of sigmoid fit to OD data for \n condition, " + exp_cond, fontsize=20)
    plt.ylabel("OD",fontsize=20)
    plt.xlabel("time (hrs)",fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()
    # plt.savefig('OD_plots/OD_sigmoid_trans_prior_uncertainty_plot_'+exp_cond, bbox_inches="tight")
    # plt.close()

std_vec_dict_df = pd.DataFrame.from_dict(std_vec_dict,orient='index',columns=['std L', 'std k', 'std A'])
std_vec_dict_df.to_csv(ROOT_PATH + '/exp_data_13pd/exp_data_13pd/data_files/OD_std_params_trans', index_label='Glycerol Init')

