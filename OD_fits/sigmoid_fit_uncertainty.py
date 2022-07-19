import numpy as np
import matplotlib.pyplot as plt
from exp_data_13pd import TIME_SAMPLES, TIME_SERIES_MEAN
from os.path import dirname, abspath
import pandas as pd
ROOT_PATH = dirname(dirname(abspath(__file__)))

OD_fit_params_df= pd.read_csv(ROOT_PATH + '/exp_data_13pd/exp_data_13pd/data_files/OD_mean_params_trans', index_col=0)

def sigmoid(t, L, k, A):
    t0 = np.log(L/A - 1)/k
    y =  L / (1 + np.exp(-k*(t-t0)))
    return y

std_vec_dict = {'WT-L': [0.075,0.005,0.015],
                'dD-L': [0.1,0.005,0.015],
                'dAJ-L': [0.1,0.025,0.02],
                'dP-L': [0.1,0.025,0.02]}

for exp_cond in  ['WT-L','dD-L','dAJ-L','dP-L']:
    OD_pts = TIME_SERIES_MEAN[exp_cond][['OD600']].T.to_numpy()[0]
    std_vec = std_vec_dict[exp_cond]
    # 10^ transformed plots
    sigmoid_fits = []
    while(len(sigmoid_fits) < 1000):
        popt = OD_fit_params_df.loc[exp_cond,:].values + np.array([np.random.normal(scale=std,size = 1)[0] for std in std_vec])
        sigmoid_t = lambda t: sigmoid(t, *popt)

        # plots
        t = np.linspace(0, TIME_SAMPLES[-1] + 3, num=int(10**3))
        sigmoid_fits.append(sigmoid_t(t))

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
    # plt.savefig('OD_plots/OD_sigmoid_trans_prior_uncertainty_plot_'+str(init_mass_conc_gly), bbox_inches="tight")
    # plt.close()

std_vec_dict_df = pd.DataFrame.from_dict(std_vec_dict,orient='index',columns=['std L', 'std k', 'std A'])
std_vec_dict_df.to_csv(ROOT_PATH + '/exp_data_13pd/exp_data_13pd/data_files/OD_std_params_trans', index_label='Glycerol Init')

