import matplotlib as mpl
import aesara
import aesara.tensor as at
import arviz as az
import matplotlib.pyplot as plt
import os
import seaborn as sns
from exp_data import *
import pymc as pm
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
from prior_constants import NORM_PRIOR_STD_RT_SINGLE_EXP,NORM_PRIOR_MEAN_SINGLE_EXP, NORM_PRIOR_STD_RT_ALL_EXP, \
    NORM_PRIOR_MEAN_ALL_EXP, LOG_UNIF_PRIOR_ALL_EXP, DATA_LOG_UNIF_PARAMETER_RANGES, NORM_PRIOR_PARAMETER_ALL_EXP_DICT, \
    LOG_UNIF_G_EXT_INIT_PRIOR_PARAMETERS
from constants import PERMEABILITY_PARAMETERS, KINETIC_PARAMETERS, ENZYME_CONCENTRATIONS, \
    GLYCEROL_EXTERNAL_EXPERIMENTAL, ALL_PARAMETERS, PARAMETER_LIST, TIME_SAMPLES_EXPANDED, VARIABLE_NAMES, HRS_TO_SECS, \
    TIME_SPACING, DATA_INDEX, N_MODEL_PARAMETERS
import numpy as np
from os.path import dirname, abspath
from rhs_funcs import RHS, lib, problem, solver
from formatting_constants import VARS_ALL_EXP_TO_TEX
ROOT_PATH = dirname(abspath(__file__))
from pandas.plotting import scatter_matrix


def plot_loglik_individual(loglik, plot_file_location, nchains):
    fig, ax = plt.subplots(2,min(5,nchains))
    for i in range(min(5,nchains)):
        ax[i,0].hist(loglik[i],alpha=0.5)
        ax[i, 0].set_title('Histogram of Log-Likelihood')
        ax[i, 1].plot(list(range(len(loglik[i]))),loglik[i],alpha=0.5)
        ax[i, 1].set_title('Trajectory of Log-Likelihood')
    fig.tight_layout()
    plt.savefig(os.path.join(plot_file_location, 'loglik_plot_individual.png'))
#

def plot_loglik_overlay(loglik, plot_file_location, nchains):
    fig, ax = plt.subplots(1,min(5,nchains))
    for i in range(nchains):
        ax[0].hist(loglik[i],alpha=0.1)
        ax[0].set_title('Histogram of Log-Likelihood')
        ax[1].plot(list(range(len(loglik[i]))), loglik[i],alpha=0.1)
        ax[1].set_title('Trajectory of Log-Likelihood')

    ax[0].legend(['chain ' + str(i) for i in range(nchains)])
    ax[1].legend(['chain ' + str(i) for i in range(nchains)])
    plt.savefig(os.path.join(plot_file_location, 'loglik_plot_overlay.png'))

def plot_time_series_distribution(samples, plot_file_location, nchains, atol, rtol, mxsteps):
    lib.CVodeSStolerances(solver._ode, atol, rtol)
    lib.CVodeSetMaxNumSteps(solver._ode, int(mxsteps))
    for exp_ind, gly_cond in enumerate([50, 60, 70, 80]):
        fig, ax = plt.subplots(5, min(5, nchains))
        for chain_ind in range(nchains):
            dataarray = samples.posterior.to_dataframe().loc[[chain_ind]]
            dataarray = dataarray[ALL_PARAMETERS]
            lower, upper = LOG_UNIF_G_EXT_INIT_PRIOR_PARAMETERS["G_EXT_INIT_" + str(gly_cond)]
            for jj in range(3):
                ax[jj,chain_ind].scatter(TIME_SAMPLES_EXPANDED[gly_cond][::TIME_SPACING], DATA_SAMPLES[gly_cond][:, jj])

            hpa_max = []
            for j in range(0,dataarray.shape[0]):
                param = dataarray.iloc[j,:].to_numpy()
                param_sample = NORM_PRIOR_MEAN_SINGLE_EXP[gly_cond]
                param_sample[:N_MODEL_PARAMETERS] = param[:N_MODEL_PARAMETERS]
                # g_ext_val = param[N_MODEL_PARAMETERS + exp_ind]
                # g_ext_val = lower + (upper - lower) / (1 + np.exp(-g_ext_val))
                # param_sample[N_MODEL_PARAMETERS] = g_ext_val

                tvals = TIME_SAMPLES_EXPANDED[gly_cond] * HRS_TO_SECS

                y0 = np.zeros((), dtype=problem.state_dtype)

                y0['G_CYTO'] = 10 ** param_sample[PARAMETER_LIST.index('G_EXT_INIT')]
                y0['H_CYTO'] = 0
                y0['P_CYTO'] = INIT_CONDS_GLY_PDO_DCW[gly_cond][1]
                y0['DHAB'] = 10 ** param_sample[PARAMETER_LIST.index('DHAB_INIT')]
                y0['DHAB_C'] = 0
                y0['DHAT'] = 10 ** param_sample[PARAMETER_LIST.index('DHAT_INIT')]
                y0['DHAT_C'] = 0
                y0['G_EXT'] = 10 ** param_sample[PARAMETER_LIST.index('G_EXT_INIT')]
                y0['H_EXT'] = 0
                y0['P_EXT'] = INIT_CONDS_GLY_PDO_DCW[gly_cond][1]
                y0['dcw'] = 10 ** param_sample[PARAMETER_LIST.index('A')]

                params_dict = {param_name: param_val for param_val, param_name in zip(param_sample, PARAMETER_LIST)}
                # # We can also specify the parameters by name:
                solver.set_params_dict(params_dict)
                yout, grad_out, lambda_out = solver.make_output_buffers(tvals)

                solver.solve_forward(t0=0, tvals=tvals, y0=y0, y_out=yout)
                jj=0
                for i, var in enumerate(VARIABLE_NAMES):
                    if i in DATA_INDEX:
                        ax[jj,chain_ind].plot(tvals / HRS_TO_SECS, yout.view(problem.state_dtype)[var], 'r', alpha=0.1)
                        jj += 1
                    elif var == 'H_CYTO':
                        ax[3,chain_ind].plot(tvals / HRS_TO_SECS, yout.view(problem.state_dtype)[var], 'r', alpha=0.1)
                        hpa_max.append(np.max(yout.view(problem.state_dtype)[var]))
            ax[4, chain_ind].hist(hpa_max)
            ax[0, chain_ind].set_title('Glycerol Time Series')
            ax[1, chain_ind].set_title('1,3-PD Distribution')
            ax[2, chain_ind].set_title('DCW Distribution')
            ax[3, chain_ind].set_title('Cytosolic 3-HPA Time Series')
            ax[4, chain_ind].set_title('Max 3-HPA Distribution')
        plt.suptitle('Initial Glycerol ' + str(gly_cond) + ' g/L')
        fig.tight_layout()
        plt.savefig(os.path.join(plot_file_location, 'time_series_results_' + str(gly_cond) + '.png'))

def plot_corr(data, directory_plot, nchains, thres=5e-2):
    for chain_ind in range(nchains):
        dataarray = data.posterior.to_dataframe().loc[[chain_ind]]
        dataarray = dataarray[ALL_PARAMETERS]

        fig, ax = plt.subplots()
        data_corr = np.corrcoef(dataarray.to_numpy().T)
        # matrix = data_corr

        for i in range(data_corr.shape[0]):
            for j in range(data_corr.shape[1]):
                if np.abs(data_corr[i,j]) < thres:
                    data_corr[i,j] = np.nan


        # using diag as mask
        mask_mat = np.ones_like(data_corr)
        mask_mat = np.triu(mask_mat,k=0)
        ax = sns.heatmap(data_corr, mask = mask_mat, annot=True, cmap="YlGnBu", vmin=-1, vmax=1, annot_kws={"size":15},fmt='.1g')
        xticks = [(i + 0.5) for i in range(len(ALL_PARAMETERS))]
        yticks = [(i + 0.5) for i in range(len(ALL_PARAMETERS))]
        plt.xticks(xticks, [VARS_ALL_EXP_TO_TEX[key] for key in ALL_PARAMETERS], fontsize=15,
                   rotation = -25)
        plt.yticks(yticks, [VARS_ALL_EXP_TO_TEX[key] for key in ALL_PARAMETERS], fontsize=15,
                   rotation = 45, ha="right")
        plt.title('Correlation Matrix of Posterior Samples', fontsize=20)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=20)
        fig.set_size_inches(15, 9.5, forward=True)
        plt.savefig(directory_plot + '/correlation_plot_chain_' + str(chain_ind), bbox_inches="tight")
        plt.close()

def plot_corr_scatter(data, directory_plot, nchains):
    for chain_ind in range(nchains):
        diverging = data.sample_stats.diverging[chain_ind].values
        dataarray = data.posterior.to_dataframe().loc[[chain_ind]]
        dataarray = dataarray[ALL_PARAMETERS]
        # data_array = pd.DataFrame(np.array([data[:, i] for i in range(len(ALL_PARAMETERS))]).T, columns=ALL_PARAMETERS)
        axes = scatter_matrix(dataarray.loc[np.invert(diverging),:], alpha=0.2, figsize=(len(ALL_PARAMETERS), len(ALL_PARAMETERS)),
                              diagonal='kde')
        for i in range(np.shape(axes)[0]):
            for j in range(np.shape(axes)[1]):
                if i < j:
                    axes[i, j].set_visible(False)
                xlab_curr = axes[i, j].get_xlabel()
                ylab_curr = axes[i, j].get_ylabel()
                if i > j:
                    axes[i, j].scatter(dataarray.loc[diverging,ALL_PARAMETERS[j]],
                                       dataarray.loc[diverging,ALL_PARAMETERS[i]], s=1,alpha=0.5)
                axes[i, j].set_xlabel(VARS_ALL_EXP_TO_TEX[xlab_curr])
                axes[i, j].set_ylabel(VARS_ALL_EXP_TO_TEX[ylab_curr])
                axes[i, j].xaxis.label.set_rotation(-25)
                axes[i, j].xaxis.label.set_fontsize(15)
                axes[i, j].yaxis.label.set_rotation(45)
                axes[i, j].yaxis.label.set_ha('right')
                axes[i, j].yaxis.label.set_fontsize(15)
                axes[i, j].tick_params(axis="x", labelsize=15, rotation=0)
                axes[i, j].tick_params(axis="y", labelsize=15)

        plt.suptitle('Scatter Plot Matrix of Posterior Samples', fontsize=20)
        plt.savefig(directory_plot + '/correlation_scatter_plot_chain_' + str(chain_ind), bbox_inches="tight")
        plt.close()

def scatter_hist(x, y, ax, ax_histx, ax_histy):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y, s=0.5)
    # now determine nice limits by hand:
    binwidth = 0.5
    xymax = max(np.max(x), np.max(y))
    xymin = min(np.min(x), np.min(y))

    lim_xymax = (int(xymax/binwidth) + 1) * binwidth
    lim_xymin = (int(xymin/binwidth) - 1) * binwidth

    bins = np.arange(lim_xymin, lim_xymax + binwidth, binwidth)
    ax_histx.hist(x, bins=bins, density=True)
    ax_histy.hist(y, bins=bins, orientation='horizontal', density=True)

def joint_Keq_distribution(KeqDhaB_chains,KeqDhaT_chains, plot_location, nchains):

    xlab = r'$\log_{10}(K_{\text{eq}}^{\text{DhaT}})$'
    ylab = r'$\log_{10}(K_{\text{eq}}^{\text{DhaB}})$'
    # # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]
    for chain_ind in range(nchains):
        KeqDhaB = KeqDhaB_chains[chain_ind]
        KeqDhaT = KeqDhaT_chains[chain_ind]

        # start with a square Figure
        fig = plt.figure(figsize=(8, 8))

        ax = fig.add_axes(rect_scatter)
        dhaT_fill_between = np.arange(2, 5.1, 0.1)
        dhaB_fill_between = np.arange(7, 8.1, 0.1)
        ax.fill_between(dhaT_fill_between, dhaB_fill_between[0], dhaB_fill_between[-1], facecolor='yellow', alpha=0.5,
                        label ="Thermodynamically\n Feasible")
        ax.set_xlabel(xlab, fontsize=10)
        ax.set_ylabel(ylab, fontsize=10)
        sns.kdeplot(x=np.log10(KeqDhaB),y=np.log10(KeqDhaT), fill=True,
                    alpha=0.5, color='blue', ax=ax)
        ax.set_xlim([min([min(np.log10(KeqDhaT)), min(dhaT_fill_between)]) - 1,
                     max([max(np.log10(KeqDhaT)), max(dhaT_fill_between)]) + 1])
        ax.set_ylim([min([min(np.log10(KeqDhaB)), min(dhaB_fill_between)]) - 1,
                     max([max(np.log10(KeqDhaB)), max(dhaB_fill_between)]) + 1])


        ax_histx = fig.add_axes(rect_histx, sharex=ax)
        ax_histx.set_ylabel('Probability', fontsize=15)
        ax_histy = fig.add_axes(rect_histy, sharey=ax)
        ax_histy.set_xlabel('Probability', fontsize=15)
        #
        # # use the previously defined function
        scatter_hist(np.log10(KeqDhaT), np.log10(KeqDhaB), ax, ax_histx, ax_histy)
        probdhaT = ax_histx.get_ylim()
        probdhaB = ax_histy.get_xlim()
        y_fill_between = np.arange(probdhaB[0], probdhaB[1], 0.05)
        ax_histx.fill_between(dhaT_fill_between, probdhaT[0], probdhaT[1],
                              facecolor='yellow', alpha=0.5)
        ax_histy.fill_between(y_fill_between, dhaB_fill_between[0], dhaB_fill_between[-1],
                               facecolor='yellow', alpha=0.5)

        ax.legend(fontsize=15)
        ax.tick_params(axis="x", labelsize=15)
        ax.tick_params(axis="y", labelsize=15, rotation=0)
        ax_histx.tick_params(axis="x", labelsize=15)
        ax_histx.tick_params(axis="y", labelsize=15, rotation=0)
        ax_histy.tick_params(axis="x", labelsize=15)
        ax_histy.tick_params(axis="y", labelsize=15, rotation=0)
        ax_histx.set_title('Joint Distribution of the reaction\n equilibrium constants', fontsize=15)
        plt.savefig(plot_location + '/K_Eq_Distribution_' + str(chain_ind), bbox_inches="tight")
        plt.close()