import matplotlib as mpl
import aesara
import aesara.tensor as at
import arviz as az
import matplotlib.pyplot as plt
import os
import seaborn as sns

import sunode
from exp_data import *
import pymc as pm
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
from prior_constants import *
from constants import *
import numpy as np
from exp_data_13pd import *
from os.path import dirname, abspath
from rhs_WT import RHS_WT, problem_WT
from rhs_delta_AJ import RHS_delta_AJ, problem_delta_AJ
ROOT_PATH = dirname(abspath(__file__))
from pandas.plotting import scatter_matrix
solver_delta_AJ = sunode.solver.AdjointSolver(problem_delta_AJ, solver='BDF')
solver_WT = sunode.solver.AdjointSolver(problem_WT, solver='BDF')

def plot_trace(samples, plot_file_location):
    n_display = 10
    for i in range(int(np.ceil(len(ALL_PARAMETERS)/n_display))):
        az.plot_trace(samples, var_names=ALL_PARAMETERS[(n_display*i):(n_display*(i+1))], compact=True)
        plt.savefig(os.path.join(plot_file_location,"trace_plot_" + str(i) + ".jpg"))
        plt.close()

def plot_loglik_individual(loglik, plot_file_location, nchains):
    fig, ax = plt.subplots(nchains,2)
    for i in range(nchains):
        ax[i,0].hist(loglik[i],alpha=0.5)
        ax[i,0].set_title('Histogram of Log-Likelihood')
        ax[i,1].plot(list(range(len(loglik[i]))),loglik[i],alpha=0.5)
        ax[i,1].set_title('Trajectory of Log-Likelihood')
    fig.tight_layout()
    plt.savefig(os.path.join(plot_file_location, 'loglik_plot_individual.png'))


def plot_loglik_overlay(loglik, plot_file_location, nchains):
    fig, ax = plt.subplots(1,2)
    for i in range(nchains):
        ax[0].hist(loglik[i],alpha=0.1)
        ax[0].set_title('Histogram of Log-Likelihood')
        ax[1].plot(list(range(len(loglik[i]))), loglik[i],alpha=0.1)
        ax[1].set_title('Trajectory of Log-Likelihood')

    ax[0].legend(['chain ' + str(i) for i in range(nchains)])
    ax[1].legend(['chain ' + str(i) for i in range(nchains)])
    plt.savefig(os.path.join(plot_file_location, 'loglik_plot_overlay.png'))

def plot_time_series_distribution(samples, plot_file_location, nchains, fwd_atol, fwd_rtol, mxsteps):
    lib.CVodeSStolerances(solver_delta_AJ._ode, fwd_rtol, fwd_atol)
    lib.CVodeSetMaxNumSteps(solver_delta_AJ._ode, mxsteps)
    lib.CVodeSStolerances(solver_WT._ode, fwd_rtol, fwd_atol)
    lib.CVodeSetMaxNumSteps(solver_WT._ode, mxsteps)
    c = ['r', 'y', 'b', 'g', 'k']
    legend_names = ['chain ' + str(i)  for i in range(min(5, nchains))]

    for exp_ind, exp_cond in enumerate(['WT-L', 'dAJ-L', 'dD-L', 'dP-L']):

        # set initial condition depending on each experiment
        if exp_cond in ['WT-L', 'dD-L', 'dP-L']:
            solver = solver_WT
            problem = problem_WT
        elif exp_cond == 'dAJ-L':
            solver = solver_delta_AJ
            problem = problem_delta_AJ

        # set initial conditions
        y0 = np.zeros((), dtype=problem.state_dtype)
        for var in VARIABLE_NAMES:
            y0[var] = 0
        y0['G_EXT'] = TIME_SERIES_MEAN[exp_cond]['glycerol'][0]
        y0['G_CYTO'] = TIME_SERIES_MEAN[exp_cond]['glycerol'][0]
        y0['G_MCP'] = TIME_SERIES_MEAN[exp_cond]['glycerol'][0]
        y0['H_EXT'] = TIME_SERIES_MEAN[exp_cond]['3-HPA'][0]
        y0['H_CYTO'] = TIME_SERIES_MEAN[exp_cond]['3-HPA'][0]
        y0['H_MCP'] = TIME_SERIES_MEAN[exp_cond]['3-HPA'][0]
        y0['P_EXT'] = TIME_SERIES_MEAN[exp_cond]['13PD'][0]
        y0['P_CYTO'] = TIME_SERIES_MEAN[exp_cond]['13PD'][0]
        y0['P_MCP'] = TIME_SERIES_MEAN[exp_cond]['13PD'][0]

        fig, ax = plt.subplots(4, min(5, nchains), figsize=(15,15)) #min(5, nchains))
        for chain_ind in range(min(5, nchains)):
            dataarray = samples.posterior.to_dataframe().loc[[chain_ind]]
            dataarray = dataarray[DEV_PARAMETER_LIST]
            for jj in range(4):
                ax[jj, chain_ind].scatter(TIME_SAMPLES, TIME_SERIES_MEAN[exp_cond].iloc[:,jj])

            for j in range(0,dataarray.shape[0],int(dataarray.shape[0]/500)):
                params = dataarray.iloc[j,:].to_numpy()
                non_enz_model_params = params[:-8]
                enz_params_WT = params[-8:-4]
                enz_params_dAJ = params[-4:]

                # alter initial conditions
                if exp_cond in ['WT-L', 'dD-L', 'dP-L']:
                    param_samples_copy = np.concatenate((non_enz_model_params,enz_params_WT,list(OD_PRIOR_PARAMETER_MEAN[exp_cond].values())))
                    y0['NADH_MCP'] = (10**(param_samples_copy[PARAMETER_LIST.index('NADH_NAD_TOTAL_MCP')]
                                           + param_samples_copy[PARAMETER_LIST.index('NADH_NAD_RATIO_MCP')]))/(10**param_samples_copy[PARAMETER_LIST.index('NADH_NAD_RATIO_MCP')] + 1)
                    y0['NAD_MCP'] = 10**param_samples_copy[PARAMETER_LIST.index('NADH_NAD_TOTAL_MCP')]/(10**param_samples_copy[PARAMETER_LIST.index('NADH_NAD_RATIO_MCP')] + 1)
                    y0['PduCDE'] = 10**param_samples_copy[PARAMETER_LIST.index('nPduCDE')]/(Avogadro * MCP_VOLUME)
                    y0['PduP'] = 10**param_samples_copy[PARAMETER_LIST.index('nPduP')]/(Avogadro * MCP_VOLUME)
                    y0['PduQ'] = 10**param_samples_copy[PARAMETER_LIST.index('nPduQ')]/(Avogadro * MCP_VOLUME)
                    y0['PduL'] = 10**param_samples_copy[PARAMETER_LIST.index('nPduL')]/(Avogadro * MCP_VOLUME)
                    y0['OD'] = 10**param_samples_copy[PARAMETER_LIST.index('A')]
                elif exp_cond == 'dAJ-L':
                    param_samples_copy = np.concatenate((non_enz_model_params,
                                                         enz_params_dAJ,
                                                         list(OD_PRIOR_PARAMETER_MEAN[exp_cond].values())))
                    POLAR_VOLUME = (4./3.)*np.pi*((10**param_samples_copy[PARAMETER_LIST.index('AJ_radius')])**3)
                    y0['NADH_MCP'] = (10**(param_samples_copy[PARAMETER_LIST.index('NADH_NAD_TOTAL_CYTO')]
                                           + param_samples_copy[PARAMETER_LIST.index('NADH_NAD_RATIO_CYTO')]))/(10**param_samples_copy[PARAMETER_LIST.index('NADH_NAD_RATIO_CYTO')] + 1)
                    y0['NAD_MCP'] = 10**param_samples_copy[PARAMETER_LIST.index('NADH_NAD_TOTAL_CYTO')]/(10**param_samples_copy[PARAMETER_LIST.index('NADH_NAD_RATIO_CYTO')] + 1)
                    y0['PduCDE'] = param_samples_copy[PARAMETER_LIST.index('nMCPs')]*(10**param_samples_copy[PARAMETER_LIST.index('nPduCDE')])/(Avogadro * POLAR_VOLUME)
                    y0['PduP'] = param_samples_copy[PARAMETER_LIST.index('nMCPs')]*(10**param_samples_copy[PARAMETER_LIST.index('nPduP')])/(Avogadro * POLAR_VOLUME)
                    y0['PduQ'] = param_samples_copy[PARAMETER_LIST.index('nMCPs')]*(10**param_samples_copy[PARAMETER_LIST.index('nPduQ')])/(Avogadro * POLAR_VOLUME)
                    y0['PduL'] = param_samples_copy[PARAMETER_LIST.index('nMCPs')]*(10**param_samples_copy[PARAMETER_LIST.index('nPduL')])/(Avogadro * POLAR_VOLUME)
                    y0['OD'] = 10**param_samples_copy[PARAMETER_LIST.index('A')]

                params_dict = { param_name : param_val for param_val,param_name in zip(param_samples_copy, PARAMETER_LIST)}
                if exp_cond == 'dD-L':
                    y0['PduCDE'] = 0
                    params_dict['nPduCDE'] = 0
                elif exp_cond == 'dP-L':
                    y0['PduP'] = 0
                    params_dict['nPduP'] = 0

                # set solver parameters
                solver.set_params_dict(params_dict)
                tvals = TIME_SAMPLES*HRS_TO_SECS
                yout, _, _ = solver.make_output_buffers(tvals)

                try:
                    solver.solve_forward(t0=0, tvals=tvals, y0=y0, y_out=yout)
                    jj=0
                    for i, var in enumerate(VARIABLE_NAMES):
                        if i in DATA_INDEX:
                            ax[jj, chain_ind].plot(tvals / HRS_TO_SECS, yout.view(problem.state_dtype)[var], 'r',
                                                  alpha=0.05)
                            jj += 1
                except sunode.solver.SolverError:
                    pass
            ax[0, chain_ind].set_title('Glycerol Time Series')
            ax[1, chain_ind].set_title('3-HPA Time Series')
            ax[2, chain_ind].set_title('1,3-PD Time Series')
            ax[3, chain_ind].set_title('DCW Distribution')

        plt.suptitle('Experimental Condition ' + str(exp_cond))
        fig.tight_layout()
        plt.savefig(os.path.join(plot_file_location, 'time_series_results_' + str(exp_cond) + '.png'))

def plot_corr(data, directory_plot, nchains, thres=5e-2):
    for chain_ind in range(nchains):
        dataarray = data.posterior.to_dataframe().loc[[chain_ind]]
        dataarray = dataarray[PLOT_PARAMETERS]

        fig, ax = plt.subplots()
        data_corr = np.corrcoef(dataarray[PLOT_PARAMETERS].to_numpy().T)
        # matrix = data_corr

        for i in range(data_corr.shape[0]):
            for j in range(data_corr.shape[1]):
                if np.abs(data_corr[i,j]) < thres:
                    data_corr[i,j] = np.nan

        # using diag as mask
        mask_mat = np.ones_like(data_corr)
        mask_mat = np.triu(mask_mat,k=0)
        ax = sns.heatmap(data_corr, mask = mask_mat, annot=True, cmap="YlGnBu", vmin=-1, vmax=1, annot_kws={"size":15},fmt='.1g')
        xticks = [(i + 0.5) for i in range(len(PLOT_PARAMETERS))]
        yticks = [(i + 0.5) for i in range(len(PLOT_PARAMETERS))]
        plt.xticks(xticks, [MODEL_PARAMS_TO_TEX[key] for key in PLOT_PARAMETERS], fontsize=15,
                   rotation = -25)
        plt.yticks(yticks, [MODEL_PARAMS_TO_TEX[key] for key in PLOT_PARAMETERS], fontsize=15,
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
        dataarray = dataarray[PLOT_PARAMETERS]
        try:
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
                        axes[i, j].scatter(dataarray.loc[diverging,PLOT_PARAMETERS[j]],
                                           dataarray.loc[diverging,PLOT_PARAMETERS[i]], s=1,alpha=0.5)
                    axes[i, j].set_xlabel(MODEL_PARAMS_TO_TEX[xlab_curr])
                    axes[i, j].set_ylabel(MODEL_PARAMS_TO_TEX[ylab_curr])
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
        except np.linalg.LinAlgError:
            pass

def scatter_hist(x, y, ax, ax_histx, ax_histy):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=True)
    ax_histy.tick_params(axis="y", labelleft=True)

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


        # print([min([min(np.log10(KeqDhaT)) - 1, min(dhaT_fill_between)]) - 1,
        #              max([max(np.log10(KeqDhaT))+ 1, max(dhaT_fill_between)]) + 1 ])

        # plt.show()

        #
        ax_histx = fig.add_axes(rect_histx, sharex=ax)
        ax_histx.set_ylabel('Probability', fontsize=15)
        ax_histy = fig.add_axes(rect_histy, sharey=ax)
        ax_histy.set_xlabel('Probability', fontsize=15)
        #
        # # use the previously defined function
        scatter_hist(np.log10(KeqDhaT), np.log10(KeqDhaB), ax, ax_histx, ax_histy)
        sns.kdeplot(x=np.log10(KeqDhaT), y=np.log10(KeqDhaB), fill=True,
                    alpha=0.5, color='blue', ax=ax)
        ax.fill_between(dhaT_fill_between, dhaB_fill_between[0], dhaB_fill_between[-1], facecolor='yellow', alpha=0.3,
                        label ="Thermodynamically\n Feasible")
        ax.set_xlim([min([min(np.log10(KeqDhaT)) - 1, min(dhaT_fill_between)]) - 1,
                     max([max(np.log10(KeqDhaT))+ 1, max(dhaT_fill_between)]) + 1 ])
        ax.set_ylim([min([min(np.log10(KeqDhaB)) - 1, min(dhaB_fill_between)]) - 1,
                     max([max(np.log10(KeqDhaB))+ 1, max(dhaB_fill_between)]) + 1])

        probdhaT = ax_histx.get_ylim()
        probdhaB = ax_histy.get_xlim()
        y_fill_between = np.arange(probdhaB[0], probdhaB[1], 0.05)
        ax_histx.fill_between(dhaT_fill_between, probdhaT[0], probdhaT[1],
                              facecolor='yellow', alpha=0.3)
        ax_histy.fill_between(y_fill_between, dhaB_fill_between[0], dhaB_fill_between[-1],
                               facecolor='yellow', alpha=0.3)
        ax.set_xlabel(xlab, fontsize=10)
        ax.set_ylabel(ylab, fontsize=10)
        ax.legend(fontsize=15)
        ax.tick_params(axis="x", labelsize=15)
        ax.tick_params(axis="y", labelsize=15, rotation=0)
        ax_histx.tick_params(axis="x", labelsize=15)
        ax_histx.tick_params(axis="y", labelsize=15, rotation=0)
        ax_histy.tick_params(axis="x", labelsize=15)
        ax_histy.tick_params(axis="y", labelsize=15, rotation=0)
        ax_histx.set_title('Joint Distribution of the reaction\n equilibrium constants', fontsize=15)
        # plt.show()
        plt.savefig(plot_location + '/K_Eq_Distribution_' + str(chain_ind), bbox_inches="tight")
        plt.close()
