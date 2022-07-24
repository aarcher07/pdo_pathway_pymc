import matplotlib as mpl
import aesara
import aesara.tensor as at
import arviz as az
import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy.constants import Avogadro
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
import sunode

solver_delta_AJ = sunode.solver.AdjointSolver(problem_delta_AJ, solver='BDF')
solver_WT = sunode.solver.AdjointSolver(problem_WT, solver='BDF')
lib = sunode._cvodes.lib

def plot_trace(samples, plot_file_location):
    n_display = 10
    for i in range(int(np.ceil(len(DEV_PARAMETER_LIST)/n_display))):
        az.plot_trace(samples, var_names=DEV_PARAMETER_LIST[(n_display*i):(n_display*(i+1))], compact=True)
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
    lib.CVodeSetMaxNumSteps(solver_delta_AJ._ode, int(mxsteps))
    lib.CVodeSStolerances(solver_WT._ode, fwd_rtol, fwd_atol)
    lib.CVodeSetMaxNumSteps(solver_WT._ode, int(mxsteps))


    c = ['r', 'y', 'b', 'g', 'k']
    legend_names = ['chain ' + str(i)  for i in range(min(5, nchains))]

    for exp_ind, exp_cond in enumerate(['WT-L', 'dAJ-L', 'dD-L', 'dP-L']):

        # set initial condition depending on each experiment
        if exp_cond in ['WT-L', 'dD-L', 'dP-L']:
            param_list = [*DEV_PARAMETER_LIST[:-4], *[param + '_WT' for param in DEV_PARAMETER_LIST[-4:]]]
            solver = solver_WT
            problem = problem_WT
        elif exp_cond == 'dAJ-L':
            param_list = [*DEV_PARAMETER_LIST[:-4], *[param + '_dAJ' for param in DEV_PARAMETER_LIST[-4:]]]
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
            dataarray = dataarray[param_list]
            for jj in range(4):
                ax[jj, chain_ind].scatter(TIME_SAMPLES, TIME_SERIES_MEAN[exp_cond].iloc[:,jj])

            for j in [-1]:#range(0,dataarray.shape[0],int(dataarray.shape[0]/500)):
                y0_copy = y0.copy()
                params = dataarray.iloc[j,:].to_numpy()
                non_enz_model_params = params[:-4]
                enz_params = params[-4:]
                param_samples_copy = np.concatenate(
                    (non_enz_model_params, enz_params, list(OD_PRIOR_PARAMETER_MEAN[exp_cond].values())))
                # alter initial conditions
                if exp_cond in ['WT-L', 'dD-L', 'dP-L']:
                    y0_copy['NADH_MCP'] = (10**(param_samples_copy[PARAMETER_LIST.index('NADH_NAD_TOTAL_MCP')]
                                           + param_samples_copy[PARAMETER_LIST.index('NADH_NAD_RATIO_MCP')]))/(10**param_samples_copy[PARAMETER_LIST.index('NADH_NAD_RATIO_MCP')] + 1)
                    y0_copy['NAD_MCP'] = 10**param_samples_copy[PARAMETER_LIST.index('NADH_NAD_TOTAL_MCP')]/(10**param_samples_copy[PARAMETER_LIST.index('NADH_NAD_RATIO_MCP')] + 1)
                    y0_copy['PduCDE'] = 10**param_samples_copy[PARAMETER_LIST.index('nPduCDE')]/(Avogadro * MCP_VOLUME)
                    y0_copy['PduP'] = 10**param_samples_copy[PARAMETER_LIST.index('nPduP')]/(Avogadro * MCP_VOLUME)
                    y0_copy['PduQ'] = 10**param_samples_copy[PARAMETER_LIST.index('nPduQ')]/(Avogadro * MCP_VOLUME)
                    y0_copy['PduL'] = 10**param_samples_copy[PARAMETER_LIST.index('nPduL')]/(Avogadro * MCP_VOLUME)
                    y0_copy['OD'] = 10**param_samples_copy[PARAMETER_LIST.index('A')]
                elif exp_cond == 'dAJ-L':
                    POLAR_VOLUME = (4./3.)*np.pi*((10**param_samples_copy[PARAMETER_LIST.index('AJ_radius')])**3)
                    y0_copy['NADH_MCP'] = (10**(param_samples_copy[PARAMETER_LIST.index('NADH_NAD_TOTAL_CYTO')]
                                           + param_samples_copy[PARAMETER_LIST.index('NADH_NAD_RATIO_CYTO')]))/(10**param_samples_copy[PARAMETER_LIST.index('NADH_NAD_RATIO_CYTO')] + 1)
                    y0_copy['NAD_MCP'] = 10**param_samples_copy[PARAMETER_LIST.index('NADH_NAD_TOTAL_CYTO')]/(10**param_samples_copy[PARAMETER_LIST.index('NADH_NAD_RATIO_CYTO')] + 1)
                    y0_copy['PduCDE'] = param_samples_copy[PARAMETER_LIST.index('nMCPs')]*(10**param_samples_copy[PARAMETER_LIST.index('nPduCDE')])/(Avogadro * POLAR_VOLUME)
                    y0_copy['PduP'] = param_samples_copy[PARAMETER_LIST.index('nMCPs')]*(10**param_samples_copy[PARAMETER_LIST.index('nPduP')])/(Avogadro * POLAR_VOLUME)
                    y0_copy['PduQ'] = param_samples_copy[PARAMETER_LIST.index('nMCPs')]*(10**param_samples_copy[PARAMETER_LIST.index('nPduQ')])/(Avogadro * POLAR_VOLUME)
                    y0_copy['PduL'] = param_samples_copy[PARAMETER_LIST.index('nMCPs')]*(10**param_samples_copy[PARAMETER_LIST.index('nPduL')])/(Avogadro * POLAR_VOLUME)
                    y0_copy['OD'] = 10**param_samples_copy[PARAMETER_LIST.index('A')]

                params_dict = { param_name : param_val for param_val,param_name in zip(param_samples_copy, PARAMETER_LIST)}
                if exp_cond == 'dD-L':
                    y0_copy['PduCDE'] = 0
                    params_dict['nPduCDE'] = 0
                elif exp_cond == 'dP-L':
                    y0_copy['PduP'] = 0
                    params_dict['nPduP'] = 0
                # set solver parameters
                solver.set_params_dict(params_dict)
                tvals = TIME_SAMPLES*HRS_TO_SECS
                yout, _, _ = solver.make_output_buffers(tvals)

                try:
                    solver.solve_forward(t0=0, tvals=tvals, y0=y0_copy, y_out=yout)
                    jj=0
                    for i, var in enumerate(VARIABLE_NAMES):
                        if i in DATA_INDEX:
                            ax[jj, chain_ind].plot(tvals / HRS_TO_SECS, yout.view(problem.state_dtype)[var], 'r',
                                                  alpha=0.05)
                            jj += 1
                    # print(TIME_SERIES_MEAN[exp_cond] - yout[:, DATA_INDEX])
                    # print(TIME_SERIES_STD[exp_cond])
                    # print(((TIME_SERIES_MEAN[exp_cond] - yout[:, DATA_INDEX])/TIME_SERIES_STD[exp_cond])**2)
                    #
                    # print((((TIME_SERIES_MEAN[exp_cond] - yout[:, DATA_INDEX])/TIME_SERIES_STD[exp_cond])**2).to_numpy().sum())
                except sunode.solver.SolverError:
                    pass
            ax[0, chain_ind].set_title('Glycerol Time Series')
            ax[1, chain_ind].set_title('3-HPA Time Series')
            ax[2, chain_ind].set_title('1,3-PD Time Series')
            ax[3, chain_ind].set_title('DCW Distribution')

        plt.suptitle('Experimental Condition ' + str(exp_cond))
        fig.tight_layout()
        plt.show()
        #plt.savefig(os.path.join(plot_file_location, 'time_series_results_' + str(exp_cond) + '.png'))

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

def joint_Keq_distribution(Keq_enz_1_chains, Keq_enz_2_chains, Keq_enz_name_1, Keq_enz_name_2,
                           plot_location, nchains):

    xlab = r'$\log_{10}(' + MODEL_PARAMS_TO_TEX[Keq_enz_name_1][1:-1] + ')$'
    ylab = r'$\log_{10}(' + MODEL_PARAMS_TO_TEX[Keq_enz_name_2][1:-1] + ')$'
    # # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]
    for chain_ind in range(nchains):
        Keq_enz_1 = Keq_enz_1_chains[chain_ind]
        Keq_enz_2 = Keq_enz_2_chains[chain_ind]

        # start with a square Figure
        fig = plt.figure(figsize=(8, 8))

        ax = fig.add_axes(rect_scatter)
        enz_1_fill_between = np.arange(KINETIC_PARAMETER_RANGES[Keq_enz_name_1][0],
                                       KINETIC_PARAMETER_RANGES[Keq_enz_name_1][1] + 0.1, 0.1)
        enz_2_fill_between = np.arange(KINETIC_PARAMETER_RANGES[Keq_enz_name_2][0],
                                       KINETIC_PARAMETER_RANGES[Keq_enz_name_2][0] + 0.1, 0.1)


        # print([min([min(np.log10(KeqDhaT)) - 1, min(enz_1_fill_between)]) - 1,
        #              max([max(np.log10(KeqDhaT))+ 1, max(enz_1_fill_between)]) + 1 ])

        # plt.show()

        #
        ax_histx = fig.add_axes(rect_histx, sharex=ax)
        ax_histx.set_ylabel('Probability', fontsize=15)
        ax_histy = fig.add_axes(rect_histy, sharey=ax)
        ax_histy.set_xlabel('Probability', fontsize=15)
        #
        # # use the previously defined function
        scatter_hist(np.log10(Keq_enz_1), np.log10(Keq_enz_2), ax, ax_histx, ax_histy)
        sns.kdeplot(x=np.log10(Keq_enz_1), y=np.log10(Keq_enz_2), fill=True,
                    alpha=0.5, color='blue', ax=ax)
        ax.fill_between(enz_1_fill_between, enz_2_fill_between[0], enz_2_fill_between[-1], facecolor='yellow', alpha=0.3,
                        label ="Thermodynamically\n Feasible")
        ax.set_xlim([min([min(np.log10(Keq_enz_1)) - 1, min(enz_1_fill_between)]) - 1,
                     max([max(np.log10(Keq_enz_1)) + 1, max(enz_1_fill_between)]) + 1])
        ax.set_ylim([min([min(np.log10(Keq_enz_2)) - 1, min(enz_2_fill_between)]) - 1,
                     max([max(np.log10(Keq_enz_2)) + 1, max(enz_2_fill_between)]) + 1])

        prob_enz_1 = ax_histx.get_ylim()
        prob_enz_2 = ax_histy.get_xlim()
        y_fill_between = np.arange(prob_enz_2[0], prob_enz_2[1], 0.05)
        ax_histx.fill_between(enz_1_fill_between, prob_enz_1[0], prob_enz_1[1],
                              facecolor='yellow', alpha=0.3)
        ax_histy.fill_between(y_fill_between, enz_2_fill_between[0], enz_2_fill_between[-1],
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
        plt.savefig(plot_location + '/K_Eq_Distribution_' + Keq_enz_name_1 + '_' + Keq_enz_name_2 + '_' +  str(chain_ind), bbox_inches="tight")
        plt.close()
