import matplotlib as mpl
import aesara
import aesara.tensor as at
import arviz as az
import matplotlib.pyplot as plt
import pymc as pm
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
from prior_constants import NORM_PRIOR_STD_RT_SINGLE_EXP,NORM_PRIOR_MEAN_SINGLE_EXP, NORM_PRIOR_STD_RT_ALL_EXP, \
    NORM_PRIOR_MEAN_ALL_EXP, LOG_UNIF_PRIOR_ALL_EXP, DATA_LOG_UNIF_PARAMETER_RANGES, NORM_PRIOR_PARAMETER_ALL_EXP_DICT
from constants import PERMEABILITY_PARAMETERS, KINETIC_PARAMETERS, ENZYME_CONCENTRATIONS, GLYCEROL_EXTERNAL
import time
from os.path import dirname, abspath
import sys
from pathlib import Path
import numpy as np
from datetime import datetime
from scipy.stats import multivariate_normal
import pickle
from likelihood_funcs_adj import likelihood, likelihood_derivative_adj

ROOT_PATH = dirname(abspath(__file__))


# define a aesara Op for our likelihood function
class LogLike(at.Op):
    """
    Specify what type of object will be passed and returned to the Op when it is
    called. In our case we will be passing it a vector of values (the parameters
    that define our model) and returning a single "scalar" value (the
    log-likelihood)
    """

    itypes = [at.dvector]  # expects a vector of parameter values when called
    otypes = [at.dscalar]  # outputs a single scalar value (the log likelihood)

    def __init__(self, likelihood):
        """
        Initialise the Op with various things that our log-likelihood function
        requires. Below are the things that are needed in this particular
        example.
        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'x') that our model requires
        sigma:
            The noise standard deviation that our function requires.
        """

        # add inputs as class attributes
        self.likelihood = likelihood
        self.logpgrad = LogLikeGrad()

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        (theta,) = inputs  # this will contain my variables
        # call the log-lik
        logl = self.likelihood(theta)
        outputs[0][0] = np.array(logl)  # output the log-likelihood

    def grad(self, inputs, g):
        # the method that calculates the gradients - it actually returns the
        # vector-Jacobian product - g[0] is a vector of parameter values
        (theta,) = inputs  # our parameters
        return [g[0] * self.logpgrad(theta)]


class LogLikeGrad(at.Op):
    """
    This Op will be called with a vector of values and also return a vector of
    values - the gradients in each dimension.
    """

    itypes = [at.dvector]
    otypes = [at.dvector]

    def __init__(self):
        """
        Initialise with various things that the function requires. Below
        are the things that are needed in this particular example.
        """
        pass

    def perform(self, node, inputs, outputs):
        (params,) = inputs
        # calculate gradients
        grads = likelihood_derivative_adj(params)

        outputs[0][0] = grads

# use PyMC to sampler from log-likelihood
nsamples = 2
burn_in = 2
logl = LogLike(likelihood)
with pm.Model():
    permeability_params = [pm.TruncatedNormal(param_name, mu=NORM_PRIOR_PARAMETER_ALL_EXP_DICT[param_name][0],
                                              sigma= NORM_PRIOR_PARAMETER_ALL_EXP_DICT[param_name][1],
                                              lower=DATA_LOG_UNIF_PARAMETER_RANGES[param_name][0],
                                              upper=DATA_LOG_UNIF_PARAMETER_RANGES[param_name][1])
                           for param_name in PERMEABILITY_PARAMETERS]

    kinetic_params = [pm.Normal(param_name, mu = NORM_PRIOR_PARAMETER_ALL_EXP_DICT[param_name][0],
                                sigma = NORM_PRIOR_PARAMETER_ALL_EXP_DICT[param_name][1])
                      for param_name in KINETIC_PARAMETERS]

    enzyme_init = [pm.TruncatedNormal(param_name, mu = NORM_PRIOR_PARAMETER_ALL_EXP_DICT[param_name][0],
                             sigma = NORM_PRIOR_PARAMETER_ALL_EXP_DICT[param_name][1], upper = 1)
                   for param_name in ENZYME_CONCENTRATIONS]

    gly_init = [pm.Normal(param_name, mu = np.log10(NORM_PRIOR_PARAMETER_ALL_EXP_DICT[param_name][0])
                                           - NORM_PRIOR_PARAMETER_ALL_EXP_DICT[param_name][1]**2/(2*NORM_PRIOR_PARAMETER_ALL_EXP_DICT[param_name][0]**2),
                          sigma = NORM_PRIOR_PARAMETER_ALL_EXP_DICT[param_name][1]**2/NORM_PRIOR_PARAMETER_ALL_EXP_DICT[param_name][0]**2)
                for param_name in GLYCEROL_EXTERNAL]

    variables = [*permeability_params, *kinetic_params, *enzyme_init, *gly_init]
    print(variables)

    # variables = pm.MvNormal("variables", mu=np.zeros(N_UNKNOWN_PARAMETERS), cov=np.diag(np.ones(N_UNKNOWN_PARAMETERS)))
    # convert m and c to a tensor vector
    theta = at.as_tensor_variable(variables)
    # use a Potential to "call" the Op and include it in the logp computation
    pm.Potential("likelihood", logl(theta))
    idata_mh = pm.sample(draws=int(nsamples),cores=1,chains=1, tune=int(burn_in))
    #idata_mh = pm.sample_smc(3,parallel=True,chains=2,n_steps=2)


# # create our Op
# PARAMETER_SAMP_PATH = '/Volumes/Wario/PycharmProjects/pdo_pathway_model/MCMC/output'
# FILE_NAME = '/MCMC_results_data/mass_action/adaptive/preset_std/lambda_0,05_beta_0,01_burn_in_n_cov_2000/nsamples_100000/date_2022_03_04_02_11_52_142790_rank_0.pkl'
#
# N_MODEL_PARAMETERS = 15
# N_DCW_PARAMETERS = 3
# N_UNKNOWN_PARAMETERS = 19
# N_TOTAL_PARAMETERS = 15 + 4 + 12
#
# param_sample = NORM_PRIOR_MEAN_ALL_EXP.copy()
# with open(PARAMETER_SAMP_PATH + FILE_NAME, 'rb') as f:
#     postdraws = pickle.load(f)
#     samples = postdraws['samples']
#     burn_in_subset_samples = samples[int(2e4):]
#     data_subset = burn_in_subset_samples[::600,:]
#     param_mean = data_subset.mean(axis=0)
#     param_mean_trans = np.matmul(NORM_PRIOR_STD_RT_ALL_EXP[:len(param_mean), :len(param_mean)].T, param_mean) + NORM_PRIOR_MEAN_ALL_EXP[
#                                                                                                                 :len(param_mean)]
# param_sample[:(N_MODEL_PARAMETERS+4)] = param_mean_trans
# param_sample[N_MODEL_PARAMETERS:] = np.log10(param_sample[N_MODEL_PARAMETERS:])
# print(likelihood(param_sample))
# logl = LogLike(likelihood)
# print(logl.likelihood(param_sample))
