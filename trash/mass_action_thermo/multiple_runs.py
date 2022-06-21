import matplotlib as mpl
import aesara
import aesara.tensor as at
import arviz as az
import matplotlib.pyplot as plt

import pymc as pm

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
from pdo_model_sympy import loglik, N_MODEL_PARAMETERS, N_DCW_PARAMETERS, TOTAL_PARAMETERS, \
    LOG_NORM_MODEL_PRIOR_MEAN, LOG_NORM_MODEL_PRIOR_STD, VARS_ALL_EXP_TO_UNITS, N_UNKNOWN_PARAMETERS
from mpi4py import MPI
import time
from os.path import dirname, abspath
import sys
from pathlib import Path
import numpy as np
from datetime import datetime
from scipy.stats import multivariate_normal
import pickle

from exp_data import TIME_SAMPLES, DATA_SAMPLES, STD_EXPERIMENTAL_DATA

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

    def __init__(self, loglik, rtol, atol):
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
        self.likelihood = lambda sample: loglik(np.concatenate((sample, np.zeros((N_UNKNOWN_PARAMETERS-2+N_DCW_PARAMETERS * 4,)))), rtol=rtol,
                                                atol=atol, type='qoi only', parallel = False)[0]
        self.logpgrad = LogLikeGrad(rtol=rtol, atol=atol)

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        (theta,) = inputs  # this will contain my variables
        # call the log-lik

        logl = self.likelihood(theta)
        print(logl)
        print(theta)

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

    def __init__(self, rtol, atol):
        """
        Initialise with various things that the function requires. Below
        are the things that are needed in this particular example.
        Parameters
        ----------
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'x') that our model requires
        sigma:
            The noise standard deviation that out function requires.
        """
        self.rtol = rtol
        self.atol = atol

    def perform(self, node, inputs, outputs):
        (theta,) = inputs
        # calculate gradients
        grads = loglik(np.concatenate((theta, np.zeros((N_UNKNOWN_PARAMETERS-2+ N_DCW_PARAMETERS * 4,)))), rtol=self.rtol,
                       atol=self.atol, type='qoi sens', parallel = False)[1]

        outputs[0][0] = grads


# create our Op
logl = LogLike(loglik, rtol=1e-5, atol=1e-3)

# use PyMC to sampler from log-likelihood
nsamples = 2
burn_in = 2
with pm.Model():
    # uniform priors on m and c

    # variables = []
    # for param_name in list(VARS_ALL_EXP_TO_UNITS.keys())[:N_UNKNOWN_PARAMETERS]:
    #    variables.append(pm.Normal(param_name, mu=0., sigma=1.))
    variables = [pm.Normal("P_G", mu=0., sigma=1.), pm.Normal("P_P", mu=0., sigma=1.)]

    # variables = pm.MvNormal("variables", mu=np.zeros(N_UNKNOWN_PARAMETERS), cov=np.diag(np.ones(N_UNKNOWN_PARAMETERS)))
    # convert m and c to a tensor vector
    theta = at.as_tensor_variable(variables)
    # use a Potential to "call" the Op and include it in the logp computation
    pm.Potential("likelihood", logl(theta))
    #idata_mh = pm.sample(draws=int(nsamples), step=pm.Metropolis(),cores=1,chains=1, tune=int(burn_in))
    idata_mh = pm.sample_smc(3,parallel=True,chains=2,n_steps=2)
date_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
folder = ROOT_PATH +  "/output/MCMC_results_data/michaelis_menten/metropolis/nsamples_" \
         + str(int(nsamples)).replace('.','_') + "_burn_in_" + str(int(burn_in)).replace('.','_')
Path(folder).mkdir(parents=True, exist_ok=True)
idata_mh.to_netcdf(folder + "/" + date_string + ".nc")

