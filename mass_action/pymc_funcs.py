import matplotlib as mpl
import aesara
import aesara.tensor as at
import arviz as az
import matplotlib.pyplot as plt
import os
import pymc as pm
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
from prior_constants import NORM_PRIOR_STD_RT_SINGLE_EXP,NORM_PRIOR_MEAN_SINGLE_EXP, NORM_PRIOR_STD_RT_ALL_EXP, \
    NORM_PRIOR_MEAN_ALL_EXP, LOG_UNIF_PRIOR_ALL_EXP, DATA_LOG_UNIF_PARAMETER_RANGES, NORM_PRIOR_PARAMETER_ALL_EXP_DICT
from constants import PERMEABILITY_PARAMETERS, KINETIC_PARAMETERS, ENZYME_CONCENTRATIONS, GLYCEROL_EXTERNAL_EXPERIMENTAL, ALL_PARAMETERS
import time
from os.path import dirname, abspath
import sys
from pathlib import Path
import numpy as np
from datetime import datetime
from likelihood_funcs_adj import likelihood_adj, likelihood_derivative_adj
from os.path import dirname, abspath

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

    def __init__(self, likelihood, atol = 1e-8, rtol = 1e-8, mxsteps = int(1e4)):
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
        self.atol = atol
        self.rtol = rtol
        self.mxsteps = mxsteps
        self.likelihood = lambda params: likelihood(params, atol=self.atol, rtol=self.rtol, mxsteps=self.mxsteps)
        self.logpgrad = LogLikeGrad(atol=self.atol, rtol=self.rtol, mxsteps=self.mxsteps)

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

    def __init__(self, atol=1e-8, rtol=1e-8, mxsteps = int(1e4)):
        """
        Initialise with various things that the function requires. Below
        are the things that are needed in this particular example.
        """
        self.atol = atol
        self.rtol = rtol
        self.mxsteps = mxsteps

    def perform(self, node, inputs, outputs):
        (params,) = inputs
        # calculate gradients
        grads = likelihood_derivative_adj(params, atol=self.atol, rtol=self.rtol, mxsteps=self.mxsteps)
        outputs[0][0] = grads

def sample(nsamples, burn_in, nchains, acc_rate=0.8, atol=1e-8, rtol=1e-8, mxsteps=int(2e4), init = 'jitter+adapt_full'):
    # use PyMC to sampler from log-likelihood

    logl = LogLike(likelihood_adj,  atol=atol, rtol=rtol, mxsteps=mxsteps)
    with pm.Model():
        permeability_params = [pm.TruncatedNormal(param_name, mu=NORM_PRIOR_PARAMETER_ALL_EXP_DICT[param_name][0],
                                                  sigma= NORM_PRIOR_PARAMETER_ALL_EXP_DICT[param_name][1],
                                                  lower=DATA_LOG_UNIF_PARAMETER_RANGES[param_name][0],
                                                  upper=DATA_LOG_UNIF_PARAMETER_RANGES[param_name][1])
                               for param_name in PERMEABILITY_PARAMETERS]

        kinetic_params = [pm.TruncatedNormal(param_name, mu = NORM_PRIOR_PARAMETER_ALL_EXP_DICT[param_name][0],
                                    sigma = NORM_PRIOR_PARAMETER_ALL_EXP_DICT[param_name][1], lower = -7, upper = 7)
                          for param_name in KINETIC_PARAMETERS]

        enzyme_init = [pm.TruncatedNormal(param_name, mu = NORM_PRIOR_PARAMETER_ALL_EXP_DICT[param_name][0],
                                 sigma = NORM_PRIOR_PARAMETER_ALL_EXP_DICT[param_name][1], lower = -4, upper = 2)
                       for param_name in ENZYME_CONCENTRATIONS]

        # gly_init = [pm.Normal(param_name, mu = 0,sigma = 4) for param_name in GLYCEROL_EXTERNAL_EXPERIMENTAL]

        variables = [*permeability_params, *kinetic_params, *enzyme_init]#, *gly_init]

        # convert m and c to a tensor vector
        theta = at.as_tensor_variable(variables)
        # use a Potential to "call" the Op and include it in the logp computation
        pm.Potential("likelihood", logl(theta))
        idata_nuts = pm.sample(draws=int(nsamples), init=init,cores=nchains,chains=nchains, tune=int(burn_in), target_accept=acc_rate)

        return idata_nuts

if __name__ == '__main__':
    nsamples = int(float(sys.argv[1]))
    burn_in = int(float(sys.argv[2]))
    nchains = int(float(sys.argv[3]))
    acc_rate = float(sys.argv[4])
    atol = float(sys.argv[5])
    rtol = float(sys.argv[6])
    mxsteps = int(float(sys.argv[7]))
    init = sys.argv[8]

    print(sys.argv)

    idata_nuts = sample(nsamples, burn_in, nchains, acc_rate=acc_rate, atol=atol, rtol = rtol, mxsteps=mxsteps, init = init)

    # save samples
    PARAMETER_SAMP_PATH = ROOT_PATH + '/samples'
    directory_name = 'nsamples_' + str(nsamples) + '_burn_in_' + str(burn_in) + '_acc_rate_' + str(acc_rate) +\
                     '_nchains_' + str(nchains) + '_atol_' + str(atol) + '_rtol_' + str(rtol) + '_mxsteps_' + \
                     str(mxsteps) + '_initialization_' + init
    directory_name = directory_name.replace('.','_').replace('-','_').replace('+','_')

    date_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f") + '.nc'
    file_name = date_string
    sample_file_location = os.path.join(PARAMETER_SAMP_PATH, directory_name)
    Path(sample_file_location).mkdir(parents=True, exist_ok=True)
    idata_nuts.to_netcdf(os.path.join(sample_file_location,date_string))

    # save trace plots
    PLOT_SAMP_PATH = ROOT_PATH + '/prelim_trace_plots'
    plot_file_location = os.path.join(PLOT_SAMP_PATH, directory_name, date_string[:-3])
    Path(plot_file_location).mkdir(parents=True, exist_ok=True)

    n_display = 10
    for i in range(int(np.ceil(len(ALL_PARAMETERS)/n_display))):
        az.plot_trace(idata_nuts, var_names=ALL_PARAMETERS[(n_display*i):(n_display*(i+1))], compact=True)
        plt.savefig(os.path.join(plot_file_location,"trace_plot_" + str(i) + ".jpg"))
    print(az.summary(idata_nuts))