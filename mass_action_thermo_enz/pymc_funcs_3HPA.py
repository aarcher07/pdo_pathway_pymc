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
from constants import PERMEABILITY_PARAMETERS, KINETIC_PARAMETERS, ENZYME_CONCENTRATIONS, GLYCEROL_EXTERNAL_EXPERIMENTAL, \
    ALL_PARAMETERS, THERMO_PARAMETERS
import time
from os.path import dirname, abspath
import sys
from pathlib import Path
import numpy as np
from datetime import datetime
from likelihood_funcs_adj_3HPA import likelihood_adj, likelihood_derivative_adj  #TODO : change to _3HPA
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

    def __init__(self, likelihood, fwd_rtol = 1e-8, fwd_atol = 1e-8, bck_rtol = 1e-4, bck_atol = 1e-4,
                 fwd_mxsteps = int(1e4), bck_mxsteps = int(1e4)):
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
        self.fwd_rtol = fwd_rtol
        self.fwd_atol = fwd_atol
        self.bck_rtol = bck_rtol
        self.bck_atol = bck_atol
        self.fwd_mxsteps = fwd_mxsteps
        self.bck_mxsteps = bck_mxsteps
        self.likelihood = lambda params: likelihood(params, fwd_rtol=self.fwd_rtol, fwd_atol = self.fwd_atol,
                                                    fwd_mxsteps=self.fwd_mxsteps)
        self.logpgrad = LogLikeGrad(fwd_rtol=self.fwd_rtol, fwd_atol = self.fwd_atol, bck_rtol=self.bck_rtol,
                                    bck_atol = self.bck_atol, fwd_mxsteps=self.fwd_mxsteps, bck_mxsteps=self.bck_mxsteps)

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

    def __init__(self, fwd_rtol=1e-8, fwd_atol=1e-8, bck_rtol=1e-4, bck_atol=1e-4, fwd_mxsteps= int(1e4),
                 bck_mxsteps=int(1e4)):
        """
        Initialise with various things that the function requires. Below
        are the things that are needed in this particular example.
        """
        self.fwd_rtol = fwd_rtol
        self.fwd_atol = fwd_atol
        self.bck_rtol = bck_rtol
        self.bck_atol = bck_atol
        self.fwd_mxsteps = fwd_mxsteps
        self.bck_mxsteps = bck_mxsteps


    def perform(self, node, inputs, outputs):
        (params,) = inputs
        # calculate gradients
        grads = likelihood_derivative_adj(params, fwd_rtol=self.fwd_rtol, fwd_atol=self.fwd_atol,
                                          bck_rtol=self.bck_rtol, bck_atol=self.bck_atol, fwd_mxsteps=self.fwd_mxsteps,
                                          bck_mxsteps=self.bck_mxsteps)
        outputs[0][0] = grads

def sample(nsamples, burn_in, nchains, acc_rate=0.8, fwd_atol=1e-8, fwd_rtol=1e-8, bck_atol=1e-4, bck_rtol=1e-4,
           fwd_mxsteps=int(1e5), bck_mxsteps=int(1e5), init = 'jitter+adapt_full', initvals = None, random_seed = None):
    # use PyMC to sampler from log-likelihood

    logl = LogLike(likelihood_adj, fwd_rtol=fwd_rtol, fwd_atol=fwd_atol, bck_rtol=bck_rtol, bck_atol=bck_atol,
                   fwd_mxsteps=fwd_mxsteps, bck_mxsteps=bck_mxsteps)
    with pm.Model():
        permeability_params = [pm.TruncatedNormal(param_name, mu=NORM_PRIOR_PARAMETER_ALL_EXP_DICT[param_name][0],
                                                  sigma= NORM_PRIOR_PARAMETER_ALL_EXP_DICT[param_name][1],
                                                  lower=DATA_LOG_UNIF_PARAMETER_RANGES[param_name][0],
                                                  upper=DATA_LOG_UNIF_PARAMETER_RANGES[param_name][1])
                               for param_name in PERMEABILITY_PARAMETERS]

        kinetic_params = [pm.TruncatedNormal(param_name, mu = NORM_PRIOR_PARAMETER_ALL_EXP_DICT[param_name][0],
                                    sigma = NORM_PRIOR_PARAMETER_ALL_EXP_DICT[param_name][1], lower = -7, upper = 7)
                          if param_name not in THERMO_PARAMETERS else pm.Uniform(param_name, lower= DATA_LOG_UNIF_PARAMETER_RANGES[param_name][0],
                                                                                 upper=DATA_LOG_UNIF_PARAMETER_RANGES[param_name][1])
                          for param_name in KINETIC_PARAMETERS]
        enzyme_init = [pm.TruncatedNormal(param_name, mu = NORM_PRIOR_PARAMETER_ALL_EXP_DICT[param_name][0],
                                 sigma = NORM_PRIOR_PARAMETER_ALL_EXP_DICT[param_name][1], lower = -4, upper = 1)
                       for param_name in ENZYME_CONCENTRATIONS]

        # gly_init = [pm.Normal(param_name, mu = 0,sigma = 4) for param_name in GLYCEROL_EXTERNAL_EXPERIMENTAL]

        variables = [*permeability_params, *kinetic_params, *enzyme_init]#, *gly_init]

        # convert m and c to a tensor vector
        theta = at.as_tensor_variable(variables)
        # use a Potential to "call" the Op and include it in the logp computation
        pm.Potential("likelihood", logl(theta))
        idata_nuts = pm.sample(draws=int(nsamples), init=init, cores=nchains, chains=nchains, tune=int(burn_in),
                               target_accept=acc_rate, initvals=initvals, random_seed=random_seed,
                               discard_tuned_samples=False)

        return idata_nuts

if __name__ == '__main__':
    nsamples = int(float(sys.argv[1]))
    burn_in = int(float(sys.argv[2]))
    nchains = int(float(sys.argv[3]))
    acc_rate = float(sys.argv[4])
    fwd_rtol = float(sys.argv[5])
    fwd_atol = float(sys.argv[6])
    bck_rtol = float(sys.argv[7])
    bck_atol = float(sys.argv[8])
    fwd_mxsteps = int(float(sys.argv[9]))
    bck_mxsteps = int(float(sys.argv[10]))
    init = sys.argv[11]

    seed = int(time.time() * 1e6)
    seed = ((seed & 0xff000000) >> 24) + ((seed & 0x00ff0000) >> 8) + ((seed & 0x0000ff00) << 8) + (
            (seed & 0x000000ff) << 24)
    random_seed = seed + np.array(list(range(nchains)))
    random_seed = list(random_seed.astype(int))
    print('seed: ' + str(random_seed))
    start_val = None
    # start_val =  {'PermCellGlycerol': -3.2387621755443825, 'PermCellPDO': -4.023320346770019,
    #              'PermCell3HPA': -4.899986741128067, 'k1DhaB': -0.6036725290144016, 'k2DhaB': -0.48615514794602044,
    #              'k3DhaB': 1.1564894912795705, 'k4DhaB': 1.5738758332657916, 'k1DhaT': 1.804214813153589,
    #              'k2DhaT': -0.5618853036728277, 'k3DhaT': 0.6856456564770114, 'k4DhaT': 1.1298098325985182,
    #              'VmaxfMetab': 0.2257880715104006, 'KmMetabG': 2.72562301357831, 'DHAB_INIT': -0.511548640059402,
    #              'DHAT_INIT': 0.28071248963437223}
    print(sys.argv)

    idata_nuts = sample(nsamples, burn_in, nchains, acc_rate=acc_rate, fwd_rtol=fwd_rtol, fwd_atol=fwd_atol,
                        bck_rtol=bck_rtol, bck_atol=bck_atol, fwd_mxsteps=fwd_mxsteps, bck_mxsteps=bck_mxsteps,
                        init=init, initvals=start_val,
                        random_seed=random_seed)


    # save samples
    PARAMETER_SAMP_PATH = ROOT_PATH + '/samples_3HPA'  #TODO : change to _3HPA
    directory_name = 'nsamples_' + str(nsamples) + '_burn_in_' + str(burn_in) + '_acc_rate_' + str(acc_rate) + \
                     '_nchains_' + str(nchains) + '_fwd_rtol_' + str(fwd_rtol) + '_fwd_atol_' + str(fwd_atol) \
                     + '_bck_rtol_' + str(bck_rtol) + '_bck_atol_' + str(bck_atol) + '_fwd_mxsteps_' + str(fwd_mxsteps) \
                     + '_bck_mxsteps_' + str(bck_mxsteps) + '_initialization_' + init
    directory_name = directory_name.replace('.','_').replace('-','_').replace('+','_')

    date_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f") + '.nc'
    file_name = date_string
    sample_file_location = os.path.join(PARAMETER_SAMP_PATH, directory_name)
    Path(sample_file_location).mkdir(parents=True, exist_ok=True)
    idata_nuts.to_netcdf(os.path.join(sample_file_location,date_string))

