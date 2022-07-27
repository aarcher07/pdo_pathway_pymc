import matplotlib as mpl
import aesara
import aesara.tensor as at
import matplotlib.pyplot as plt
import os
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

import numpy as np
from datetime import datetime
from likelihood_funcs_adj import likelihood_adj, likelihood_derivative_adj  #TODO : change to _3HPA
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

    def __init__(self, likelihood, fwd_rtol = 1e-8, fwd_atol = 1e-8,
                 bck_rtol = 1e-4, bck_atol = 1e-4, mxsteps = int(1e4)):
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
        self.mxsteps = mxsteps
        self.likelihood = lambda params: likelihood(params, fwd_rtol=self.fwd_rtol, fwd_atol = self.fwd_atol,
                                                    mxsteps=self.mxsteps)
        self.logpgrad = LogLikeGrad(fwd_rtol=self.fwd_rtol, fwd_atol = self.fwd_atol, bck_rtol=self.bck_rtol,
                                    bck_atol = self.bck_atol, mxsteps=self.mxsteps)

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

    def __init__(self, fwd_rtol=1e-8, fwd_atol=1e-8, bck_rtol=1e-4, bck_atol=1e-4, mxsteps = int(1e5)):
        """
        Initialise with various things that the function requires. Below
        are the things that are needed in this particular example.
        """
        self.fwd_rtol = fwd_rtol
        self.fwd_atol = fwd_atol
        self.bck_rtol = bck_rtol
        self.bck_atol = bck_atol
        self.mxsteps = mxsteps

    def perform(self, node, inputs, outputs):
        (params,) = inputs
        # calculate gradients
        grads = likelihood_derivative_adj(params, fwd_rtol=self.fwd_rtol, fwd_atol=self.fwd_atol,
                                          bck_rtol=self.bck_rtol, bck_atol=self.bck_atol, mxsteps=self.mxsteps)
        outputs[0][0] = grads

