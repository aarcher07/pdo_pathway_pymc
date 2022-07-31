from prior_constants import *
import pymc as pm
import arviz as az
from constants import *
import time
from os.path import dirname, abspath
import sys
import aesara.tensor as at
from pathlib import Path
from likelihood_funcs_adj_true import likelihood_adj_true
from pymc_funcs import LogLike
from pymc_funcs_true import LogLikeTrue

import numpy as np

ROOT_PATH = dirname(abspath(__file__))
from datetime import datetime
import os


def sample(nsamples, burn_in, nchains, nsamps_prior=int(1e4), acc_rate=0.8, fwd_rtol=1e-8, fwd_atol=1e-8, bck_rtol=1e-4,
           bck_atol=1e-4,
           fwd_mxsteps=int(1e5), bck_mxsteps=int(1e5), init='jitter+adapt_diag', initvals=None, random_seed=None):
    # use PyMC to sampler from log-likelihood

    logl = LogLikeTrue(likelihood_adj_true, fwd_rtol=fwd_rtol, fwd_atol=fwd_atol, bck_rtol=bck_rtol,
                       bck_atol=bck_atol,
                       fwd_mxsteps=fwd_mxsteps, bck_mxsteps=bck_mxsteps)
    with pm.Model() as model:
        # set prior parameters
        permeability_cell_params = [pm.TruncatedNormal(param_name, mu=LOG_NORM_PRIOR_ALL_EXP_MEAN[param_name],
                                                       sigma=LOG_NORM_PRIOR_ALL_EXP_STD[param_name],
                                                       lower=LOG_UNIF_PRIOR_ALL_EXP[param_name][0],
                                                       upper=LOG_UNIF_PRIOR_ALL_EXP[param_name][1])
                                    for param_name in PERMEABILITY_CELL_PARAMETERS]

        permeability_mcp_params = [pm.TruncatedNormal(param_name, mu=LOG_NORM_PRIOR_ALL_EXP_MEAN[param_name],
                                                      sigma=LOG_NORM_PRIOR_ALL_EXP_STD[param_name],
                                                      lower=LOG_UNIF_PRIOR_ALL_EXP[param_name][0],
                                                      upper=LOG_UNIF_PRIOR_ALL_EXP[param_name][1])
                                   for param_name in PERMEABILITY_MCP_PARAMETERS]

        kinetic_params = [pm.TruncatedNormal(param_name, mu=LOG_NORM_PRIOR_ALL_EXP_MEAN[param_name],
                                             sigma=LOG_NORM_PRIOR_ALL_EXP_STD[param_name], lower=-7, upper=7)
                          if param_name not in THERMO_PARAMETERS else pm.Uniform(param_name, lower= LOG_UNIF_PRIOR_ALL_EXP[param_name][0],
                                                                                 upper=LOG_UNIF_PRIOR_ALL_EXP[param_name][1])

                          for param_name in KINETIC_PARAMETERS]

        glpk_kinetic_params = [pm.TruncatedNormal(param_name, mu=LOG_NORM_PRIOR_ALL_EXP_MEAN[param_name],
                                             sigma=LOG_NORM_PRIOR_ALL_EXP_STD[param_name], lower=-7, upper=7)
                          for param_name in GLOBAL_GlpK_PARAMETERS]

        nmcps = pm.Uniform('nMCPs', lower=LOG_UNIF_PRIOR_ALL_EXP['nMCPs'][0],
                           upper=LOG_UNIF_PRIOR_ALL_EXP['nMCPs'][1])

        mcp_geometry_params = [nmcps, pm.Uniform('AJ_radius', lower=np.log10(MCP_RADIUS * (10 ** (nmcps / 3.))),
                                                 upper=np.log10(MCP_RADIUS * (10 ** (nmcps / 2.))))]

        cofactor_params = [pm.TruncatedNormal(param_name, mu=LOG_NORM_PRIOR_ALL_EXP_MEAN[param_name],
                                              sigma=COFACTOR_NUMBER_PARAMETER_STD[param_name],
                                              lower=LOG_UNIF_PRIOR_ALL_EXP[param_name][0],
                                              upper=LOG_UNIF_PRIOR_ALL_EXP[param_name][1])
                           for param_name in GLOBAL_COFACTOR_PARAMETERS]


        enzyme_init = [pm.TruncatedNormal(param_name, mu=PDU_ENZ_NUMBERS_PARAMETER_MEAN[param_name],
                                             sigma=PDU_ENZ_NUMBERS_PARAMETER_STD[param_name],
                                             lower=LOG_UNIF_PRIOR_ALL_EXP[param_name][0] + np.log10(0.25),
                                             upper=LOG_UNIF_PRIOR_ALL_EXP[param_name][1] + np.log10(1.5))
                        for param_name in GLOBAL_ENZYME_PARAMETERS]

        variables = [*permeability_cell_params, *permeability_mcp_params, *kinetic_params, *glpk_kinetic_params,
                     *mcp_geometry_params, *cofactor_params, *enzyme_init]

        theta = at.as_tensor_variable(variables)

        # Compute Michaelis-Menten Parameters

        #############################
        ########## PduCDE
        #############################
        k4PduCDE = pm.Deterministic('k4PduCDE', variables[GLOBAL_DEV_PARAMETERS.index('k1PduCDE')]
                                    + variables[GLOBAL_DEV_PARAMETERS.index('k3PduCDE')] - variables[
                                        GLOBAL_DEV_PARAMETERS.index('KeqPduCDE')]
                                    - variables[GLOBAL_DEV_PARAMETERS.index('k2PduCDE')])

        pm.Deterministic('kcat_PduCDE_f', variables[GLOBAL_DEV_PARAMETERS.index('k3PduCDE')])
        pm.Deterministic('Km_PduCDE_Glycerol', np.log10((np.power(10, variables[GLOBAL_DEV_PARAMETERS.index('k2PduCDE')])
                                                           + np.power(10, variables[
                    GLOBAL_DEV_PARAMETERS.index('k3PduCDE')])) / np.power(10,
                                                                     variables[GLOBAL_DEV_PARAMETERS.index('k1PduCDE')])))
        pm.Deterministic('kcat_PduCDE_r', variables[GLOBAL_DEV_PARAMETERS.index('k2PduCDE')])
        pm.Deterministic('Km_PduCDE_HPA', np.log10((np.power(10, variables[GLOBAL_DEV_PARAMETERS.index('k2PduCDE')])
                                                      + np.power(10, variables[
                    GLOBAL_DEV_PARAMETERS.index('k3PduCDE')])) / np.power(10, k4PduCDE)))

        #############################
        ########## PduQ
        #############################
        k8PduQ = pm.Deterministic('k8PduQ', variables[GLOBAL_DEV_PARAMETERS.index('k1PduQ')]
                                  + variables[GLOBAL_DEV_PARAMETERS.index('k3PduQ')]
                                  + variables[GLOBAL_DEV_PARAMETERS.index('k5PduQ')]
                                  + variables[GLOBAL_DEV_PARAMETERS.index('k7PduQ')]
                                  - variables[GLOBAL_DEV_PARAMETERS.index('KeqPduQ')]
                                  - variables[GLOBAL_DEV_PARAMETERS.index('k2PduQ')]
                                  - variables[GLOBAL_DEV_PARAMETERS.index('k4PduQ')]
                                  - variables[GLOBAL_DEV_PARAMETERS.index('k6PduQ')])
        pm.Deterministic('kcat_PduQ_f', np.log10(
            np.power(10, variables[GLOBAL_DEV_PARAMETERS.index('k5PduQ')]) * np.power(10, variables[
                GLOBAL_DEV_PARAMETERS.index('k7PduQ')]) / (
                    np.power(10, variables[GLOBAL_DEV_PARAMETERS.index('k5PduQ')]) + np.power(10, variables[
                GLOBAL_DEV_PARAMETERS.index('k7PduQ')]))))
        pm.Deterministic('Km_PduQ_NADH', np.log10(
            np.power(10, variables[GLOBAL_DEV_PARAMETERS.index('k5PduQ')]) * np.power(10, variables[
                GLOBAL_DEV_PARAMETERS.index('k7PduQ')]) / (
                    np.power(10, variables[GLOBAL_DEV_PARAMETERS.index('k1PduQ')]) * (
                    np.power(10, variables[GLOBAL_DEV_PARAMETERS.index('k5PduQ')]) + np.power(10, variables[
                GLOBAL_DEV_PARAMETERS.index('k7PduQ')])))))
        pm.Deterministic('Km_PduQ_HPA', np.log10(
            (np.power(10, variables[GLOBAL_DEV_PARAMETERS.index('k4PduQ')]) + np.power(10, variables[
                GLOBAL_DEV_PARAMETERS.index('k5PduQ')])) * np.power(10,
                                                               variables[GLOBAL_DEV_PARAMETERS.index('k7PduQ')]) / (
                    np.power(10, variables[GLOBAL_DEV_PARAMETERS.index('k3PduQ')]) * (
                    np.power(10, variables[GLOBAL_DEV_PARAMETERS.index('k5PduQ')]) + np.power(10, variables[
                GLOBAL_DEV_PARAMETERS.index('k7PduQ')])))))
        pm.Deterministic('kcat_PduQ_r', np.log10(
            np.power(10, variables[GLOBAL_DEV_PARAMETERS.index('k2PduQ')]) * np.power(10, variables[
                GLOBAL_DEV_PARAMETERS.index('k4PduQ')]) / (
                    np.power(10, variables[GLOBAL_DEV_PARAMETERS.index('k2PduQ')]) + np.power(10, variables[
                GLOBAL_DEV_PARAMETERS.index('k4PduQ')]))))
        pm.Deterministic('Km_PduQ_PDO', np.log10(
            (np.power(10, variables[GLOBAL_DEV_PARAMETERS.index('k4PduQ')]) + np.power(10, variables[
                GLOBAL_DEV_PARAMETERS.index('k5PduQ')])) * np.power(10,
                                                               variables[GLOBAL_DEV_PARAMETERS.index('k2PduQ')]) / (
                    np.power(10, variables[GLOBAL_DEV_PARAMETERS.index('k6PduQ')]) * (
                    np.power(10, variables[GLOBAL_DEV_PARAMETERS.index('k2PduQ')]) + np.power(10, variables[
                GLOBAL_DEV_PARAMETERS.index('k4PduQ')])))))
        pm.Deterministic('Km_PduQ_NAD', np.log10(
            np.power(10, variables[GLOBAL_DEV_PARAMETERS.index('k2PduQ')]) * np.power(10, variables[
                GLOBAL_DEV_PARAMETERS.index('k4PduQ')]) / (
                    np.power(10,
                             k8PduQ) * (
                            np.power(10, variables[GLOBAL_DEV_PARAMETERS.index('k2PduQ')]) + np.power(10, variables[
                        GLOBAL_DEV_PARAMETERS.index('k4PduQ')])))))

        #############################
        ########## PduP
        #############################
        k8PduP = pm.Deterministic('k8PduP', variables[GLOBAL_DEV_PARAMETERS.index('k1PduP')]
                                  + variables[GLOBAL_DEV_PARAMETERS.index('k3PduP')]
                                  + variables[GLOBAL_DEV_PARAMETERS.index('k5PduP')]
                                  + variables[GLOBAL_DEV_PARAMETERS.index('k7PduP')]
                                  - variables[GLOBAL_DEV_PARAMETERS.index('KeqPduP')]
                                  - variables[GLOBAL_DEV_PARAMETERS.index('k2PduP')]
                                  - variables[GLOBAL_DEV_PARAMETERS.index('k4PduP')]
                                  - variables[GLOBAL_DEV_PARAMETERS.index('k6PduP')])

        pm.Deterministic('kcat_PduP_f', np.log10(
            np.power(10, variables[GLOBAL_DEV_PARAMETERS.index('k5PduP')]) * np.power(10, variables[
                GLOBAL_DEV_PARAMETERS.index('k7PduP')]) / (
                    np.power(10, variables[GLOBAL_DEV_PARAMETERS.index('k5PduP')]) + np.power(10, variables[
                GLOBAL_DEV_PARAMETERS.index('k7PduP')]))))
        pm.Deterministic('Km_PduP_NAD', np.log10(
            np.power(10, variables[GLOBAL_DEV_PARAMETERS.index('k5PduP')]) * np.power(10, variables[
                GLOBAL_DEV_PARAMETERS.index('k7PduP')]) / (
                    np.power(10, variables[GLOBAL_DEV_PARAMETERS.index('k1PduP')]) * (
                    np.power(10, variables[GLOBAL_DEV_PARAMETERS.index('k5PduP')]) + np.power(10, variables[
                GLOBAL_DEV_PARAMETERS.index('k7PduP')])))))
        pm.Deterministic('Km_PduP_HPA', np.log10(
            (np.power(10, variables[GLOBAL_DEV_PARAMETERS.index('k4PduP')]) + np.power(10, variables[
                GLOBAL_DEV_PARAMETERS.index('k5PduP')])) * np.power(10,
                                                               variables[GLOBAL_DEV_PARAMETERS.index('k7PduP')]) / (
                    np.power(10, variables[GLOBAL_DEV_PARAMETERS.index('k3PduP')]) * (
                    np.power(10, variables[GLOBAL_DEV_PARAMETERS.index('k5PduP')]) + np.power(10, variables[
                GLOBAL_DEV_PARAMETERS.index('k7PduP')])))))
        pm.Deterministic('kcat_PduP_r', np.log10(
            np.power(10, variables[GLOBAL_DEV_PARAMETERS.index('k2PduP')]) * np.power(10, variables[
                GLOBAL_DEV_PARAMETERS.index('k4PduP')]) / (
                    np.power(10, variables[GLOBAL_DEV_PARAMETERS.index('k2PduP')]) + np.power(10, variables[
                GLOBAL_DEV_PARAMETERS.index('k4PduP')]))))
        pm.Deterministic('Km_PduP_HCoA', np.log10(
            (np.power(10, variables[GLOBAL_DEV_PARAMETERS.index('k4PduP')]) + np.power(10, variables[
                GLOBAL_DEV_PARAMETERS.index('k5PduP')])) * np.power(10,
                                                               variables[GLOBAL_DEV_PARAMETERS.index('k2PduP')]) / (
                    np.power(10, variables[GLOBAL_DEV_PARAMETERS.index('k6PduP')]) * (
                    np.power(10, variables[GLOBAL_DEV_PARAMETERS.index('k2PduP')]) + np.power(10, variables[
                GLOBAL_DEV_PARAMETERS.index('k4PduP')])))))
        pm.Deterministic('Km_PduP_NADH', np.log10(
            np.power(10, variables[GLOBAL_DEV_PARAMETERS.index('k2PduP')]) * np.power(10, variables[
                GLOBAL_DEV_PARAMETERS.index('k4PduP')]) / (
                    np.power(10, k8PduP) * (
                    np.power(10, variables[GLOBAL_DEV_PARAMETERS.index('k2PduP')]) + np.power(10, variables[
                GLOBAL_DEV_PARAMETERS.index('k4PduP')])))))

        #############################
        ########## PduL
        #############################
        k4PduL = pm.Deterministic('k4PduL', variables[GLOBAL_DEV_PARAMETERS.index('k1PduL')]
                                  + variables[GLOBAL_DEV_PARAMETERS.index('k3PduL')] - variables[
                                      GLOBAL_DEV_PARAMETERS.index('KeqPduL')]
                                  - variables[GLOBAL_DEV_PARAMETERS.index('k2PduL')])
        pm.Deterministic('kcat_PduL_f', variables[GLOBAL_DEV_PARAMETERS.index('k3PduL')])
        pm.Deterministic('Km_PduL_HCoA', np.log10((np.power(10, variables[GLOBAL_DEV_PARAMETERS.index('k2PduL')])
                                                     + np.power(10, variables[
                    GLOBAL_DEV_PARAMETERS.index('k3PduL')])) / np.power(10,
                                                                   variables[GLOBAL_DEV_PARAMETERS.index('k1PduL')])))
        pm.Deterministic('kcat_PduL_r', variables[GLOBAL_DEV_PARAMETERS.index('k2PduL')])
        pm.Deterministic('Km_PduL_HPhosph', np.log10((np.power(10, variables[GLOBAL_DEV_PARAMETERS.index('k2PduL')])
                                                        + np.power(10, variables[
                    GLOBAL_DEV_PARAMETERS.index('k3PduL')])) / np.power(10, k4PduL)))

        idata_nuts_prior = pm.sample_prior_predictive(samples=nsamps_prior, random_seed=[random_seed[0]])


        # set external potential
        pm.Potential("likelihood", logl(theta))

        idata_nuts_post = pm.sample(draws=int(nsamples), init=init, cores=nchains, chains=nchains,
                                    tune=int(burn_in),
                                    target_accept=acc_rate, initvals=initvals, random_seed=random_seed,
                                    discard_tuned_samples=False)
        return idata_nuts_prior, idata_nuts_post


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

    print(sys.argv)

    idata_nuts_prior, idata_nuts_post = sample(nsamples, burn_in, nchains, acc_rate=acc_rate, fwd_rtol=fwd_rtol,
                                               fwd_atol=fwd_atol,
                                               bck_rtol=bck_rtol, bck_atol=bck_atol, fwd_mxsteps=fwd_mxsteps,
                                               bck_mxsteps=bck_mxsteps,
                                               init=init, initvals=start_val,
                                               random_seed=random_seed)

    # define directory
    PARAMETER_SAMP_PATH = ROOT_PATH + '/samples_true'
    directory_name = 'nsamples_' + str(nsamples) + '_burn_in_' + str(burn_in) + '_acc_rate_' + str(
        acc_rate) + '_nchains_' + str(nchains) + '_fwd_rtol_' + str(fwd_rtol) + '_fwd_atol_' + str(
        fwd_atol) + '_bck_rtol_' + str(bck_rtol) + '_bck_atol_' + str(bck_atol) + '_fwd_mxsteps_' + str(
        fwd_mxsteps) + '_bck_mxsteps_' + str(bck_mxsteps) + '_initialization_' + init
    directory_name = directory_name.replace('.', '_').replace('-', '_').replace('+', '_')
    date_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    sample_file_location = os.path.join(PARAMETER_SAMP_PATH, directory_name, date_string)
    Path(sample_file_location).mkdir(parents=True, exist_ok=True)

    # save samples
    idata_nuts_prior.to_netcdf(os.path.join(sample_file_location, 'prior.nc'))
    idata_nuts_post.to_netcdf(os.path.join(sample_file_location, 'post.nc'))

    file = open(os.path.join(sample_file_location, "seed.txt"), "w")
    file.write(str(random_seed))
    file.close()
