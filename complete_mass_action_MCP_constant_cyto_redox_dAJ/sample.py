from prior_constants import CELL_PERMEABILITY_MEAN, CELL_PERMEABILITY_STD, MCP_PERMEABILITY_MEAN, MCP_PERMEABILITY_STD, \
    KINETIC_PARAMETER_MEAN, KINETIC_PARAMETER_STD, PDU_WT_ENZ_NUMBERS_PARAMETER_MEAN, PDU_WT_ENZ_NUMBERS_PARAMETER_STD, \
    dPDU_AJ_ENZ_NUMBER_PARAMETER_MEAN, dPDU_AJ_ENZ_NUMBER_PARAMETER_STD, COFACTOR_NUMBER_PARAMETER_MEAN, \
    COFACTOR_NUMBER_PARAMETER_STD, GEOMETRY_PARAMETER_MEAN, GEOMETRY_PARAMETER_STD, CELL_PERMEABILITY_PARAMETER_RANGES, \
    MCP_PERMEABILITY_PARAMETER_RANGES, KINETIC_PARAMETER_RANGES, GEOMETRY_PARAMETER_RANGES, \
    COFACTOR_NUMBER_PARAMETER_RANGES, PDU_WT_ENZ_NUMBERS_PARAMETER_RANGES, dPDU_AJ_ENZ_NUMBER_PARAMETER_RANGES
import pymc as pm
import arviz as az
from constants import PERMEABILITY_CELL_PARAMETERS, PERMEABILITY_MCP_PARAMETERS, KINETIC_PARAMETERS, MCP_PARAMETERS, \
    COFACTOR_PARAMETERS, ENZYME_CONCENTRATIONS, THERMO_PARAMETERS, MCP_RADIUS, MODEL_PARAMETERS
import time
from os.path import dirname, abspath
import sys
import aesara.tensor as at
from pathlib import Path
from likelihood_funcs_adj_true import likelihood_adj_true
from likelihood_funcs_adj import likelihood_adj
from pymc_funcs import LogLike
from pymc_funcs_true import LogLikeTrue

import numpy as np

ROOT_PATH = dirname(abspath(__file__))
from datetime import datetime
import os


def sample(nsamples, burn_in, nchains, acc_rate=0.8, fwd_rtol=1e-8, fwd_atol=1e-8, bck_rtol=1e-4, bck_atol=1e-4,
           fwd_mxsteps=int(1e5), bck_mxsteps=int(1e5), init='jitter+adapt_diag', initvals=None, random_seed=None):
    # use PyMC to sampler from log-likelihood

    logl = LogLike(likelihood_adj, fwd_rtol=fwd_rtol, fwd_atol=fwd_atol, bck_rtol=bck_rtol, bck_atol=bck_atol,
                   fwd_mxsteps=fwd_mxsteps, bck_mxsteps=bck_mxsteps)
    logltrue = LogLikeTrue(likelihood_adj_true, fwd_rtol=fwd_rtol, fwd_atol=fwd_atol, bck_rtol=bck_rtol,
                           bck_atol=bck_atol,
                           fwd_mxsteps=fwd_mxsteps, bck_mxsteps=bck_mxsteps)
    with pm.Model():
        # set prior parameters
        permeability_cell_params = [pm.TruncatedNormal(param_name, mu=CELL_PERMEABILITY_MEAN[param_name],
                                                       sigma=CELL_PERMEABILITY_STD[param_name],
                                                       lower=CELL_PERMEABILITY_PARAMETER_RANGES[param_name][0],
                                                       upper=CELL_PERMEABILITY_PARAMETER_RANGES[param_name][1])
                                    for param_name in PERMEABILITY_CELL_PARAMETERS]

        permeability_mcp_params = [pm.TruncatedNormal(param_name, mu=MCP_PERMEABILITY_MEAN[param_name],
                                                      sigma=MCP_PERMEABILITY_STD[param_name],
                                                      lower=MCP_PERMEABILITY_PARAMETER_RANGES[param_name][0],
                                                      upper=MCP_PERMEABILITY_PARAMETER_RANGES[param_name][1])
                                   for param_name in PERMEABILITY_MCP_PARAMETERS]

        kinetic_params = [pm.TruncatedNormal(param_name, mu=KINETIC_PARAMETER_MEAN[param_name],
                                             sigma=KINETIC_PARAMETER_STD[param_name], lower=-7, upper=7)
                          if param_name not in THERMO_PARAMETERS else pm.Uniform(param_name, lower=
        KINETIC_PARAMETER_RANGES[param_name][0],
                                                                                 upper=
                                                                                 KINETIC_PARAMETER_RANGES[param_name][
                                                                                     1])
                          for param_name in KINETIC_PARAMETERS]
        nmcps = pm.Uniform('nMCPs', lower=GEOMETRY_PARAMETER_RANGES['nMCPs'][0],
                           upper=GEOMETRY_PARAMETER_RANGES['nMCPs'][1])

        mcp_geometry_params = [nmcps, pm.Uniform('AJ_radius', lower=np.log10(MCP_RADIUS * (10 ** (nmcps / 3.))),
                                                 upper=np.log10(MCP_RADIUS * (10 ** (nmcps / 2.))))]

        cofactor_params = [pm.TruncatedNormal(param_name, mu=COFACTOR_NUMBER_PARAMETER_MEAN[param_name],
                                              sigma=COFACTOR_NUMBER_PARAMETER_STD[param_name],
                                              lower=COFACTOR_NUMBER_PARAMETER_RANGES[param_name][0],
                                              upper=COFACTOR_NUMBER_PARAMETER_RANGES[param_name][1])
                           for param_name in COFACTOR_PARAMETERS]

        enzyme_init_WT = [pm.TruncatedNormal(param_name + '_WT', mu=PDU_WT_ENZ_NUMBERS_PARAMETER_MEAN[param_name],
                                             sigma=PDU_WT_ENZ_NUMBERS_PARAMETER_STD[param_name],
                                             lower=PDU_WT_ENZ_NUMBERS_PARAMETER_RANGES[param_name][0] + np.log10(0.25),
                                             upper=PDU_WT_ENZ_NUMBERS_PARAMETER_RANGES[param_name][1] + np.log10(1.5))
                          for param_name in ENZYME_CONCENTRATIONS]

        enzyme_init_dAJ = [pm.TruncatedNormal(param_name + '_dAJ', mu=dPDU_AJ_ENZ_NUMBER_PARAMETER_MEAN[param_name],
                                              sigma=dPDU_AJ_ENZ_NUMBER_PARAMETER_STD[param_name],
                                              lower=PDU_WT_ENZ_NUMBERS_PARAMETER_RANGES[param_name][0] + np.log10(0.25),
                                              upper=PDU_WT_ENZ_NUMBERS_PARAMETER_RANGES[param_name][1] + np.log10(1.5))
                           for param_name in ENZYME_CONCENTRATIONS]

        variables = [*permeability_cell_params, *permeability_mcp_params, *kinetic_params, *mcp_geometry_params,
                     *cofactor_params, *enzyme_init_WT, *enzyme_init_dAJ]

        theta = at.as_tensor_variable(variables)

        # Compute Michaelis-Menten Parameters

        #############################
        ########## PduCDE
        #############################
        k4PduCDE = pm.Deterministic('k4PduCDE', variables[MODEL_PARAMETERS.index('k1PduCDE')]
                                    + variables[MODEL_PARAMETERS.index('k3PduCDE')] - variables[
                                        MODEL_PARAMETERS.index('KeqPduCDE')]
                                    - variables[MODEL_PARAMETERS.index('k2PduCDE')])

        pm.Deterministic('kcat_PduCDE_f', variables[MODEL_PARAMETERS.index('k3PduCDE')])
        pm.Deterministic('kcat_PduCDE_Glycerol', np.log10((np.power(10, variables[MODEL_PARAMETERS.index('k2PduCDE')])
                                                           + np.power(10, variables[
                    MODEL_PARAMETERS.index('k3PduCDE')])) / np.power(10,
                                                                     variables[MODEL_PARAMETERS.index('k1PduCDE')])))
        pm.Deterministic('kcat_PduCDE_r', variables[MODEL_PARAMETERS.index('k2PduCDE')])
        pm.Deterministic('kcat_PduCDE_HPA', np.log10((np.power(10, variables[MODEL_PARAMETERS.index('k2PduCDE')])
                                                      + np.power(10, variables[
                    MODEL_PARAMETERS.index('k3PduCDE')])) / np.power(10, k4PduCDE)))

        #############################
        ########## PduQ
        #############################
        k8PduQ = pm.Deterministic('k8PduQ', variables[MODEL_PARAMETERS.index('k1PduQ')]
                                  + variables[MODEL_PARAMETERS.index('k3PduQ')]
                                  + variables[MODEL_PARAMETERS.index('k5PduQ')]
                                  + variables[MODEL_PARAMETERS.index('k7PduQ')]
                                  - variables[MODEL_PARAMETERS.index('KeqPduQ')]
                                  - variables[MODEL_PARAMETERS.index('k2PduQ')]
                                  - variables[MODEL_PARAMETERS.index('k4PduQ')]
                                  - variables[MODEL_PARAMETERS.index('k6PduQ')])
        pm.Deterministic('kcat_PduQ_f', np.log10(
            np.power(10, variables[MODEL_PARAMETERS.index('k5PduQ')]) * np.power(10, variables[
                MODEL_PARAMETERS.index('k7PduQ')]) / (
                    np.power(10, variables[MODEL_PARAMETERS.index('k5PduQ')]) + np.power(10, variables[
                MODEL_PARAMETERS.index('k7PduQ')]))))
        pm.Deterministic('kcat_PduQ_NADH', np.log10(
            np.power(10, variables[MODEL_PARAMETERS.index('k5PduQ')]) * np.power(10, variables[
                MODEL_PARAMETERS.index('k7PduQ')]) / (
                    np.power(10, variables[MODEL_PARAMETERS.index('k1PduQ')]) * (
                    np.power(10, variables[MODEL_PARAMETERS.index('k5PduQ')]) + np.power(10, variables[
                MODEL_PARAMETERS.index('k7PduQ')])))))
        pm.Deterministic('kcat_PduQ_HPA', np.log10(
            (np.power(10, variables[MODEL_PARAMETERS.index('k4PduQ')]) + np.power(10, variables[
                MODEL_PARAMETERS.index('k5PduQ')])) * np.power(10,
                                                               variables[MODEL_PARAMETERS.index('k7PduQ')]) / (
                    np.power(10, variables[MODEL_PARAMETERS.index('k3PduQ')]) * (
                    np.power(10, variables[MODEL_PARAMETERS.index('k5PduQ')]) + np.power(10, variables[
                MODEL_PARAMETERS.index('k7PduQ')])))))
        pm.Deterministic('kcat_PduQ_r', np.log10(
            np.power(10, variables[MODEL_PARAMETERS.index('k2PduQ')]) * np.power(10, variables[
                MODEL_PARAMETERS.index('k4PduQ')]) / (
                    np.power(10, variables[MODEL_PARAMETERS.index('k2PduQ')]) + np.power(10, variables[
                MODEL_PARAMETERS.index('k4PduQ')]))))
        pm.Deterministic('kcat_PduQ_PDO', np.log10(
            (np.power(10, variables[MODEL_PARAMETERS.index('k4PduQ')]) + np.power(10, variables[
                MODEL_PARAMETERS.index('k5PduQ')])) * np.power(10,
                                                               variables[MODEL_PARAMETERS.index('k2PduQ')]) / (
                    np.power(10, variables[MODEL_PARAMETERS.index('k6PduQ')]) * (
                    np.power(10, variables[MODEL_PARAMETERS.index('k2PduQ')]) + np.power(10, variables[
                MODEL_PARAMETERS.index('k4PduQ')])))))
        pm.Deterministic('kcat_PduQ_NAD', np.log10(
            np.power(10, variables[MODEL_PARAMETERS.index('k2PduQ')]) * np.power(10, variables[
                MODEL_PARAMETERS.index('k4PduQ')]) / (
                    np.power(10,
                             k8PduQ) * (
                            np.power(10, variables[MODEL_PARAMETERS.index('k2PduQ')]) + np.power(10, variables[
                        MODEL_PARAMETERS.index('k4PduQ')])))))

        #############################
        ########## PduP
        #############################
        k8PduP = pm.Deterministic('k8PduP', variables[MODEL_PARAMETERS.index('k1PduP')]
                                  + variables[MODEL_PARAMETERS.index('k3PduP')]
                                  + variables[MODEL_PARAMETERS.index('k5PduP')]
                                  + variables[MODEL_PARAMETERS.index('k7PduP')]
                                  - variables[MODEL_PARAMETERS.index('KeqPduP')]
                                  - variables[MODEL_PARAMETERS.index('k2PduP')]
                                  - variables[MODEL_PARAMETERS.index('k4PduP')]
                                  - variables[MODEL_PARAMETERS.index('k6PduP')])

        pm.Deterministic('kcat_PduP_f', np.log10(
            np.power(10, variables[MODEL_PARAMETERS.index('k5PduP')]) * np.power(10, variables[
                MODEL_PARAMETERS.index('k7PduP')]) / (
                    np.power(10, variables[MODEL_PARAMETERS.index('k5PduP')]) + np.power(10, variables[
                MODEL_PARAMETERS.index('k7PduP')]))))
        pm.Deterministic('kcat_PduP_NAD', np.log10(
            np.power(10, variables[MODEL_PARAMETERS.index('k5PduP')]) * np.power(10, variables[
                MODEL_PARAMETERS.index('k7PduP')]) / (
                    np.power(10, variables[MODEL_PARAMETERS.index('k1PduP')]) * (
                    np.power(10, variables[MODEL_PARAMETERS.index('k5PduP')]) + np.power(10, variables[
                MODEL_PARAMETERS.index('k7PduP')])))))
        pm.Deterministic('kcat_PduP_HPA', np.log10(
            (np.power(10, variables[MODEL_PARAMETERS.index('k4PduP')]) + np.power(10, variables[
                MODEL_PARAMETERS.index('k5PduP')])) * np.power(10,
                                                               variables[MODEL_PARAMETERS.index('k7PduP')]) / (
                    np.power(10, variables[MODEL_PARAMETERS.index('k3PduP')]) * (
                    np.power(10, variables[MODEL_PARAMETERS.index('k5PduP')]) + np.power(10, variables[
                MODEL_PARAMETERS.index('k7PduP')])))))
        pm.Deterministic('kcat_PduP_r', np.log10(
            np.power(10, variables[MODEL_PARAMETERS.index('k2PduP')]) * np.power(10, variables[
                MODEL_PARAMETERS.index('k4PduP')]) / (
                    np.power(10, variables[MODEL_PARAMETERS.index('k2PduP')]) + np.power(10, variables[
                MODEL_PARAMETERS.index('k4PduP')]))))
        pm.Deterministic('kcat_PduP_HCoA', np.log10(
            (np.power(10, variables[MODEL_PARAMETERS.index('k4PduP')]) + np.power(10, variables[
                MODEL_PARAMETERS.index('k5PduP')])) * np.power(10,
                                                               variables[MODEL_PARAMETERS.index('k2PduP')]) / (
                    np.power(10, variables[MODEL_PARAMETERS.index('k6PduP')]) * (
                    np.power(10, variables[MODEL_PARAMETERS.index('k2PduP')]) + np.power(10, variables[
                MODEL_PARAMETERS.index('k4PduP')])))))
        pm.Deterministic('kcat_PduP_NADH', np.log10(
            np.power(10, variables[MODEL_PARAMETERS.index('k2PduP')]) * np.power(10, variables[
                MODEL_PARAMETERS.index('k4PduP')]) / (
                    np.power(10, k8PduP) * (
                    np.power(10, variables[MODEL_PARAMETERS.index('k2PduP')]) + np.power(10, variables[
                MODEL_PARAMETERS.index('k4PduP')])))))

        #############################
        ########## PduL
        #############################
        k4PduL = pm.Deterministic('k4PduL', variables[MODEL_PARAMETERS.index('k1PduL')]
                                  + variables[MODEL_PARAMETERS.index('k3PduL')] - variables[
                                      MODEL_PARAMETERS.index('KeqPduL')]
                                  - variables[MODEL_PARAMETERS.index('k2PduL')])
        pm.Deterministic('kcat_PduL_f', variables[MODEL_PARAMETERS.index('k3PduL')])
        pm.Deterministic('kcat_PduL_HCoA', np.log10((np.power(10, variables[MODEL_PARAMETERS.index('k2PduL')])
                                                     + np.power(10, variables[
                    MODEL_PARAMETERS.index('k3PduL')])) / np.power(10,
                                                                   variables[MODEL_PARAMETERS.index('k1PduL')])))
        pm.Deterministic('kcat_PduL_r', variables[MODEL_PARAMETERS.index('k2PduL')])
        pm.Deterministic('kcat_PduL_HPhosph', np.log10((np.power(10, variables[MODEL_PARAMETERS.index('k2PduL')])
                                                        + np.power(10, variables[
                    MODEL_PARAMETERS.index('k3PduL')])) / np.power(10, k4PduL)))

        pm.Deterministic('loglik_true_sd', logltrue(theta))

        # set external potential
        pm.Potential("likelihood", logl(theta))

        # use a Potential to "call" the Op and include it in the logp computation
        idata_nuts = pm.sample(draws=int(nsamples), init=init, cores=nchains, chains=nchains, tune=int(burn_in),
                               target_accept=acc_rate, initvals=initvals, random_seed=random_seed,
                               discard_tuned_samples=False)
        print(idata_nuts.posterior)
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

    print(sys.argv)

    idata_nuts = sample(nsamples, burn_in, nchains, acc_rate=acc_rate, fwd_rtol=fwd_rtol, fwd_atol=fwd_atol,
                        bck_rtol=bck_rtol, bck_atol=bck_atol, fwd_mxsteps=fwd_mxsteps, bck_mxsteps=bck_mxsteps,
                        init=init, initvals=start_val,
                        random_seed=random_seed)

    # save samples
    PARAMETER_SAMP_PATH = ROOT_PATH + '/samples'
    directory_name = 'nsamples_' + str(nsamples) + '_burn_in_' + str(burn_in) + '_acc_rate_' + str(acc_rate) + \
                     '_nchains_' + str(nchains) + '_fwd_rtol_' + str(fwd_rtol) + '_fwd_atol_' + str(fwd_atol) \
                     + '_bck_rtol_' + str(bck_rtol) + '_bck_atol_' + str(bck_atol) + '_fwd_mxsteps_' + str(fwd_mxsteps) \
                     + '_bck_mxsteps_' + str(bck_mxsteps) + '_initialization_' + init
    directory_name = directory_name.replace('.', '_').replace('-', '_').replace('+', '_')
    date_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f") + '.nc'
    file_name = date_string
    sample_file_location = os.path.join(PARAMETER_SAMP_PATH, directory_name)
    Path(sample_file_location).mkdir(parents=True, exist_ok=True)
    idata_nuts.to_netcdf(os.path.join(sample_file_location, date_string))
