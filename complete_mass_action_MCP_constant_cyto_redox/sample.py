from prior_constants import CELL_PERMEABILITY_MEAN, CELL_PERMEABILITY_STD, MCP_PERMEABILITY_MEAN, MCP_PERMEABILITY_STD, \
    KINETIC_PARAMETER_MEAN, KINETIC_PARAMETER_STD, PDU_WT_ENZ_NUMBERS_PARAMETER_MEAN, PDU_WT_ENZ_NUMBERS_PARAMETER_STD, \
    dPDU_AJ_ENZ_NUMBER_PARAMETER_MEAN, dPDU_AJ_ENZ_NUMBER_PARAMETER_STD, COFACTOR_NUMBER_PARAMETER_MEAN, \
    COFACTOR_NUMBER_PARAMETER_STD, GEOMETRY_PARAMETER_MEAN, GEOMETRY_PARAMETER_STD, CELL_PERMEABILITY_PARAMETER_RANGES,\
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
from likelihood_funcs_adj import likelihood_adj
from pymc_funcs import LogLike
import numpy as np
ROOT_PATH = dirname(abspath(__file__))
from datetime import datetime
import os

def sample(nsamples, burn_in, nchains, acc_rate=0.8, fwd_rtol=1e-8, fwd_atol=1e-8, bck_rtol=1e-4, bck_atol=1e-4,
           mxsteps=int(1e5), init = 'jitter+adapt_diag', initvals = None, random_seed = None):
    # use PyMC to sampler from log-likelihood

    logl = LogLike(likelihood_adj, fwd_rtol=fwd_rtol, fwd_atol=fwd_atol, bck_rtol=bck_rtol, bck_atol=bck_atol,
                   mxsteps=mxsteps)
    with pm.Model():
        permeability_cell_params = [pm.TruncatedNormal(param_name, mu=CELL_PERMEABILITY_MEAN[param_name],
                                                  sigma= CELL_PERMEABILITY_STD[param_name],
                                                  lower=CELL_PERMEABILITY_PARAMETER_RANGES[param_name][0],
                                                  upper=CELL_PERMEABILITY_PARAMETER_RANGES[param_name][1])
                               for param_name in PERMEABILITY_CELL_PARAMETERS]

        permeability_mcp_params = [pm.TruncatedNormal(param_name, mu=MCP_PERMEABILITY_MEAN[param_name],
                                                       sigma= MCP_PERMEABILITY_STD[param_name],
                                                       lower=MCP_PERMEABILITY_PARAMETER_RANGES[param_name][0],
                                                       upper=MCP_PERMEABILITY_PARAMETER_RANGES[param_name][1])
                                    for param_name in PERMEABILITY_MCP_PARAMETERS]

        kinetic_params = [pm.TruncatedNormal(param_name, mu = KINETIC_PARAMETER_MEAN[param_name],
                                             sigma = KINETIC_PARAMETER_STD[param_name], lower = -7, upper = 7)
                          if param_name not in THERMO_PARAMETERS else pm.Uniform(param_name, lower= KINETIC_PARAMETER_RANGES[param_name][0],
                                                                                 upper=KINETIC_PARAMETER_RANGES[param_name][1])
                          for param_name in KINETIC_PARAMETERS]
        nmcps = pm.Uniform('nMCPs', lower=GEOMETRY_PARAMETER_RANGES['nMCPs'][0], upper=GEOMETRY_PARAMETER_RANGES['nMCPs'][1])

        mcp_geometry_params = [nmcps, pm.Uniform('AJ_radius', lower=np.log10(MCP_RADIUS*(10**(nmcps/3.))),
                                                 upper=np.log10(MCP_RADIUS*(10**(nmcps/2.))))]

        cofactor_params = [pm.TruncatedNormal(param_name, mu = COFACTOR_NUMBER_PARAMETER_MEAN[param_name],
                                             sigma = COFACTOR_NUMBER_PARAMETER_STD[param_name],
                                              lower = COFACTOR_NUMBER_PARAMETER_RANGES[param_name][0],
                                              upper = COFACTOR_NUMBER_PARAMETER_RANGES[param_name][1])
                          for param_name in COFACTOR_PARAMETERS]

        enzyme_init_WT = [pm.TruncatedNormal(param_name + '_WT', mu = PDU_WT_ENZ_NUMBERS_PARAMETER_MEAN[param_name],
                                          sigma = PDU_WT_ENZ_NUMBERS_PARAMETER_STD[param_name],
                                          lower = PDU_WT_ENZ_NUMBERS_PARAMETER_RANGES[param_name][0] + np.log10(0.25),
                                          upper = PDU_WT_ENZ_NUMBERS_PARAMETER_RANGES[param_name][1] + np.log10(1.5))
                       for param_name in ENZYME_CONCENTRATIONS]

        enzyme_init_dAJ = [pm.TruncatedNormal(param_name + '_dAJ', mu = dPDU_AJ_ENZ_NUMBER_PARAMETER_MEAN[param_name],
                                             sigma = dPDU_AJ_ENZ_NUMBER_PARAMETER_STD[param_name],
                                             lower = PDU_WT_ENZ_NUMBERS_PARAMETER_RANGES[param_name][0] + np.log10(0.25),
                                             upper = PDU_WT_ENZ_NUMBERS_PARAMETER_RANGES[param_name][1] + np.log10(1.5))
                           for param_name in ENZYME_CONCENTRATIONS]

        variables = [*permeability_cell_params, *permeability_mcp_params, *kinetic_params, *mcp_geometry_params,
                     *cofactor_params, *enzyme_init_WT, *enzyme_init_dAJ]

        # convert m and c to a tensor vector
        theta = at.as_tensor_variable(variables)
        k4PduCDE = pm.Deterministic('k4PduCDE', variables[MODEL_PARAMETERS.index('k1PduCDE')]
                                    + variables[MODEL_PARAMETERS.index('k3PduCDE')] - variables[MODEL_PARAMETERS.index('KeqPduCDE')]
                                    - variables[MODEL_PARAMETERS.index('k2PduCDE')])

        pm.Deterministic('kcat_PduCDE_f', variables[MODEL_PARAMETERS.index('k3PduCDE')])
        pm.Deterministic('kcat_PduCDE_Glycerol', np.log10((np.power(10, variables[MODEL_PARAMETERS.index('k2PduCDE')])
                                                           + np.power(10,variables[MODEL_PARAMETERS.index('k3PduCDE')]))/np.power(10,variables[MODEL_PARAMETERS.index('k1PduCDE')])))
        pm.Deterministic('kcat_PduCDE_r', variables[MODEL_PARAMETERS.index('k2PduCDE')] )
        pm.Deterministic('kcat_PduCDE_HPA', np.log10((np.power(10, variables[MODEL_PARAMETERS.index('k2PduCDE')])
                                                      + np.power(10,variables[MODEL_PARAMETERS.index('k3PduCDE')]))/np.power(10,k4PduCDE)))


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
    mxsteps = int(float(sys.argv[9]))
    init = sys.argv[10]

    seed = int(time.time() * 1e6)
    seed = ((seed & 0xff000000) >> 24) + ((seed & 0x00ff0000) >> 8) + ((seed & 0x0000ff00) << 8) + (
            (seed & 0x000000ff) << 24)
    random_seed = seed + np.array(list(range(nchains)))
    random_seed = list(random_seed.astype(int))
    print('seed: ' + str(random_seed))
    start_val = None

    print(sys.argv)

    idata_nuts = sample(nsamples, burn_in, nchains, acc_rate=acc_rate, fwd_rtol=fwd_rtol, fwd_atol=fwd_atol,
                        bck_rtol=bck_rtol, bck_atol=bck_atol, mxsteps=mxsteps, init=init, initvals=start_val,
                        random_seed=random_seed)

    # save samples
    PARAMETER_SAMP_PATH = ROOT_PATH + '/samples'
    directory_name = 'nsamples_' + str(nsamples) + '_burn_in_' + str(burn_in) + '_acc_rate_' + str(acc_rate) + \
                     '_nchains_' + str(nchains) + '_fwd_atol_' + str(fwd_rtol) + '_fwd_rtol_' + str(fwd_rtol) \
                     + '_bck_atol_' + str(bck_rtol) + '_bck_rtol_' + str(fwd_rtol) + '_mxsteps_' + str(mxsteps) +\
                     '_initialization_' + init
    directory_name = directory_name.replace('.','_').replace('-','_').replace('+','_')
    date_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f") + '.nc'
    file_name = date_string
    sample_file_location = os.path.join(PARAMETER_SAMP_PATH, directory_name)
    Path(sample_file_location).mkdir(parents=True, exist_ok=True)
    idata_nuts.to_netcdf(os.path.join(sample_file_location,date_string))

