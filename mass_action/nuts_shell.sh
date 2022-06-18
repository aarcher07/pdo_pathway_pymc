#!/bin/bash
#SBATCH --account=b1020
#SBATCH --partition=b1020
#SBATCH --nodes=1
#SBATCH --array=10
#SBATCH --ntasks=5
#SBATCH --time=00-01:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andrearcher2017@u.northwestern.edu
#SBATCH --mem-per-cpu=10GB
#SBATCH --job-name=mass_action_
#SBATCH --output=out/mass_action_out_%A_%a
#SBATCH --error=err/mass_action_err_%A_%a

nsamples=1e4
burn_in=1e4
nchains=2
acc_rate=0.8
tol=1e-8
mxsteps=2e4

module purge all
module load texlive/2020

python pymc_funcs.py "${nsamples}" "${burn_in}" "${nchains}" "${acc_rate}" "${tol}" "${mxsteps}"