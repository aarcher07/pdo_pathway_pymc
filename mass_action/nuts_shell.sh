#!/bin/bash
#SBATCH --account=b1020
#SBATCH --partition=b1020
#SBATCH --nodes=1
#SBATCH --array=1-12
#SBATCH --ntasks=2
#SBATCH --time=05-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andrearcher2017@u.northwestern.edu
#SBATCH --mem-per-cpu=10GB
#SBATCH --job-name=mass_action_%A_%a
#SBATCH --output=out/mass_action_out_%A_%a
#SBATCH --error=err/mass_action_err_%A_%a

nsamples=(5e5 1e4)
burn_in=(5e5 1e4)
nchains=2
acc_rate=(0.8 0.85 0.9)
tol=1e-8
mxsteps=5e4

len_nsamples=${#nsamples[@]}
len_burn_in=${#burn_in[@]}
len_acc_rate=${#acc_rate[@]}

sublen_burn_in_acc_rate=$(( $len_burn_in * $len_acc_rate))

zero_index=$(( $SLURM_ARRAY_TASK_ID-1))

module purge all
module load texlive/2020

python pymc_funcs.py "${nsamples[(($zero_index / $sublen_burn_in_acc_rate) % $len_nsamples)]}" "${burn_in[($zero_index / $len_acc_rate) % $len_burn_in]}" "${nchains}" "${acc_rate[$zero_index % $len_acc_rate]}" "${tol}" "${mxsteps}"