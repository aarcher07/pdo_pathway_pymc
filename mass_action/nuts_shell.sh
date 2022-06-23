#!/bin/bash
#SBATCH --account=b1020
#SBATCH --partition=b1020
#SBATCH --nodes=1
#SBATCH --array=1-16
#SBATCH --ntasks=3
#SBATCH --time=05-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andrearcher2017@u.northwestern.edu
#SBATCH --mem-per-cpu=10GB
#SBATCH --job-name=mass_action_%A_%a
#SBATCH --output=out/mass_action_out_%A_%a
#SBATCH --error=err/mass_action_err_%A_%a

nsamples=(2 3)
burn_in=(3 4)
nchains=2
acc_rate=(0.65 0.7 )
atol=(1e-8 1e-10)
rtol=(1e-8)
mxsteps=1e5

len_nsamples=${#nsamples[@]}
len_burn_in=${#burn_in[@]}
len_acc_rate=${#acc_rate[@]}
len_atol=${#atol[@]}
len_rtol=${#rtol[@]}

sublen_burn_in_acc_rate=$(( $len_burn_in * $len_acc_rate))
sublen_burn_in_acc_rate_nsamples=$(( $len_burn_in * $len_acc_rate * $len_nsamples))
sublen_burn_in_acc_rate_nsamples_atol=$(( $len_burn_in * $len_acc_rate * $len_nsamples * $len_atol))
sublen_burn_in_acc_rate_nsamples_atol_rtol=$(( $len_burn_in * $len_acc_rate * $len_nsamples * $len_atol * $len_rtol))

zero_index=$(( $SLURM_ARRAY_TASK_ID-1))

module purge all
module load texlive/2020

python pymc_funcs.py "${nsamples[(($zero_index / $sublen_burn_in_acc_rate_nsamples) % $len_nsamples)]}" "${burn_in[$zero_index % $len_burn_in]}" "${nchains}" "${acc_rate[($zero_index / $sublen_burn_in_acc_rate) % $len_acc_rate]}" "${atol[(($zero_index / $sublen_burn_in_acc_rate_nsamples_atol) % $len_atol)]}" "${atol[(($zero_index / $sublen_burn_in_acc_rate_nsamples_atol) % $len_atol)]}" "${rtol[(($zero_index / $sublen_burn_in_acc_rate_nsamples_atol_rtol) % $len_rtol)]}" "${mxsteps}"