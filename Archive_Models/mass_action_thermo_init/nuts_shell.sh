#!/bin/bash
#SBATCH --account=b1020
#SBATCH --partition=b1020
#SBATCH --nodes=1
#SBATCH --array=1-8
#SBATCH --ntasks=6
#SBATCH --time=05-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andrearcher2017@u.northwestern.edu
#SBATCH --mem-per-cpu=15GB
#SBATCH --job-name=mass_action_thermo_init_%A_%a
#SBATCH --output=out/mass_action_thermo_init_out_%A_%a
#SBATCH --error=err/mass_action_thermo_init_err_%A_%a

nsamples=(3e3 1e4)
burn_in=(3e3 4e3)
nchains=2
acc_rate=(0.8)
atol=(1e-8)
rtol=(1e-8)
mxsteps=1e5
init=("jitter+adapt_diag")

len_burn_in=${#burn_in[@]}
len_acc_rate=${#acc_rate[@]}
len_nsamples=${#nsamples[@]}
len_atol=${#atol[@]}
len_rtol=${#rtol[@]}
len_init=${#init[@]}

sublen_init_burn_in=$(($len_init * $len_burn_in))
sublen_init_burn_in_acc_rate=$(($len_init * $len_burn_in * $len_acc_rate))
sublen_init_burn_in_acc_rate_nsamples=$(($len_init *  $len_burn_in * $len_acc_rate * $len_nsamples))
sublen_init_burn_in_acc_rate_nsamples_atol=$(($len_init * $len_burn_in * $len_acc_rate * $len_nsamples * $len_atol))
zero_index=$(( $SLURM_ARRAY_TASK_ID-1))

module purge all
module load texlive/2020

python pymc_funcs.py "${nsamples[(($zero_index / $sublen_init_burn_in_acc_rate) % $len_nsamples)]}" "${burn_in[($zero_index / $len_init) % $len_burn_in]}" "${nchains}" "${acc_rate[($zero_index / $sublen_init_burn_in) % $len_acc_rate]}" "${atol[(($zero_index / $sublen_init_burn_in_acc_rate_nsamples) % $len_atol)]}" "${rtol[(($zero_index / $sublen_init_burn_in_acc_rate_nsamples_atol) % $len_rtol)]}" "${mxsteps}" "${init[$zero_index % $len_init]}"