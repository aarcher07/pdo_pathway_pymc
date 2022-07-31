#!/bin/bash
#SBATCH --account=b1020
#SBATCH --partition=b1020
#SBATCH --nodes=1
#SBATCH --array=1-12
#SBATCH --ntasks=2
#SBATCH --time=07-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andrearcher2017@u.northwestern.edu
#SBATCH --mem-per-cpu=15GB
#SBATCH --job-name=pdu_constant_cyto_dAJ_%A_%a
#SBATCH --output=out/pdu_constant_cyto_dAJ_out_%A_%a
#SBATCH --error=err/pdu_constant_cyto_dAJ_err_%A_%a


nsamples=(2e3 5e3)
burn_in=(2e3 5e3)
nchains=2
acc_rate=(0.75 0.8 0.85)
fwd_rtol=(1e-10)
fwd_atol=(1e-10)
bck_rtol=(1e-4)
bck_atol=(1e-4)
fwd_mxsteps=1e5
bck_mxsteps=1e5
init=("adapt_diag")

len_burn_in=${#burn_in[@]}
len_acc_rate=${#acc_rate[@]}
len_nsamples=${#nsamples[@]}
len_fwd_rtol=${#fwd_rtol[@]}
len_fwd_atol=${#fwd_atol[@]}
len_bck_rtol=${#bck_rtol[@]}
len_bck_atol=${#bck_atol[@]}
len_init=${#init[@]}

sublen_init_burn_in=$(($len_init * $len_burn_in))
sublen_init_burn_in_acc_rate=$(($len_init * $len_burn_in * $len_acc_rate))
sublen_init_burn_in_acc_rate_nsamples=$(($len_init *  $len_burn_in * $len_acc_rate * $len_nsamples))
sublen_init_burn_in_acc_rate_nsamples_fwd_rtol=$(($len_init * $len_burn_in * $len_acc_rate * $len_nsamples * $len_fwd_rtol))
sublen_init_burn_in_acc_rate_nsamples_fwd_rtol_fwd_atol=$(($len_init * $len_burn_in * $len_acc_rate * $len_nsamples * $len_fwd_rtol * $len_fwd_atol))
sublen_init_burn_in_acc_rate_nsamples_fwd_rtol_fwd_atol_bck_rtol=$(($len_init * $len_burn_in * $len_acc_rate * $len_nsamples * $len_fwd_rtol * $len_fwd_atol * $len_bck_rtol))

zero_index=$(( $SLURM_ARRAY_TASK_ID-1))

module purge all
module load texlive/2020

python sample.py "${nsamples[(($zero_index / $sublen_init_burn_in_acc_rate) % $len_nsamples)]}" "${burn_in[($zero_index / $len_init) % $len_burn_in]}" "${nchains}" "${acc_rate[($zero_index / $sublen_init_burn_in) % $len_acc_rate]}" "${fwd_rtol[(($zero_index / $sublen_init_burn_in_acc_rate_nsamples) % $len_fwd_rtol)]}" "${fwd_atol[(($zero_index / $sublen_init_burn_in_acc_rate_nsamples_fwd_rtol) % $len_fwd_atol)]}" "${bck_rtol[(($zero_index / $sublen_init_burn_in_acc_rate_nsamples_fwd_rtol_fwd_atol) % $len_bck_rtol)]}" "${bck_atol[(($zero_index / $sublen_init_burn_in_acc_rate_nsamples_fwd_rtol_fwd_atol_bck_rtol) % $len_bck_atol)]}" "${fwd_mxsteps}" "${bck_mxsteps}" "${init[$zero_index % $len_init]}"
