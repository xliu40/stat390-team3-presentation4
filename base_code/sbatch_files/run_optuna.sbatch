#!/bin/bash
#SBATCH --account=e32998                  # Use your Quest project account (check with 'groups')
#SBATCH --partition=long                  # 'short', 'normal', or 'long' partition
#SBATCH --time=96:00:00                    # max run time
#SBATCH --nodes=1                         # 1 node
#SBATCH --ntasks=16                        # n nores
#SBATCH --mem=128G                          # n GB RAM
#SBATCH --job-name=parallel1              # Job name (no spaces)
#SBATCH --output=mar12_tuning_parallel1.out
#SBATCH --error=mar12_tuning_parallel1.err
#SBATCH --mail-type=ALL                   # Send email on begin, end, and fail
#SBATCH --mail-user=xiaolinliu2026@u.northwestern.edu  # Your NU email

module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env   # you might want to copy the env and put it in your new project directory, if different from e32998

# Print system info
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

# Run training
python3 -m pip install -r /projects/e32998/STAT390_Winter2026/Presentation4/Team3/base_code/requirements.txt
python3 -m pip install --upgrade certifi
export SSL_CERT_FILE="$(python -m certifi)"

cd /projects/e32998/STAT390_Winter2026/Presentation4/Team3/base_code

# without loading checkpoints, trying 5 diff train / test splits

mkdir -p /projects/e32998/MIL_training/optuna_5fold_runs/

python3 -u optuna_training.py \
    --study_name optuna_presentation4_mar12_parallel1 \
    --output_dir /projects/e32998/STAT390_Winter2026/Presentation4/Team3/base_code/optuna_results_team3 \
    --storage sqlite:////projects/e32998/MIL_training/optuna_5fold_runs/optuna_5fold.db \
    --splits_dir /projects/e32998/STAT390_Winter2026/Presentation4/Team3/base_code/no_train_val_leakage_splits \
    --splits_pattern 'data_splits_new_0*.npz' \
    --n_trials 64 \
    --sobol_startup_trials 16 \
    --epochs 30 \
    --pruning_warmup_epochs 6 \
    --no_prune_first_trials 6 \
    --batch_size 1 \
    --num_workers 2 \
    --per_slice_cap 500 \
    --max_slices_per_stain 5 \
    --seed 42 \

echo "End time: $(date)"