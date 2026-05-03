#!/bin/bash
#SBATCH --job-name=render_modal
#SBATCH --partition=sapphire
#SBATCH --account=mweber_lab
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=logs/render_modal/%j.out
#SBATCH --error=logs/render_modal/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=camilobrownpinilla@college.harvard.edu

set -euo pipefail

module load python/3.10.13-fasrc
conda deactivate
conda activate sae_topology
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd /n/holylabs/mweber_lab/Everyone/cbrownpinilla/sae_topology

RESULTS_ROOT=${1:-results/sweep_v1}
echo "[render_modal] results_root=$RESULTS_ROOT"
echo "[render_modal] cores=$SLURM_CPUS_PER_TASK node=$(hostname)"

mkdir -p logs/render_modal

python -u -m scripts.render_modal_mapper \
  --results_root "$RESULTS_ROOT" \
  --n_jobs "$SLURM_CPUS_PER_TASK"
