#!/bin/bash
#SBATCH --job-name=calibrate_refine
#SBATCH --partition=mweber_compute
#SBATCH --account=mweber_lab
#SBATCH --array=0-5%6
#SBATCH --exclude=holy8a14402
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/calibration_fine/refine_%A_%a.out
#SBATCH --error=logs/calibration_fine/refine_%A_%a.err

set -euo pipefail

module load python/3.10.13-fasrc
conda deactivate
conda activate sae_topology
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd /n/holylabs/mweber_lab/Everyone/cbrownpinilla/sae_topology

MANIFEST=configs/calibration_refine_manifest.tsv
ROW=$(awk -v i="$SLURM_ARRAY_TASK_ID" -F'\t' 'NR == i + 2' "$MANIFEST")
if [ -z "$ROW" ]; then
  echo "ERROR: no manifest row for index $SLURM_ARRAY_TASK_ID" >&2
  exit 1
fi

TOPO=$(echo "$ROW" | cut -f2)
ARCH=$(echo "$ROW" | cut -f3)
COEFF=$(echo "$ROW" | cut -f4)
OUT=$(echo "$ROW" | cut -f5)

echo "[refine] task=$SLURM_ARRAY_TASK_ID topo=$TOPO arch=$ARCH coeff=$COEFF"
python -u -m scripts.calibrate_one \
  --topology "$TOPO" --arch "$ARCH" --coeff "$COEFF" \
  --n_steps 50000 --seed 0 --out_path "$OUT"
