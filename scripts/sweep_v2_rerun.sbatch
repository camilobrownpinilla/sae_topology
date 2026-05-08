#!/bin/bash
#SBATCH --job-name=sweep_v2_rerun
#SBATCH --partition=mweber_compute
#SBATCH --account=mweber_lab
#SBATCH --array=0-16%16
#SBATCH --exclude=holy8a14402
#SBATCH --cpus-per-task=24
#SBATCH --mem=180G
#SBATCH --time=03:00:00
#SBATCH --output=logs/sweep_v2_rerun/%A_%a.out
#SBATCH --error=logs/sweep_v2_rerun/%A_%a.err
#SBATCH --mail-type=ALL
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

MANIFEST=configs/sweep_v2_rerun/manifest.tsv
ROW=$(awk -v i="$SLURM_ARRAY_TASK_ID" -F'\t' 'NR == i + 2' "$MANIFEST")
if [ -z "$ROW" ]; then
  echo "ERROR: no manifest row for index $SLURM_ARRAY_TASK_ID" >&2
  exit 1
fi

RUN_ID=$(echo "$ROW" | cut -f2)
CFG=$(echo "$ROW" | cut -f3)
OUT=$(echo "$ROW" | cut -f4)

echo "[sweep_v2_rerun] task=$SLURM_ARRAY_TASK_ID run_id=$RUN_ID"
echo "[sweep_v2_rerun] config=$CFG"
echo "[sweep_v2_rerun] out_dir=$OUT"
echo "[sweep_v2_rerun] node=$(hostname)"

mkdir -p "$OUT"

python -u -m sae_topology.experiments.stage1_bundle \
  --config "$CFG" \
  --out_dir "$OUT" \
  --n_jobs "$SLURM_CPUS_PER_TASK" \
  --baseline_path configs/stage0/baseline.yaml \
  --spectral_K 25
