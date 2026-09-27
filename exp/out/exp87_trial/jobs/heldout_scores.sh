#!/bin/bash
# exp87 held-out fixtures: three new runs of Ai2's fine-tuned AWF model through scripts/score_area.py
# (same area, two new years; one new area). Outputs stay on scratch until fetched through a CPU srun.
#SBATCH -A p2026_0089_neu
#SBATCH -p b200-devel,rtx-devel
#SBATCH -q interactive
#SBATCH --gpus=1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:30:00
#SBATCH -J oeix-scores-h
#SBATCH -o /home/qi_zim_neu/olmoearth_inferenceX/slurm/%x-%j.out
set -u
cd /home/qi_zim_neu/olmoearth_inferenceX || exit 1
git fetch -q origin main; git reset -q --hard origin/main
export PYTHONUNBUFFERED=1
export HF_HOME=/scratch/qi_zim_neu/olmoearth_inferenceX/hf
export UV_CACHE_DIR=/scratch/qi_zim_neu/olmoearth_inferenceX/uv-cache
export PATH="$HOME/.local/bin:$PATH"
RUN="uv run --extra encoder --extra geo"
BASE=/scratch/qi_zim_neu/olmoearth_inferenceX/scores
echo "== $(date -Is) job ${SLURM_JOB_ID} on $(hostname), commit $(git rev-parse --short HEAD) =="
nvidia-smi --query-gpu=name --format=csv,noheader
for spec in "namanga_2021 -2.55 36.81 2021-01-01 2021-12-31" "namanga_2024 -2.55 36.81 2024-01-01 2024-12-31" "amboseli_2023 -2.65 37.26 2023-01-01 2023-12-31"; do
  set -- $spec; OUT=$BASE/awf_$1_${SLURM_JOB_ID}
  echo "== $1 =="
  $RUN python scripts/score_area.py --out "$OUT" --lat "$2" --lon "$3" --start "$4" --end "$5" || { echo "FAILED $1"; continue; }
  $RUN oe-inferencex assess "$OUT/scores.tif" --logits --out "$OUT/assess" >/dev/null && echo "assess ok $1"
  sha256sum "$OUT/scores.tif"
done
echo "== done $(date -Is) =="
