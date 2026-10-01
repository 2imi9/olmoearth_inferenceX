#!/bin/bash
# exp89 (docs/plan/finetuned_checkpoints.md): Ai2's fine-tuned Mangrove model (arm M), FT-AWF (arm A, P2 and P3 only),
# Nandi (arm N, frozen and not run: the Hub answers 401). One script; the mode and arm come from the environment.
#
#   E89_MODE=inv    the inventory: downloads the pinned checkpoint, tar and geojson to scratch, checks their sha256,
#                   extracts the tar, lists the checkpoint keys and counts windows and labels per split. No model.
#   E89_MODE=smoke  S1 (numpy), the torch smoke on a synthetic tar and checkpoint, then S2: the pinned checkpoint on
#                   512 real TRAINING windows (in-sample, never graded). Allowed before freezing.
#   E89_MODE=gate   G: accuracy only on the validation windows. The script refuses it until the page is frozen.
#   E89_MODE=run    the full run. Refused until the page is frozen AND exp89_gate_<arm>.json records a pass.
#   E89_ARM=mangrove|awf|nandi (default mangrove); nandi is reported as not run while the Hub answers 401.
#
# Submit from the Mac; the login node runs one sbatch and nothing else. The script travels on stdin, and the job resets
# the checkout to origin/main, so commit and push first. The partition is given per mode: the inventory on cpu, every
# model pass on a GPU partition, as scripts/score_area.sh:
#   ssh aicr 'E89_MODE=inv E89_ARM=mangrove sbatch --parsable -p cpu -t 01:00:00' < exp/jobs/e89.sh
#   ssh aicr 'E89_MODE=smoke E89_ARM=mangrove sbatch --parsable -p b200-devel,rtx-devel -q interactive --gpus=1 -t 00:45:00' < exp/jobs/e89.sh
#   ssh aicr 'E89_MODE=gate E89_ARM=mangrove sbatch --parsable -p b200-devel,rtx-devel -q interactive --gpus=1 -t 01:00:00' < exp/jobs/e89.sh
#   ssh aicr 'E89_MODE=run E89_ARM=mangrove sbatch --parsable -p b200-devel,rtx-devel -q interactive --gpus=1 -t 03:00:00 --dependency=afterok:GATE_JOB' < exp/jobs/e89.sh
#
# Every mode writes tracked files in exp/out, and the hard reset below restores tracked files: never let two of these
# jobs, or this and another job that writes tracked outputs, overlap (chain with --dependency=afterany).
#SBATCH -A p2026_0089_neu
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -J e89ft
#SBATCH -o /home/qi_zim_neu/olmoearth_inferenceX/slurm/%x-%j.out
set -u
MODE=${E89_MODE:-inv}
ARM=${E89_ARM:-mangrove}
cd /home/qi_zim_neu/olmoearth_inferenceX || exit 1
git fetch -q origin main; git reset -q --hard origin/main
SCRATCH=/scratch/qi_zim_neu/olmoearth_inferenceX
export PYTHONUNBUFFERED=1
export HF_HOME=$SCRATCH/hf                 # every download (checkpoint, tar, geojson, encoder config) lands on scratch
export UV_CACHE_DIR=$SCRATCH/uv-cache
export E89_DATA=$SCRATCH/data/exp89        # each arm's tar is extracted under $E89_DATA/<arm>
export PATH="$HOME/.local/bin:$PATH"
if [ "$ARM" = "awf" ]; then DATA=$SCRATCH/data/awf; else DATA=$E89_DATA/$ARM; fi
RUN="uv run --extra encoder --extra geo"
PY="$RUN python exp/exp89_finetuned_checkpoints.py"

echo "== $(date -Is) job ${SLURM_JOB_ID:-none} on $(hostname): mode $MODE, arm $ARM, data $DATA, commit $(git rev-parse --short HEAD) =="
case "$MODE" in
  smoke|gate|run)
    if ! nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; then
      echo "mode $MODE runs the model and needs a GPU: submit with -p b200-devel,rtx-devel -q interactive --gpus=1"
      exit 2
    fi ;;
esac
$RUN python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || exit 1
mkdir -p "$E89_DATA"

case "$MODE" in
  inv)
    echo "== S1 =="
    $PY --smoke --out-dir "$SCRATCH/exp89_smoke" || exit 1
    echo "== inventory (no model on any window) =="
    $PY --inventory --arm "$ARM" --data "$DATA" || exit 1
    ls -la "exp/out/exp89_inventory_$ARM.json" ;;
  smoke)
    echo "== S1 =="
    $PY --smoke --out-dir "$SCRATCH/exp89_smoke" || exit 1
    echo "== the torch smoke on a synthetic tar and checkpoint =="
    $PY --smoke-torch --out-dir "$SCRATCH/exp89_smoke" || exit 1
    echo "== S2: the pinned checkpoint on real training windows =="
    $PY --smoke-torch --real --arm "$ARM" --data "$DATA"
    rc=$?
    ls -la exp/out/exp89_s2_"$ARM".json 2>/dev/null
    exit $rc ;;
  gate)
    echo "== G: accuracy only =="
    $PY --gate --arm "$ARM" --data "$DATA"
    rc=$?
    cat exp/out/exp89_gate_"$ARM".json 2>/dev/null
    exit $rc ;;
  run)
    echo "== the full run =="
    $PY --arm "$ARM" --data "$DATA"
    rc=$?
    ls -la exp/out/exp89_summary.json exp/out/exp89_units_"$ARM".npz 2>/dev/null
    exit $rc ;;
  *)
    echo "unknown E89_MODE '$MODE': inv, smoke, gate or run"; exit 2 ;;
esac
