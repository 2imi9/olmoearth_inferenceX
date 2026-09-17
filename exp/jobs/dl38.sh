#!/bin/bash
#SBATCH -A p2026_0089_neu
#SBATCH -p cpu
#SBATCH -c 8
#SBATCH --mem=8G
#SBATCH -t 04:00:00
#SBATCH -J dl38
#SBATCH -o /home/qi_zim_neu/olmoearth_inferenceX/slurm/dl38-%j.out
export HF_HOME=/scratch/qi_zim_neu/olmoearth_inferenceX/hf
unset HF_HUB_OFFLINE
df -h /scratch/qi_zim_neu | tail -1
/scratch/qi_zim_neu/olmoearth_inferenceX/agentvenv/bin/python - <<'PY'
import time
from huggingface_hub import snapshot_download
t=time.time()
p=snapshot_download("Qwen/Qwen3.8-27B", max_workers=8,
                    allow_patterns=["*.json","*.safetensors","*.txt","*.jinja","*.py","*.md"])
print("downloaded to", p, f"in {(time.time()-t)/60:.1f} min")
PY
du -sh /scratch/qi_zim_neu/olmoearth_inferenceX/hf/hub/models--Qwen--Qwen3.8-27B
