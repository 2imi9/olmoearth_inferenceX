#!/bin/bash
#SBATCH -A p2026_0089_neu
#SBATCH -p b200-batch
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=160G
#SBATCH -t 01:30:00
#SBATCH -J vllm38
#SBATCH -o /home/qi_zim_neu/olmoearth_inferenceX/slurm/vllm38-%j.out
set -u
export HF_HOME=/scratch/qi_zim_neu/olmoearth_inferenceX/hf
unset HF_HUB_OFFLINE
VENV=/scratch/qi_zim_neu/olmoearth_inferenceX/agentvenv
UV=$HOME/.local/bin/uv
MODEL=nvidia/Qwen3.8-27B-NVFP4
PORT=8077
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader

serve() {  # $1 = attempt label
  LOG=/home/qi_zim_neu/olmoearth_inferenceX/slurm/vllm38-$SLURM_JOB_ID-$1.log
  echo "=== attempt $1: vllm serve ==="
  $VENV/bin/vllm serve $MODEL --port $PORT --tensor-parallel-size 1 --max-model-len 65536 \
     --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
     --seed 0 --gpu-memory-utilization 0.85 --max-num-seqs 8 --no-enable-log-requests > $LOG 2>&1 &
  VPID=$!
  for i in $(seq 1 90); do
    curl -sf http://localhost:$PORT/v1/models >/dev/null 2>&1 && { echo "attempt $1: endpoint up after ${i}0s"; return 0; }
    kill -0 $VPID 2>/dev/null || break
    sleep 10
  done
  kill $VPID 2>/dev/null; wait $VPID 2>/dev/null
  echo "attempt $1 failed; diagnostic lines:"; grep -iE "error|nvcc|ninja|not found|unsupported|No such|Traceback" $LOG | grep -v "^INFO" | head -12
  return 1
}

probe() {
  echo "=== tool-call round trip ==="
  curl -s http://localhost:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d '{
   "model": "'"$MODEL"'", "temperature": 0, "max_tokens": 300,
   "messages": [{"role":"user","content":"Rank this card. Call the assess tool with budget 0.05, then answer."}],
   "tools": [{"type":"function","function":{"name":"assess","description":"Rank windows by suspicion.",
      "parameters":{"type":"object","properties":{"budget":{"type":"number"}},"required":["budget"]}}}],
   "tool_choice": "auto"}' | $VENV/bin/python -c "import sys,json; m=json.load(sys.stdin)['choices'][0]['message']; print(json.dumps({k:m.get(k) for k in ('content','tool_calls','reasoning_content')}, indent=1)[:800])"
}

if serve native; then probe; kill $VPID; exit 0; fi

CU=$($VENV/bin/python -c "import torch;print('cu'+torch.version.cuda.replace('.',''))")
echo "=== installing prebuilt flashinfer kernels for $CU ==="
cd $VENV/.. && $UV pip install --python $VENV/bin/python -q flashinfer-cubin flashinfer-jit-cache --extra-index-url https://flashinfer.ai/whl/$CU/ 2>&1 | tail -3
if serve prebuilt-kernels; then probe; kill $VPID; exit 0; fi

echo "=== installing nvcc + CUDA headers from pip wheels, synthetic CUDA_HOME ==="
$UV pip install --python $VENV/bin/python -q ninja nvidia-cuda-nvcc-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-cccl-cu12 nvidia-cuda-nvrtc-cu12 2>&1 | tail -3
SITE=$($VENV/bin/python -c "import nvidia, os; print(os.path.dirname(nvidia.__path__[0]))")/nvidia
CH=/scratch/qi_zim_neu/olmoearth_inferenceX/cuda_home; rm -rf $CH; mkdir -p $CH/include $CH/lib64
ln -sfn $SITE/cuda_nvcc/bin $CH/bin
for d in cuda_runtime cuda_cccl cuda_nvrtc cuda_nvcc; do [ -d $SITE/$d/include ] && cp -rn $SITE/$d/include/. $CH/include/ 2>/dev/null; [ -d $SITE/$d/lib ] && cp -rn $SITE/$d/lib/. $CH/lib64/ 2>/dev/null; done
export CUDA_HOME=$CH PATH=$CH/bin:$PATH LD_LIBRARY_PATH=$CH/lib64:${LD_LIBRARY_PATH:-}
$CH/bin/nvcc --version | tail -1
if serve pip-nvcc; then probe; kill $VPID; exit 0; fi
echo "=== all three attempts failed ==="; exit 1
