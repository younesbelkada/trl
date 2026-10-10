#!/bin/bash -l
# Four AsyncGRPO + Harbor runs on one 8-GPU node, one harness per run, 2 GPUs each (vLLM + trainer).
#
# Once, on a login node, from the TRL repo root (venv, task subset, E2B templates):
#   export E2B_API_KEY=...
#   DATASET=FineEnvs/SmolDataEnvs-harbor-train N_TASKS=256 OUT=/fsx/$USER/harbor-multi PREP_ONLY=1 \
#       bash examples/async_grpo_harbor/smoke_local.sh
#   venv-ship $PWD/.venv
#
# Then, from the same directory (E2B_API_KEY is taken from the submitting shell; never pass --export on hopper-atl):
#   sbatch examples/async_grpo_harbor/slurm_multi_harness.sh
#   HARNESSES="opencode openclaw opencode openclaw" sbatch examples/async_grpo_harbor/slurm_multi_harness.sh
#
# Per run (i = 0..3): GPUs 2i (vLLM) and 2i+1 (trainer), ports 8000/8200/8300 + 10*i, everything under $ROOT/<i>-<harness>:
#   launcher.log, logs/{openenv-server,vllm,warm-template}.log, trackio/, runs/<run-name>/checkpoint-*.
#SBATCH --job-name=harbor-4x
#SBATCH --partition=hopper-atl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=88
#SBATCH --time=2-00:00:00
#SBATCH --output=/fsx/%u/logs/%x-%j.out
set -euo pipefail

HARNESSES=(${HARNESSES:-opencode openclaw pi openhands})
MODEL="${MODEL:-Qwen/Qwen3.5-2B}"
N_TASKS="${N_TASKS:-256}"
MAX_STEPS="${MAX_STEPS:-100}"
MAX_INFLIGHT="${MAX_INFLIGHT:-32}"  # E2B sandboxes per run: 4 runs use 4x this, check your E2B concurrency limit
ROOT="${ROOT:-/fsx/$USER/harbor-multi}"
SUBSET="$ROOT/tasks-subset-$N_TASKS"
REPO="$SLURM_SUBMIT_DIR"

[ -n "${E2B_API_KEY:-}" ] || { echo "E2B_API_KEY is unset" >&2; exit 1; }
[ -d "$SUBSET" ] || { echo "$SUBSET missing, run the PREP_ONLY step first" >&2; exit 1; }

# hopper-atl reads /fsx from ~15 ms away: run the venv from node-local /scratch (see `venv-ship` above).
source "$(venv-load "$REPO/.venv")"
export PYTHONPATH="$REPO/OpenEnv/envs${PYTHONPATH:+:$PYTHONPATH}"
# Download the model once so the 4 vLLM servers do not race on the same HF cache entry.
python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL')"

trap 'kill $(jobs -p) 2>/dev/null; wait' TERM INT
for i in "${!HARNESSES[@]}"; do
    h="${HARNESSES[$i]}"
    run_dir="$ROOT/runs-$SLURM_JOB_ID/$i-$h"
    mkdir -p "$run_dir"
    # The launcher downloads the training script into its cwd, so each run gets its own.
    (
        cd "$run_dir"
        TRACKIO_DIR="$run_dir/trackio" python "$REPO/examples/async_grpo_harbor/launcher.py" \
            --split "$SUBSET" \
            --model "$MODEL" \
            --harness "$h" \
            --vllm-device "$((2 * i))" \
            --train-device "$((2 * i + 1))" \
            --vllm-port "$((8000 + 10 * i))" \
            --server-port "$((8200 + 10 * i))" \
            --capture-port "$((8300 + 10 * i))" \
            --data-root "$run_dir" \
            --train-script-url "file://$REPO/examples/async_grpo_harbor/async_grpo_harbor.py" \
            --tunnel-check-s 0 \
            --max-inflight "$MAX_INFLIGHT" \
            --max-model-len 131072 \
            --max-steps "$MAX_STEPS" \
            --n-tasks "$N_TASKS" \
            --num-generations 8 \
            --learning-rate 3e-6 --temperature 0.8 \
            --max-completion-length 16384 \
            --agent-step-limit 17 --agent-timeout 600 \
            --per-device-train-batch-size 4 --gradient-accumulation-steps 4 \
            > "$run_dir/launcher.log" 2>&1
    ) &
    echo "run $i: $h on GPUs $((2 * i)),$((2 * i + 1)) -> $run_dir"
    sleep 30  # stagger vLLM startups and the first burst of sandboxes
done

rc=0
for pid in $(jobs -p); do wait "$pid" || rc=1; done
exit "$rc"
