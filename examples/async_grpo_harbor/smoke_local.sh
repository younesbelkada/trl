#!/usr/bin/env bash
# Smoke run of AsyncGRPO + Harbor on a local 2-GPU box, reusing launcher.py (server + vLLM + trainer).
#
# Usage (from the TRL repo root):
#   export E2B_API_KEY=...
#   bash examples/async_grpo_harbor/smoke_local.sh
#
# HF_TOKEN is only needed for gated/private models or datasets, or --trackio-space-id. The launcher sets
# HF_HOME=$OUT/hf-cache unless HF_HOME is already set, which hides a cached `hf auth login`.
#
# Override any knob via env, e.g. MODEL=Qwen/Qwen3.5-0.8B GPU_VLLM=2 GPU_TRAIN=3 bash ...
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.5-2B}"
GPU_VLLM="${GPU_VLLM:-0}"
GPU_TRAIN="${GPU_TRAIN:-1}"
MAX_STEPS="${MAX_STEPS:-2}"
DATASET="${DATASET:-AdithyaSK/data_agent_rl_environment_train}"
N_TASKS="${N_TASKS:-4}"
SANDBOX_MEMORY_MB="${SANDBOX_MEMORY_MB:-4096}"  # tasks ship 1024, too little for OpenCode + pandas; the blog used 4 GB
OUT="${OUT:-$PWD/smoke-out}"
OPENENV_REVISION=86a180ede21e044f7929b9a7783ad83aa67d83a3

# 1. Environment: Python 3.12 venv with this TRL checkout, pinned OpenEnv + harbor extras, vLLM.
#    The install runs every time (a no-op when satisfied) so an existing .venv missing an extra gets fixed.
command -v uv >/dev/null || pip install uv
[ -d .venv ] || uv venv --python 3.12 .venv
# shellcheck disable=SC1091
source .venv/bin/activate
# OpenEnv's harbor extra is `harbor[e2b,...]>=0.22.0; python_version >= '3.12'`: it silently installs nothing below 3.12.
python -c 'import sys; assert sys.version_info >= (3, 12), f"need Python 3.12+, .venv has {sys.version}; rm -rf .venv"'
[ -d OpenEnv ] || git clone https://github.com/huggingface/OpenEnv.git OpenEnv
git -C OpenEnv checkout --quiet --detach "$OPENENV_REVISION"
uv pip install -e . trackio "datasets>=3.2" "./OpenEnv[harbor]" \
    "transformers>=5.2" "huggingface_hub>=1.22"

uv pip install vllm
uv pip install flashinfer-jit-cache==0.6.13 --index-url https://flashinfer.ai/whl/cu130
uv pip install kernels==0.16.0

# 2. Run the launcher locally. It falls back to ./out when --data-root is not a directory, so create it.
#    The training script is read from this checkout (file://) instead of GitHub main.
#    Unknown flags (--max-steps, --n-tasks, --num-generations) are forwarded to async_grpo_harbor.py.
#    Logs: $OUT/logs/{openenv-server,vllm,warm-template}.log; checkpoints: $OUT/runs/<run-name>.
mkdir -p "$OUT"
# trackio defaults to $HF_HOME/trackio and the launcher moves HF_HOME into $OUT, so pin it explicitly.
export TRACKIO_DIR="$OUT/trackio"
echo "dashboard: TRACKIO_DIR=$TRACKIO_DIR trackio show --project async-grpo-harbor"


# 3. Download only the first $N_TASKS tasks instead of the whole repo (~13k files). OpenEnv accepts a local
#    directory of `tasks/<name>/` as a dataset and indexes tasks in sorted order, same as for the Hub repo.
SUBSET="$OUT/tasks-subset-$N_TASKS"
python - "$DATASET" "$N_TASKS" "$SUBSET" "$SANDBOX_MEMORY_MB" <<'EOF'
import re
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

repo, n, target, memory_mb = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
names = sorted(e.path for e in HfApi().list_repo_tree(repo, path_in_repo="tasks", repo_type="dataset"))[:n]
# local_dir writes real files: Harbor tars tests/ into the sandbox and HF cache symlinks would dangle.
snapshot_download(repo, repo_type="dataset", allow_patterns=[f"{p}/**" for p in names], local_dir=target)
# The grader treats OPENAI_API_KEY (its LLM judge) as optional, but `${VAR}` makes Harbor require it on the host.
# `${VAR:-}` gives it an empty default: the judge runs when the key is exported and is skipped otherwise.
for p in names:
    toml = Path(target, p, "task.toml")
    text = toml.read_text().replace('"${OPENAI_API_KEY}"', '"${OPENAI_API_KEY:-}"')
    # OpenEnv passes no memory override to Harbor, so the sandbox size comes from task.toml.
    text = re.sub(r"^memory_mb = .*$", f"memory_mb = {memory_mb}", text, flags=re.M)
    toml.write_text(text)
print(f"downloaded {len(names)} tasks to {target}")
EOF

# 4. Build each task's E2B template once, up front. A group's rollouts otherwise race to build the same template on
#    the task's first group and cancel each other. Templates persist on E2B, so later runs skip existing ones.
#    The template name ignores task.toml, but its CPU/memory are fixed at build time: after changing
#    SANDBOX_MEMORY_MB, run once with PREBUILD_FORCE=1 so existing templates are rebuilt at the new size.
python examples/async_grpo_harbor/prebuild_e2b_templates.py "$SUBSET" ${PREBUILD_FORCE:+--force}

# PREP_ONLY=1 stops here: env, task subset and templates are ready, e.g. on a login node before a Slurm job.
[ -n "${PREP_ONLY:-}" ] && exit 0

python examples/async_grpo_harbor/launcher.py \
    --split "$SUBSET" \
    --model "$MODEL" \
    --vllm-device "$GPU_VLLM" \
    --train-device "$GPU_TRAIN" \
    --data-root "$OUT" \
    --train-script-url "file://$PWD/examples/async_grpo_harbor/async_grpo_harbor.py" \
    --max-model-len 32768 \
    --max-inflight 2 \
    --tunnel-check-s 0 \
    --max-steps "$MAX_STEPS" \
    --n-tasks "$N_TASKS" \
    --num-generations 4 \
    "$@"
