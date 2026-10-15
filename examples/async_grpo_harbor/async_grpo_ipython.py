# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "trl[harbor]",
#     "peft",
#     "trackio",
#     "ipyharness @ git+https://github.com/huggingface/ipython-harness.git@9a06fd8b62e5e9c89e8b0f37de1d3e96694ba19f",
# ]
# ///

"""AsyncGRPO on Harbor tasks with the ipython-harness (`ipyh`) tool surface, the loop driven by TRL.

[ipython-harness](https://github.com/huggingface/ipython-harness) gives the model a single `ipython` tool: a
persistent IPython kernel (shell through `!cmd` / `%%bash`, files through `%%writefile`, `edit`/`view` helpers), its
system prompt, and its output handling (ANSI and progress-bar cleanup, head+tail truncation, `# timeout: N` cells).
Here that tool runs as a `trl.experimental.harbor` external agent: TRL samples every turn itself (exact tokens and
logprobs, no capture proxy), the cells run in an IPython kernel inside the task's Harbor sandbox, and the task's
verifier is the reward.

The training loop is TRL's, not ipyh's: a reply without a tool call ends the rollout, and ipyh's nudges (on an empty or
truncated reply) are not sent. The tool, its schema, the system prompt and the output handling are ipyh's, so a model
trained here is evaluated with ipyh on the same interface.

Run (2 GPUs: vLLM on one, trainer on the other; tasks prepared as in smoke_local.sh, `--split` a local task dir):

```sh
CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3.5-0.8B --port 8000 \
    --logprobs-mode processed_logprobs --weight-transfer-config '{"backend":"nccl"}'
CUDA_VISIBLE_DEVICES=1 python examples/async_grpo_harbor/async_grpo_ipython.py \
    --vllm-url http://localhost:8000 --model Qwen/Qwen3.5-0.8B --split <tasks dir> --sandbox e2b --max-steps 20
```
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shlex
import tempfile
import time
import uuid
from pathlib import Path

from ipyh.constants import CELL_TIMEOUT_SEC, DEFAULT_SYSTEM_PROMPT, HELPERS, MAX_OUTPUT_CHARS, OUTPUT_DIR, TASK_PYTHON
from ipyh.harness import _PYTHON_LINE, _TIMEOUT_LINE
from ipyh.normalizers import QUIET_ENV, python_default_normalizer, truncate
from peft import LoraConfig
from transformers import AutoTokenizer

from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer
from trl.experimental.async_grpo.vllm_client import VLLMClient
from trl.experimental.harbor import HarborEnv, HarborSpec


# IPython kernel inside the sandbox
# From ipython-harness `integrations/harbor/kernels.py` at 9a06fd8b (Apache-2.0), which the published package does not
# ship: install a kernel venv with uv, attach it to the task's Python, and run each cell through a tiny client.
KERNEL_ROOT = "/opt/ipy"  # kernel venv (and uv), kept apart from the task's own environment
KERNEL_RUN = "/tmp/ipy"  # kernel connection file, pid, logs and cell scratch files

KERNEL_INSTALL = f"""set -euo pipefail
mkdir -p {KERNEL_ROOT}/bin {KERNEL_RUN} && chmod 777 {KERNEL_RUN}
curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL={KERNEL_ROOT}/bin sh >/dev/null
export UV_PYTHON_INSTALL_DIR={KERNEL_ROOT}/python
uv={KERNEL_ROOT}/bin/uv venv={KERNEL_ROOT}/venv
sys=$(command -v python3 || true)
if [ -n "$sys" ] && "$sys" -c 'import sys; assert sys.version_info >= (3, 8)' 2>/dev/null \\
   && $uv venv --quiet --python "$sys" $venv \\
   && $uv pip install --quiet --python $venv/bin/python ipykernel jupyter_client; then
  "$sys" -c 'import site; print("\\n".join(site.getsitepackages() + [site.getusersitepackages()]))' \\
    > {KERNEL_ROOT}/task-site-packages.txt
  echo "$sys"
else
  rm -rf $venv
  $uv venv --quiet --seed --python 3.12 $venv >&2
  $uv pip install --quiet --python $venv/bin/python ipykernel jupyter_client >&2
fi
chmod -R a+rX {KERNEL_ROOT}
"""

# Puts the task interpreter's site-packages ahead of the kernel venv's, and reports it as the kernel's own.
KERNEL_STARTUP = f"""def _use_task_python():
    import site, sys
    kernel = list(sys.path)
    for path in open("{KERNEL_ROOT}/task-site-packages.txt").read().splitlines():
        site.addsitedir(path)
    at = next((i for i, p in enumerate(kernel) if p.startswith("{KERNEL_ROOT}/")), len(kernel))
    sys.path[:] = kernel[:at] + [p for p in sys.path if p not in kernel] + kernel[at:]
    sys.executable = {{path!r}}
_use_task_python()
del _use_task_python
"""

KERNEL_RUN_CELL = r"""
import os, re, signal, sys
from jupyter_client import BlockingKernelClient

conn, code_path, timeout = sys.argv[1], sys.argv[2], float(sys.argv[3])
kc = BlockingKernelClient(connection_file=conn)
kc.load_connection_file()
kc.start_channels()
kc.wait_for_ready(timeout=120)
out = []

def hook(msg):
    kind, content = msg["msg_type"], msg["content"]
    if kind == "stream":
        out.append(content["text"])
    elif kind in ("execute_result", "display_data"):
        out.append(content["data"].get("text/plain", "") + "\n")
    elif kind == "error":
        out.append("\n".join(content["traceback"]) + "\n")

try:
    kc.execute_interactive(open(code_path).read(), output_hook=hook, timeout=timeout, allow_stdin=False)
except TimeoutError:
    os.kill(int(open(sys.argv[4]).read()), signal.SIGINT)
    out.append(f"\n[cell timed out after {timeout:.0f}s; kernel interrupted, state kept]\n")
text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", "".join(out))
sys.stdout.write(text or "[no output]")
"""

KERNEL_LAUNCH = (
    "$(command -v setsid || true) "  # detach where available; minimal images may lack it
    f"nohup {KERNEL_ROOT}/venv/bin/python -m ipykernel_launcher -f {KERNEL_RUN}/kernel.json "
    f"--IPKernelApp.exec_files={KERNEL_RUN}/startup.py "
    f"> {KERNEL_RUN}/kernel.log 2>&1 < /dev/null & echo $! > {KERNEL_RUN}/kernel.pid; "
    f"for i in $(seq 120); do [ -s {KERNEL_RUN}/kernel.json ] && exit 0; sleep 1; done; "
    f"cat {KERNEL_RUN}/kernel.log; exit 1"
)


def write_file_command(path: str, content: str) -> str:
    return f"echo {shlex.quote(base64.b64encode(content.encode()).decode())} | base64 -d > {path}"


class IPythonEnv(HarborEnv):
    """ipyh's `ipython` tool over a Harbor sandbox: one persistent IPython kernel per rollout.

    Defined in the training script so the async rollout worker, a spawned process, can unpickle its factory.
    """

    def reset(self, task_dir: str | None = None, **kwargs) -> str:
        started = time.monotonic()
        instruction = super().reset(task_dir, **kwargs)
        # One line per rollout in the job logs: the sandbox start and kernel install are most of a rollout's wait.
        print(f"[ipython] {Path(task_dir).name}: sandbox ready in {time.monotonic() - started:.0f}s", flush=True)
        return instruction

    def get_reward(self) -> float:
        # AsyncGRPO scores an environment through `get_reward`: `HarborSpec.reward_funcs` reads `environments=`, which
        # only `GRPOTrainer` passes. The verifier's scalar, as `HarborEnv.reward` computes it.
        return self.reward

    async def _shell(self, command: str, timeout: int | None = None, user: str | None = None) -> tuple[str, str, int]:
        result = await self._env.exec(command, timeout_sec=timeout, user=user)
        return result.stdout or "", result.stderr or "", result.return_code

    async def _setup(self) -> None:
        stdout, stderr, rc = await self._shell(KERNEL_INSTALL, user="root")
        if rc != 0:
            raise RuntimeError(f"kernel install failed ({rc}): {stderr or stdout}")
        python = stdout.strip() or None
        startup = KERNEL_STARTUP.format(path=python) if python else ""
        for name, content in (("run_cell.py", KERNEL_RUN_CELL), ("startup.py", startup)):
            await self._shell(write_file_command(f"{KERNEL_RUN}/{name}", content), user="root")
        stdout, stderr, rc = await self._shell(KERNEL_LAUNCH)
        if rc != 0:
            raise RuntimeError(f"kernel launch failed ({rc}): {stderr or stdout}")
        # Quiet progress bars and colour, and define ipyh's `edit` / `view` helpers, as ipyh does at task start.
        await self._run_cell(f"import os; os.environ.update({json.dumps(QUIET_ENV)})\n{HELPERS}", CELL_TIMEOUT_SEC)
        self._saved_outputs = 0

    async def _run_cell(self, code: str, timeout: int) -> str:
        cell = f"{KERNEL_RUN}/cell-{uuid.uuid4().hex}.py"
        stdout, stderr, rc = await self._shell(
            f"{write_file_command(cell, code)} && {KERNEL_ROOT}/venv/bin/python {KERNEL_RUN}/run_cell.py "
            f"{KERNEL_RUN}/kernel.json {cell} {timeout} {KERNEL_RUN}/kernel.pid; rm -f {cell}",
            timeout + 120,
        )
        text = stdout + (f"\n[stderr]\n{stderr}" if stderr else "")
        return text + (f"\n[cell runner exited with {rc}]" if rc != 0 else "")

    async def _execute(self, code: str) -> str:
        # ipyh's `Harness.execute_tool`, with the kernel in the sandbox.
        override = _TIMEOUT_LINE.match(code)
        timeout = int(override.group(1)) if override else CELL_TIMEOUT_SEC
        started = time.monotonic()
        output = await self._run_cell(code, timeout)
        # Shell lines in a Python cell (`!cmd` then bare shell) fail to compile, so nothing ran: run them as bash.
        lines = code.strip().splitlines()
        if (
            "SyntaxError" in output
            and lines[0].lstrip().startswith("!")
            and len(lines) > 1
            and not any(_PYTHON_LINE.match(line) for line in lines[1:])
        ):
            shell = "\n".join(line.lstrip()[1:] if line.lstrip().startswith("!") else line for line in lines)
            output = "[ran as %%bash: the cell held shell lines]\n" + await self._run_cell("%%bash\n" + shell, timeout)
        elapsed = time.monotonic() - started
        if elapsed > 10:
            output += f"\n[ran in {elapsed:.0f}s]"
        saved_to = None
        if len(output) > MAX_OUTPUT_CHARS:  # the full output goes to a file the model can read in the sandbox
            self._saved_outputs += 1
            saved_to = f"{OUTPUT_DIR}/out-{self._saved_outputs}.log"
            with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
                f.write(output)
            await self._shell(f"mkdir -p {OUTPUT_DIR} && chmod 777 {OUTPUT_DIR}", user="root")
            await self._env.upload_file(Path(f.name), saved_to)
            os.unlink(f.name)
        return python_default_normalizer(truncate(output, saved_to=saved_to))

    def ipython(self, code: str) -> str:
        """
        Execute code in a persistent IPython kernel and return its output.

        Args:
            code: IPython cell to run.
        """
        return self._run(self._execute(code))


class IPCWeightTransfer:
    """Push weights to a vLLM server on the trainer's own GPU through CUDA IPC (NCCL refuses two ranks per device).

    Implements the `weight_transfer` protocol of `AsyncGRPOTrainer`, which still merges a LoRA adapter before sending.
    The server needs `--weight-transfer-config '{"backend":"ipc"}'` and `VLLM_ALLOW_INSECURE_SERIALIZATION=1`: the
    IPC handles travel pickled over HTTP.
    """

    def __init__(self, vllm_url: str):
        self.vllm_url = vllm_url
        self.vllm = VLLMClient(vllm_url)

    def init_weight_transfer(self) -> None:
        self.vllm.wait_for_server_ready()
        self.vllm.init_weight_transfer_engine({})

    def pause(self) -> None:
        self.vllm.pause()

    def send_weights(self, iterator) -> None:
        from vllm.distributed.weight_transfer.ipc_engine import IPCTrainerSendWeightsArgs, IPCWeightTransferEngine

        self.vllm.start_weight_update()
        try:
            # Unpacked: vLLM reads the parameters' own storage, so a sync allocates nothing. The packed mode copies
            # into fresh buffers every sync, and CUDA IPC keeps them alive until the server releases them; a packed
            # run ran out of memory after ~80 steps with the trainer holding ~90 GB.
            IPCWeightTransferEngine.trainer_send_weights(
                iterator, IPCTrainerSendWeightsArgs(send_mode="http", url=self.vllm_url, packed=False)
            )
        finally:
            self.vllm.finish_weight_update()

    def resume(self) -> None:
        self.vllm.resume()

    def destroy(self) -> None:
        pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vllm-url", required=True, help="the engine AsyncGRPO generates from and syncs weights into")
    p.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    p.add_argument("--split", required=True, help="Harbor task dataset: HF repo id or local dir with tasks/")
    p.add_argument("--sandbox", default="e2b", help="Harbor sandbox backend (e2b, hf-sandbox, docker, ...)")
    p.add_argument("--n-tasks", type=int, default=None)
    # The interpreter named in ipyh's system prompt; SmolDataEnvs images ship it here.
    p.add_argument("--task-python", default="/usr/local/bin/python3")
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--max-inflight", type=int, default=32)
    p.add_argument("--max-completion-length", type=int, default=8192, help="tokens per model turn")
    p.add_argument("--max-tool-calls", type=int, default=17)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max-staleness", type=int, default=4)
    p.add_argument("--per-device-train-batch-size", type=int, default=4)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--lora-r", type=int, default=32, help="train a LoRA adapter of this rank; 0 trains all weights")
    p.add_argument("--lora-alpha", type=int, default=None, help="defaults to 2 * --lora-r")
    # `ipc` lets vLLM and the trainer share one GPU (see IPCWeightTransfer); `nccl` needs one GPU each.
    p.add_argument("--weight-transfer", choices=("nccl", "ipc"), default="nccl")
    p.add_argument("--token-budget", type=int, default=None, help="max tokens per packed row; bounds trainer memory")
    # flash-attn3 (the default) needs Hopper; pass kernels-community/flash-attn2 on A10G, L40S or A100.
    p.add_argument("--attn-implementation", default=None)
    p.add_argument("--save-steps", type=int, default=25)
    p.add_argument("--save-total-limit", type=int, default=0, help="0 keeps every checkpoint")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--project", default="async-grpo-ipython")
    p.add_argument("--trackio-space-id", default=None, help="host the trackio dashboard on a HF Space")
    p.add_argument("--run-name", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    stamp = os.environ.get("JOB_ID") or os.environ.get("SLURM_JOB_ID") or "local"
    run_name = args.run_name or f"{args.model.split('/')[-1]}-ipython-{stamp}-{uuid.uuid4().hex[:8]}"
    output_dir = args.output_dir or f"runs/async_grpo_ipython/{run_name}"

    spec = HarborSpec(args.split, agent=IPythonEnv, environment_type=args.sandbox, num_tasks=args.n_tasks)
    # ipyh's system prompt goes first; TRL appends the task instruction (from `reset`) to the empty user message.
    system_prompt = DEFAULT_SYSTEM_PROMPT.format(python=TASK_PYTHON.format(path=args.task_python))
    system = {"role": "system", "content": system_prompt}
    dataset = spec.train_dataset.map(lambda row: {"prompt": [system, *row["prompt"]]})
    print(f"tasks {len(dataset)} from {args.split} on {args.sandbox}; output {output_dir}")

    config = AsyncGRPOConfig(
        output_dir=output_dir,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit or None,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_tool_calling_iterations=args.max_tool_calls,
        max_inflight_tasks=args.max_inflight,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        top_p=1.0,
        top_k=-1,
        max_staleness=args.max_staleness,
        token_budget=args.token_budget,
        model_init_kwargs={"attn_implementation": args.attn_implementation} if args.attn_implementation else None,
        chat_template_kwargs={"enable_thinking": False},
        vllm_server_base_url=args.vllm_url,
        optim="adamw_torch",
        dtype="bfloat16",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="trackio",
        project=args.project,
        trackio_space_id=args.trackio_space_id,
        run_name=run_name,
        log_completions=True,
        logging_steps=1,
        seed=0,
    )
    peft_config = None
    if args.lora_r:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha or 2 * args.lora_r,
            target_modules="all-linear",
            # The agent sends text only, so adapters on a vision tower would never train.
            exclude_modules=r".*visual.*",
            task_type="CAUSAL_LM",
        )

    trainer = AsyncGRPOTrainer(
        model=args.model,
        args=config,
        train_dataset=dataset,
        processing_class=AutoTokenizer.from_pretrained(args.model),
        environment_factory=spec.environment_factory,
        peft_config=peft_config,
        weight_transfer=IPCWeightTransfer(args.vllm_url) if args.weight_transfer == "ipc" else None,
    )
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()
