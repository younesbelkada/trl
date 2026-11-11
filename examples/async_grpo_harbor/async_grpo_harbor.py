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
#     "trl",
#     "trackio",
#     "datasets",
#     "peft",
#     "openenv @ git+https://github.com/openenv/OpenEnv.git@86a180ede21e044f7929b9a7783ad83aa67d83a3",
# ]
# ///

"""AsyncGRPO on Harbor tasks, with an off-the-shelf coding agent, served through OpenEnv.

A Harbor task contains a container image, instruction, and held-out verifier. This example uses
`mini-swe-agent` in an E2B sandbox:

    mini-swe-agent solves a Harbor task in a sandbox
      -> every model call it makes goes through the OpenEnv server's capture proxy to your vLLM
      -> the proxy records exact token ids and the sampling distribution's logprobs
      -> AsyncGRPO trains on them and syncs new weights back into that same vLLM

The agent owns the tool loop; TRL consumes OpenEnv's captured token IDs, logprobs, and masks.
Rewritten histories fork into separate training rows to preserve eligible tokens. See README.md
for the contract, dependency pin, and memory/weighting limitations.

Select another qualified agent with `--harness`. Rollouts and weight updates must use the same vLLM
instance and full-vocabulary sampling policy. Weight updates pause active inference requests without waiting for them to finish.

The reward adds `0.3 * tool_efficiency` only when correctness is at least 1.0 (`--tool-efficiency-weight`). Use `--reward-key` to
select a component when the verifier returns a reward dictionary.

Requirements:
  - A running OpenEnv Harbor server, which owns the dataset and the sandbox templates:
        MAX_CONCURRENT_ENVS=9 openenv harbor serve --dataset <hf-dataset> --port 8200 --capture-port 8300 --expose gradio
  - A sandbox backend credential for the server's environment, e.g. `E2B_API_KEY`.
  - An OpenAI-compatible vLLM server (below) reachable at `--vllm-url`.
  - The pinned OpenEnv checkout and `PYTHONPATH` setup in this example's README.md.

Run (2 GPUs: vLLM on one, trainer on the other):

```sh
# Terminal 1: serve the policy with token capture and weight updates.
CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3.5-2B \
    --host 0.0.0.0 --port 8000 \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --logprobs-mode processed_logprobs \
    --return-tokens-as-token-ids \
    --weight-transfer-config '{"backend":"nccl"}'

# Terminal 2 - train.
CUDA_VISIBLE_DEVICES=1 python examples/async_grpo_harbor/async_grpo_harbor.py \
    --server http://localhost:8200 \
    --vllm-url http://localhost:8000 \
    --model Qwen/Qwen3.5-2B \
    --split <hf-dataset> \
    --max-steps 20
```

To train a LoRA adapter instead, start the server with `--enable-lora` and `VLLM_ALLOW_RUNTIME_LORA_UPDATING=1`, so each
weight sync publishes only the adapter. `--max-lora-rank` must be at least `--lora-r`, and `--max-loras` must hold the
`--max-staleness + 1` adapter versions requests may still name, plus the one being loaded (6 for the default of 4):

```sh
CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 vllm serve Qwen/Qwen3.5-2B \
    --host 0.0.0.0 --port 8000 \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --logprobs-mode processed_logprobs \
    --return-tokens-as-token-ids \
    --weight-transfer-config '{"backend":"nccl"}' \
    --enable-lora --max-lora-rank 32 --max-loras 6

CUDA_VISIBLE_DEVICES=1 python examples/async_grpo_harbor/async_grpo_harbor.py \
    --server http://localhost:8200 \
    --vllm-url http://localhost:8000 \
    --model Qwen/Qwen3.5-2B \
    --split <hf-dataset> \
    --lora-r 32 --learning-rate 1e-5 \
    --max-steps 20
```
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import re
import threading
import time
import uuid
from functools import partial

from datasets import Dataset
from harbor_env.harness import HarborSessionFactory
from peft import LoraConfig
from transformers import AutoTokenizer, TrainerCallback

from trl.experimental.async_grpo import AsyncGRPOConfig
from trl.experimental.async_grpo.openenv_harness import HarnessAsyncGRPOTrainer, HarnessRolloutOutcome


logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Credentials as they appear in commands: Hugging Face, OpenAI-style, GitHub, AWS and Slack tokens, and bearer headers
SECRET = re.compile(
    r"hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"
    r"|AKIA[0-9A-Z]{16}|xox[abprs]-[A-Za-z0-9-]{10,}|(?i:bearer)\s+[A-Za-z0-9._~+/=-]{16,}"
)
W_TOOL_EFFICIENCY = 0.3
TOOL_BUDGET = 15.0


def tool_efficiency(n_tool_calls: int | None) -> float | None:
    """`clip(1 - n/TOOL_BUDGET, 0, 1)`, or `None` when the tool count is unknown."""
    if n_tool_calls is None or TOOL_BUDGET <= 0:
        return None
    return max(0.0, min(1.0, 1.0 - n_tool_calls / TOOL_BUDGET))


class LiveLog(TrainerCallback):
    """`live.json` in the output dir, rewritten every 30 s for the livestream Space to read from the bucket.

    It holds the run's logged metrics and, as safeagents' monitor, how many tool calls were checked and every call
    that was not simply passed.
    """

    def __init__(self, path: str, info: dict):
        self.path = pathlib.Path(path)
        self.lock = threading.Lock()
        self.state = {
            **info,
            "started_at": time.time(),
            "step": 0,
            "finished": False,
            "metrics": [],
            "safeagents": {"requests": 0, "passed": 0, "suspicious": 0, "blocked": 0, "events": []},
        }
        threading.Thread(target=self._flush_every_30s, daemon=True).start()

    def on_log(self, args, state, control, logs=None, **kwargs):
        numbers = {key: value for key, value in (logs or {}).items() if isinstance(value, (int, float))}
        with self.lock:
            self.state["step"] = state.global_step
            self.state["metrics"].append({"step": state.global_step, "time": time.time(), **numbers})

    def on_train_end(self, args, state, control, **kwargs):
        with self.lock:
            self.state["finished"] = True
        self.flush()

    # safeagents' monitor interface: the proxy counts model turns, the guard reports each call's verdict.
    def request(self) -> None:
        with self.lock:
            self.state["safeagents"]["requests"] += 1

    def record(self, name, arguments, verdict, reason, signals, judge, call_id) -> None:
        if verdict != "passed":
            # With a monitor the guard logs nothing itself
            logger.warning("[safeagents] %s call to `%s`: %s", verdict, name, reason or ", ".join(signals))
        with self.lock:
            safeagents = self.state["safeagents"]
            safeagents[verdict] = safeagents.get(verdict, 0) + 1
            if verdict != "passed":
                safeagents["events"].append(
                    {
                        "time": time.time(),
                        "step": self.state["step"],
                        "verdict": verdict,
                        "tool": name,
                        # live.json is read by the livestream Space: never let a credential through
                        "arguments": SECRET.sub("[redacted]", arguments[:2000]),
                        "reason": reason,
                        "signals": signals,
                    }
                )

    def flush(self) -> None:
        with self.lock:
            text = json.dumps({**self.state, "updated_at": time.time()})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(text)

    def _flush_every_30s(self) -> None:
        while True:
            time.sleep(30)
            # One failed write (the bucket mount can error briefly) must not end the thread and freeze the livestream
            try:
                self.flush()
            except Exception as e:
                logger.warning("could not write %s: %s", self.path, e)


def start_safeagents(upstream: str, plugins: list[str], allowed_hosts: list[str], monitor: LiveLog) -> str:
    """Serve safeagents in front of vLLM on a background thread and return the URL harnesses should call.

    Every tool call the model makes is checked before the harness sees it: undeclared tools, hosts from nowhere and
    atif-scan findings at `high` or above (built-in checks plus `plugins`, `module:factory` or `file.py:factory`)
    are replaced by a notice, so the harness runs nothing. Token ids and logprobs pass through untouched, so the
    blocked tokens are still trained on, at the reward `harbor_reward` gives them. Every verdict goes to `monitor`.
    """
    import uvicorn
    from safeagents import DEFAULT_ALLOWED_HOSTS, Config, Scanner, load_plugin
    from safeagents.proxy import create_app

    scanner = Scanner([check for plugin in plugins for check in load_plugin(plugin)])
    config = Config(allowed_hosts=[*DEFAULT_ALLOWED_HOSTS, *allowed_hosts], scanner=scanner, monitor=monitor)
    # Port 0: the OS picks a free one, so trainers sharing a host don't collide.
    server = uvicorn.Server(uvicorn.Config(create_app(upstream, config), host="127.0.0.1", port=0, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.1)
    port = server.servers[0].sockets[0].getsockname()[1]
    return f"http://127.0.0.1:{port}"


def harbor_reward(outcome: HarnessRolloutOutcome, tool_efficiency_weight: float = W_TOOL_EFFICIENCY) -> float | None:
    """Reward for one rollout, or `None` when it is unscorable.

    Args:
        outcome (`HarnessRolloutOutcome`):
            The verifier's reward, transcript, tool-call count, and
            whether the agent ran out of wall clock.
        tool_efficiency_weight (`float`, *optional*, defaults to `0.3`):
            Weight of the tool-efficiency bonus added to correct rollouts. `0` rewards correctness only.

    Returns:
        `float` or `None`: Unscored rollouts are excluded from the group baseline.
    """
    correctness = outcome.env_reward
    if correctness is None:
        logger.warning("verifier did not run (tool_calls=%d); rollout unscorable", outcome.tool_call_count)
        return None

    if outcome.timed_out:
        logger.warning("agent timed out; keeping the verifier's score of %.3f on the partial work", correctness)

    correctness = float(correctness)
    # With --safeagents, a blocked tool call leaves its notice in the transcript: the attempt scores 0, whatever the
    # verifier says, so the policy learns not to make it.
    if any("[safeagents] blocked call" in str(message.get("content")) for message in outcome.completion):
        logger.warning("safeagents blocked a tool call; reward 0 (verifier: %.3f)", correctness)
        return 0.0
    reward = correctness

    eff = tool_efficiency(outcome.tool_call_count)
    if eff is not None and correctness >= 1.0 and tool_efficiency_weight:
        reward += tool_efficiency_weight * eff

    return reward


def task_indices(spec: str) -> list[int] | None:
    """Read comma-separated task indices, directly or from `@path` (safe for Slurm exports)."""
    spec = (spec or "").strip()
    if not spec:
        return None
    if spec.startswith("@"):
        spec = pathlib.Path(spec[1:]).read_text(encoding="utf-8")
    return [int(x) for x in spec.replace("\n", ",").split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--server", default="http://127.0.0.1:8200", help="a running `openenv harbor serve`")
    p.add_argument("--vllm-url", required=True, help="the engine AsyncGRPO generates from and syncs weights into")
    p.add_argument("--model", default="Qwen/Qwen3.5-2B")
    p.add_argument("--trust-remote-code", action="store_true", help="for models shipping their code, e.g. LFM3")
    # TRL parses tool calls only for chat templates it knows by their text. TwIL-8B's template is unknown to it, but its
    # output is Qwen3.5's format: ChatML turns, a <think> block and <tool_call><function=...> calls.
    p.add_argument("--qwen3-5-response-template", action="store_true", help="parse responses as Qwen3.5's, e.g. TwIL-8B")
    p.add_argument("--split", required=True, help="the Harbor task dataset the server was started with")
    p.add_argument("--harness", default="mini-swe-agent", help="any harness the server reports; see the docstring")
    p.add_argument("--sandbox", default="e2b")
    p.add_argument("--reward-key", default="")
    # The group-std normalization scales this bonus up in groups that all solved the task, so it can dominate the
    # gradient on easy tasks; 0 trains on correctness alone.
    p.add_argument("--tool-efficiency-weight", type=float, default=W_TOOL_EFFICIENCY)
    # safeagents in front of vLLM: blocked tool calls never run, and their rollouts score 0.
    p.add_argument("--safeagents", action="store_true", help="check every tool call with safeagents")
    p.add_argument("--safeagents-plugin", action="append", default=[], help="MODULE:FACTORY or file.py:FACTORY")
    p.add_argument("--safeagents-allow-host", action="append", default=[], help="host tool calls may reach")
    p.add_argument("--n-tasks", type=int, default=32)
    p.add_argument("--task-indices", default="", help="comma-separated indices, or @path to a file of them")
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--max-inflight", type=int, default=8)
    p.add_argument("--max-completion-length", type=int, default=1024)
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--save-steps", type=int, default=50)
    p.add_argument("--save-total-limit", type=int, default=3, help="checkpoints kept; 0 keeps every one")
    p.add_argument("--learning-rate", type=float, default=1e-6)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-staleness", type=int, default=4)
    # Tokens per packed training row; longer samples are dropped. Unset, it is the vLLM server's max_model_len.
    p.add_argument("--token-budget", type=int, default=None)
    p.add_argument(
        "--vllm-lora-name", default="trl-policy", help="adapter name prefix; unique per trainer on a shared vLLM"
    )
    # Sandbox setup has its own timeouts; this bounds the agent run.
    p.add_argument("--agent-timeout", type=float, default=300.0)
    # Bound model calls; rewritten histories can still produce multiple training rows.
    p.add_argument("--agent-step-limit", type=int, default=12)
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    # 0 trains every weight. With LoRA, start vLLM with `--enable-lora` so syncs publish only the adapter.
    p.add_argument("--lora-r", type=int, default=0)
    p.add_argument("--lora-alpha", type=int, default=None, help="defaults to 2 * --lora-r")
    p.add_argument("--lora-target-modules", default="all-linear", help="`all-linear`, or comma-separated names")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--project", default="async-grpo-harbor")
    p.add_argument("--trackio-space-id", default=None, help="host the trackio dashboard on a HF Space")
    p.add_argument("--run-name", default=None)
    p.add_argument("--resume-from-checkpoint", default=None, help="a checkpoint-N folder, possibly of another run")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    stamp = os.environ.get("SLURM_JOB_ID") or os.environ.get("HF_JOB_ID") or os.environ.get("JOB_ID") or "local"
    stamp = f"{stamp}-{uuid.uuid4().hex[:8]}"
    run_name = args.run_name or f"{args.model.split('/')[-1]}-{args.harness}-{args.max_steps}steps-{stamp}"
    output_dir = args.output_dir or f"runs/async_grpo_harbor/{run_name}"

    live = LiveLog(
        f"{output_dir}/live.json",
        {
            "run_name": run_name,
            "job_id": os.environ.get("HF_JOB_ID"),
            "model": args.model,
            "harness": args.harness,
            "split": args.split,
            "max_steps": args.max_steps,
            "num_generations": args.num_generations,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "lora_r": args.lora_r,
            "safeagents_enabled": args.safeagents,
            "safeagents_plugins": args.safeagents_plugin,
        },
    )
    # Harnesses reach the model through safeagents; weight syncs still go to vLLM directly.
    llm_url = args.vllm_url
    if args.safeagents:
        llm_url = start_safeagents(args.vllm_url, args.safeagents_plugin, args.safeagents_allow_host, live)
    factory = partial(
        HarborSessionFactory,
        args.server,
        split=args.split,
        harness=args.harness,
        sandbox=args.sandbox,
        llm_url=llm_url,
        model=args.model,
        agent_timeout_sec=args.agent_timeout,
        agent_step_limit=args.agent_step_limit,
        reward_key=args.reward_key,
        num_tasks=args.n_tasks,
        indices=task_indices(args.task_indices),
    )
    metadata_factory = factory()
    try:
        # The rollout worker walks the dataset in order, and the server lists tasks sorted by name: unshuffled, a run that
        # covers fewer tasks than it has would never reach the last ones (a mixed split's second dataset)
        dataset = Dataset.from_list(metadata_factory.prompt_rows()).shuffle(seed=0)
    finally:
        # The pinned OpenEnv factory has no public close method.
        if metadata_factory._env is not None:
            metadata_factory._env.close()

    print(f"server    {args.server}")
    print(f"vllm      {args.vllm_url}   model {args.model}")
    print(f"rollouts  {args.harness} on {args.sandbox}, {args.num_generations}x{args.max_inflight}")
    print(f"tasks     {len(dataset)} from {args.split}")
    print(f"output    {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if args.qwen3_5_response_template:
        # As `add_response_schema` sets it, which the rollout worker then skips
        from trl.chat_template_utils import _SUPPORTS_RESPONSE_TEMPLATE, qwen3_5_schema, qwen3_5_template

        if _SUPPORTS_RESPONSE_TEMPLATE:
            tokenizer.response_template = qwen3_5_template
        else:
            tokenizer.response_schema = qwen3_5_schema

    config = AsyncGRPOConfig(
        output_dir=output_dir,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit or None,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        top_p=1.0,
        top_k=-1,
        chat_template_kwargs={"enable_thinking": False},
        max_concurrent_rollouts=args.max_inflight,
        max_staleness=args.max_staleness,
        vllm_server_base_url=args.vllm_url,
        vllm_lora_name=args.vllm_lora_name,
        token_budget=args.token_budget,
        optim="adamw_torch",
        dtype="bfloat16",
        trust_remote_code=args.trust_remote_code,
        bf16=True,
        gradient_checkpointing=True,
        # Support layers whose inputs are passed as keyword arguments.
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="trackio",
        project=args.project,
        trackio_space_id=args.trackio_space_id,
        run_name=run_name,
        log_completions=True,
        num_completions_to_print=2,
        logging_steps=1,
        seed=0,
    )

    peft_config = None
    if args.lora_r:
        # PEFT reads `all-linear` only as a string, and anything else as module names
        target_modules = args.lora_target_modules
        if target_modules != "all-linear":
            target_modules = target_modules.split(",")
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha or 2 * args.lora_r,
            target_modules=target_modules,
            task_type="CAUSAL_LM",
        )

    trainer = HarnessAsyncGRPOTrainer(
        model=args.model,
        harness_session_factory=factory,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        rollout_reward_fn=partial(harbor_reward, tool_efficiency_weight=args.tool_efficiency_weight),
        peft_config=peft_config,
    )
    trainer.add_callback(live)
    if args.resume_from_checkpoint:
        # The resumed TrainerState keeps the checkpoint's save interval; apply this run's `--save-steps` instead.
        class _SaveSteps(TrainerCallback):
            def on_train_begin(self, args_, state, control, **kwargs):
                state.save_steps = args.save_steps

        trainer.add_callback(_SaveSteps())
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model()


if __name__ == "__main__":
    main()
