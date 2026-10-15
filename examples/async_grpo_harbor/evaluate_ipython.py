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

"""Evaluate a policy on held-out Harbor tasks with the ipython tool of `async_grpo_ipython.py`: each task once, pass@1.

A rollout runs as in training: the same `IPythonEnv` and system prompt (imported from `async_grpo_ipython.py`, which
must sit next to this file), and the same loop as TRL's rollout worker, turn by turn: the whole conversation rendered
through the training chat template, `/v1/completions` on its token ids, the reply parsed with TRL's `parse_response`,
each tool call answered with a tool message, until a reply without a tool call or `--max-tool-calls` calls. The task's
verifier grades `/workdir/answer.txt`.

    python examples/async_grpo_harbor/evaluate_ipython.py --vllm-url http://127.0.0.1:8000 \
        --model Qwen/Qwen3.5-0.8B --split <test tasks dir> --sandbox hf-sandbox --output-dir runs/eval-ipython

Each task's result is written to `<output-dir>/pairs/` as soon as it finishes; rerunning retries only ungraded tasks.
The summary has the layout of `evaluate_harbor.py`'s, with one harness: `ipython`.
"""

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from async_grpo_ipython import IPythonEnv, system_prompt
from transformers import AutoTokenizer

from trl.chat_template_utils import get_training_chat_template, is_chat_template_prefix_preserving, parse_response
from trl.experimental.harbor import HarborSpec


def run_task(args, tokenizer, chat_template, max_model_len, env_factory, system, row) -> dict:
    """One rollout of one task, as TRL's rollout worker runs it; returns its grade and counts."""
    start = time.monotonic()
    env = env_factory()
    messages = [system, {"role": "user", "content": env.reset(task_dir=row["task_dir"])}]
    tool_calls = model_calls = generated_tokens = tool_turns = 0
    while True:
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            return_dict=False,
            add_generation_prompt=True,
            tools=[env.ipython],
            chat_template=chat_template,
            enable_thinking=False,
        )
        if len(prompt_ids) >= max_model_len:  # the conversation fills the context window
            break
        payload = {
            "model": args.model,
            "prompt": prompt_ids,
            "max_tokens": min(args.max_completion_length, max_model_len - len(prompt_ids)),
            "temperature": args.temperature,
            "top_p": 1.0,
            "top_k": -1,
            "n": 1,
            "return_token_ids": True,
        }
        response = requests.post(f"{args.vllm_url}/v1/completions", json=payload, timeout=args.request_timeout)
        response.raise_for_status()
        turn_ids = response.json()["choices"][0]["token_ids"]
        model_calls += 1
        generated_tokens += len(turn_ids)
        message = parse_response(tokenizer, turn_ids, prefix=prompt_ids)
        messages.append(message)
        calls = message.get("tool_calls")
        # As the worker's `max_tool_calling_iterations`: a limit on tool-calling turns, the last request refused.
        if calls is None or tool_turns >= args.max_tool_calls:
            break
        tool_turns += 1
        for call in calls:
            tool_calls += 1
            function = call["function"]
            try:
                if function["name"] != "ipython":
                    raise ValueError(f"unknown tool {function['name']}")
                result = env.ipython(**function.get("arguments", {}))
            except Exception as error:  # noqa: BLE001 - reported to the model, as the rollout worker does
                result = {"error": str(error)}
            messages.append({"role": "tool", "name": function["name"], "content": str(result)})
    return {
        "reward": env.get_reward(),
        "model_calls": model_calls,
        "tool_calls": tool_calls,
        "generated_tokens": generated_tokens,
        "elapsed_s": round(time.monotonic() - start, 1),
    }


def difficulty(split, task):
    """The task's declared difficulty, recorded in each result so it stays readable without the task folder."""
    toml = Path(split, "tasks", task, "task.toml")
    m = re.search(r'^difficulty = "(\w+)"', toml.read_text(), re.M) if toml.exists() else None
    return m[1] if m else None


def summarize(rows, expected):
    """`evaluate_harbor.py`'s summary: a task is graded when the verifier returned a reward, solved when it is >= 1."""
    graded = [row for row in rows if row.get("reward") is not None]

    def mean(key):
        values = [row[key] for row in graded if row.get(key) is not None]
        return sum(values) / len(values) if values else None

    return {
        "pairs": expected,
        "graded": len(graded),
        "coverage": len(graded) / expected,
        "pass_at_1": sum(row["reward"] >= 1.0 for row in graded) / len(graded) if graded else None,
        "mean_reward": mean("reward"),
        "mean_tool_calls": mean("tool_calls"),
        "mean_model_calls": mean("model_calls"),
        "mean_generated_tokens": mean("generated_tokens"),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vllm-url", required=True)
    p.add_argument("--model", required=True, help="the name vLLM serves the policy under")
    p.add_argument("--checkpoint", default="", help="label recorded in summary.json, e.g. the checkpoint path")
    p.add_argument("--split", required=True, help="Harbor task dataset: HF repo id or local dir with tasks/")
    p.add_argument("--sandbox", default="e2b")
    p.add_argument("--n-tasks", type=int, default=250)
    p.add_argument("--concurrency", type=int, default=32)
    # Training's defaults, so a checkpoint is evaluated under the rollout limits it was trained with.
    p.add_argument("--task-python", default="/usr/local/bin/python3")
    p.add_argument("--no-magics-prompt", action="store_true", help="ipyh's system prompt alone, as in training")
    p.add_argument("--max-tool-calls", type=int, default=17)
    p.add_argument("--max-completion-length", type=int, default=8192, help="tokens per model turn")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--request-timeout", type=float, default=600.0)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    (output_dir / "pairs").mkdir(parents=True, exist_ok=True)

    spec = HarborSpec(args.split, agent=IPythonEnv, environment_type=args.sandbox, num_tasks=args.n_tasks)
    rows = list(spec.train_dataset)
    system = {"role": "system", "content": system_prompt(args.task_python, magics=not args.no_magics_prompt)}
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # The rollout worker's choice: a training-safe template when the tokenizer's own is not prefix-preserving.
    chat_template = None if is_chat_template_prefix_preserving(tokenizer) else get_training_chat_template(tokenizer)
    max_model_len = requests.get(f"{args.vllm_url}/v1/models").json()["data"][0]["max_model_len"]

    def pair_path(row):
        return output_dir / "pairs" / f"{Path(row['task_dir']).name}--ipython.json"

    def graded(row):
        return pair_path(row).exists() and json.loads(pair_path(row).read_text())["reward"] is not None

    pending = [row for row in rows if not graded(row)]
    print(f"{len(rows)} tasks: {len(pending)} to run, {args.concurrency} at a time", flush=True)

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        context = (args, tokenizer, chat_template, max_model_len, spec.environment_factory, system)
        futures = {pool.submit(run_task, *context, row): row for row in pending}
        for done, future in enumerate(as_completed(futures), 1):
            row = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 - record the failure, keep evaluating the other tasks
                result = {"reward": None, "error": f"{type(exc).__name__}: {exc}"}
            task = Path(row["task_dir"]).name
            record = {"task_name": task, "task_index": row["task_index"], "harness": "ipython",
                      "difficulty": difficulty(args.split, task), **result}
            pair_path(row).write_text(json.dumps(record, indent=2))
            print(f"[{done}/{len(pending)}] {task}  reward={result.get('reward')}", flush=True)

    records = [json.loads(pair_path(row).read_text()) for row in rows if pair_path(row).exists()]
    summary = {
        "model": args.model,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "sandbox": args.sandbox,
        "temperature": args.temperature,
        "max_tool_calls": args.max_tool_calls,
        "magics_prompt": not args.no_magics_prompt,
        "overall": summarize(records, len(rows)),
        "harnesses": {"ipython": summarize(records, len(rows))},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'harness':16s} {'pass@1':>7s} {'coverage':>9s} {'tool calls':>11s}")
    for name, s in [*summary["harnesses"].items(), ("overall", summary["overall"])]:
        pass_at_1 = f"{s['pass_at_1']:.3f}" if s["pass_at_1"] is not None else "n/a"
        tools = f"{s['mean_tool_calls']:.1f}" if s["mean_tool_calls"] is not None else "n/a"
        print(f"{name:16s} {pass_at_1:>7s} {s['coverage']:>9.2f} {tools:>11s}")


if __name__ == "__main__":
    main()
