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

# Notes:
# - Qwen3.8-Flash-Next is Qwen4ExpForConditionalGeneration: gated DeltaNet + indexed attention (QSA) with a per-layer
#   n-gram embedding (PLE), hyper-connections and a 512-expert MoE.
# - 4 layers: 3 linear-attention layers, the second one with PLE (`ple_layer_ids` is one-indexed and PLE must sit on a
#   linear-attention layer), then one indexed-attention layer.
# - `indexer_budget=128` covers whole test samples, so every query sees all its tokens: the top-k block selection is a
#   step function, and tests comparing two numerically different forwards would otherwise flip near-tied blocks.
#   Tests of the sparse selection lower the budget on the loaded model.
# - The n-gram vocabulary is cut from 20M rows per head to about 1k, or the table alone would be hundreds of MB.
# - Run on a GPU node (see the `.to(device="cuda")` below).
# - head_dim stays 256: the mRoPE sections ([11, 11, 10]) are sized for its 64 rotary dims, which the index head must fit.

import torch
from transformers import AutoConfig, AutoProcessor, GenerationConfig, Qwen4ExpForConditionalGeneration

from .._common import (
    check_dtype_pattern,
    check_transformers_version,
    print_config_diff,
    push_to_hub,
    set_seed,
    smoke_test,
)


TRANSFORMERS_VERSION = "5.23.0.dev0"
check_transformers_version(TRANSFORMERS_VERSION)

set_seed()

MODEL_ID = "Qwen/Qwen3.8-Flash-Next"

processor = AutoProcessor.from_pretrained(MODEL_ID)
generation_config = GenerationConfig.from_pretrained(MODEL_ID)
reference = AutoConfig.from_pretrained(MODEL_ID)

text_config = reference.text_config.to_dict()
text_config.update(
    {
        "num_hidden_layers": 4,
        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "indexed_attention"],
        "hidden_size": 16,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 32,
        "shared_expert_intermediate_size": 32,
        "hc_lowrank": 8,
        "ple_layer_ids": [2],
        "ple_embed_dim": 16,
        "heads_per_ngram": 2,
        "ngram_vocab_size_base": 1000,
        "make_ngram_vocab_size_divisible_by": 8,
        "split_ngram_parts": 1,
        "indexer_n_heads": 2,
        "indexer_head_dim": 64,
        "indexer_budget": 128,
    }
)
vision_config = reference.vision_config.to_dict()
vision_config.update({"hidden_size": 16, "depth": 2, "num_heads": 4, "intermediate_size": 32, "out_hidden_size": 16})

config = AutoConfig.from_pretrained(MODEL_ID, text_config=text_config, vision_config=vision_config)
# On the GPU: with flash-linear-attention installed, the gated DeltaNet layers run its Triton kernels, which have no CPU
# path
model = Qwen4ExpForConditionalGeneration(config).to(dtype=torch.bfloat16, device="cuda")

smoke_test(model, processor)
check_dtype_pattern(MODEL_ID, model)
print_config_diff(MODEL_ID, model)
push_to_hub(model, processor, generation_config, "tiny")
