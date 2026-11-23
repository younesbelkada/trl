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

"""
Inkling 2 Loop's sequence-mixing steps over a tree-packed row, patched in by `patch_tree_inkling2_loop`
(`trl/trainer/async_grpo_trainer.py`).

Three steps read across tokens, and each follows transformers' reference (`modeling_inkling2_loop.py`) with a token's
past taken along its own path in the tree:

- Attention (sliding and full): a learned bias per query-key distance (zero past `rel_extent`), one sink logit per query
  and head, and a window for the sliding layers. The reference builds the bias as a dense `(query, key)` matrix from
  flat positions; here each segment gathers its root path (`tree_kv_index`, whose index is the depth) and computes the
  bias and window from depth differences, chunk by chunk.
- Short convolutions (on keys, values and after the attention): their 4 taps run back along the token's path.
- Dynamic short convolutions (keys and values of the full layers): the same, with per-token weights.

The looped core, its messages and the attention residuals all work position by position, so they need nothing.
"""

import math
import sys

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .tree_gated_delta_rule import tree_causal_conv1d


# Upper bound on one chunk's `(queries, heads, keys)` attention scores, in elements (256 MB in float32)
_MAX_SCORES = 2**26


def tree_parent_index(tree, n_tokens):
    """
    Packed position of the token before each one in its samples, `-1` for a first token: the previous position inside a
    segment, the end of the parent segment at a segment start.
    """
    cu_seqlens, parents = tree["tree_cu_seqlens_q"].long(), tree["tree_segment_parents"].long()
    parent_index = torch.arange(-1, n_tokens - 1, device=cu_seqlens.device)
    parent_index[cu_seqlens[:-1]] = torch.where(parents >= 0, cu_seqlens[parents.clamp_min(0) + 1] - 1, -1)
    return parent_index


def _chunk_attention(module, eager, query, key, value, rel_logits, sinks, first_depth, key_start, window, scaling):
    """
    Queries at depths `first_depth, first_depth + 1, ...` over the keys at depths `key_start, key_start + 1, ...`: a
    dense attention over one window, so it runs the model's own `eager_attention_forward` (scores plus relative bias and
    mask, softmax with the sink as one extra logit), which keeps the reference's rounding in bfloat16.
    """
    n_queries, n_heads = query.shape[0], query.shape[1]
    n_keys = key.shape[0]
    distance = (first_depth + torch.arange(n_queries, device=query.device))[:, None] - (
        key_start + torch.arange(n_keys, device=query.device)
    )[None, :]
    rel_extent = rel_logits.shape[-1]
    index = distance.clamp(0, rel_extent - 1)[:, None, :].expand(-1, n_heads, -1)
    bias = rel_logits.gather(-1, index).to(query.dtype)
    bias = bias.masked_fill(((distance < 0) | (distance >= rel_extent))[:, None, :], 0.0)
    visible = distance >= 0
    if window is not None:
        visible = visible & (distance < window)
    mask = torch.zeros(visible.shape, dtype=query.dtype, device=query.device)
    mask = mask.masked_fill(~visible, torch.finfo(query.dtype).min)

    # (tokens, heads, head_dim) -> (1, heads, tokens, head_dim), as the attention interface takes them
    out, _ = eager(
        module,
        query.transpose(0, 1)[None],
        key.to(query.dtype).transpose(0, 1)[None],
        value.to(query.dtype).transpose(0, 1)[None],
        mask[None, None],
        scaling=scaling,
        position_bias=bias.transpose(0, 1)[None],
        sinks=sinks.to(query.dtype).T[None, :, :, None],
    )
    return out[0]


def _segment_attention(module, eager, query, key, value, rel_logits, sinks, path, window, scaling):
    """
    A segment's queries (the last `len(query)` tokens of `path`) over the K/V rows of their path. Checkpointed per chunk
    of queries, so the backward holds one chunk's scores at a time.
    """
    n_queries, n_heads = query.shape[0], query.shape[1]
    first_depth = len(path) - n_queries
    key_start = 0 if window is None else max(0, first_depth - window + 1)
    keys = key.index_select(0, path[key_start:])
    values = value.index_select(0, path[key_start:])
    chunk = max(1, _MAX_SCORES // (n_heads * len(keys)))
    outputs = []
    for s in range(0, n_queries, chunk):
        depth = first_depth + s
        # The keys this chunk can see: its window (or all of the path) up to its last query
        lo = 0 if window is None else max(0, depth - window + 1) - key_start
        hi = min(depth + chunk, len(path)) - key_start
        outputs.append(
            checkpoint(
                _chunk_attention,
                module,
                eager,
                query[s : s + chunk],
                keys[lo:hi],
                values[lo:hi],
                rel_logits[s : s + chunk],
                sinks[s : s + chunk],
                depth,
                key_start + lo,
                window,
                scaling,
                use_reentrant=False,
            )
        )
    return torch.cat(outputs)


def tree_inkling2_loop_attention(module, hidden_states, key_hidden_states, value_hidden_states, conv_mask, kwargs):
    """`Inkling2LoopSlidingAttention.forward` / `Inkling2LoopFullAttention.forward` over a tree-packed row."""
    position_ids = kwargs["position_ids"]
    input_shape = hidden_states.shape[:-1]
    key_hidden_states = hidden_states if key_hidden_states is None else key_hidden_states
    value_hidden_states = hidden_states if value_hidden_states is None else value_hidden_states

    query = module.q_proj(hidden_states, position_ids)
    key = module.k_proj(key_hidden_states, conv_mask=conv_mask, **kwargs)
    value = module.v_proj(value_hidden_states, conv_mask=conv_mask, **kwargs)
    # (1, heads, N, head_dim) -> (N, heads, head_dim). K/V are gathered from float32 copies: `index_select`'s backward
    # adds the many queries' contributions to a row atomically, which would lose precision in bfloat16
    query = query[0].transpose(0, 1)
    key, value = (x[0].transpose(0, 1).float().contiguous() for x in (key, value))

    # The bias of each query for every distance, `(N, heads, rel_extent)`, instead of the reference's dense matrix
    bias = module.relative_position_bias
    relative_states = bias.linear(hidden_states).view(*input_shape, bias.num_heads, -1)
    rel_logits = (bias.norm(relative_states, position_ids) @ bias.proj)[0]
    sinks = module.sink_proj(hidden_states)[0]

    eager = sys.modules[type(module).__module__].eager_attention_forward
    cu_q, cu_k = kwargs["tree_cu_seqlens_q"].tolist(), kwargs["tree_cu_seqlens_k"].tolist()
    attn_output = torch.cat(
        [
            _segment_attention(
                module,
                eager,
                query[q_start:q_end],
                key,
                value,
                rel_logits[q_start:q_end],
                sinks[q_start:q_end],
                kwargs["tree_kv_index"][k_start:k_end],
                module.sliding_window,
                module.scaling,
            )
            for q_start, q_end, k_start, k_end in zip(cu_q[:-1], cu_q[1:], cu_k[:-1], cu_k[1:], strict=True)
        ]
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    return module.o_proj(attn_output, hidden_states), None


def tree_short_convolution(module, conv, hidden_states, tree):
    """`Inkling2LoopShortConvolution.forward` over a tree-packed row, its taps run by TRL's tree convolution."""
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.float()
    residual = hidden_states
    weight, bias = module.conv1d.weight.squeeze(1), module.conv1d.bias
    hidden_states = tree_causal_conv1d(conv, hidden_states.transpose(1, 2), weight, bias, None, **tree).transpose(1, 2)
    return ((hidden_states + residual) / math.sqrt(module.conv_kernel_size + 1)).to(input_dtype)


def tree_dynamic_short_convolution(module, hidden_states, dynamic_input, tree):
    """`Inkling2LoopDynamicShortConvolution.forward` over a tree-packed row: each window runs back along the path."""
    input_dtype = hidden_states.dtype
    _, n_tokens, hidden_size = hidden_states.shape
    width = module.conv_kernel_size
    dynamic_weights = module.weight_proj(dynamic_input).float().view(1, n_tokens, module.num_heads, 1, width)
    weights = module.static_weight.float().T.view(module.num_heads, -1, width) + dynamic_weights

    hidden_states = hidden_states.float()
    residual = hidden_states
    # taps[t, lag]: the packed position `lag` steps back along t's path, `-1` past its first token
    parent_index = tree_parent_index(tree, n_tokens)
    taps, index = [], torch.arange(n_tokens, device=hidden_states.device)
    for _ in range(width):
        taps.append(index)
        index = torch.where(index >= 0, parent_index[index.clamp_min(0)], -1)
    taps = torch.stack(taps, dim=1)
    rows = F.pad(hidden_states[0], (0, 0, 0, 1))  # a zero row for the taps past a first token
    windows = rows[taps.where(taps >= 0, n_tokens)].transpose(1, 2)  # (N, hidden, width)
    windows = windows.reshape(1, n_tokens, module.num_heads, -1, width)
    hidden_states = (windows * weights).sum(-1).reshape(1, n_tokens, hidden_size)
    return ((hidden_states + residual) / math.sqrt(2 * width + 1)).to(input_dtype)
