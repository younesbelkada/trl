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
Qwen4-Exp's indexed attention (QSA) and per-layer n-gram embedding (PLE) over a tree-packed row, patched in by
`patch_tree_qwen4_exp` (`trl/trainer/async_grpo_trainer.py`).

Both follow transformers' reference implementation (`modeling_qwen4_exp.py`), which loops over every query position and
reads the tokens before each one from the flat row. Here a token's past is its path in the tree: the K/V rows tree
attention gathers for its segment (`tree_kv_index`), and the chain of `tree_parent_index`.

QSA: each query sees at most `indexer_budget + compress_ratio - 1` tokens. Its visible tokens are cut into blocks of
`compress_ratio` consecutive ones, the indexer scores the blocks it sees whole, the query keeps the `block_topk` best
and the 0 to `compress_ratio - 1` tokens after the last whole block, and attends over those alone. The selection carries
no gradient, as in the reference (it is a boolean mask there), so it runs under `no_grad`.
"""

import math

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# Queries per chunk: bounds the indexer's score matrix and the gathered K/V of the sparse attention
_CHUNK = 1024


def _select_tokens(indexer, q_index, k_index, cos, sin, path, apply_rotary_pos_emb):
    """
    Packed positions each of a segment's queries attends to, `-1`-padded: its `block_topk` best whole blocks and the
    tokens after the last one. `path` is the segment's root-to-end path, the queries are its last `len(q_index)` tokens.
    """
    ratio, n_queries = indexer.compress_ratio, len(q_index)
    n_blocks = len(path) // ratio
    depths = torch.arange(len(path) - n_queries, len(path), device=path.device)
    complete = (depths + 1) // ratio  # whole blocks each query sees
    k = min(indexer.block_topk, n_blocks)
    if k:
        blocks = path[: n_blocks * ratio].view(n_blocks, ratio)
        pooled = indexer.k_layernorm(k_index[blocks].float().mean(dim=1).to(k_index.dtype))
        starts = blocks[:, 0]
        pooled = apply_rotary_pos_emb(pooled.unsqueeze(1), cos=cos[starts], sin=sin[starts]).squeeze(1).float()
        selected = []
        for start in range(0, n_queries, _CHUNK):
            q = q_index[start : start + _CHUNK].float()
            scores = torch.relu(torch.einsum("qhd,bd->qhb", q, pooled)).sum(dim=1) / math.sqrt(indexer.index_head_dim)
            visible = torch.arange(n_blocks, device=path.device) < complete[start : start + _CHUNK, None]
            top = scores.masked_fill(~visible, -math.inf).topk(k, dim=-1)
            tokens = torch.where(torch.isfinite(top.values)[..., None], blocks[top.indices], -1)
            selected.append(tokens.flatten(1))
        selected = torch.cat(selected)
    else:
        selected = path.new_empty((n_queries, 0))
    tail_depths = (complete * ratio)[:, None] + torch.arange(ratio - 1, device=path.device)
    tail = torch.where(tail_depths <= depths[:, None], path[tail_depths.clamp_max(len(path) - 1)], -1)
    return torch.cat([selected, tail], dim=1)


def _sparse_attention(query, key, value, selected, scaling):
    """
    Each query attends over its own selected K/V rows. Grouped by K/V head, so a group's query heads share the
    gathered rows instead of repeating them, and checkpointed per chunk, so the backward holds one chunk's rows at a
    time.

    The rows are gathered with `index_select` from a float32 copy of K/V: each row is selected by up to `budget`
    queries, and the backward of advanced indexing (`key[index]`) sums those duplicates by sorting every index, which
    took 97% of the layer's forward + backward. `index_select`'s backward adds them atomically instead, in float32 so
    that thousands of contributions per row keep their precision. The copy is small: K/V have few heads.
    """
    n_kv_heads, head_dim = key.shape[1], key.shape[2]
    key, value = key.float().contiguous(), value.float().contiguous()

    def chunk(query, key, value, selected):
        rows = selected.clamp_min(0).flatten()
        shape = (*selected.shape, n_kv_heads, head_dim)
        out = F.scaled_dot_product_attention(
            query.reshape(len(query), n_kv_heads, -1, head_dim),
            key.index_select(0, rows).view(shape).to(query.dtype).transpose(1, 2),
            value.index_select(0, rows).view(shape).to(query.dtype).transpose(1, 2),
            attn_mask=(selected >= 0)[:, None, None, :],
            scale=scaling,
        )
        return out.reshape(len(query), -1, head_dim)

    return torch.cat(
        [
            checkpoint(chunk, query[s : s + _CHUNK], key, value, selected[s : s + _CHUNK], use_reentrant=False)
            for s in range(0, len(query), _CHUNK)
        ]
    )


def tree_qsa_attention(module, hidden_states, position_embeddings, tree, apply_rotary_pos_emb):
    """`Qwen4ExpTextAttention.forward` over a tree-packed row: the projections as in the reference, then QSA per
    segment."""
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)

    query_states, gate = torch.chunk(
        module.q_proj(hidden_states).view(*input_shape, -1, module.head_dim * 2), 2, dim=-1
    )
    gate = gate.reshape(*input_shape, -1)

    query_states = module.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
    key_states = module.k_norm(module.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    # (1, heads, N, head_dim) -> (N, heads, head_dim)
    query, key, value = (x[0].transpose(0, 1) for x in (query_states, key_states, value_states))
    cos, sin = cos[0], sin[0]

    indexer = module.indexer
    with torch.no_grad():
        q_index, k_index = torch.split(
            indexer.index_qk_proj(hidden_states[0]),
            [indexer.index_n_heads * indexer.index_head_dim, indexer.index_kv_heads * indexer.index_head_dim],
            dim=-1,
        )
        q_index = indexer.q_layernorm(q_index.view(len(q_index), -1, indexer.index_head_dim))
        q_index = apply_rotary_pos_emb(q_index, cos=cos, sin=sin, unsqueeze_dim=1)
        k_index = k_index.view(len(k_index), indexer.index_head_dim)  # one indexer K head, as the reference assumes

    cu_q, cu_k = tree["tree_cu_seqlens_q"].tolist(), tree["tree_cu_seqlens_k"].tolist()
    attn_output = []
    for q_start, q_end, k_start, k_end in zip(cu_q[:-1], cu_q[1:], cu_k[:-1], cu_k[1:], strict=True):
        path = tree["tree_kv_index"][k_start:k_end]
        with torch.no_grad():
            selected = _select_tokens(indexer, q_index[q_start:q_end], k_index, cos, sin, path, apply_rotary_pos_emb)
        attn_output.append(_sparse_attention(query[q_start:q_end], key, value, selected, module.scaling))

    attn_output = torch.cat(attn_output).reshape(*input_shape, -1).contiguous()
    attn_output = attn_output * torch.sigmoid(gate)
    return module.o_proj(attn_output)


def tree_parent_index(tree, n_tokens):
    """
    Packed position of the token before each one in its samples, `-1` for a first token: the previous position inside a
    segment, the end of the parent segment at a segment start.
    """
    cu_seqlens, parents = tree["tree_cu_seqlens_q"].long(), tree["tree_segment_parents"].long()
    parent_index = torch.arange(-1, n_tokens - 1, device=cu_seqlens.device)
    parent_index[cu_seqlens[:-1]] = torch.where(parents >= 0, cu_seqlens[parents.clamp_min(0) + 1] - 1, -1)
    return parent_index


def _ancestor(parent_index, index):
    """The packed position before each of `index` in its samples, `-1` past a first token."""
    return torch.where(index >= 0, parent_index[index.clamp_min(0)], -1)


def _tree_ngram_embedding(embedding, input_ids, parent_index):
    """`Qwen4ExpTextNGramEmbedding.forward` over a tree-packed row: each token's n-gram is read up its path."""
    tokens, eos = input_ids[0], embedding.eos_token_id
    # The token `shift` before each one, or EOS where it is missing or an EOS lies in between, as
    # `_shift_right_ignore_eos` does on a flat row
    shifted_tokens = [tokens]
    index, valid = parent_index, parent_index >= 0
    for _ in range(1, embedding.ngram_size):
        previous = tokens[index.clamp_min(0)]
        valid = valid & (index >= 0)
        shifted_tokens.append(torch.where(valid, previous, eos))
        valid = valid & (previous != eos)
        index = _ancestor(parent_index, index)
    shifted_tokens = [t.unsqueeze(0) for t in shifted_tokens]

    blocks = []
    for ngram in range(2, embedding.ngram_size + 1):
        start_idx = (ngram - 2) * embedding.heads_per_ngram
        end_idx = start_idx + embedding.heads_per_ngram
        mixed_ids = shifted_tokens[0] * embedding.layer_multipliers[0]
        for position in range(1, ngram):
            mixed_ids = torch.bitwise_xor(
                mixed_ids,
                shifted_tokens[position] * embedding.layer_multipliers[position],
            )
        head_vocab_sizes = embedding.ngram_heads_vocab_sizes[start_idx:end_idx]
        head_offsets = embedding.ngram_heads_offsets[start_idx:end_idx]
        ngram_ids = torch.remainder(mixed_ids.unsqueeze(-1), head_vocab_sizes.view(1, 1, -1))
        blocks.append(ngram_ids + head_offsets.view(1, 1, -1))

    ngram_ids = torch.cat(blocks, dim=-1)
    execution_device = (
        embedding.ngram_embedding.weight.device if embedding.ngram_embedding.weight.device.type != "meta" else None
    )
    return embedding.ngram_embedding(ngram_ids.to(execution_device)).to(ngram_ids.device).flatten(-2)


def _tree_short_conv(ple, hidden_states, parent_index):
    """`Qwen4ExpTextPLELayer._short_conv` over a tree-packed row: each tap reads the token `k * dilation` before up the
    path, zero past the first token, as the reference's left padding."""
    weight = ple.conv1d.weight[:, 0]  # (channels, kernel_size), depthwise
    kernel_size, dilation = weight.shape[1], ple.conv1d.dilation[0]
    hidden_states = hidden_states[0]
    out = hidden_states * weight[:, kernel_size - 1]
    index = torch.arange(len(hidden_states), device=hidden_states.device)
    for tap in range(kernel_size - 2, -1, -1):
        for _ in range(dilation):
            index = _ancestor(parent_index, index)
        out = out + torch.where((index >= 0)[:, None], hidden_states[index.clamp_min(0)], 0) * weight[:, tap]
    return F.silu(out).unsqueeze(0)


def tree_ple(ple, hidden_states, input_ids, parent_index):
    """`Qwen4ExpTextPLELayer.forward` over a tree-packed row (no padding, no cache)."""
    embeddings = _tree_ngram_embedding(ple.ple_embedding, input_ids, parent_index)
    key_normed = ple.norm_key(ple.key_proj(embeddings)).unflatten(-1, (ple.hc_count, ple.hidden_size))
    value = ple.value_proj(embeddings)
    query_normed = ple.norm_query(hidden_states).unflatten(-1, (ple.hc_count, ple.hidden_size))
    gate = (key_normed * query_normed).sum(dim=-1, keepdim=True) / math.sqrt(ple.hidden_size)
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    gated_value = torch.sigmoid(gate) * value.unsqueeze(-2)
    gated_value_normed = ple.norm_conv(gated_value.flatten(-2))
    gated_value = gated_value.flatten(-2)
    return gated_value + _tree_short_conv(ple, gated_value_normed, parent_index)
