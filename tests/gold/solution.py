"""Gold solution — batched KV-cached greedy decoding for TinyGPT.

Design, and why each piece is shaped the way it is:

  * **Prefill runs per sequence, not as a padded batch.** Each row's prompt
    goes through a [1, T_i, d] forward that reproduces `Block.forward`
    operation for operation, so the prefilled K/V are bit-identical to the
    reference's — no padded-softmax reduction to perturb them. The prefill is
    ~1/64 of the naive path's work, so this costs almost nothing.
  * **Step 0's logits come from one stacked [B, d] @ [d, 66] GEMM.** The
    reference gets them from a [T, d] @ [d, 66] GEMM. Those agree bit-for-bit,
    whereas a per-row [1, d] GEMV disagrees by ~4e-5 (a different BLAS kernel
    reduces the 528-wide dot product in a different order, and the logits here
    run to |x| ~ 45). The contract guarantees B >= 2, so the stacked form is
    always available.
  * **Decode is batched with left-aligned caches.** Row i's prompt occupies
    cache columns [Tmax - T_i, Tmax); generated tokens extend from Tmax. Every
    row's newest token therefore lands in the same column each step, so one
    [B, 1, d] pass serves the whole batch.
  * **Per-row RoPE positions** come from one [B, S] table: row i's column j
    carries true position j - (Tmax - T_i). cos/sin are precomputed at import
    for 0..MAX_CONTEXT-1 — prompt-independent, and bit-identical to
    `starter.model.apply_rope`, which forms the same fp32 products.
  * **Pad columns are masked with an additive -inf bias** and the K/V cache is
    zero-initialized, so a masked column contributes exactly 0 * 0 and can
    never leak (or turn into NaN) across sequences.

Step semantics match the reference exactly: step 0's logits are the ones at
the last prompt position, and step t > 0 feeds token t-1 at true position
T_i + t - 1.
"""

import math

import torch
import torch.nn.functional as F

from starter.model import (D_MODEL, HEAD_DIM, LN_EPS, MAX_CONTEXT, N_HEAD,
                           ROPE_BASE)

_HALF = HEAD_DIM // 2
_INV_FREQ = ROPE_BASE ** (-torch.arange(_HALF, dtype=torch.float32) * 2.0 / HEAD_DIM)
_ANG = torch.arange(MAX_CONTEXT, dtype=torch.float32).unsqueeze(-1) * _INV_FREQ
_COS = torch.cos(_ANG)  # [MAX_CONTEXT, HEAD_DIM // 2]
_SIN = torch.sin(_ANG)
_SCALE = math.sqrt(HEAD_DIM)


def _rope(x, pos_idx):
    """x [B, H, T, Dh]; pos_idx [B, T] long -> rotated x, per-row positions."""
    cos = _COS[pos_idx].unsqueeze(1)  # [B, 1, T, half]
    sin = _SIN[pos_idx].unsqueeze(1)
    x1, x2 = x[..., :_HALF], x[..., _HALF:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def _ln(x, weight):
    return F.layer_norm(x, weight.shape, weight, None, LN_EPS)


@torch.no_grad()
def generate_cached(model, prompt_ids, max_new_tokens):
    B = len(prompt_ids)
    N = int(max_new_tokens)
    lens = [int(p.numel()) for p in prompt_ids]
    t_max = max(lens)
    s_max = t_max + N - 1  # cache columns: prompt + the N-1 fed-back tokens

    blocks = list(model.blocks)
    emb_w = model.tok_emb.weight
    lnf_w = model.ln_f.weight
    n_layer = len(blocks)

    # Zero-init so masked pad columns contribute exactly 0, never NaN.
    k_cache = [torch.zeros(B, N_HEAD, s_max, HEAD_DIM) for _ in range(n_layer)]
    v_cache = [torch.zeros(B, N_HEAD, s_max, HEAD_DIM) for _ in range(n_layer)]

    pos = torch.zeros(B, s_max, dtype=torch.long)
    key_ok = torch.ones(B, s_max, dtype=torch.bool)
    last_hidden = torch.empty(B, D_MODEL)

    # ---- prefill, one sequence at a time (matches the reference exactly) ----
    for i, prompt in enumerate(prompt_ids):
        t_i = lens[i]
        off = t_max - t_i
        pos[i, off:t_max] = torch.arange(t_i, dtype=torch.long)
        pos[i, t_max:] = torch.arange(t_i, t_i + N - 1, dtype=torch.long)
        key_ok[i, :off] = False

        row_pos = torch.arange(t_i, dtype=torch.long)
        x = model.tok_emb(prompt.view(1, t_i))
        causal = torch.tril(torch.ones(t_i, t_i, dtype=torch.bool))
        for li, blk in enumerate(blocks):
            h = _ln(x, blk.ln1.weight)
            q, k, v = F.linear(h, blk.attn_qkv.weight).split(D_MODEL, dim=-1)
            q = q.view(1, t_i, N_HEAD, HEAD_DIM).transpose(1, 2)
            k = k.view(1, t_i, N_HEAD, HEAD_DIM).transpose(1, 2)
            v = v.view(1, t_i, N_HEAD, HEAD_DIM).transpose(1, 2)
            q = _rope(q, row_pos.view(1, t_i))
            k = _rope(k, row_pos.view(1, t_i))
            k_cache[li][i, :, off:t_max] = k[0]
            v_cache[li][i, :, off:t_max] = v[0]
            att = (q @ k.transpose(-2, -1)) / _SCALE
            att = att.masked_fill(~causal, float("-inf"))
            y = F.softmax(att, dim=-1) @ v
            x = x + F.linear(y.transpose(1, 2).reshape(1, t_i, D_MODEL),
                             blk.attn_out.weight)
            h = _ln(x, blk.ln2.weight)
            x = x + F.linear(F.gelu(F.linear(h, blk.mlp_up.weight), approximate="none"),
                             blk.mlp_down.weight)
        last_hidden[i] = x[0, -1]

    # Additive attention bias: 0 for usable keys, -inf for pad columns.
    bias = torch.zeros(B, 1, 1, s_max)
    bias.masked_fill_(~key_ok.view(B, 1, 1, s_max), float("-inf"))

    logits = _ln(last_hidden, lnf_w) @ emb_w.t()  # [B, 66], GEMM (B >= 2)
    new_tokens = torch.empty(B, N, dtype=torch.long)
    step_logits = torch.empty(B, N, logits.shape[-1])
    step_logits[:, 0] = logits
    cur = torch.argmax(logits, dim=-1)
    new_tokens[:, 0] = cur

    # ---- batched decode ------------------------------------------------------
    for t in range(1, N):
        slot = t_max + t - 1
        step_pos = pos[:, slot:slot + 1]
        b = bias[:, :, :, :slot + 1]
        x = model.tok_emb(cur).view(B, 1, D_MODEL)
        for li, blk in enumerate(blocks):
            h = _ln(x, blk.ln1.weight)
            q, k, v = F.linear(h, blk.attn_qkv.weight).split(D_MODEL, dim=-1)
            q = _rope(q.view(B, 1, N_HEAD, HEAD_DIM).transpose(1, 2), step_pos)
            k = _rope(k.view(B, 1, N_HEAD, HEAD_DIM).transpose(1, 2), step_pos)
            v = v.view(B, 1, N_HEAD, HEAD_DIM).transpose(1, 2)
            kc, vc = k_cache[li], v_cache[li]
            kc[:, :, slot:slot + 1] = k
            vc[:, :, slot:slot + 1] = v
            K = kc[:, :, :slot + 1]
            V = vc[:, :, :slot + 1]
            att = (q @ K.transpose(-2, -1)) / _SCALE + b
            y = F.softmax(att, dim=-1) @ V
            x = x + F.linear(y.transpose(1, 2).reshape(B, 1, D_MODEL),
                             blk.attn_out.weight)
            h = _ln(x, blk.ln2.weight)
            x = x + F.linear(F.gelu(F.linear(h, blk.mlp_up.weight), approximate="none"),
                             blk.mlp_down.weight)
        logits = _ln(x[:, 0], lnf_w) @ emb_w.t()
        step_logits[:, t] = logits
        cur = torch.argmax(logits, dim=-1)
        new_tokens[:, t] = cur

    return new_tokens, step_logits
