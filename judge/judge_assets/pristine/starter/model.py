"""TinyGPT — the normative model for the KV-cache decoding task.

This file is the ground truth: `task_prompt.md`'s model description is an
informative summary of exactly this code. The judge always uses its own
pristine copy of this file and `weights.pt`; edits made to the workspace copy
are never used at judge time.

Architecture: pre-LN GPT-style decoder. vocab 66 (char-level), d_model 528,
3 layers, 4 heads (head_dim 132), GELU (exact erf) MLP at 4x, no biases
anywhere, gain-only LayerNorm (eps 1e-5), tied input/output embeddings, fp32,
max context 512. Positions via RoPE on q and k over the full head dim
(66 rotation pairs, base 10000); no learned positional embedding.
Parameter count: 10,074,768 (asserted at construction).
"""

import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB_SIZE = 66
D_MODEL = 528
N_LAYER = 3
N_HEAD = 4
HEAD_DIM = D_MODEL // N_HEAD  # 132
MAX_CONTEXT = 512
ROPE_BASE = 10000.0
LN_EPS = 1e-5
PARAM_COUNT = 10_074_768

_DIR = os.path.dirname(os.path.abspath(__file__))


def apply_rope(x, positions):
    """Apply rotary position embedding over the full head dim.

    x:         FloatTensor [..., T, HEAD_DIM]
    positions: LongTensor broadcastable to x's T dimension — [T] or [..., T].

    Rotation pairs are (x[..., i], x[..., i + HEAD_DIM//2]) for
    i in 0..HEAD_DIM//2-1 (66 pairs); the angle for pair i at position p is
    p * ROPE_BASE ** (-2 i / HEAD_DIM). fp32 throughout.
    """
    half = HEAD_DIM // 2
    inv_freq = ROPE_BASE ** (
        -torch.arange(half, dtype=torch.float32) * 2.0 / HEAD_DIM
    )
    ang = positions.to(torch.float32).unsqueeze(-1) * inv_freq  # [..., T, half]
    cos, sin = torch.cos(ang), torch.sin(ang)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class LayerNormGain(nn.Module):
    """Gain-only LayerNorm: normalized * weight, no bias. eps 1e-5."""

    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, None, LN_EPS)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = LayerNormGain(D_MODEL)
        self.attn_qkv = nn.Linear(D_MODEL, 3 * D_MODEL, bias=False)
        self.attn_out = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.ln2 = LayerNormGain(D_MODEL)
        self.mlp_up = nn.Linear(D_MODEL, 4 * D_MODEL, bias=False)
        self.mlp_down = nn.Linear(4 * D_MODEL, D_MODEL, bias=False)

    def forward(self, x, positions):
        B, T, _ = x.shape
        h = self.ln1(x)
        qkv = self.attn_qkv(h)
        q, k, v = qkv.split(D_MODEL, dim=-1)
        q = q.view(B, T, N_HEAD, HEAD_DIM).transpose(1, 2)  # [B, H, T, Dh]
        k = k.view(B, T, N_HEAD, HEAD_DIM).transpose(1, 2)
        v = v.view(B, T, N_HEAD, HEAD_DIM).transpose(1, 2)
        q = apply_rope(q, positions)
        k = apply_rope(k, positions)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(HEAD_DIM)  # [B, H, T, T]
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
        att = att.masked_fill(~causal, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v  # [B, H, T, Dh]
        y = y.transpose(1, 2).contiguous().view(B, T, D_MODEL)
        x = x + self.attn_out(y)
        h = self.ln2(x)
        h = self.mlp_down(F.gelu(self.mlp_up(h), approximate="none"))
        return x + h


class TinyGPT(nn.Module):
    """forward(input_ids: LongTensor[B, T]) -> FloatTensor[B, T, 66].

    Applies NO padding mask: every position attends causally to all earlier
    tensor positions, and RoPE positions are 0..T-1 per row. Ragged-batch
    handling is entirely the caller's responsibility.
    """

    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.blocks = nn.ModuleList(Block() for _ in range(N_LAYER))
        self.ln_f = LayerNormGain(D_MODEL)
        n_params = sum(p.numel() for p in self.parameters())
        assert n_params == PARAM_COUNT, (
            f"parameter count {n_params} != pinned {PARAM_COUNT}"
        )

    def forward(self, input_ids):
        B, T = input_ids.shape
        assert T <= MAX_CONTEXT, f"sequence length {T} > max context {MAX_CONTEXT}"
        positions = torch.arange(T, dtype=torch.long)
        x = self.tok_emb(input_ids)
        for block in self.blocks:
            x = block(x, positions)
        x = self.ln_f(x)
        return x @ self.tok_emb.weight.t()  # tied embeddings


def load_model(weights_path=None):
    """Build TinyGPT and load the pinned weights; returns the model in eval mode."""
    model = TinyGPT()
    path = weights_path or os.path.join(_DIR, "weights.pt")
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def load_vocab(vocab_path=None):
    """Returns (stoi, itos) for the 66-char vocabulary."""
    path = vocab_path or os.path.join(_DIR, "vocab.json")
    chars = json.load(open(path))["chars"]
    return {c: i for i, c in enumerate(chars)}, chars


@torch.no_grad()
def generate(model, prompt_ids, max_new_tokens):
    """Reference greedy decoder — correct but slow (the correctness ground truth).

    Re-runs the full forward pass over the entire sequence for every new
    token, one sequence at a time.

    prompt_ids: 1-D LongTensor. Returns (new_tokens LongTensor[N],
    step_logits FloatTensor[N, 66]) — only the newly generated tokens and the
    logits used to pick each one. Greedy: argmax over the 66 logits at the
    last position; on an exact tie the lowest index wins (torch.argmax CPU).
    """
    seq = prompt_ids.to(torch.long)
    new_tokens, step_logits = [], []
    for _ in range(max_new_tokens):
        logits = model(seq.unsqueeze(0))[0, -1]  # [66]
        tok = torch.argmax(logits)
        new_tokens.append(tok)
        step_logits.append(logits)
        seq = torch.cat([seq, tok.view(1)])
    return torch.stack(new_tokens), torch.stack(step_logits)
