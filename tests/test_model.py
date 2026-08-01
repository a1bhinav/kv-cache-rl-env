"""Starter-model unit tests: pinned tie-break semantics and parameter count."""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "env", "starter"))

import model as M  # noqa: E402


def test_argmax_lowest_index_tiebreak():
    """The contract pins greedy tie-break to torch.argmax CPU behavior:
    on exactly equal maxima, the lowest index wins."""
    v = torch.zeros(66)
    v[3] = 5.0
    v[41] = 5.0
    assert torch.argmax(v).item() == 3
    # and in the batched form the judge uses: argmax over the last dim
    m = torch.stack([v, v.flip(0)])
    assert torch.argmax(m, dim=-1).tolist() == [3, 66 - 1 - 41]


def test_param_count_pinned():
    net = M.TinyGPT()  # __init__ itself asserts PARAM_COUNT
    assert sum(p.numel() for p in net.parameters()) == M.PARAM_COUNT == 10_074_768


def test_rope_position_zero_is_identity():
    x = torch.randn(1, 1, 4, M.HEAD_DIM)
    out = M.apply_rope(x, torch.zeros(4, dtype=torch.long))
    assert torch.equal(out, x) or torch.allclose(out, x, atol=0, rtol=0)
