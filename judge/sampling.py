"""Nonce-driven prompt sampling (docs/design.md, "Prompt sampling").

All judge RNG derives from a single uint64 nonce via numpy PCG64. Prompt text
is generated at judge time by the corpus grammar (gen_corpus.py) with a
nonce-derived seed — never the shipped seed-42 text, never shipped text at
all. 2/3 clean slices, 1/3 perturbed (i.i.d. char substitution at 10% per
position, chars drawn from the 66-char vocab).

`local_check.py` uses this same module with a locally chosen nonce; the judge
uses a fresh nonce the agent never sees. Only the nonce differs.
"""

import json
import os
import random

import numpy as np
import torch

import gen_corpus

_STARTER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "env", "starter")

# name, B, T_lo, T_hi, N   (bounds published in task_prompt.md; sizes per spec §3)
SET_SPECS = [
    ("W", 2, 24, 48, 16),
    ("E1", 4, 8, 96, 32),
    ("E2", 8, 16, 192, 32),
    ("E3", 6, 64, 192, 16),
    ("S1", 8, 96, 192, 64),
    ("S2", 8, 96, 192, 64),
    ("S3", 8, 96, 192, 64),
]

PERTURB_FRAC = 1 / 3
PERTURB_RATE = 0.10
TEXT_POOL_CHARS = 120_000


def load_stoi(vocab_path=None):
    path = vocab_path or os.path.join(_STARTER, "vocab.json")
    chars = json.load(open(path))["chars"]
    return {c: i for i, c in enumerate(chars)}


def _grammar_text(seed, n_chars):
    rng = random.Random(seed)
    parts, n = [], 0
    while n < n_chars:
        p = gen_corpus.make_paragraph(rng)
        parts.append(p)
        n += len(p) + 2
    return "\n\n".join(parts) + "\n"


def sample_sets(nonce, vocab_path=None):
    """Returns {set_name: (prompts, N)} where prompts is a list of B 1-D
    LongTensors, sampled per SET_SPECS from the given nonce."""
    rng = np.random.Generator(np.random.PCG64(int(nonce)))
    stoi = load_stoi(vocab_path)
    text = _grammar_text(int(rng.integers(0, 2**63)), TEXT_POOL_CHARS)
    ids = np.array([stoi[c] for c in text], dtype=np.int64)

    sets = {}
    for name, B, tlo, thi, N in SET_SPECS:
        prompts = []
        for _ in range(B):
            T = int(rng.integers(tlo, thi + 1))
            assert T + N <= 256
            start = int(rng.integers(0, len(ids) - T))
            p = ids[start:start + T].copy()
            if rng.random() < PERTURB_FRAC:
                mask = rng.random(T) < PERTURB_RATE
                p[mask] = rng.integers(0, 66, size=int(mask.sum()))
            prompts.append(torch.from_numpy(p))
        sets[name] = (prompts, N)
    return sets
