#!/usr/bin/env python3
"""Asset prep: train starter/weights.pt and run the skip-rate acceptance test.

Not part of the agent environment or the judge — this is the offline tool that
produced the pinned weights. Training text comes from the corpus grammar
(judge/gen_corpus.py, the same library the judge samples prompts with) at a
pinned seed. The acceptance test (docs/design.md, "Calibration"): on freshly
sampled decode sets (judge-like distribution, held-out seed), the fraction of
generated positions whose reference top-2 logit gap is < 1e-3 must be < 0.5%.

Training may use all machine threads (asset prep only); the shipped artifact
is pinned by sha256, so cross-machine training reproducibility is not claimed.

Usage: python tools/train_weights.py [--steps 300] [--resume]
"""

import argparse
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "judge"))
sys.path.insert(0, os.path.join(ROOT, "env", "starter"))

import gen_corpus  # noqa: E402
import model as M  # noqa: E402

TRAIN_TEXT_SEED = 777      # pinned; distinct from the judge's nonce-derived seeds
TRAIN_TEXT_CHARS = 3_000_000
TORCH_SEED = 777
ACCEPT_TEXT_SEED = 90_001  # held-out seed for the acceptance decode sets
SEQ, BATCH = 256, 32
PEAK_LR, WARMUP, WD, CLIP = 6e-4, 50, 0.1, 1.0
WEIGHTS = os.path.join(ROOT, "env", "starter", "weights.pt")


def grammar_text(seed, n_chars):
    rng = random.Random(seed)
    parts, n = [], 0
    while n < n_chars:
        p = gen_corpus.make_paragraph(rng)
        parts.append(p)
        n += len(p) + 2
    return "\n\n".join(parts) + "\n"


def train(steps, resume):
    torch.manual_seed(TORCH_SEED)
    stoi, _ = M.load_vocab()
    text = grammar_text(TRAIN_TEXT_SEED, TRAIN_TEXT_CHARS)
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    print(f"train text: {len(text):,} chars")

    net = M.TinyGPT()
    if resume and os.path.exists(WEIGHTS):
        net.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
        print("resumed from", WEIGHTS)
    net.train()
    decay = [p for n, p in net.named_parameters() if p.dim() >= 2]
    nodecay = [p for n, p in net.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": WD}, {"params": nodecay, "weight_decay": 0.0}],
        lr=PEAK_LR, betas=(0.9, 0.95), eps=1e-8,
    )
    rng = torch.Generator().manual_seed(TORCH_SEED + 1)
    t0 = time.time()
    for step in range(1, steps + 1):
        lr = PEAK_LR * (step / WARMUP if step <= WARMUP else
                        0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - WARMUP) / max(1, steps - WARMUP))))
        for g in opt.param_groups:
            g["lr"] = lr
        ix = torch.randint(0, len(data) - SEQ - 1, (BATCH,), generator=rng)
        xb = torch.stack([data[i:i + SEQ] for i in ix])
        yb = torch.stack([data[i + 1:i + SEQ + 1] for i in ix])
        logits = net(xb)
        loss = F.cross_entropy(logits.view(-1, M.VOCAB_SIZE), yb.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP)
        opt.step()
        if step % 50 == 0 or step == 1:
            print(f"step {step:5d}  loss {loss.item():.4f}  lr {lr:.2e}  {time.time()-t0:.0f}s")
    net.eval()
    torch.save(net.state_dict(), WEIGHTS)
    print("saved", WEIGHTS)
    return net


@torch.no_grad()
def acceptance(net):
    """Skip-rate measurement on judge-like decode sets (held-out text seed)."""
    stoi, _ = M.load_vocab()
    text = grammar_text(ACCEPT_TEXT_SEED, 400_000)
    ids = [stoi[c] for c in text]
    rng = random.Random(ACCEPT_TEXT_SEED)
    gaps = []
    sets = [(8, 96, 192, 64)] * 3 + [(8, 16, 192, 32)]  # S-like x3 + E2-like
    for B, tlo, thi, N in sets:
        for _ in range(B):
            T = rng.randint(tlo, thi)
            start = rng.randint(0, len(ids) - T - 1)
            prompt = ids[start:start + T]
            if rng.random() < 1 / 3:  # perturbed variant, 10% i.i.d. substitution
                prompt = [rng.randrange(66) if rng.random() < 0.10 else c for c in prompt]
            seq = torch.tensor(prompt, dtype=torch.long)
            for _ in range(N):
                logits = net(seq.unsqueeze(0))[0, -1]
                top2 = torch.topk(logits, 2).values
                gaps.append((top2[0] - top2[1]).item())
                seq = torch.cat([seq, torch.argmax(logits).view(1)])
    gaps = torch.tensor(gaps)
    skip = (gaps < 1e-3).float().mean().item()
    print(f"acceptance: {len(gaps)} positions, skip rate {skip*100:.3f}% "
          f"(bar < 0.5%), min gap {gaps.min().item():.2e}, "
          f"p1 gap {gaps.quantile(0.01).item():.4f}")
    print("ACCEPT" if skip < 0.005 else "REJECT — train longer")
    return skip


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--accept-only", action="store_true")
    args = ap.parse_args()
    if args.accept_only:
        net = M.load_model()
    else:
        net = train(args.steps, args.resume)
    acceptance(net)
