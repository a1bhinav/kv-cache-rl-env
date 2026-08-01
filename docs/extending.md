# Extending the environment

## Difficulty levers

The contract in `env/task_prompt.md` is pinned tightly enough that difficulty can be
raised without making the task vaguer. Each lever below leaves the agent-facing
specification exactly as precise as it is now:

- **Sliding-window attention.** Bounds the cache the agent must maintain and adds an
  eviction boundary that must line up exactly with the reference. The interesting bugs
  are off-by-ones at the window edge, which the per-sequence equivalence gate catches.
- **A cache memory cap.** Forces preallocation and in-place writes rather than repeated
  `torch.cat`, which is the naive way to grow a cache. Enforce it with the existing RSS
  budget so no new judge machinery is needed.
- **Longer contexts.** Raises the arithmetic-intensity ceiling and with it the
  achievable multiple, so the speed bar can rise without becoming unreachable.
- **A higher speed bar.** The cheapest lever: raise the 8× pass line. Re-measure the
  reference first and keep it at ≥ 1.5× the new bar, or the environment starts denying
  honest work.
- **Continuous batching.** Sequences finish at different lengths, so the agent must
  retire and refill rows mid-flight. This is where real inference engines live, and it
  makes cross-sequence contamination much easier to introduce by accident.
- **Beam or speculative decoding.** Changes the reference semantics from greedy argmax
  to something with its own tie-breaking rules, which must be pinned as exactly as the
  current argmax rule is.

The first four require no judge changes at all. The last two need a new reference
implementation, and the reference is what every threshold is calibrated against — so
recalibrate the logit tolerance and skip rate before trusting the results.

## Other environments in this family

The property that makes this environment cheap to trust is that **the reference is the
shipped code itself**, so there is no convention left to guess and equivalence is exactly
checkable. Designs that keep that property inherit the judge's robustness; designs that
give it up have to buy verification some other way.

**Debug a seeded-bug training run.** Ship a trainer with several *interacting* planted
bugs — a causal-mask off-by-one that makes train loss look good and eval garbage,
gradient accumulation that sums without dividing, an integer-division warm-up that pins
the learning rate at zero, unshifted labels against broken weight tying. The judge
re-runs training from its own seed on the agent's *code*, verifies a frozen config hash
and parameter count, and evaluates on a shard injected only at judge time. The hacking
surface is small for a good reason: training on the eval set is impossible because it
never exists in the agent's environment, and smuggled weights are irrelevant because the
judge retrains. The bugs mask each other, which is what makes it genuinely hard.

*What makes it hard to judge well:* run-to-run variance. At the ~10M-parameter scale a
short compute budget allows, seed and data-order luck move final validation loss by
around half a nat — measured at 0.51 nats of spread across three seeds, five times the
0.10 that a loss-threshold scoring rule can absorb. A threshold loose enough to cover
that variance guts the difficulty it exists to create. It needs either larger batches on
dedicated hardware, where the spread tightens, or a scoring rule that is robust to
variance by construction — ranking against a seed-matched reference run rather than an
absolute threshold. The judge is also a full retrain per scoring run, which makes the
red-teaming loop that this repo depends on impractical.

**Implement a paper technique against a hidden reference** (LoRA merging, a RoPE-scaling
scheme). Structurally squeezed from both sides: pin every convention and it degenerates
into spec translation with the difficulty gone; leave conventions open and honest
solutions diverge from the hidden reference, generating false negatives. It is viable
only for techniques with exactly one canonical formulation — which are the ones strong
models already know cold. This environment keeps the good half of the idea, comparing
against a reference the agent cannot see, while dodging the bad half.

**Near-duplicate detection under budget.** MinHash/LSH dedup over a corpus, hitting
precision and recall targets on hidden labeled pairs within runtime and memory budgets.
Verification is clean and the hacking surface is small, so the judge is easy to trust.

*What makes it hard to judge well:* the algorithm is well-trodden enough that strong
models likely one-shot it, and scaling the corpus to restore difficulty makes the judge
slow and flaky — trading away exactly the cheapness that makes adversarial testing
affordable.
