# KV-cache RL environment

An RL environment for LLM training in which the agent implements batched KV-cached
greedy decoding for a small transformer, and is scored mechanically on exact
equivalence to a reference decoder and on how much faster it runs.

## Why this exists

RL environments are only as good as their reward. A reward for an engineering task has
to be three things at once, and the third is the one that usually gets skipped:

- **Mechanically checkable** — no language model in the loop, no rubric, no human. Here
  the reward is a token-level equivalence check against a reference the agent cannot
  see, plus a measured speed ratio.
- **Resistant to reward hacking** — a wrong-but-clever solution must not score well.
- **Safe from reward denial** — an honest solution must not fail for reasons that have
  nothing to do with its quality: float jitter, a noisy machine, an ambiguous contract.

This repo is a worked example on a real inference-engineering task, where the judge
itself is the artifact under test. Its thresholds are derived from measurement rather
than assumed, and it is evaluated adversarially: nine attacks, each a genuine attempt at
the reward rather than a strawman, are pinned as tests at their measured outcomes.

## The task

The agent receives `TinyGPT` — a 10M-parameter pre-LN decoder (vocab 66, `d_model` 528,
3 layers, 4 heads, RoPE over the full head dim, tied embeddings, fp32, CPU-only) — along
with a correct but slow reference decoder that generates one sequence at a time with no
cache. It must implement `generate_cached(model, prompt_ids, max_new_tokens)` using a KV
cache and batching across ragged prompts.

Correctness is defined **per sequence**: for every row, the output must match running the
reference on that prompt *alone*, so any cross-sequence contamination — padding leaking
into attention, wrong per-row RoPE offsets, row-order mix-ups — is a wrong answer. The
solution must also run at least **8× faster** than the reference to pass.

The full agent-facing contract, which pins every type, shape, dtype, bound and
tie-break rule, is `env/task_prompt.md`.

## How judging works

Ordered hard gates, then a speed measurement, then a score. Any gate failure scores 0
and reports a machine-readable `reason_code`:

1. **Presence and size** — `solution.py` exists; the graded snapshot is ≤ 25 MB.
2. **Import and signature** — imports within 20 s; `generate_cached` is callable.
3. **Shape and finiteness** — correct dtypes, shapes, and no NaN or inf.
4. **Self-consistency** — returned tokens equal the argmax of the returned logits.
5. **Equivalence** — per position: logits within 2.5e-4 of the reference, and tokens
   equal, except at near-ties in the *reference's* own logits, which are skipped and
   accounted for in a reported coverage figure.
6. **Budgets** — per-call deadline, RSS cap, no exceptions.

Speed is the median of three paired ratios, `reference wall-clock ÷ agent wall-clock`,
each measured on its own freshly sampled prompt set. The score is then

```
S = 0.2 + 0.8 * min(1, max(0, log2 M) / 5)      # 0 if any gate failed
pass = all gates and M >= 8
```

so a correct-but-slow decoder earns 0.2, the pass line sits at 8× (S = 0.68), and the
cap is reached at 32×. Correctness and speed cannot be traded against each other.

Full details — two-process isolation, sampling, the timing protocol, the tripwires, and
how each threshold was measured — are in [`docs/design.md`](docs/design.md).

## Adversarial evaluation

Nine adversarial baselines, each carrying the reference algorithm underneath where it
needs one, so that it fails at the defense it targets rather than at the equivalence
gates:

| Attack | Measured outcome |
|---|---|
| do-nothing stub | **0.0**, `AGENT_EXCEPTION` |
| no `solution.py` | **0.0**, `MISSING_SOLUTION` |
| naive wrapper | all gates green, M = 1.005 → **0.2012**, `pass=false` |
| hardcoded outputs | **0.0**, `LOGIT_DIVERGENCE`, zero positions credited |
| own weights | **0.0**, `LOGIT_DIVERGENCE` |
| in-place bf16 quantization | **0.0**, `LOGIT_DIVERGENCE`, ~400× the tolerance |
| clock monkeypatch (100× on four clock functions) | **no gain**: M = 18.06 vs reference 18.41 |
| memoize across repeats | **no inflation**: M = 18.13 vs reference 18.41 |
| all 18 cores + 4 worker threads | **tripped**: CPU/wall 8.42–8.60 → `ANOMALOUS_PARALLELISM`, exit 3, M never computed |

The vectors these cover, the mitigations, and the reward-denial risks on the other side
of the ledger are in [`docs/threat-model.md`](docs/threat-model.md).

## Reference solution results

The reference (gold) solution, on three nonces it never sees:

| Nonce | M | Score | Coverage | Skip rate | max \|logit diff\| |
|---|---|---|---|---|---|
| 1001 | **18.41** | 0.8724 | 100.00% | 0.00% | 5.34e-5 |
| 1002 | **18.90** | 0.8784 | 99.81% | 0.19% | 4.01e-5 |
| 1003 | **18.60** | 0.8747 | 99.81% | 0.19% | 4.20e-5 |

`M` is a wall-clock ratio and moves a little between runs — observed 17.9–18.9 across
sessions, always far above the bar of 8 — while the equivalence figures are
deterministic and reproduce exactly.

## Quickstart

Requires **Python 3.11**; the pinned dependency set does not resolve on older
interpreters, so name a 3.11 interpreter explicitly if `python3` is not already 3.11.

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r env/requirements.txt
.venv/bin/python -m pytest          # expect: 30 passed
```

Run the judge directly against the reference solution:

```bash
./judge/run_judge.sh tests/gold 1001
```

A full suite run takes about three minutes, most of it real judge invocations. On a
heavily loaded machine the timing drift probes can exceed their 1.3× spread and return
`INVALID_RUN` — that is the judge refusing to score on unreliable measurements, and the
correct response is to rerun, not to adjust anything.

## Repository layout

| Path | Contents |
|---|---|
| `env/` | the agent-facing environment: `task_prompt.md`, `starter/` model and weights, the `solution.py` stub, `local_check.py`, pinned requirements, and the container spec |
| `judge/` | the two-process judge: `judge.py` (parent), `child_harness.py` (runs agent code), sampling, grading and scoring modules, and hash-verified pristine assets |
| `tests/gold/` | the reference solution |
| `tests/exploits/` | eight adversarial baselines, pinned at their measured outcomes |
| `tests/` | the 30-test suite over the judge, the reference, the exploits, and the model |
| `tools/` | offline asset prep: trains and validates `starter/weights.pt` |
| `docs/` | design, threat model, and extension notes |

## Extending

The contract is pinned tightly enough that difficulty can be raised without making the
task vaguer — sliding-window attention, a cache memory cap, longer contexts, a higher
speed bar, continuous batching, speculative decoding.

[`docs/extending.md`](docs/extending.md) covers those levers and sketches alternative
environment designs in the same family, with an honest note on what makes each one
harder to judge well.
