# Judge design

The judge decides, mechanically and without a language model in the loop, whether a
candidate `generate_cached` is *equivalent* to the reference decoder and *how much
faster* it is. This document describes how it is built and why each threshold has the
value it has.

Two principles govern everything below:

1. **The judge trusts only its own assets and its own arithmetic.** Reference outputs
   are computed live from pristine, hash-verified copies. The only thing read from the
   candidate is `/workspace/solution.py` and whatever it imports from the workspace
   snapshot.
2. **Thresholds are derived from measurement, never tuned to pass.** Every constant in
   `judge/` traces back to a measured quantity, recorded in [Calibration](#calibration).
   When a measurement contradicts a threshold, the threshold moves and the reason is
   written down — the test result never moves.

A corollary worth stating on its own: **judge faults never score the candidate.** Asset
mismatch, nondeterminism, or timing instability all exit `INVALID_RUN` (exit 3) with no
score, which a caller should treat as "run it again", never as a zero.

## Two-process isolation

The security model is a process boundary, not a sandbox.

**Parent** (`judge/judge.py`) holds all judge logic and never executes a line the
candidate wrote. It verifies the manifest, runs the determinism control, samples every
prompt set from the run nonce, computes all reference tokens, logits and per-position
top-2 gaps, and times the naive reference path — **all in memory, never written to
disk**. Only then does it spawn the child. After the child exits, the parent loads the
child's raw outputs, runs every comparison, and applies the score formula.

**Child** (`judge/child_harness.py`) is the only process that runs candidate code: a
~130-line harness with no judge logic and no reference data, driven by JSON lines over
stdin. Four properties fall out of this arrangement rather than being bolted on:

- **The reference cannot be stolen.** It never exists in the process the candidate
  controls, and never exists on disk at all. There is no file to read, and
  `gc.get_objects()` in the child sees only the child's own heap.
- **The clock is captured before agent import.** The child binds
  `_clock = time.perf_counter` at module load, before `solution.py` is imported, so
  import-time monkeypatching of `time.*` cannot reach recorded timings. The naive path
  is timed in the parent, before the child exists at all.
- **Prompts are revealed just in time.** Each call's prompts are written by the parent
  and announced over stdin only as that call begins, so candidate code can never see a
  timed set during an untimed slot.
- **Results travel in a sha-verified envelope.** Every `done` line carries the sha256 of
  the output file it just wrote, and the parent timestamps the line's arrival. That
  gives a parent-measured bound on true compute time, independent of anything the child
  reports about itself.

Limits on the child: RSS ≤ 4 GB, 20 s import deadline, 60 s per-call deadline, 12 min
global deadline. Model load and construction are excluded from all timing; only the
generation loop (parent) and the `generate_cached` call (child) are timed.

## Prompt sampling

All randomness derives from a **run nonce** the candidate never sees; every scoring run
gets a fresh one. Prompts are character slices from text generated *at judge time* by
the corpus grammar with a nonce-derived seed — never the shipped sample text, never
shipped text at all. Two-thirds are clean slices, one-third are perturbed by i.i.d.
character substitution at 10% per position, which stresses off-distribution correctness
while keeping reference logits sharp.

Seven sets per run, all within the bounds published in `env/task_prompt.md`:

| Set | Purpose | B | T_i | N |
|---|---|---|---|---|
| W | warm-up, smoke, stability probe | 2 | [24, 48] | 16 |
| E1 | equivalence, short | 4 | [8, 96] | 32 |
| E2 | equivalence, wide spread | 8 | [16, 192] | 32 |
| E3 | equivalence, long-prompt | 6 | [64, 192] | 16 |
| S1–S3 | timing, a fresh set each | 8 | [96, 192] | 64 |

S1–S3 being independently sampled is what makes memoization across timing repeats
worthless.

## Ordered gates

Failures that carry a `reason_code` score 0. Judge-side faults exit `INVALID_RUN` with
no score.

| # | Gate | Fails as |
|---|---|---|
| 0 | Preflight: manifest hashes match; naive-decode set W twice, bit-identical | `INVALID_RUN` |
| 1 | Presence: `solution.py` exists; snapshot (workspace minus shipped `starter/`) ≤ 25 MB | `MISSING_SOLUTION`, `SNAPSHOT_TOO_LARGE` |
| 2 | Import and signature: imports within 20 s; `generate_cached(model, prompt_ids, max_new_tokens)` callable | `IMPORT_TIMEOUT`, `IMPORT_ERROR`, `BAD_SIGNATURE` |
| 3 | Smoke: returns a 2-tuple; `new_tokens` int64 `[B, N]`; `step_logits` fp32 `[B, N, 66]`; all finite | `BAD_RETURN`, `NONFINITE_LOGITS` |
| 4 | Equivalence, per sequence (below) | `INCONSISTENT_ARGMAX`, `LOGIT_DIVERGENCE`, `TOKEN_MISMATCH` |
| 5 | Timing (see [protocol](#timing-protocol)) | — |
| 6 | Budgets: per-call deadline, RSS, exceptions | `CALL_TIMEOUT`, `RSS_EXCEEDED`, `AGENT_EXCEPTION` |
| 7 | Plausibility: M > 128 | `INVALID_RUN` + `ANOMALOUS_SPEED` |
| 7b | Parallelism: CPU/wall > 2.5 on any timed call | `INVALID_RUN` + `ANOMALOUS_PARALLELISM` |

Gate 0's manifest check covers the pristine `starter/` assets and the corpus grammar
module, and it is deliberately unforgiving: any hash that does not match the file on
disk stops the run at `INVALID_RUN` before a single line of candidate code executes.
That is the intended failure mode for silent tampering — an altered reference model or
a drifted prompt grammar produces a refusal to run rather than a quietly wrong score.
The cost is that *deliberate* edits to a pinned file must be re-recorded on purpose:
run `python tools/make_manifest.py` to see a pinned-vs-current table (exit 1 on drift)
and `--write` to re-pin. Re-pinning is a real change to an integrity record, so it
belongs in its own reviewable commit, and the equivalence figures in
[Calibration](#calibration) should reproduce exactly afterwards — if they move, the
edit was not as cosmetic as it looked.

Gate 4 grades **every** output the child ever returns — including the warm-up and the
timing sets, not just E1–E3 — walking each row independently:

1. **Consistency.** `new_tokens[i,t] == argmax(step_logits[i,t])`, always checked.
2. **Logit closeness**, while the row's prefix still matches the reference:
   `max |candidate − reference| ≤ 2.5e-4` across the 66 logits.
3. **Token equality.** If the *reference's* top-2 gap at that position is `< 1e-3` it is
   a near-tie and the comparison is skipped; if the candidate diverges at a skipped
   position, grading of that row stops and the remaining positions count as uncovered.
   Otherwise the tokens must be equal.

Skips are decided from reference logits only, so a candidate cannot tell which positions
qualify without already matching the reference's numerics. **Coverage** (checked ÷
generated positions) is reported every run: below 95% warns, below 90% is
`INVALID_RUN` — a signal to recalibrate the assets, never a zero for the candidate.

The two tolerances are chosen to be coherent with each other: a 2.5e-4 logit
perturbation can only flip an argmax where the reference top-2 gap is below 5e-4, and
the skip line sits 2× above that. No position can therefore both pass the logit gate and
fail the token gate on float jitter.

## Score

```
if any hard gate failed:   S = 0.0            # reason_code says which
else:                      S = 0.2 + 0.8 * min(1, max(0, log2(M)) / 5)
pass  = all_gates and (M >= 8.0)              # equivalently S >= 0.68
```

| M | ≤ 1 | 2 | 4 | **8** | 16 | ≥ 32 |
|---|---|---|---|---|---|---|
| S | 0.20 | 0.36 | 0.52 | **0.68 (pass)** | 0.84 | 1.00 (cap) |

The 0.2 floor pays for the genuinely hard part — an exactly equivalent cache — so a
correct-but-slow decoder is not scored as a failure. The `log2` curve pays speed work
most where it is hardest to fake, and the cap bounds the incentive beyond the intended
regime. A fast but subtly wrong solution earns 0. The two cannot be traded against each
other, which is what makes the score a useful learning signal rather than a coin flip.

The report is a single JSON object on stdout carrying `score`, `pass`, `reason_code`,
`multiple`, the three `ratios`, per-set `naive_s`/`cached_s`/`cpu_s`, every gate's
state, position accounting, `coverage`, `skip_rate`, `max_logit_diff`, and
`stability_probes`. Exit 0 means scored (pass or fail); exit 3 means `INVALID_RUN`.

## Timing protocol

- The parent times the naive reference: one wall-clock measurement per set S_r around
  the full loop of 8 sequences decoded one at a time with the shipped `generate()`.
- The child times the candidate: one `_clock()` measurement around the single
  `generate_cached(model, prompts_r, 64)` call, on a fresh model instance.
- Both paths get a warm-up call on set W first, and the warm-up output is still graded.
- **Paired ratio per set**, `ratio_r = naive_r / cached_r`, then **M = median of the
  three**. Pairing within a set and taking a median is what makes the measure robust to
  a machine whose step rate wanders mid-run.
- **Drift probes.** The parent re-times a W naive decode at start, after S2, and at end.
  If max/min of the three probes exceeds 1.3, the run exits `INVALID_RUN` — the machine
  is too noisy to time on, so the caller retries rather than recording a bad number.
- **Plausibility cap.** M > 128 is beyond this model's arithmetic-intensity ceiling, so
  it is set aside for review rather than scored.
- **Parallelism tripwire.** The child records process CPU time (`RUSAGE_SELF` +
  `RUSAGE_CHILDREN`, user + sys) alongside wall time around every timed call. Both paths
  are pinned to 2 torch threads, so a compliant solution's CPU/wall ratio cannot exceed
  ~2 however it is written. Above 2.5 the run is set aside as `ANOMALOUS_PARALLELISM`.
  This makes the thread cap *harness-measured on any machine* rather than a property of
  the deployment container.

Note that the speed measure is **relative and never absolute**. There is no tokens/sec
threshold anywhere, so the environment transfers to a new machine without recalibration.

## Calibration

Every threshold below was measured against the reference (gold) solution. The rule is
that a threshold moves only when a *measurement* contradicts it, never because a test is
failing.

**The logit tolerance is the interesting one.** Cached decoding **cannot** reproduce
full recompute bit-for-bit: holding the hidden state fixed, the final projection
contributes exactly 0.0 to the divergence, so the entire residual comes from the
decode's single-token GEMMs reducing in a different order than full-recompute GEMMs —
something every correct cached decoder must do. The divergence is therefore
**irreducible**, and a hair-trigger equality gate would fail honest work.

Measured over 405,504 logit values (6,144 positions across 3 nonces × 7 sets), with
reference logits reaching |x| = 50 where one fp32 ulp is 3.8e-6:

| Statistic | Value | In ulps |
|---|---|---|
| max | 5.34e-5 | 14 |
| p999 | 2.67e-5 | 7 |
| p99 | 1.91e-5 | 5 |
| median | 3.8e-6 | 1 |

An earlier 1e-4 line sat only 1.87× above that measured honest maximum — passing, but
close enough to the honest floor to be a latent source of false failures. The line is
set by an explicit rule: **the smallest round τ that is ≥ 4× the measured honest max and
≤ half the near-tie skip line**, giving **2.5e-4** — a 4.7× margin above honest noise
and 2× of coherence headroom below the skip line.

The rule is then checked against a second honest implementation rather than left as an
assertion: `torch.compile` lands at **4.196e-5**, a **5.96× margin** under the line.
Admitting it to the honest envelope leaves the maximum at 5.34e-5, so both legs of the
rule still hold (`4 × 5.34e-5 = 2.14e-4 ≤ 2.5e-4`, and `2.5e-4 ≤ 1e-3 / 2`). Meanwhile
reduced-precision internals cost 1e-2 to 1e0 at this logit scale — 40× to 4000× above
the line — so bf16 or fp16 internals fail by design, which is stated to the agent in
`env/task_prompt.md` rather than left as a trap.

The remaining constants:

| Constant | Value | How it is fixed |
|---|---|---|
| Speed bar | 8× | Reference clears 1.5× the bar on every pinned nonce (measured M ≈ 18.4–18.9 against a required ≥ 12) |
| Near-tie skip | 1e-3 gap | Weights are trained until the measured skip rate is < 0.5%; observed 0.056% |
| Coverage floor | 90% invalid, 95% warn | Reference coverage measures > 99% on all three nonces |
| CPU/wall tripwire | 2.5 | `max(2.5, 1.25 × measured reference max)`; reference measures 1.043–1.049 over 18 timed calls across 6 runs, so the rule yields `max(2.5, 1.31)` = 2.5 |
| Drift bound | 1.3× probe spread | Confirmed against a false-invalid rate of ~0 across 11 runs |
| Plausibility cap | M > 128 | Held ≥ 3× above measured reference M |

Two baselines are pinned as tests alongside the adversarial ones. The do-nothing stub
must score 0. The naive wrapper — which calls the reference internally, so it is exactly
correct and exactly not faster — must clear every equivalence gate and be stopped by the
speed gate alone: measured **M ∈ [0.997, 1.015]** over 5 nonces, score ∈ [0.2000,
0.2033], never a pass. The score sits a few thousandths above the floor because the
wrapper is timed in a different process than the naive path it duplicates, so the paired
ratio fluctuates around 1.0 by about ±1.5%.

## Runtime

A full judge run takes roughly 12 s of wall-clock on the reference solution. Worst case
is bounded by the child's 12-minute global deadline plus about 3 minutes of parent work,
regardless of how the candidate behaves. That cheapness is a design goal, not an
accident: it is what makes an adversarial test suite affordable to run on every change.
