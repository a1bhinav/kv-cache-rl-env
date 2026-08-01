# Threat model

A reward signal is only as good as its worst failure mode. This document maps every
known **reward-hacking** vector (a wrong solution that scores well) and every known
**reward-denial** risk (an honest solution that scores badly) to the mitigation that
handles it, and then to the artifact that turns the claim into evidence.

Where a row says *structural*, no test exists because none can be written: the attack
targets something that does not exist in the attacked process. Those rows carry the
argument instead. Where a row names an exploit, it lives in `tests/exploits/` and is
asserted at its documented outcome by `tests/test_judge.py`.

## A. Reward hacking — a wrong solution that scores well

| ID | Vector | The attack | Mitigation | Residual risk / evidence |
|---|---|---|---|---|
| R1 | Hardcoded outputs | Ship a lookup table of outputs for prompts guessed from the shipped sample text | Prompts are sampled at judge time from a nonce the candidate never sees, over grammar text generated at judge time — never shipped text | Negligible: the prompt space is 66^8 and up. `tests/exploits/hardcoded_outputs` |
| R2 | Memoization across timing repeats | Cache by prompt hash so repeats 2–3 return instantly and inflate M | S1–S3 are three independently sampled sets; ratios are paired per set, then median | Nothing left to exploit. `tests/exploits/memoize_timing` |
| R3 | Secretly calling the reference | Wrap the pristine `generate()` for correctness at no effort | The speed gate: M ≥ 8 to pass, and a wrapper lands at M ≈ 1 | Intended floor, not a pass. Measured over 5 nonces: M ∈ [0.997, 1.015], score ∈ [0.2000, 0.2033], all equivalence gates green. `tests/exploits/naive_wrapper` |
| R4 | Judge tampering at import | Monkeypatch `torch`, `time.perf_counter`, the model class, or builtins to slow the baseline or shrink cached timings | Naive timing completes in the **parent** before any candidate code exists anywhere; the child binds `perf_counter` pre-import; scoring runs in the parent after the child exits | In-child tampering can distort only *cached* timings, bounded by the M > 128 plausibility cap. Equivalence is untouchable — graded in the parent against an in-memory reference. `tests/exploits/monkeypatch_clock` |
| R5 | Sabotaging the baseline copy | Slow down or break `starter/generate()` in the workspace so the naive path looks worse | The judge never executes workspace copies: the snapshot *excludes* `starter/` outright, and the parent uses hash-verified pristine assets | None. **Structural** — the attacked artifact is never read, and the naive path runs before candidate code exists |
| R6 | Logit laundering | Generate tokens with a subtly wrong fast cache, then recompute the returned `step_logits` with a clean teacher-forced forward so the logit gate cannot see the cache's numerics | Token equality against the parent's reference stays binding at every non-skipped position; the consistency gate forces `tokens == argmax(returned logits)`; laundering costs extra forwards, which lowers M | **Accepted residual, documented.** The logit gate is defense in depth; the load-bearing gate is token equality on fresh inputs. A cache wrong only *below* argmax on every judged input is, by construction, what the contract calls equivalent. **Structural** |
| R7 | In-process theft of reference data | Import-time code walks `gc.get_objects()` or reads judge files to find reference tokens and echoes them | The child holds reference outputs at no point: they exist only in the parent's memory and are never written to disk | None short of OS-level escalation. **Structural** — the data the attack would steal is not in the attacked process, or on disk at all |
| R8 | Hidden-data access during the session | Read judge seeds or assets while working | Judge assets and the nonce are injected only at judge time, never mounted in the agent's environment | None. **Structural** |
| R9 | Model or config swap | Ignore the passed `model` and use re-trained, quantized, or distilled weights | Equivalence is computed against the parent's reference on the judge's weights; different weights diverge immediately at the 2.5e-4 line | None. `tests/exploits/own_weights` |
| R10 | Mutating the judge's model | In-place quantization or weight edits to speed later timed calls | Every output from every call is graded, including timed ones; timing calls get a fresh model instance | None. `tests/exploits/inplace_quantization` |
| R11 | Extra processes or threads | Spawn workers or raise thread counts to beat a 2-thread baseline | **Harness-measured:** the child records CPU time around every timed call; a solution honoring the pin cannot exceed CPU/wall ≈ 2, so above 2.5 the run is set aside. **Deployment-capped too:** the container cpuset is 2 cores | Detection holds on any machine, not only in the container. A ratio just under the line bounds the extra cores to ~1.25× over a compliant solution. Second finding: at this model size the attack is **self-defeating** — 16 threads runs at 0.71× the reference and +4 workers at 0.44×, because the decode GEMMs are small and memory-bound. `tests/exploits/set_threads` |
| R12 | RSS-funded precompute | Multi-GB tables to shortcut compute | 4 GB RSS cap on the child; prompt-dependent precompute is impossible (R1) | Prompt-independent tables within RSS are legitimate optimization. Accepted by design |
| R13 | Import-time compute abuse | Do heavy work at import, where it is untimed | 20 s import deadline; only prompt-independent work is possible there (RoPE tables and the like — intended) | Accepted by design, bounded by the deadline |
| R14 | Tie-divergence laundering | Deliberately pick the other token at detected near-ties to escape grading of the row's tail | Skips are determined from **reference** logits only, so the candidate cannot identify them without already matching reference numerics; the coverage floor plus a measured 0.056% skip rate bound the unchecked tail to a couple of rows | Bounded and monitored — coverage is in every report |
| R15 | Judge-mode detection | Behave correctly only under graded conditions | There is no ungraded scored path: the only scored execution *is* the judge run, on fresh inputs. Local runs are the candidate's own tooling | Moot by construction |

## B. Reward denial — an honest solution that scores badly

| ID | Risk | What it would cause | Mitigation | Evidence |
|---|---|---|---|---|
| R16 | Float jitter: single-token GEMMs vs full recompute | Honest cached logits differ from the per-sequence reference, and a hair-trigger gate fails them | **Measured, not assumed:** max 5.34e-5 (14 ulps) over 405,504 values, irreducible because the final projection contributes exactly 0.0 once the hidden state is fixed. Tolerance set at 2.5e-4, 4.7× above that; skip line at 1e-3; fp32, CPU-only, deterministic algorithms, 2 threads on both paths | Reference max ≤ τ/4 and p99 ≤ τ/5 on 3 nonces |
| R17 | Machine speed variance | An honest 15× solution measured at 7.9× on a noisy machine | Relative multiple only, never absolute tokens/sec; paired per-set ratios; median of 3; warm-ups; drift probes with a 1.3× invalid line; parent and child run consecutively on one machine | Reference margin ≥ 1.5× the bar on 3 nonces. Residual: extreme host noise yields `INVALID_RUN` and a retry, never a 0 |
| R18 | Judge breakage or nondeterministic build | Mismatched assets or a flaky reference would zero honest work | Manifest hashes plus a decode-twice bit-identity control; any judge fault exits `INVALID_RUN` with no score | Covered by the `INVALID_RUN` path test |
| R19 | Near-tie inflation from under-trained weights | Flat logits produce many skips, weakening the token gate and dropping coverage | Weights are trained until the measured skip rate is < 0.5% and pinned by sha256; the coverage floor triggers recalibration, not a zero | Measured skip rate 0.056% |
| R20 | Contract ambiguity | An honest solution fails on a defensible reading of return shape, row order, tie-break, or N semantics | The prompt pins types, shapes, dtypes, order, tie-break and bounds exactly; a signature stub ships; `env/local_check.py` runs the same gates and formula as the judge, so misreadings surface early | The reference passes through the same code path a user runs locally |
| R21 | Denial of legitimate optimization | A "morally correct" low-precision cache fails the logit line | **Intended bar, stated in the prompt** rather than hidden: reduced-precision internals cost 40× to 4000× the line. The 2.5e-4 re-derivation does not soften it | By design; measured by the in-place quantization exploit |
| R22 | Budget knife-edges | An honest but modest solution killed by the per-call cap | The 60 s cap sits ≥ 30× above reference per-call cost, and above the naive path itself on every graded set | Caps sized from measured rates |

## C. Standing structural guarantees

1. Judge assets live outside every agent-readable path and are injected only at judge
   time.
2. The judge runs from read-only pristine copies in fresh processes; the candidate's
   only input is the workspace snapshot.
3. Reference outputs and candidate code are never co-resident in one process, and
   reference outputs are never written to disk.
4. No threshold is ever adjusted to make a failing test pass. Changes go through a
   re-measurement against the reference solution.

**Enforcement scope.** Two classes of property, deliberately kept distinct:

- **Harness-enforced**, and therefore true wherever this repo runs: parent/child
  isolation with the reference only ever in parent memory; pre-import clock capture;
  live reference computation from hash-verified assets; fresh nonce-derived prompts;
  every returned output graded; paired relative timing with drift probes; the CPU/wall
  parallelism tripwire.
- **Deployment-specified**, enforced by the container rather than the harness (see
  `env/DEPLOYMENT.md`): no network, the memory cap, read-only judge mounts. For this
  task no-network enables no known exploit — the reference is computed live from local
  assets and judge prompts never leave the judge process — so it is defense in depth
  rather than load-bearing.
- **Both:** the CPU cap. `--cpus=2` *prevents* a parallelism gain at deployment, and the
  CPU/wall tripwire *detects* the attempt anywhere (R11).

## Measured attack outcomes

Nine adversarial baselines, each a real attack with the reference algorithm underneath
where it needs one — otherwise it would die at the equivalence gates and prove nothing
about the defense it targets.

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
