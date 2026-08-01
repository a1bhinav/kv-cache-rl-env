# Task: Batched KV-cache greedy decoding for a small transformer LM

You are working in `/workspace` inside an offline Linux container (2 CPU cores, 6 GB RAM, no network, no GPU). Python 3.11 and CPU-only PyTorch (pinned; see `/workspace/ENV.txt` for exact versions) are installed. You have a command-line shell and can read, write, and run files anywhere under `/workspace` and `/tmp`.

## The task

The repository at `/workspace/starter/` contains a small decoder-only transformer language model and a **correct but slow** text generator, `generate()`, which re-runs the full forward pass over the entire sequence for every new token, one sequence at a time.

Your job is to write a **fast, batched, KV-cached greedy decoder** for this model:

- Create the file **`/workspace/solution.py`**.
- It must define a function with exactly this call signature:

```python
def generate_cached(model, prompt_ids, max_new_tokens):
    ...
    return new_tokens, step_logits
```

- Your function must produce **the same tokens the reference `generate()` produces** (grading tolerances below) while being **at least 8× faster** than it, measured live on this machine.

A stub `/workspace/solution.py` containing this signature and a `NotImplementedError` is already present; replace its body.

## The model (informative summary — `starter/model.py` is normative)

`TinyGPT`, a pre-LN GPT-style decoder: vocab **66** (character-level; table in `starter/vocab.json`), `d_model` **528**, **3** layers, **4** heads (head_dim **132**), GELU (exact erf) MLP at 4×, **no biases anywhere**, gain-only LayerNorm (eps 1e-5), **tied** input/output embeddings, fp32, 10,074,768 parameters, max context **512**.

Positions are encoded with **RoPE** (rotary embeddings) applied to q and k over the full head dim (66 rotation pairs, base θ = 10000). There is **no** learned positional embedding. `starter/model.py` exposes the helper `apply_rope(x, positions)` used by `TinyGPT.forward`.

`TinyGPT.forward(input_ids: LongTensor[B, T]) -> FloatTensor[B, T, 66]` computes logits for every position. **It applies no padding mask** — every position attends causally to all earlier tensor positions, and RoPE positions are `0..T-1` per row. Correct handling of ragged batches (padding, masking, per-row RoPE offsets) is entirely your responsibility; solve it using the model's parameters and submodules however you like.

`load_model()` in `starter/model.py` builds the model and loads the pinned weights `starter/weights.pt` (lightly trained; sharp enough that greedy argmax is decisive at almost every position). SHA-256 checksums of all starter assets are in `starter/ASSETS.json`.

## Interface contract (exact)

Inputs — your function must accept a call of the form `generate_cached(model, prompt_ids, max_new_tokens)` where:

- `model` — a `TinyGPT` instance, constructed by the judge from the **pristine** `starter/model.py` and `starter/weights.pt` (as shipped — your edits to `/workspace/starter/` are never used by the judge). Treat it as the object you receive; imports of `starter.*` inside your solution also resolve to the judge's pristine copies at judge time.
- `prompt_ids` — a Python **list** of `B` one-dimensional `torch.LongTensor`s on CPU, one prompt per list element, values in `[0, 65]`. Lengths vary **within** the list and are **not sorted**. Guaranteed bounds: `B ∈ [2, 8]`, every length `T_i ∈ [8, 192]`.
- `max_new_tokens` — a Python `int`, `N ∈ [16, 64]`. Guaranteed: `max(T_i) + N ≤ 256`.

Return — a tuple `(new_tokens, step_logits)`:

- `new_tokens`: `torch.LongTensor` of shape `[B, N]` on CPU — **only** the newly generated tokens, row `i` in the same order as `prompt_ids[i]`. Every row contains exactly `N` tokens (there is no EOS / early stop).
- `step_logits`: `torch.FloatTensor` (fp32) of shape `[B, N, 66]` on CPU — for each row `i` and step `t`, the logits your decoder used to select `new_tokens[i, t]` (i.e., the logits at the last position of row `i`'s context before appending that token).

Greedy decoding semantics — at each step, the next token is `argmax` over the 66 logits at the last position. If several logits were exactly equal at the maximum, the lowest index wins (`torch.argmax` CPU behavior); exact ties do not occur in practice, and near-ties are excluded from grading by the skip rule below.

Reference semantics — correctness is defined **per sequence, independently**: for every `i`, your outputs for row `i` must match running the pristine reference `generate()` on `prompt_ids[i]` **alone**. Any cross-sequence contamination from your batching (padding tokens leaking into attention, wrong per-row RoPE offsets, row-order mix-ups) is a wrong answer.

Determinism — your function must be deterministic: same inputs → same outputs, with no dependence on RNG state, wall-clock, environment variables, or call count.

## Constraints and resource budgets

- Allowed imports: Python stdlib, `torch`, `numpy`, and your own files under `/workspace`. Nothing else.
- Filesystem: at judge time your code runs from a **read-only** snapshot of `/workspace`; write only to `/tmp`. The graded snapshot is `/workspace` **excluding** the shipped `starter/` tree (the judge always substitutes its own pristine copy of `starter/`), and it must total ≤ **25 MB** — `local_check.py` verifies this.
- Do not spawn subprocesses or extra threads; do not change thread counts. The harness pins `torch.set_num_threads(2)` and the container to 2 CPU cores for **both** your path and the reference path — the comparison is fair.
- Do not monkeypatch or mutate anything outside your own module's state: no patching of `torch`, `time`, builtins, or the model object's methods. Registering read-only precomputed buffers you attach to your own module (e.g., RoPE tables) is fine; import-time precomputation of prompt-independent tables is fine.
- Budgets, enforced at judge time: module import ≤ **20 s**; each `generate_cached` call ≤ **60 s** (the reference path itself stays well under this on every graded input, so any solution at least as fast as the reference is unaffected); peak RSS of the grading process ≤ **4 GB**. Note that a one-time warm-up cost lands inside whichever call triggers it — `torch.compile`, for instance, can spend minutes compiling on its first call, and that time counts against that call's 60 s. `local_check.py` reports per-call times, so you can see this before you submit.
- Your function is also expected to honour the 2-thread pin: the judge records CPU time alongside wall time on every timed call, and a run whose CPU/wall ratio exceeds 2.5 is set aside for manual review rather than scored.
- Numeric duty: outputs must match the fp32 reference within the tolerances below. (Internally computing in reduced precision will fail the logit tolerance by design.)

## How you will be scored (the judge's exact rule)

The judge runs after your session ends, on this same machine, using **fresh prompts sampled from seeds you never see** (in-distribution text slices plus lightly perturbed variants, within the bounds above). It computes the reference outputs live with its own pristine copy of the model and reference decoder. Your `/workspace/solution.py` is the only thing of yours it uses.

Hard gates, in order — failing any of them scores **0**:

1. `/workspace/solution.py` exists, imports in ≤ 20 s, and exposes a callable `generate_cached` accepting the call above.
2. A smoke call returns the exact contract: a 2-tuple with the shapes, dtypes, and devices specified; all logits finite.
3. Self-consistency: `new_tokens == argmax(step_logits, dim=-1)` at every position, every call.
4. **Token equality**: at every generated position, your token equals the reference token — except positions where the reference's top-2 logit gap is `< 1e-3` (a "near-tie"), which are skipped, and grading of a row stops if you and the reference legitimately diverge at a skipped position. Skips are determined by reference logits only; with the shipped weights they are rare.
5. **Logit closeness**: at every position that is checked (your prefix matches the reference so far), `max |your logit − reference logit| ≤ 2.5e-4` across the 66 logits. (For scale: these logits run to about 50 in magnitude, where one fp32 ulp is 3.8e-6, and an honest KV-cached decoder lands within ~14 ulps of the full-recompute path. The tolerance is set several times above that, and far below what any reduced-precision arithmetic would cost you.)
6. Budgets respected (per-call time, RSS); every call returns without exception.

Speed multiple — the judge times, on **three fresh prompt sets** (B = 8, lengths 96–192, N = 64): the reference path (its own copy, sequences one at a time, as shipped) and your path, paired per set, after a warm-up call for each. `M = median over the 3 sets of (reference wall-clock / your wall-clock)`. Timing runs are also equivalence-checked — every output your function ever returns to the judge is graded by gates 3–5.

Score (continuous, in [0, 1]):

- If any hard gate fails: **S = 0**.
- Otherwise: **S = 0.2 + 0.8 × min(1, max(0, log2 M) / 5)**.

So: correct-but-not-faster ⇒ 0.2; M = 8× ⇒ 0.68; the cap S = 1.0 is reached at M = 32×. **Pass** is reported iff all gates hold **and M ≥ 8** (equivalently S ≥ 0.68). Implausible multiples (M > 128, beyond this model's arithmetic-intensity ceiling) invalidate the run for manual review rather than scoring.

## What's in the repo

- `starter/model.py` — `TinyGPT`, `apply_rope`, `load_model()`, and the reference `generate()` (the correctness ground truth).
- `starter/weights.pt`, `starter/vocab.json`, `starter/ASSETS.json` — pinned weights, char table, checksums.
- `starter/sample_text.txt` — a small text sample in the training distribution, for building your own test prompts.
- `local_check.py` — run `python local_check.py` to grade your current `solution.py` with **the same checks and formula as the judge** (gates 1–6, skip rule, tolerances, paired timing) on locally generated prompts. It prints per-gate results, the estimated multiple, and the score. The judge uses different, unseen prompt sets and seeds — passing locally does not guarantee passing, but the logic is identical.
- `ENV.txt` — pinned interpreter and package versions (identical at judge time).

## Suggested workflow (informative)

Read `starter/model.py` closely, especially `apply_rope` and how `forward` computes positions. Get a single-sequence cached decoder exact first (token-for-token against `generate()`), then batch it, then make it fast. `local_check.py` will tell you which gate you are failing. The classic failure modes here are worth a careful look: RoPE position offsets once a cache is in play, and what padding does to attention and to per-row positions in a ragged batch.
