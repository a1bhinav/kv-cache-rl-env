#!/usr/bin/env python3
"""Grade your current solution.py with the judge's own logic.

This tool imports the judge's grading, scoring, and sampling modules — the
same code the real judge runs. Only the prompts differ: the judge uses fresh
sets from a nonce you never see; this tool samples from a local nonce
(--nonce, default random). Passing here does not guarantee passing there,
but the checks, tolerances, skip rule, timing protocol, and score formula are
identical by construction.

Differences from the real judge (documented, not gameable): your solution
runs in this same process (no child isolation), and the reference comes from
YOUR ./starter copy (the judge always uses its pristine one).

Usage: python local_check.py [--nonce N] [--quick]
"""

import argparse
import importlib.util
import os
import statistics
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")

import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
# The judge's grading/scoring/sampling modules: shipped into the workspace as
# checklib/ at deployment (see Dockerfile); in the repo they live in ../judge.
# Neither copy is secret — the run nonce is the secret, not this code.
for cand in [os.path.join(HERE, "checklib"), os.path.join(os.path.dirname(HERE), "judge")]:
    if os.path.exists(os.path.join(cand, "grading.py")):
        sys.path.insert(0, cand)
        break
else:
    sys.exit("local_check: cannot find grading modules (checklib/ or ../judge)")
import grading  # noqa: E402
import sampling  # noqa: E402
import scoring  # noqa: E402

SNAPSHOT_CAP_BYTES = 25 * 1024 * 1024
CALL_ORDER = ["W", "E1", "E2", "E3", "W_warm", "S1", "S2", "S3"]


def workspace_size():
    """Published Check 1: workspace minus starter/ must be <= 25 MB."""
    total = 0
    for root, dirs, files in os.walk(HERE):
        for skip in ("starter", "__pycache__", ".venv"):
            if skip in dirs:
                dirs.remove(skip)
        total += sum(os.path.getsize(os.path.join(root, f)) for f in files)
    return total


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        print(f"\nScore: 0.0  pass=False  (failed gate: {name})")
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nonce", type=int,
                    default=int.from_bytes(os.urandom(6), "big"))
    ap.add_argument("--quick", action="store_true",
                    help="one timing set instead of three")
    args = ap.parse_args()

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    print(f"local_check: nonce {args.nonce}")

    # Check 1 — presence & published snapshot cap
    size = workspace_size()
    gate("presence: solution.py exists", os.path.exists(os.path.join(HERE, "solution.py")))
    gate("snapshot size (workspace minus starter/) <= 25 MB",
         size <= SNAPSHOT_CAP_BYTES, f"{size/1e6:.2f} MB")

    sys.path.insert(0, HERE)
    from starter import model as M
    model = M.load_model()

    # Check 2 — import & signature
    t0 = time.perf_counter()
    spec = importlib.util.spec_from_file_location("solution", os.path.join(HERE, "solution.py"))
    solution = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(solution)
    except Exception as e:
        gate("import solution.py", False, f"{type(e).__name__}: {e}")
    import_s = time.perf_counter() - t0
    gate("import <= 20 s", import_s <= 20, f"{import_s:.2f}s")
    fn = getattr(solution, "generate_cached", None)
    gate("generate_cached callable", callable(fn))

    sets = sampling.sample_sets(args.nonce, os.path.join(HERE, "starter", "vocab.json"))
    timing_keys = ["S1"] if args.quick else ["S1", "S2", "S3"]

    # Reference + naive timings (same protocol as the judge)
    ref, naive_s = {}, []
    for name in ["W", "E1", "E2", "E3"]:
        prompts, n = sets[name]
        outs = [M.generate(model, p, n) for p in prompts]
        tok = torch.stack([o[0] for o in outs])
        log = torch.stack([o[1] for o in outs])
        top2 = torch.topk(log, 2, dim=-1).values
        ref[name] = (tok, log, top2[..., 0] - top2[..., 1])
    for name in timing_keys:
        prompts, n = sets[name]
        t0 = time.perf_counter()
        outs = [M.generate(model, p, n) for p in prompts]
        naive_s.append(time.perf_counter() - t0)
        tok = torch.stack([o[0] for o in outs])
        log = torch.stack([o[1] for o in outs])
        top2 = torch.topk(log, 2, dim=-1).values
        ref[name] = (tok, log, top2[..., 0] - top2[..., 1])
    ref["W_warm"] = ref["W"]

    # Agent calls: smoke + equivalence, then warmup + timed sets (fresh model)
    stats = grading.GradeStats()
    cached_s = []
    order = ["W", "E1", "E2", "E3", "W_warm"] + timing_keys
    model_t = None
    for key in order:
        set_name = "W" if key == "W_warm" else key
        prompts, n = sets[set_name]
        use_model = model
        if key == "W_warm":
            model_t = M.load_model()
        if key in ("W_warm", *timing_keys):
            use_model = model_t
        t0 = time.perf_counter()
        try:
            out = fn(use_model, [p.clone() for p in prompts], n)
        except Exception as e:
            gate(f"{key}: call returns", False, f"{type(e).__name__}: {e}")
        wall = time.perf_counter() - t0
        if key in timing_keys:
            cached_s.append(wall)
        gate(f"{key}: per-call time <= 60 s", wall <= 60, f"{wall:.2f}s")
        ok, code, detail = grading.check_shapes(out, len(prompts), n)
        gate(f"{key}: return contract", ok, detail or "")
        grading.grade_call(key, out[0], out[1], *ref[set_name], stats)
        if stats.failures:
            code, detail = stats.failures[0]
            gate(f"{key}: equivalence ({code})", False, detail)

    print(f"  [PASS] equivalence: all {stats.generated} positions "
          f"(checked {stats.checked}, skipped {stats.skipped_ties} near-ties, "
          f"max |logit diff| {stats.max_logit_diff:.2e})")
    print(f"  coverage {stats.coverage*100:.2f}%  skip rate {stats.skip_rate*100:.3f}%")

    ratios = [a / b for a, b in zip(naive_s, cached_s)]
    mult = statistics.median(ratios)
    s = scoring.score(True, mult)
    print(f"\n  naive_s  {[round(x,3) for x in naive_s]}")
    print(f"  cached_s {[round(x,3) for x in cached_s]}")
    print(f"  ratios   {[round(x,2) for x in ratios]}   M = {mult:.2f}x"
          + ("  (--quick: single set, judge uses median of 3)" if args.quick else ""))
    print(f"\nScore: {s:.4f}  pass={scoring.passed(True, mult)}"
          f"  (pass needs all gates AND M >= {scoring.SPEED_BAR})")


if __name__ == "__main__":
    main()
