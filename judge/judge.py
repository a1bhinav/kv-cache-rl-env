#!/usr/bin/env python3
"""Judge — parent process of the two-process architecture (docs/design.md).

The parent holds all judge logic and never runs agent code: it verifies the
MANIFEST, runs the determinism control, samples all prompt sets from the run
nonce, computes all reference outputs and naive timings IN MEMORY ONLY (never
written to disk), then drives the child harness (child_harness.py — the only
process that touches agent code) over a just-in-time pipe protocol: each
call's prompts are revealed to the child only when that call starts, and every
result line is hash-committed and timestamped on arrival, giving a
parent-measured envelope that bounds true compute time. After the child
exits, the parent grades the child's raw outputs, applies the score formula,
and prints the JSON report to stdout.

Usage:
  judge.py --workspace <dir> --nonce <uint64>     # snapshots <dir> itself
  judge.py --snapshot  <dir> --nonce <uint64>     # dir is already a snapshot

Exit codes: 0 = scored (pass or fail; see report); 3 = INVALID_RUN (no score,
harness retries with a fresh nonce).
"""

import argparse
import hashlib
import json
import os
import queue
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")

import torch  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import grading  # noqa: E402
import sampling  # noqa: E402
import scoring  # noqa: E402

ASSETS = os.path.join(_HERE, "judge_assets")
PRISTINE_PARENT = os.path.join(ASSETS, "pristine")

JUDGE_VERSION = "3.0-two-process-jit"
SCHEMA_VERSION = 1
SNAPSHOT_CAP_BYTES = 25 * 1024 * 1024
IMPORT_DEADLINE_S = 20.0
CALL_DEADLINE_S = 60.0
CHILD_DEADLINE_S = 720.0
RSS_CAP_BYTES = 4 * 1024**3
DRIFT_BOUND = 1.3
COVERAGE_WARN, COVERAGE_INVALID = 0.95, 0.90
# Parallelism tripwire, gate 7b (docs/design.md). Both paths are pinned to 2
# torch threads, so a compliant solution's CPU-time / wall-time ratio cannot
# exceed ~2. Calibrated against the reference: max measured ratio 1.049 across
# nonces {1001,1002,1003} (the decode is memory-bound and does not even
# saturate its two threads), line = max(2.5, 1.25 * 1.049) = 2.5, which also
# leaves room for an honest solution that genuinely uses both. Exceeding it
# means the solution found more cores than it was given -- INVALID_RUN for
# manual review, never an auto-score, exactly like the M > 128 tripwire.
CPU_WALL_CAP = 2.5
# Envelope tolerance: child-reported wall_s below (envelope - this) => tamper
ENV_TOL = lambda env: 0.75 + 0.05 * env  # noqa: E731
IMPORT_ENV_TOL = 10.0

CALL_ORDER = ["W", "E1", "E2", "E3", "W_warm", "S1", "S2", "S3"]


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def new_report(nonce):
    return {
        "schema_version": SCHEMA_VERSION, "nonce": nonce, "score": 0.0,
        "pass": False, "reason_code": None, "multiple": None,
        "ratios": None, "naive_s": None, "cached_s": None,
        "gates": {g: None for g in ["presence", "import", "signature", "smoke",
                                    "consistency", "token_equality", "logit_tol",
                                    "budgets"]},
        "positions": {"generated": 0, "checked": 0, "skipped_ties": 0, "uncovered": 0},
        "coverage": None, "skip_rate": None, "max_logit_diff": None,
        "stability_probes": None, "coverage_warning": False,
        "cpu_s": None, "cpu_wall_ratios": None,
        "manifest_ok": None, "judge_version": JUDGE_VERSION, "wall_s": None,
        "detail": None,
    }


class Judged(Exception):
    """Ends the run: scored failure (exit 0) or INVALID_RUN (exit 3)."""

    def __init__(self, exit_code, reason, detail=""):
        self.exit_code, self.reason, self.detail = exit_code, reason, detail


class Child:
    """Drives child_harness.py over the just-in-time JSON-line protocol."""

    def __init__(self, tmp):
        self.tmp = tmp
        self.deadline = time.perf_counter() + CHILD_DEADLINE_S
        env = dict(os.environ, OMP_NUM_THREADS="2")
        self.proc = subprocess.Popen(
            [sys.executable, os.path.join(_HERE, "child_harness.py")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, env=env)
        self.lines = queue.Queue()
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        for line in self.proc.stdout:
            self.lines.put(line)
        self.lines.put(None)  # EOF

    def send(self, obj):
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()

    def recv(self, step_timeout, on_timeout_reason, on_eof=None):
        timeout = min(step_timeout, self.deadline - time.perf_counter())
        if timeout <= 0:
            self.kill()
            raise Judged(0, "CHILD_DEADLINE", f"child exceeded {CHILD_DEADLINE_S}s")
        try:
            line = self.lines.get(timeout=timeout)
        except queue.Empty:
            self.kill()
            if time.perf_counter() >= self.deadline:
                raise Judged(0, "CHILD_DEADLINE", f"child exceeded {CHILD_DEADLINE_S}s")
            raise Judged(*on_timeout_reason)
        if line is None:
            # EOF before 'ready' is unattributable (judge-side retry); after a
            # successful import, only agent code can kill the child mid-call.
            raise Judged(*(on_eof or (3, "INVALID_RUN", "child harness exited unexpectedly")))
        return json.loads(line)

    def kill(self):
        try:
            self.proc.kill()
        except Exception:
            pass


def check_manifest(report):
    manifest = json.load(open(os.path.join(ASSETS, "MANIFEST.json")))
    for rel, want in manifest.items():
        path = (os.path.join(PRISTINE_PARENT, rel) if rel.startswith("starter/")
                else os.path.join(_HERE, rel))
        if not os.path.exists(path) or sha256(path) != want:
            report["manifest_ok"] = False
            raise Judged(3, "INVALID_RUN", f"manifest mismatch: {rel}")
    report["manifest_ok"] = True


def make_snapshot(workspace, tmp):
    """Copy workspace excluding the shipped starter/ tree (Check 1 definition)."""
    snap = os.path.join(tmp, "snapshot")
    shutil.copytree(workspace, snap,
                    ignore=shutil.ignore_patterns("starter", "__pycache__", ".venv"))
    for root, _dirs, files in os.walk(snap):
        for f in files:
            os.chmod(os.path.join(root, f), 0o444)
    return snap


def snapshot_size(snap):
    total = 0
    for root, dirs, files in os.walk(snap):
        if "starter" in dirs:
            dirs.remove("starter")
        total += sum(os.path.getsize(os.path.join(root, f)) for f in files)
    return total


def naive_decode_set(M, model, prompts, n):
    """Reference decode, one sequence at a time, the shipped generate().
    Returns (tokens, logits, gaps, wall_s); timed region = the loop only."""
    outs = []
    t0 = time.perf_counter()
    for p in prompts:
        outs.append(M.generate(model, p, n))
    wall = time.perf_counter() - t0
    tokens = torch.stack([o[0] for o in outs])
    logits = torch.stack([o[1] for o in outs])
    top2 = torch.topk(logits, 2, dim=-1).values
    gaps = top2[..., 0] - top2[..., 1]
    return tokens, logits, gaps, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace")
    ap.add_argument("--snapshot")
    ap.add_argument("--nonce", type=int, required=True)
    ap.add_argument("--keep-tmp", action="store_true")
    args = ap.parse_args()
    if bool(args.workspace) == bool(args.snapshot):
        ap.error("exactly one of --workspace / --snapshot required")

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)

    t_start = time.perf_counter()
    report = new_report(args.nonce)
    tmp = tempfile.mkdtemp(prefix="judge_")
    code = 0
    try:
        run(args, report, tmp)
        report["reason_code"] = "OK"
    except Judged as j:
        report["reason_code"] = j.reason
        report["detail"] = j.detail
        report["score"] = 0.0
        report["pass"] = False
        code = j.exit_code
    finally:
        if not args.keep_tmp:
            shutil.rmtree(tmp, ignore_errors=True)
    report["wall_s"] = round(time.perf_counter() - t_start, 3)
    print(json.dumps(report))
    sys.exit(code)


def run(args, report, tmp):
    # ---- Check 0: preflight --------------------------------------------------
    check_manifest(report)
    sys.path.insert(0, PRISTINE_PARENT)
    from starter import model as M
    model = M.load_model()

    # ---- Check 1: presence & snapshot size ------------------------------------
    snap = args.snapshot or make_snapshot(args.workspace, tmp)
    if not os.path.exists(os.path.join(snap, "solution.py")):
        report["gates"]["presence"] = False
        raise Judged(0, "MISSING_SOLUTION", "solution.py not found in snapshot")
    size = snapshot_size(snap)
    if size > SNAPSHOT_CAP_BYTES:
        report["gates"]["presence"] = False
        raise Judged(0, "SNAPSHOT_TOO_LARGE",
                     f"{size} bytes > {SNAPSHOT_CAP_BYTES} (starter/ excluded)")
    report["gates"]["presence"] = True

    # ---- Sampling + reference computation (parent memory only) ----------------
    sets = sampling.sample_sets(
        args.nonce, os.path.join(PRISTINE_PARENT, "starter", "vocab.json"))
    ref = {}
    wp, wn = sets["W"]

    # Determinism control + first stability probe.
    ref_a = naive_decode_set(M, model, wp, wn)
    ref_b = naive_decode_set(M, model, wp, wn)
    if not (torch.equal(ref_a[0], ref_b[0]) and torch.equal(ref_a[1], ref_b[1])):
        raise Judged(3, "INVALID_RUN", "determinism control failed on set W")
    probes = [ref_a[3]]
    ref["W"] = ref_a
    for name in ["E1", "E2", "E3"]:
        prompts, n = sets[name]
        ref[name] = naive_decode_set(M, model, prompts, n)

    # ---- Naive timing on S-sets (+ warmup and drift probes) -------------------
    naive_decode_set(M, model, wp, wn)  # warmup before S1, per spec §5
    naive_s = []
    for name in ["S1", "S2", "S3"]:
        prompts, n = sets[name]
        ref[name] = naive_decode_set(M, model, prompts, n)
        naive_s.append(ref[name][3])
        if name == "S2":
            probes.append(naive_decode_set(M, model, wp, wn)[3])
    probes.append(naive_decode_set(M, model, wp, wn)[3])
    report["naive_s"] = [round(x, 4) for x in naive_s]  # visible even on early exit
    report["stability_probes"] = [round(p, 4) for p in probes]
    if max(probes) / min(probes) > DRIFT_BOUND:
        raise Judged(3, "INVALID_RUN",
                     f"stability probes drift {max(probes)/min(probes):.2f}x > {DRIFT_BOUND}x")
    ref["W_warm"] = ref["W"]  # same prompts; reference identical (control-verified)

    # ---- Child: just-in-time scripted calls -----------------------------------
    child = Child(tmp)
    t0 = time.perf_counter()
    child.send({"op": "init", "pristine_parent": PRISTINE_PARENT, "snapshot": snap})
    ready = child.recv(IMPORT_DEADLINE_S + 40,
                       (0, "IMPORT_TIMEOUT", "no ready within import window"))
    import_env = time.perf_counter() - t0
    import_s = ready.get("import_s", 0.0)
    fatal = ready.get("fatal")
    if fatal and fatal.startswith("IMPORT_ERROR"):
        report["gates"]["import"] = False
        code = "IMPORT_TIMEOUT" if import_s > IMPORT_DEADLINE_S else "IMPORT_ERROR"
        raise Judged(0, code, fatal)
    if import_s > IMPORT_DEADLINE_S:
        report["gates"]["import"] = False
        raise Judged(0, "IMPORT_TIMEOUT", f"import took {import_s:.1f}s > 20s")
    if import_env - import_s > IMPORT_ENV_TOL + 5.0:  # model load+torch import slack
        raise Judged(3, "INVALID_RUN",
                     f"TIMING_MISMATCH: import envelope {import_env:.1f}s vs reported {import_s:.1f}s")
    report["gates"]["import"] = True
    if fatal:
        report["gates"]["signature"] = False
        raise Judged(0, "BAD_SIGNATURE", fatal)
    report["gates"]["signature"] = True

    recs, envelopes = {}, {}
    err_key = None
    for key in CALL_ORDER:
        if key == "W_warm":
            child.send({"op": "fresh_model"})
            child.recv(30, (3, "INVALID_RUN", "fresh_model stalled"),
                       on_eof=(0, "AGENT_ABORT", "child died before fresh_model"))
        set_name = "W" if key == "W_warm" else key
        prompts, n = sets[set_name]
        in_path = os.path.join(tmp, f"call_{key}_in.pt")
        out_path = os.path.join(tmp, f"call_{key}_out.pt")
        t0 = time.perf_counter()  # clock starts before the prompts exist on disk
        torch.save({"prompts": prompts, "N": n}, in_path)
        child.send({"op": "call", "key": key, "in": in_path, "out": out_path})
        done = child.recv(CALL_DEADLINE_S + 20,
                          (0, "CALL_TIMEOUT", f"{key}: no result within window"),
                          on_eof=(0, "AGENT_ABORT", f"{key}: child died mid-call"))
        envelopes[key] = time.perf_counter() - t0
        if done.get("done") != key:
            raise Judged(3, "INVALID_RUN", f"protocol breach at {key}: {done}")
        if not os.path.exists(out_path) or sha256(out_path) != done.get("sha"):
            raise Judged(3, "INVALID_RUN",
                         f"TAMPER_SUSPECT: result hash mismatch at {key} — manual review")
        recs[key] = torch.load(out_path, weights_only=False)
        wall = recs[key]["wall_s"]
        if wall is not None and envelopes[key] - wall > ENV_TOL(envelopes[key]):
            raise Judged(3, "INVALID_RUN",
                         f"TIMING_MISMATCH at {key}: envelope {envelopes[key]:.2f}s vs "
                         f"reported {wall:.2f}s — manual review")
        if recs[key].get("err") is not None:
            err_key = key  # abort remaining calls; graded below in order
            break
    child.send({"op": "exit"})
    bye = child.recv(15, (3, "INVALID_RUN", "child did not exit cleanly"),
                     on_eof=(0, "AGENT_ABORT", "child died before exit"))
    child.proc.wait(timeout=10)

    # ---- Checks 3+4: smoke, then grade every recorded output ------------------
    stats = grading.GradeStats()
    gate_of = {"INCONSISTENT_ARGMAX": "consistency",
               "TOKEN_MISMATCH": "token_equality",
               "LOGIT_DIVERGENCE": "logit_tol"}
    for key in CALL_ORDER:
        if key not in recs:
            break  # aborted after err_key; the AGENT_EXCEPTION raise below covers it
        rec = recs[key]
        set_name = "W" if key == "W_warm" else key
        prompts, n = sets[set_name]
        if rec.get("err") is not None:
            if key == "W":
                report["gates"]["smoke"] = False
            raise Judged(0, "AGENT_EXCEPTION", f"{key}: {rec['err']}")
        if rec.get("new_tokens") is None:
            if key == "W":
                report["gates"]["smoke"] = False
            raise Judged(0, "BAD_RETURN",
                         f"{key}: returned {rec.get('shape_note')}, not a 2-tuple of tensors")
        if rec.get("nt_device", "cpu") != "cpu" or rec.get("sl_device", "cpu") != "cpu":
            if key == "W":
                report["gates"]["smoke"] = False
            raise Judged(0, "BAD_RETURN",
                         f"{key}: tensors on ({rec.get('nt_device')}, {rec.get('sl_device')}), want cpu")
        out = (rec["new_tokens"], rec["step_logits"])
        ok, code, detail = grading.check_shapes(out, len(prompts), n)
        if not ok:
            if key == "W":
                report["gates"]["smoke"] = False
            raise Judged(0, code, f"{key}: {detail}")
        if key == "W":
            report["gates"]["smoke"] = True
        rt, rl, rg, _ = ref[set_name]
        grading.grade_call(key, out[0], out[1], rt, rl, rg, stats)
        if stats.failures:
            fcode, fdetail = stats.failures[0]
            report["gates"][gate_of[fcode]] = False
            raise Judged(0, fcode, fdetail)

    for gate in ["consistency", "token_equality", "logit_tol"]:
        report["gates"][gate] = True
    report["positions"] = {"generated": stats.generated, "checked": stats.checked,
                           "skipped_ties": stats.skipped_ties, "uncovered": stats.uncovered}
    report["coverage"] = round(stats.coverage, 6)
    report["skip_rate"] = round(stats.skip_rate, 6)
    report["max_logit_diff"] = stats.max_logit_diff
    if stats.coverage < COVERAGE_INVALID:
        raise Judged(3, "INVALID_RUN",
                     f"coverage {stats.coverage:.3f} < {COVERAGE_INVALID} — recalibrate assets")
    report["coverage_warning"] = stats.coverage < COVERAGE_WARN

    # ---- Check 6: budgets ------------------------------------------------------
    for key in CALL_ORDER:
        if recs[key]["wall_s"] > CALL_DEADLINE_S:
            report["gates"]["budgets"] = False
            raise Judged(0, "CALL_TIMEOUT",
                         f"{key}: {recs[key]['wall_s']:.1f}s > {CALL_DEADLINE_S}s")
    rss = bye.get("peak_rss", 0)
    rss_bytes = rss if sys.platform == "darwin" else rss * 1024
    if rss_bytes > RSS_CAP_BYTES:
        report["gates"]["budgets"] = False
        raise Judged(0, "RSS_EXCEEDED", f"peak RSS {rss_bytes/1e9:.2f} GB > 4 GB")
    report["gates"]["budgets"] = True

    # ---- Check 7b: parallelism tripwire ------------------------------------------
    timed = ["S1", "S2", "S3"]
    cpu_s, cpu_ratios = [], []
    for k in timed:
        w, c = recs[k]["wall_s"], recs[k].get("cpu_s")
        cpu_s.append(c)
        cpu_ratios.append((c / w) if (c is not None and w and w > 0) else None)
    report["cpu_s"] = [None if c is None else round(c, 4) for c in cpu_s]
    report["cpu_wall_ratios"] = [None if r is None else round(r, 3) for r in cpu_ratios]
    seen = [r for r in cpu_ratios if r is not None]
    if seen and max(seen) > CPU_WALL_CAP:
        raise Judged(3, "INVALID_RUN",
                     f"ANOMALOUS_PARALLELISM: CPU/wall ratio {max(seen):.2f} > {CPU_WALL_CAP} "
                     f"under a 2-thread pin — the solution used more cores than it was "
                     f"given; manual review")

    # ---- Checks 5+7: speed multiple & plausibility ------------------------------
    cached_s = [recs[k]["wall_s"] for k in ["S1", "S2", "S3"]]
    ratios = [a / b for a, b in zip(naive_s, cached_s)]
    mult = statistics.median(ratios)
    report["naive_s"] = [round(x, 4) for x in naive_s]
    report["cached_s"] = [round(x, 4) for x in cached_s]
    report["ratios"] = [round(x, 3) for x in ratios]
    report["multiple"] = round(mult, 3)
    if mult > scoring.PLAUSIBILITY_CAP:
        raise Judged(3, "INVALID_RUN",
                     f"ANOMALOUS_SPEED: M={mult:.1f} > {scoring.PLAUSIBILITY_CAP} — manual review")

    report["score"] = round(scoring.score(True, mult), 6)
    report["pass"] = scoring.passed(True, mult)


if __name__ == "__main__":
    main()
