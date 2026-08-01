"""Judge acceptance suite.

Runs the real judge end to end against the gold solution on three pinned
nonces, against both baselines, and against every exploit in tests/exploits/,
asserting each one's documented outcome. Also unit-tests the score formula,
the grading rules, and the INVALID_RUN path.

Each judge invocation takes ~12-25 s, so reports are cached per
(workspace, nonce) for the session.
"""

import json
import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD = os.path.join(ROOT, "tests", "gold", "solution.py")
EXPLOITS = os.path.join(ROOT, "tests", "exploits")
PY = os.path.join(ROOT, ".venv", "bin", "python")
if not os.path.exists(PY):
    PY = sys.executable

NONCES = (1001, 1002, 1003)
SPEED_BAR = 8.0
CALIBRATION_BAR = 12.0  # calibration: gold must clear 1.5x the speed bar

_CACHE = {}


def run_judge(workspace, nonce):
    key = (workspace, nonce)
    if key not in _CACHE:
        env = dict(os.environ, OMP_NUM_THREADS="2")
        proc = subprocess.run(
            [PY, os.path.join(ROOT, "judge", "judge.py"),
             "--workspace", workspace, "--nonce", str(nonce)],
            capture_output=True, text=True, env=env, timeout=1800)
        assert proc.stdout.strip(), f"judge emitted no report; stderr:\n{proc.stderr[-2000:]}"
        _CACHE[key] = (json.loads(proc.stdout), proc.returncode)
    return _CACHE[key]


@pytest.fixture(scope="session", autouse=True)
def _dump_reports():
    """Persist every judge report this run produced, for the calibration log."""
    yield
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    out = {f"{os.path.basename(ws)}@{n}": rep
           for (ws, n), (rep, _rc) in _CACHE.items()}
    with open(os.path.join(ROOT, "logs", "reports.json"), "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)


@pytest.fixture(scope="session")
def build_workspace(tmp_path_factory):
    """Assemble a judge-gradable workspace: the candidate's solution.py plus
    _base.py (the gold algorithm), which the exploits that need a *correct*
    decoder underneath their cheat import from."""
    def _build(solution_path, name):
        ws = tmp_path_factory.mktemp(name)
        shutil.copy(solution_path, ws / "solution.py")
        shutil.copy(GOLD, ws / "_base.py")
        return str(ws)
    return _build


@pytest.fixture(scope="session")
def gold_ws(build_workspace):
    return build_workspace(GOLD, "gold")


@pytest.fixture(scope="session")
def gold_multiple(gold_ws):
    return sorted(run_judge(gold_ws, n)[0]["multiple"] for n in NONCES)[1]  # median


def exploit_ws(build_workspace, name):
    return build_workspace(os.path.join(EXPLOITS, name, "solution.py"), name)


# --------------------------------------------------------------------------
# Gold: no reward denial. Passes with margin on three judge seeds.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("nonce", NONCES)
def test_gold_passes_with_margin(gold_ws, nonce):
    from judge import grading
    r, rc = run_judge(gold_ws, nonce)
    assert rc == 0 and r["reason_code"] == "OK", r["detail"]
    assert r["pass"] is True, r
    assert all(r["gates"].values()), r["gates"]
    assert r["multiple"] >= CALIBRATION_BAR, (
        f"gold M={r['multiple']} below the 1.5x-bar calibration line {CALIBRATION_BAR}")
    assert r["score"] >= 0.68
    # calibration targets from docs/design.md, "Calibration"
    assert r["coverage"] > 0.99, r["coverage"]
    assert r["skip_rate"] < 0.005, r["skip_rate"]
    assert r["max_logit_diff"] <= grading.LOGIT_TOL / 4, r["max_logit_diff"]
    assert max(r["stability_probes"]) / min(r["stability_probes"]) <= 1.3


def test_gold_never_hits_the_plausibility_cap(gold_ws):
    from judge import scoring
    for n in NONCES:
        m = run_judge(gold_ws, n)[0]["multiple"]
        assert m * 3 <= scoring.PLAUSIBILITY_CAP, (
            f"plausibility cap {scoring.PLAUSIBILITY_CAP} is under 3x gold M={m}")


def test_per_call_budget_has_headroom_over_the_naive_path(gold_ws):
    """R22: the 60 s per-call cap must sit far above the naive path itself, so
    that even an M=1 solution is nowhere near it."""
    from judge.judge import CALL_DEADLINE_S
    worst_naive = max(max(run_judge(gold_ws, n)[0]["naive_s"]) for n in NONCES)
    assert CALL_DEADLINE_S >= 10 * worst_naive, (worst_naive, CALL_DEADLINE_S)


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------

def test_do_nothing_stub_scores_zero(build_workspace):
    r, rc = run_judge(exploit_ws(build_workspace, "do_nothing_stub"), 1001)
    assert rc == 0
    assert r["score"] == 0.0 and r["pass"] is False, r
    assert r["reason_code"] in {"MISSING_SOLUTION", "IMPORT_ERROR", "IMPORT_TIMEOUT",
                                "BAD_SIGNATURE", "AGENT_EXCEPTION"}, r["reason_code"]
    assert r["gates"]["smoke"] is False


def test_missing_solution_scores_zero(tmp_path):
    r, rc = run_judge(str(tmp_path), 1001)
    assert rc == 0 and r["score"] == 0.0
    assert r["reason_code"] == "MISSING_SOLUTION"


def test_naive_wrapper_lands_on_the_correctness_floor(build_workspace):
    """R3: wrapping the reference buys full correctness and no speed. Every
    equivalence gate must pass — it *is* the reference — and the speed gate
    alone must hold it at the floor with pass=false.

    The score is 0.2 exactly only when M <= 1. The wrapper does the same work
    as the naive path but is measured in a different process, so the paired
    ratio wobbles around 1.0 by ~1.5% (measured M in [0.997, 1.015] over 5
    nonces, score in [0.2000, 0.2033]). The bound pinned at calibration is below."""
    r, rc = run_judge(exploit_ws(build_workspace, "naive_wrapper"), 1001)
    assert rc == 0 and r["reason_code"] == "OK", r["detail"]
    assert all(r["gates"].values()), r["gates"]
    assert r["multiple"] <= 1.15, r["multiple"]
    assert r["score"] <= 0.21, r["score"]
    assert r["pass"] is False


# --------------------------------------------------------------------------
# Exploits — every one must land at or below its documented outcome
# --------------------------------------------------------------------------

def test_hardcoded_outputs_fails_equivalence(build_workspace):
    """R1: the lookup table never hits, because judge prompts come from an
    unseen nonce and text generated at judge time. It fails on the logit gate
    rather than the token gate purely because logits are checked first (see
    test_coherence_...): the bluffed outputs are nowhere near the reference."""
    r, rc = run_judge(exploit_ws(build_workspace, "hardcoded_outputs"), 1001)
    assert rc == 0
    assert r["score"] == 0.0 and r["pass"] is False, r
    assert r["reason_code"] in {"LOGIT_DIVERGENCE", "TOKEN_MISMATCH"}, r["reason_code"]
    assert r["gates"]["logit_tol"] is False or r["gates"]["token_equality"] is False
    assert r["positions"]["checked"] == 0, "no position should have been credited"


def test_own_weights_fails_equivalence(build_workspace):
    """R9: substituting the agent's own weights diverges immediately."""
    r, rc = run_judge(exploit_ws(build_workspace, "own_weights"), 1001)
    assert rc == 0
    assert r["score"] == 0.0 and r["pass"] is False, r
    assert r["reason_code"] in {"LOGIT_DIVERGENCE", "TOKEN_MISMATCH"}, r["reason_code"]


def test_inplace_quantization_fails_the_logit_gate(build_workspace):
    """R10 + R21: mutating the judge's model to bf16 precision is caught, and
    caught by a wide margin rather than a hair."""
    r, rc = run_judge(exploit_ws(build_workspace, "inplace_quantization"), 1001)
    assert rc == 0
    assert r["score"] == 0.0 and r["pass"] is False, r
    assert r["reason_code"] in {"LOGIT_DIVERGENCE", "TOKEN_MISMATCH"}, r["reason_code"]


def test_monkeypatch_clock_gains_nothing(build_workspace, gold_multiple):
    """R4: the child binds perf_counter before importing agent code, the naive
    baseline is timed in a process this code never enters, and the parent
    timestamps every result independently. A correct solution that also
    compresses every clock by 100x must measure the same as plain gold."""
    r, rc = run_judge(exploit_ws(build_workspace, "monkeypatch_clock"), 1001)
    assert rc == 0, r
    if r["reason_code"] == "INVALID_RUN":
        assert "TIMING_MISMATCH" in (r["detail"] or "")  # caught, not scored
        return
    assert r["reason_code"] == "OK", r["detail"]
    assert r["multiple"] <= 1.25 * gold_multiple, (
        f"clock patch inflated M to {r['multiple']} vs gold {gold_multiple}")


def test_set_threads_trips_the_parallelism_tripwire(build_workspace):
    """R11 / Check 7b: taking more cores than the 2-thread pin allows is
    detected natively, not merely prevented by a container cpuset. It must
    land in manual review (exit 3) rather than being scored at all."""
    r, rc = run_judge(exploit_ws(build_workspace, "set_threads"), 1001)
    assert rc == 3, (rc, r)
    assert r["reason_code"] == "INVALID_RUN"
    assert "ANOMALOUS_PARALLELISM" in (r["detail"] or ""), r["detail"]
    assert r["pass"] is False
    assert r["multiple"] is None, "a tripped run must never reach the score formula"
    assert max(r["cpu_wall_ratios"]) > 2.5


def test_gold_cpu_wall_ratio_sits_far_below_the_tripwire(gold_ws):
    """The other half of Check 7b: an honest solution must be nowhere near
    the line, or the tripwire would be a false-positive generator."""
    from judge.judge import CPU_WALL_CAP
    for n in NONCES:
        r = run_judge(gold_ws, n)[0]
        assert max(r["cpu_wall_ratios"]) < 0.6 * CPU_WALL_CAP, r["cpu_wall_ratios"]


def test_memoize_across_timing_repeats_gains_nothing(build_workspace, gold_multiple):
    """R2: S1-S3 are three independent fresh sets, so a prompt-keyed cache
    never hits on a timed call."""
    r, rc = run_judge(exploit_ws(build_workspace, "memoize_timing"), 1001)
    assert rc == 0 and r["reason_code"] == "OK", r["detail"]
    assert r["multiple"] <= 1.25 * gold_multiple, (
        f"memoization inflated M to {r['multiple']} vs gold {gold_multiple}")


# --------------------------------------------------------------------------
# Judge-side invariants
# --------------------------------------------------------------------------

@pytest.mark.parametrize("nonce", NONCES)
def test_timed_sets_are_pairwise_distinct(nonce):
    """The structural basis of the memoize defense (R2): a prompt-keyed cache
    can only pay off if a timed set repeats. S1-S3 must be independent draws,
    and the only set the judge ever repeats is W, which is never timed."""
    from judge import sampling
    from judge.judge import CALL_ORDER
    sets = sampling.sample_sets(nonce)

    def sig(name):
        prompts, n = sets[name]
        return (tuple(tuple(p.tolist()) for p in prompts), n)

    timed = ["S1", "S2", "S3"]
    assert len({sig(s) for s in timed}) == 3, "timed sets must be independent draws"
    repeats = [k for k in CALL_ORDER if CALL_ORDER.count(k.replace("_warm", "")) > 1]
    assert set(repeats) <= {"W", "W_warm"}
    assert not set(timed) & set(repeats)


def test_score_formula_waypoints():
    from judge import scoring
    assert scoring.score(False, 100.0) == 0.0
    assert scoring.score(True, 1.0) == pytest.approx(0.2)
    assert scoring.score(True, 0.5) == pytest.approx(0.2)      # clamped at the floor
    assert scoring.score(True, 8.0) == pytest.approx(0.68)     # the pass line
    assert scoring.score(True, 32.0) == pytest.approx(1.0)     # cap
    assert scoring.score(True, 1000.0) == pytest.approx(1.0)
    assert scoring.passed(True, 8.0) and not scoring.passed(True, 7.99)
    assert not scoring.passed(False, 100.0)


def _three_position_case():
    """Row of 3 positions: decisive, near-tie (gap 1e-4), decisive."""
    import torch
    ref_logits = torch.zeros(1, 3, 66)
    ref_logits[0, 0, 5] = 10.0                                     # decisive
    ref_logits[0, 1, 5], ref_logits[0, 1, 6] = 10.0, 10.0 - 1e-4   # near-tie
    ref_logits[0, 2, 7] = 10.0                                     # decisive
    top2 = torch.topk(ref_logits, 2, dim=-1).values
    return ref_logits, top2[..., 0] - top2[..., 1], torch.argmax(ref_logits, dim=-1)


def test_near_tie_divergence_stops_the_row_instead_of_failing_it():
    from judge import grading
    ref_logits, gaps, ref_tokens = _three_position_case()
    # Agent stays within the logit tolerance but the near-tie tips the other way.
    ag_logits = ref_logits.clone()
    ag_logits[0, 1, 5] = 10.0 - 1e-4
    ag_logits[0, 1, 6] = 10.0
    ag_tokens = ref_tokens.clone()
    ag_tokens[0, 1] = 6
    st = grading.GradeStats()
    grading.grade_call("T", ag_tokens, ag_logits, ref_tokens, ref_logits, gaps, st)
    assert st.ok, st.failures
    assert (st.skipped_ties, st.uncovered, st.checked, st.generated) == (1, 1, 1, 3)
    assert st.max_logit_diff <= grading.LOGIT_TOL


def test_token_mismatch_fires_at_a_decisive_position(monkeypatch):
    """The token gate's own code path. It can only be reached by relaxing the
    logit gate, which is exactly the coherence property asserted below: with
    the real tolerance, nothing can pass logits and fail tokens."""
    from judge import grading
    monkeypatch.setattr(grading, "LOGIT_TOL", 50.0)
    ref_logits, gaps, ref_tokens = _three_position_case()
    ag_logits = ref_logits.clone()
    ag_logits[0, 2, 9] = 11.0
    ag_tokens = ref_tokens.clone()
    ag_tokens[0, 2] = 9
    st = grading.GradeStats()
    grading.grade_call("T", ag_tokens, ag_logits, ref_tokens, ref_logits, gaps, st)
    assert not st.ok and st.failures[0][0] == "TOKEN_MISMATCH"


def test_coherence_no_position_passes_logits_and_fails_tokens():
    """Gate 4 coherence: a LOGIT_TOL perturbation can only flip an argmax where the
    reference top-2 gap is < 2*LOGIT_TOL, and every such position is below the
    1e-3 near-tie line and therefore skipped. So TOKEN_MISMATCH is unreachable
    while the logit gate holds -- verified here by brute force."""
    import torch

    from judge import grading
    assert 2 * grading.LOGIT_TOL <= grading.NEAR_TIE_GAP
    g = torch.Generator().manual_seed(7)
    for _ in range(200):
        ref_logits = (torch.randn(1, 4, 66, generator=g) * 20).float()
        top2 = torch.topk(ref_logits, 2, dim=-1).values
        gaps = top2[..., 0] - top2[..., 1]
        ref_tokens = torch.argmax(ref_logits, dim=-1)
        # worst-case admissible perturbation: exactly +/- LOGIT_TOL everywhere
        pert = (torch.randint(0, 2, ref_logits.shape, generator=g).float() * 2 - 1)
        ag_logits = ref_logits + pert * grading.LOGIT_TOL
        ag_tokens = torch.argmax(ag_logits, dim=-1)
        st = grading.GradeStats()
        grading.grade_call("T", ag_tokens, ag_logits, ref_tokens, ref_logits, gaps, st)
        assert not any(f[0] == "TOKEN_MISMATCH" for f in st.failures), st.failures


def test_consistency_gate_catches_tokens_not_matching_returned_logits():
    import torch

    from judge import grading
    ref_logits = torch.zeros(2, 2, 66)
    ref_logits[..., 3] = 5.0
    gaps = torch.full((2, 2), 5.0)
    ref_tokens = torch.full((2, 2), 3, dtype=torch.long)
    ag_tokens = ref_tokens.clone()
    ag_tokens[1, 1] = 4                            # not argmax of its own logits
    st = grading.GradeStats()
    grading.grade_call("T", ag_tokens, ref_logits.clone(), ref_tokens, ref_logits, gaps, st)
    assert not st.ok and st.failures[0][0] == "INCONSISTENT_ARGMAX"


def test_manifest_mismatch_is_invalid_run_not_a_zero(tmp_path, monkeypatch):
    """R18: judge-side breakage must exit 3 (retry), never score the agent."""
    import judge.judge as J
    assets = tmp_path / "judge_assets"
    shutil.copytree(os.path.join(ROOT, "judge", "judge_assets"), assets)
    manifest = json.load(open(assets / "MANIFEST.json"))
    manifest["starter/weights.pt"] = "0" * 64      # corrupt one hash
    json.dump(manifest, open(assets / "MANIFEST.json", "w"))
    monkeypatch.setattr(J, "ASSETS", str(assets))
    monkeypatch.setattr(J, "PRISTINE_PARENT", str(assets / "pristine"))
    report = J.new_report(1001)
    with pytest.raises(J.Judged) as e:
        J.check_manifest(report)
    assert e.value.exit_code == 3 and e.value.reason == "INVALID_RUN"
    assert report["manifest_ok"] is False


def test_snapshot_excludes_starter_and_enforces_the_published_cap(tmp_path):
    """Gate 1: the graded snapshot drops the shipped starter/ tree, so the
    ~40 MB weights file cannot push an honest solution over the 25 MB cap."""
    import judge.judge as J
    ws = tmp_path / "ws"
    (ws / "starter").mkdir(parents=True)
    shutil.copy(GOLD, ws / "solution.py")
    (ws / "starter" / "weights.pt").write_bytes(b"\0" * (40 * 1024 * 1024))
    snap = J.make_snapshot(str(ws), str(tmp_path / "out"))
    assert not os.path.exists(os.path.join(snap, "starter"))
    assert J.snapshot_size(snap) < J.SNAPSHOT_CAP_BYTES


def test_no_judge_asset_is_reachable_from_agent_paths():
    """Brief acceptance criterion 3 / threat R8: the agent's workspace must
    contain no judge asset, and the snapshot handed to the judge must carry
    none either."""
    import judge.judge as J
    env = os.path.join(ROOT, "env")
    assert not os.path.commonpath([J.ASSETS, env]).startswith(env)
    leaked = []
    for root, dirs, files in os.walk(env):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in files:
            low = f.lower()
            if "manifest" in low or "nonce" in low or low == "judge.py":
                leaked.append(os.path.join(root, f))
    assert not leaked, leaked
    # The corpus grammar is deliberately NOT secret (the nonce is); assert the
    # thing that must be secret is absent from the workspace.
    assert not os.path.exists(os.path.join(env, "judge_assets"))


def test_local_check_agrees_with_the_judge_on_gold(tmp_path):
    """R20: local_check.py must import the judge's own grading and scoring, so
    an honest agent's local verdict matches the real one."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    shutil.copy(GOLD, ws / "solution.py")
    shutil.copy(os.path.join(ROOT, "env", "local_check.py"), ws / "local_check.py")
    os.symlink(os.path.join(ROOT, "env", "starter"), ws / "starter")
    (ws / "checklib").mkdir()
    for m in ("grading.py", "scoring.py", "sampling.py", "gen_corpus.py"):
        shutil.copy(os.path.join(ROOT, "judge", m), ws / "checklib" / m)
    proc = subprocess.run([PY, str(ws / "local_check.py"), "--nonce", "4242", "--quick"],
                          capture_output=True, text=True,
                          env=dict(os.environ, OMP_NUM_THREADS="2"), timeout=900)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "pass=True" in proc.stdout, proc.stdout
    assert "[FAIL]" not in proc.stdout
