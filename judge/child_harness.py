"""Child harness — the only judge process that runs agent code (docs/design.md).

Minimal by design: no judge logic, no reference data, no grading. It imports
the agent's solution from the workspace snapshot and executes calls the parent
scripts over stdin, recording raw outputs and raw timings for the parent to
grade after exit.

Anti-tamper properties this file provides:
- The wall clock (`_clock`) is captured BEFORE any agent code is imported, so
  import-time monkeypatching of time.* cannot affect recorded timings.
- Prompts arrive JUST IN TIME: each call's prompts are written by the parent
  and announced over stdin only when that call starts, so agent code can never
  see a timed set before its timed call begins (kills precompute/memoization
  of timing sets during untimed slots).
- Every result line carries the sha256 of the just-written output file; the
  parent timestamps the line's arrival, giving a parent-measured envelope that
  bounds the true compute time independent of anything this process reports.

Protocol (JSON lines on stdin/stdout):
  parent -> child: {"op": "init", "pristine_parent": ..., "snapshot": ...}
                   {"op": "fresh_model"}
                   {"op": "call", "key": ..., "in": path, "out": path}
                   {"op": "exit"}
  child -> parent: {"ready": true, "import_s": ...} | {"done": key,
                   "wall_s": ..., "sha": ...} | {"bye": true, "peak_rss": ...}
"""

import hashlib
import json
import os
import resource
import sys
import time

_clock = time.perf_counter  # captured pre-import of any agent code

import torch  # noqa: E402

torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _cpu_seconds():
    """Total CPU consumed by this process AND any child it spawned
    (user + sys). Paired with wall time this gives a parallelism ratio: a
    solution obeying the 2-thread pin cannot exceed ~2, whatever it does."""
    me = resource.getrusage(resource.RUSAGE_SELF)
    kids = resource.getrusage(resource.RUSAGE_CHILDREN)
    return me.ru_utime + me.ru_stime + kids.ru_utime + kids.ru_stime


def record_call(fn, model, prompts, n):
    rec = {"ok": False, "err": None, "wall_s": None, "cpu_s": None,
           "new_tokens": None, "step_logits": None}
    c0 = _cpu_seconds()
    t0 = _clock()
    try:
        out = fn(model, [p.clone() for p in prompts], n)
    except BaseException as e:  # incl. SystemExit; graded as AGENT_EXCEPTION
        rec["wall_s"] = _clock() - t0
        rec["cpu_s"] = _cpu_seconds() - c0
        rec["err"] = f"{type(e).__name__}: {e}"
        return rec
    rec["wall_s"] = _clock() - t0
    rec["cpu_s"] = _cpu_seconds() - c0
    rec["ok"] = True
    if isinstance(out, tuple) and len(out) == 2 and all(isinstance(x, torch.Tensor) for x in out):
        nt, sl = out
        rec["nt_device"], rec["sl_device"] = nt.device.type, sl.device.type
        rec["new_tokens"] = nt.detach().to("cpu")
        rec["step_logits"] = sl.detach().to("cpu")
    else:
        rec["shape_note"] = type(out).__name__
    return rec


def main():
    fn = None
    model = None
    pristine_model_mod = None
    for line in sys.stdin:
        cmd = json.loads(line)
        op = cmd["op"]
        if op == "init":
            own_dir = os.path.dirname(os.path.abspath(__file__))
            sys.path = [cmd["pristine_parent"], cmd["snapshot"]] + [
                p for p in sys.path if p not in ("", own_dir, os.getcwd())
            ]
            t0 = _clock()
            try:
                import importlib.util
                mod_spec = importlib.util.spec_from_file_location(
                    "solution", os.path.join(cmd["snapshot"], "solution.py"))
                solution = importlib.util.module_from_spec(mod_spec)
                sys.modules["solution"] = solution
                mod_spec.loader.exec_module(solution)
            except Exception as e:
                emit({"ready": False, "import_s": _clock() - t0,
                      "fatal": f"IMPORT_ERROR:{type(e).__name__}: {e}"})
                return
            import_s = _clock() - t0
            fn = getattr(solution, "generate_cached", None)
            if not callable(fn):
                emit({"ready": False, "import_s": import_s,
                      "fatal": "BAD_SIGNATURE:generate_cached missing or not callable"})
                return
            from starter import model as pristine_model_mod  # noqa: F811
            model = pristine_model_mod.load_model()  # excluded from all timing
            emit({"ready": True, "import_s": import_s})
        elif op == "fresh_model":
            model = pristine_model_mod.load_model()
            emit({"fresh": True})
        elif op == "call":
            data = torch.load(cmd["in"], weights_only=True)
            prompts = [t for t in data["prompts"]]
            rec = record_call(fn, model, prompts, int(data["N"]))
            torch.save(rec, cmd["out"])
            sha = hashlib.sha256(open(cmd["out"], "rb").read()).hexdigest()
            emit({"done": cmd["key"], "wall_s": rec["wall_s"],
                  "cpu_s": rec["cpu_s"], "err": rec["err"], "sha": sha})
        elif op == "exit":
            emit({"bye": True,
                  "peak_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
            return


if __name__ == "__main__":
    main()
