# Deployment — container flags and mounts

The environment runs in a 2-CPU / 6 GB / no-network container built from
`env/Dockerfile`. The **same image serves both the agent session and the judge run**,
so the paired timing ratio `M` is measured under identical conditions on both sides.

The judge is machine-relative by design — it computes the reference live, scores speed
as a ratio rather than an absolute rate, and derives every threshold from measurement
against the reference solution. Nothing about it needs recalibration to move between a
container and a development machine. The figures published in this repo are measured
natively with threads pinned to 2 (`torch.set_num_threads(2)`, `OMP_NUM_THREADS=2`);
see `env/ENV.txt` for exact versions.

## What lands in /workspace

`env/` holds both the agent-visible workspace contents and this deployment spec.
The Dockerfile is authoritative about which of them the agent actually receives:

| Path in image | Source | Purpose |
|---|---|---|
| `/workspace/starter/` | `env/starter/` | `model.py`, `weights.pt`, `vocab.json`, `sample_text.txt` |
| `/workspace/solution.py` | `env/solution.py` | the stub carrying the required signature |
| `/workspace/local_check.py` | `env/local_check.py` | self-check with the judge's own gates and formula |
| `/workspace/ENV.txt` | `env/ENV.txt` | pinned versions and process settings |
| `/workspace/checklib/` | `judge/{grading,scoring,sampling,gen_corpus}.py` | shared grading code |

`checklib/` is deliberately not secret. Shipping the judge's real grading, scoring and
sampling modules is what makes the local check *be* the judge's check rather than an
approximation of it. The secret is the run nonce, not the code. The judge never reads
these copies — it hash-verifies its own against `judge/judge_assets/MANIFEST.json`.

`Dockerfile`, `DEPLOYMENT.md`, `requirements.txt` and `task_prompt.md` are build and
specification inputs, not workspace files. Asset-prep tooling in `tools/` is never
shipped into the container.

## Agent session

```
docker run --cpus=2 --memory=6g --network=none \
  -v <workspace-volume>:/workspace \
  <image> bash
```

Only `/workspace` and `/tmp` are writable. **No judge asset exists anywhere in this
container**: judge code, `judge_assets/`, and the run nonce are injected only at judge
time. That is what makes the "prompts sampled from an unseen nonce" guarantee hold —
there is nothing to read ahead.

## Judge run

```
docker run --cpus=2 --memory=6g --network=none \
  -v <workspace-snapshot>:/snapshot:ro \
  -v <judge-repo>/judge:/judge:ro \
  <image> python /judge/judge.py --snapshot /snapshot --nonce <fresh-nonce>
```

Both mounts are **read-only**: the workspace snapshot and the judge tree alike. The
judge takes its own read-only copy of the snapshot before any agent code runs, and the
graded snapshot excludes the shipped `starter/` tree — the judge always substitutes its
own pristine copy, so edits to `/workspace/starter/` are never used.

`--nonce` must be fresh for every scoring run. `--snapshot` grades a directory as-is;
`--workspace` (used by `judge/run_judge.sh` for local runs) snapshots the directory
first and then grades it.

Exit codes:

| Code | Meaning |
|---|---|
| 0 | scored — report JSON on stdout, `pass` true or false |
| 3 | `INVALID_RUN` — no score; the caller retries with a fresh nonce, never records a 0 |

Treating exit 3 as a retry rather than a failure is load-bearing. It is how judge-side
faults (asset mismatch, machine timing noise, coverage collapse) stay off the agent's
reward signal.

## Enforcement scope

Two classes of property, kept distinct so it is clear what survives outside a container:

**Enforced by the judge harness** — holds wherever this repo runs, container or not:
parent/child process isolation; reference outputs only ever in parent memory and never
written to disk; the clock captured before agent import; just-in-time prompt reveal;
live reference computation from hash-verified pristine assets; fresh nonce-derived
prompts each run; every returned output graded; paired relative timing with drift
probes; the CPU/wall parallelism tripwire.

**Enforced by these container flags** — a deployment property, not a harness one:
`--network=none`, `--cpus=2 --memory=6g`, read-only mounts. For this task, network
access enables no known exploit: the reference is computed live from local assets and
judge prompts never exist outside the judge process. `--network=none` is therefore
defense in depth rather than load-bearing.

**Both:** the CPU cap. `--cpus=2` *prevents* a parallelism gain at deployment, while the
harness's CPU/wall tripwire *detects* the attempt on any machine. See
`docs/threat-model.md` for the full mapping.
