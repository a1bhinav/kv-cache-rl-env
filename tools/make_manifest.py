#!/usr/bin/env python3
"""Check or re-pin the judge's asset manifest.

The judge refuses to run when any hash in judge_assets/MANIFEST.json does not
match the file on disk (INVALID_RUN at gate 0), which is what makes silent
tampering with the pristine assets or the corpus grammar a hard stop rather
than a quietly wrong score. The cost of that guarantee is that a *deliberate*
edit to a pinned file -- fixing a docstring, retraining the weights -- has to
be re-recorded on purpose.

This tool is that step. It never invents the key set: it reads MANIFEST.json
and checks exactly the keys already pinned there, resolving each one the same
way judge.py does, so the two cannot disagree about which file a key means.

  python tools/make_manifest.py            # check only; exit 1 on drift
  python tools/make_manifest.py --write    # re-pin drifted hashes

Usage: python tools/make_manifest.py [--write] [--manifest PATH]
"""

import argparse
import hashlib
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JUDGE = os.path.join(ROOT, "judge")
MANIFEST = os.path.join(JUDGE, "judge_assets", "MANIFEST.json")
PRISTINE_PARENT = os.path.join(JUDGE, "judge_assets", "pristine")


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def resolve(rel):
    """Map a manifest key to its file, mirroring judge.py's check_manifest."""
    if rel.startswith("starter/"):
        return os.path.join(PRISTINE_PARENT, rel)
    return os.path.join(JUDGE, rel)


def read_pinned(raw):
    """Pinned hashes in file order. Parsed from the raw text rather than via
    json.load so the on-disk key order and formatting stay authoritative."""
    return re.findall(r'"([^"]+)"\s*:\s*"([0-9a-f]{64})"', raw)


def repin(raw, rel, current):
    """Replace one key's hash in place, leaving every other byte untouched."""
    pattern = re.compile(r'("' + re.escape(rel) + r'"\s*:\s*")[0-9a-f]{64}(")')
    new_raw, n = pattern.subn(lambda m: m.group(1) + current + m.group(2), raw)
    if n != 1:
        raise SystemExit(f"error: expected exactly 1 entry for {rel}, found {n}")
    return new_raw


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true",
                    help="re-pin drifted hashes instead of only reporting them")
    ap.add_argument("--manifest", default=MANIFEST,
                    help="path to MANIFEST.json (default: the judge's own)")
    args = ap.parse_args()

    raw = open(args.manifest, encoding="utf-8").read()
    pinned = read_pinned(raw)
    if not pinned:
        raise SystemExit(f"error: no pinned entries found in {args.manifest}")

    rows, drifted, missing = [], [], []
    for rel, want in pinned:
        path = resolve(rel)
        if not os.path.exists(path):
            rows.append((rel, want, None))
            missing.append(rel)
            continue
        got = sha256(path)
        rows.append((rel, want, got))
        if got != want:
            drifted.append((rel, got))

    width = max(len(rel) for rel, _, _ in rows)
    print(f"{'asset'.ljust(width)}  {'pinned':<12}  {'current':<12}  status")
    print(f"{'-' * width}  {'-' * 12}  {'-' * 12}  ------")
    for rel, want, got in rows:
        if got is None:
            status, shown = "MISSING", "-"
        elif got == want:
            status, shown = "ok", got[:12]
        else:
            status, shown = "DRIFT", got[:12]
        print(f"{rel.ljust(width)}  {want[:12]:<12}  {shown:<12}  {status}")

    if missing:
        print(f"\n{len(missing)} pinned asset(s) missing from disk: "
              + ", ".join(missing), file=sys.stderr)
        return 1

    if not drifted:
        print(f"\nall {len(rows)} assets match the manifest")
        return 0

    print(f"\n{len(drifted)} asset(s) drifted from the manifest: "
          + ", ".join(rel for rel, _ in drifted))

    if not args.write:
        print("the judge will refuse to run (INVALID_RUN at gate 0) until this "
              "is resolved.\nre-pin deliberate edits with: "
              "python tools/make_manifest.py --write", file=sys.stderr)
        return 1

    new_raw = raw
    for rel, got in drifted:
        new_raw = repin(new_raw, rel, got)
    if new_raw == raw:                      # nothing to say that we did not
        print("manifest already current; left byte-for-byte unchanged")
        return 0
    with open(args.manifest, "w", encoding="utf-8", newline="") as f:
        f.write(new_raw)
    print(f"re-pinned {len(drifted)} entr(y/ies) in {args.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
