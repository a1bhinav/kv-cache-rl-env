# Container image for the KV-cache decoding environment: the same image serves
# the agent session and the judge run. Development and measurement here are
# native, with threads pinned to 2; the judge is machine-relative (live
# reference, relative speed bar, calibrated thresholds), so nothing about its
# behavior depends on which of the two it runs in. See DEPLOYMENT.md for run
# flags and mounts.

FROM python:3.11-slim

# Pinned dependencies (frozen from the natively-tested environment).
COPY env/requirements.txt /opt/requirements.txt
RUN pip install --no-cache-dir -r /opt/requirements.txt

# The agent's workspace: starter repo, stub, local checker, env pin.
COPY env/starter/       /workspace/starter/
COPY env/solution.py    /workspace/solution.py
COPY env/local_check.py /workspace/local_check.py
COPY env/ENV.txt        /workspace/ENV.txt

# local_check.py grades with the judge's OWN grading/scoring/sampling code.
# These modules are not secret (the run nonce is); shipping them lets the
# agent's local checker be the judge's checker by construction. The judge
# never reads these copies -- it hash-verifies its own (MANIFEST.json).
COPY judge/grading.py judge/scoring.py judge/sampling.py judge/gen_corpus.py \
     /workspace/checklib/

# Judge code + private assets live OUTSIDE /workspace and are mounted
# read-only only at judge time (see DEPLOYMENT.md); they are never baked
# into the agent-visible image layer list above this line in an agent VM.

WORKDIR /workspace
