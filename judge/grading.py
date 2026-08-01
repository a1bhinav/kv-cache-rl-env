"""Equivalence grading — gate 4, plus gate 3's shape checks (docs/design.md).

Pure functions over recorded tensors; no I/O, no timing. Used verbatim by both
the judge parent process and the agent-facing local_check.py, so local grading
is the judge's grading by construction.
"""

from dataclasses import dataclass, field

import torch

# Calibrated against the reference solution (see docs/design.md, "Calibration"):
# honest cached-vs-recompute divergence maxes at 5.34e-5
# (14 ulps at this model's |logit| <= 50) over 405,504 measured logit values.
# 2.5e-4 sits 4.7x above that and 2x below the near-tie line, so no position
# can both pass the logit gate and fail the token gate on float jitter.
LOGIT_TOL = 2.5e-4
NEAR_TIE_GAP = 1e-3
VOCAB_SIZE = 66


@dataclass
class GradeStats:
    generated: int = 0
    checked: int = 0
    skipped_ties: int = 0
    uncovered: int = 0
    max_logit_diff: float = 0.0
    failures: list = field(default_factory=list)  # (reason_code, detail)

    @property
    def coverage(self):
        return self.checked / self.generated if self.generated else 1.0

    @property
    def skip_rate(self):
        return self.skipped_ties / self.generated if self.generated else 0.0

    @property
    def ok(self):
        return not self.failures


def check_shapes(out, B, N):
    """Check 3: return-contract validation. Returns (ok, reason_code, detail)."""
    if not (isinstance(out, tuple) and len(out) == 2):
        return False, "BAD_RETURN", f"expected 2-tuple, got {type(out).__name__}"
    nt, sl = out
    if not (isinstance(nt, torch.Tensor) and isinstance(sl, torch.Tensor)):
        return False, "BAD_RETURN", "elements are not tensors"
    if nt.dtype != torch.int64 or nt.device.type != "cpu" or tuple(nt.shape) != (B, N):
        return False, "BAD_RETURN", f"new_tokens {nt.dtype} {nt.device} {tuple(nt.shape)}, want int64 cpu {(B, N)}"
    if sl.dtype != torch.float32 or sl.device.type != "cpu" or tuple(sl.shape) != (B, N, VOCAB_SIZE):
        return False, "BAD_RETURN", f"step_logits {sl.dtype} {sl.device} {tuple(sl.shape)}, want float32 cpu {(B, N, VOCAB_SIZE)}"
    if not torch.isfinite(sl).all():
        return False, "NONFINITE_LOGITS", "step_logits contain non-finite values"
    return True, None, None


def grade_call(set_name, agent_tokens, agent_logits, ref_tokens, ref_logits,
               ref_gaps, stats):
    """Grade one recorded call against the reference, per sequence, walking
    t = 0..N-1. Mutates `stats`; appends to stats.failures on gate breaks.

    agent_tokens [B,N] int64, agent_logits [B,N,66] fp32;
    ref_tokens/ref_logits same shapes; ref_gaps [B,N] fp32 (reference top-2 gap).
    """
    B, N = agent_tokens.shape
    # Consistency: always checked, every position, every row.
    recomputed = torch.argmax(agent_logits, dim=-1)
    if not torch.equal(recomputed, agent_tokens):
        bad = (recomputed != agent_tokens).nonzero()
        i, t = int(bad[0][0]), int(bad[0][1])
        stats.failures.append(("INCONSISTENT_ARGMAX",
                               f"{set_name}: tokens != argmax(step_logits), first at row {i} step {t}"))
        stats.generated += B * N
        return

    for i in range(B):
        prefix_ok = True
        for t in range(N):
            stats.generated += 1
            if not prefix_ok:
                stats.uncovered += 1
                continue
            # Logit closeness (prefix matches so far)
            diff = float((agent_logits[i, t] - ref_logits[i, t]).abs().max())
            if diff > stats.max_logit_diff:
                stats.max_logit_diff = diff
            if diff > LOGIT_TOL:
                stats.failures.append(("LOGIT_DIVERGENCE",
                                       f"{set_name}: row {i} step {t} max|diff|={diff:.3e} > {LOGIT_TOL}"))
                return
            # Token equality with near-tie skip (from reference logits only)
            if float(ref_gaps[i, t]) < NEAR_TIE_GAP:
                stats.skipped_ties += 1
                if int(agent_tokens[i, t]) != int(ref_tokens[i, t]):
                    prefix_ok = False  # legitimate divergence: rest of row uncovered
            else:
                if int(agent_tokens[i, t]) != int(ref_tokens[i, t]):
                    stats.failures.append(("TOKEN_MISMATCH",
                                           f"{set_name}: row {i} step {t} agent={int(agent_tokens[i,t])} ref={int(ref_tokens[i,t])} gap={float(ref_gaps[i,t]):.3e}"))
                    return
                stats.checked += 1
