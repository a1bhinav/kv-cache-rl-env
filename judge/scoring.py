"""Score formula and pass rule (docs/design.md). Shared by judge and local_check."""

import math

SPEED_BAR = 8.0
SCORE_FLOOR = 0.2
CAP_LOG2 = 5.0          # S = 1.0 at M = 32x
PLAUSIBILITY_CAP = 128.0  # M beyond this -> INVALID_RUN, never scored


def score(all_gates_ok, M):
    if not all_gates_ok:
        return 0.0
    return SCORE_FLOOR + 0.8 * min(1.0, max(0.0, math.log2(M)) / CAP_LOG2)


def passed(all_gates_ok, M):
    return bool(all_gates_ok and M >= SPEED_BAR)
