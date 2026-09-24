"""Mutation testing: reproduce each known fault and check that the suite detects it.

A passing suite only shows that the tests agree with the code. Each mutant below reproduces a
fault that produced a plausible number during development, and the suite must reject it; a
mutant that survives marks a gap in the tests.
"""
import sys, pathlib, itertools
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np

DIMS = [4, 64, 1024, 12288, 20000]
SPARSITIES = [0.001, 0.01, 0.1, 0.5, 1.0]
RESULTS = []


def check(name, detector):
    """`detector` returns True if the buggy behaviour is DETECTED."""
    caught = detector()
    RESULTS.append((name, caught))
    print(f"  [{'CAUGHT' if caught else 'SURVIVED'}] {name}")


# ---- MUTANT 1: consistency via norm over all dimensions (the exp(-55) = 1e-22 bug) --------
def mutant_consistency_norm(pred, past):
    seam = np.linalg.norm(pred[0] - past[-1])          # NOT per-dimension
    return float(np.exp(-seam))


def detect_norm_bug():
    """Property: DIMENSION INVARIANCE. With the SAME per-coordinate discrepancy, consistency
    must not change as D grows.

    Stating it this way removes my judgement from the test. An earlier version of this
    detector picked a discrepancy of 1e-3, which at D=12288 gives a seam of only 0.11 and
    exp(-0.11)=0.89 -- so the mutant SURVIVED. The real data had a per-pixel discrepancy of
    ~0.5, i.e. a seam of ~55 and exp(-55)=1e-22. I had once again chosen a fixture gentler
    than reality. The invariant below needs no such choice: it compares the metric against
    ITSELF across D, so any dimension coupling is exposed whatever the magnitude.
    """
    for eps in (1e-3, 1e-2, 0.1, 0.5):        # sweep the magnitude too, do not pick one
        vals = []
        for D in DIMS:
            rng = np.random.default_rng(D)
            past = rng.normal(0, 1, (4, D))
            pred = np.repeat(past[-1:], 8, axis=0) + eps * rng.normal(0, 1, (8, D))
            vals.append(mutant_consistency_norm(pred, past))
        if max(vals) - min(vals) > 0.05:      # varies with D alone -> dimension coupled
            return True
    return False


# ---- MUTANT 2: quantile threshold on sparse difference images -----------------------------
def mutant_support_quantile(v, q=0.90):
    return np.abs(v) >= np.quantile(np.abs(v), q)


def detect_quantile_bug():
    """Property: the support of a sparse vector must not exceed its live-coordinate count."""
    for D, sp in itertools.product(DIMS, SPARSITIES):
        rng = np.random.default_rng(hash((D, sp)) % 2**32)
        k = max(1, int(D * sp)); v = np.zeros(D)
        v[rng.choice(D, k, replace=False)] = rng.normal(0, 1, k)
        if mutant_support_quantile(v).sum() > max(1, k):
            return True
    return False


# ---- MUTANT 3: relative decay rule when m0 == 0 -------------------------------------------
def mutant_verdict(m0, mn, causal_lost=0.3):
    return mn < causal_lost * m0        # 0 < 0 is False -> strongest result called a failure


def detect_verdict_bug():
    """Property: a model with zero response at every delay must not be reported as 'no
    separation' -- that is the STRONGEST version of the finding."""
    return mutant_verdict(0.0, 0.0) is False


# ---- MUTANT 4: unpaired stochastic sampling (the .rsample() 15/255 bug) --------------------
def mutant_unpaired_rollout(actions, rng):
    return rng.normal(0, 1, (8, 64))     # fresh noise per call -> arms differ with no cause


def detect_unpaired_bug():
    """Property: identical actions must produce identical rollouts (null control == 0)."""
    rng = np.random.default_rng(0)
    a = mutant_unpaired_rollout([0, 1], rng); b = mutant_unpaired_rollout([0, 1], rng)
    return float(np.abs(a - b).max()) > 0


# ---- MUTANT 5: guard threshold from an assumed scaling law --------------------------------
def mutant_guard_floor(divergences, D):
    return 0.25 * float(np.std(divergences)) * np.sqrt(D)   # ~13.9 vs a real ~3.0


def detect_formula_guard_bug():
    """Property: a guard must not reject every cell of real data."""
    pong_like = [2.85, 3.40, 3.13, 2.92, 2.60, 2.84, 2.27, 2.10]
    floor = mutant_guard_floor(pong_like, 12288)
    return sum(d >= floor for d in pong_like) == 0


# ---- MUTANT 6: dataset gate that ignores the degenerate fraction --------------------------
def mutant_gate(true_div, min_div=0.1):
    return len([v for v in true_div.values() if v > min_div]) >= 2   # no majority check


def detect_gate_bug():
    """Property: a suite where only 31% of units carry a live intervention must not pass."""
    td = {f"live{i}": 2.0 for i in range(8)}
    td.update({f"dead{i}": 0.0 for i in range(18)})
    return mutant_gate(td) is True      # buggy gate says OK on 8/26


if __name__ == "__main__":
    print("Reproducing each known fault; the suite must catch every one:\n")
    check("consistency via norm over all dims (exp(-55) = 1e-22)", detect_norm_bug)
    check("quantile threshold on sparse difference images", detect_quantile_bug)
    check("relative decay rule degenerates when m0 = 0", detect_verdict_bug)
    check("unpaired stochastic sampling reads as causality", detect_unpaired_bug)
    check("guard floor from an assumed scaling law", detect_formula_guard_bug)
    check("dataset gate ignoring the degenerate fraction", detect_gate_bug)
    survived = [n for n, c in RESULTS if not c]
    print(f"\n{sum(c for _, c in RESULTS)}/{len(RESULTS)} reproduced faults are detected")
    assert not survived, f"SUITE HAS GAPS: these faults are not detected: {survived}"
    print("NO MUTANT SURVIVES: every reproduced fault is detected")
