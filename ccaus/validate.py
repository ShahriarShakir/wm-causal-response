"""Validity gates that a statistic must pass before it is reported.

They guard against two failures of an analysis layer:
  * a guard floor calibrated from a formula rejects every cell, and a sweep reports n=0 as though
    it were a measurement;
  * a correlation is computed over games in which the intervention changed nothing (median true
    divergence exactly 0), so a near-zero response there is an absent cause, not a blind model.

Every statistic passes `gate_correlation` or `gate_dataset`, and a claim that fails a gate is
reported as UNSUPPORTED with the reason attached. The gates return
verdicts, they do not raise -- a failed gate is a result about the experiment, not a crash.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np


@dataclass
class Gate:
    """One pass/fail check with the reason attached, so a failure is never silent."""

    name: str
    passed: bool
    detail: str

    def __str__(self) -> str:
        return f"[{'PASS' if self.passed else 'FAIL'}] {self.name}: {self.detail}"


@dataclass
class Report:
    gates: List[Gate] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str) -> "Report":
        self.gates.append(Gate(name, passed, detail)); return self

    @property
    def ok(self) -> bool:
        return all(g.passed for g in self.gates)

    def failures(self) -> List[Gate]:
        return [g for g in self.gates if not g.passed]

    def __str__(self) -> str:
        head = "ALL GATES PASSED" if self.ok else f"{len(self.failures())} GATE(S) FAILED"
        return head + "\n" + "\n".join(f"  {g}" for g in self.gates)


def _midrank(v: np.ndarray) -> np.ndarray:
    """Average ranks for ties (the standard Spearman convention)."""
    order = np.argsort(v, kind="mergesort"); ranks = np.empty(len(v), float); sv = v[order]
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and sv[j + 1] == sv[i]: j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j); i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation with average ranks for ties (ordinal ranks, used before 2026-09-21, broke ties in
    array order). A constant input carries no ranking information and returns 0.0 rather than an
    arbitrary value."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or np.all(x == x[0]) or np.all(y == y[0]): return 0.0
    return float(np.corrcoef(_midrank(x), _midrank(y))[0, 1])


def permutation_p(x: Sequence[float], y: Sequence[float], n_perm: int = 20000,
                  seed: int = 0) -> float:
    """Two-sided permutation p for the rank correlation. No normality assumed at n<30."""
    r = abs(spearman(x, y))
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)
    null = np.array([abs(spearman(x, rng.permutation(y))) for _ in range(n_perm)])
    return float((null >= r).mean())


def detectable_rho(n: int, alpha: float = 0.05) -> float:
    """Smallest |rho| a two-sided test at this n could call significant.

    Reported alongside every correlation so an underpowered null is never mistaken for
    evidence of absence. At n=8 this is ~0.71: a true rho of 0.5 CANNOT reach significance,
    so "p=0.217" says nothing about whether the correlation is real.
    """
    if n < 4:
        return float("nan")
    from math import sqrt
    t = {4: 4.303, 5: 3.182, 6: 2.776, 7: 2.571, 8: 2.447, 9: 2.365, 10: 2.306,
         11: 2.262, 12: 2.228, 15: 2.160, 20: 2.101, 25: 2.069, 30: 2.048}
    tc = t.get(n, 1.96 if n > 30 else 2.2)
    return float(tc / sqrt(n - 2 + tc * tc))


def mde_rho(n: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Minimum detectable Spearman/Pearson correlation at the given power (Fisher z approximation,
    two-sided alpha): rho = tanh((z_{1-alpha/2} + z_{power}) / sqrt(n - 3)).

    `detectable_rho` is the smaller number at which an observed correlation first reaches significance
    (50% power); this is the effect size an experiment of size n detects with probability `power`.
    Note that the two were conflated in the text; both are now reported."""
    if n < 4:
        return float("nan")
    from math import sqrt, tanh
    z = {0.05: 1.959964, 0.01: 2.575829}.get(alpha, 1.959964) + {0.8: 0.841621, 0.9: 1.281552}.get(power, 0.841621)
    return float(tanh(z / sqrt(n - 3)))


def gate_dataset(true_divergence: Dict[str, float], min_divergence: float = 0.1) -> Report:
    """Which units carry a real intervention at all.

    A unit whose GROUND-TRUTH divergence is ~0 had no cause to track, so its near-zero response
    is uninformative about the model. Excluding it is decided by ground truth alone and so
    cannot favour any model -- but it must be COUNTED, because an exclusion that removes most
    of the suite destroys the power of anything computed afterwards.
    """
    live = {k: v for k, v in true_divergence.items()
            if np.isfinite(v) and v > min_divergence}
    dead = sorted(set(true_divergence) - set(live))
    n, N = len(live), len(true_divergence)
    rep = Report()
    rep.add("units carry a real intervention", n >= 2,
            f"{n}/{N} above true-divergence {min_divergence}; degenerate: {dead}")
    rep.add("majority of the suite survives", n >= 0.5 * N,
            f"{100*n/N:.0f}% survive; below 50% means the intervention design, not the "
            f"models, is limiting what can be measured")
    return rep


def gate_correlation(x: Sequence[float], y: Sequence[float], alpha: float = 0.05) -> Report:
    """A correlation may only be REPORTED if the test could have detected it."""
    n = len(x)
    rep = Report()
    r = spearman(x, y) if n >= 4 else float("nan")
    dr = detectable_rho(n, alpha)
    rep.add("enough units for a correlation", n >= 10,
            f"n={n} (need >=10 before a rank correlation is worth quoting)")
    rep.add("test is powered for this effect", np.isfinite(dr) and abs(r) >= dr,
            f"|rho|={abs(r):.3f} vs smallest detectable {dr:.3f} at n={n}; "
            f"a null here is NOT evidence of absence")
    return rep


def claim(text: str, report: Report) -> str:
    """Render a claim with its gates. A failed gate turns it into UNSUPPORTED, with reasons."""
    if report.ok:
        return f"SUPPORTED: {text}\n{report}"
    reasons = "; ".join(g.detail for g in report.failures())
    return f"UNSUPPORTED: {text}\n  blocked by: {reasons}\n{report}"
