"""The joint measurement: consistency and causal response from ONE pass over ONE rollout pair.

This is the contribution, so the code is written to make the central claim impossible to fake.

The claim is a SEPARATION: on identical data, a consistency-style score stays flat while the
model's causal response collapses. Decay alone is not a finding -- MBench already reports that
world models degrade over long horizons. So the only thing worth measuring is the gap between
the two curves, and the gap is only meaningful if nothing except the metrics differs between
them.

Two design rules follow, and both are enforced in code rather than left to discipline:

1. ONE PASS, ONE PAIR. Both metrics are computed inside `measure_pair` from the same four
   arrays. There is no code path that computes them from separate rollouts, so a difference in
   seeds, conditioning or sampling noise cannot masquerade as a separation.

2. INVARIANTS ARE ASSERTED, NOT ASSUMED. A cause cannot act backwards: before the intervention
   step, the two futures must be bit-identical. If they are not, the harness is leaking and
   every number downstream is meaningless. `measure_pair` raises rather than returning a
   plausible-looking result.

Failure modes this file exists to prevent, each of which actually occurred earlier in this
project:
  * a metric that returns ~0 everywhere -> we report MAGNITUDE, which is dimension-robust,
    alongside alignment. NOTE (2026-08-28): the earlier belief that alignment necessarily
    collapses in high dimension is FALSIFIED. In the same 12,288 pixel dimensions where an
    RSSM scores 0.0008, DIAMOND scores 0.80. Alignment is near zero when the model is not
    tracking the cause, not because the space is large. Both are reported; neither is
    "secondary".
  * a guard whose threshold came from an assumed scaling law rather than the data
    -> `Guard` calibrates against the observed divergence and REPORTS what it excluded;
  * a mean dragged around by a handful of exploding ratios near a zero denominator
    -> `bootstrap_median` is the reported statistic;
  * stochastic sampling read as a causal response
    -> `assert_null` gives callers a one-line way to prove their sampler is paired.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Sequence

import numpy as np

from ccaus.metrics import consistency, response_magnitude, causal_response


# ----------------------------------------------------------------------------- statistics --

@dataclass(frozen=True)
class Estimate:
    """A median with a bootstrap CI, plus the sample size it rests on."""

    median: float
    lo: float
    hi: float
    n: int

    def as_dict(self) -> dict:
        return asdict(self)


def bootstrap_median(values: Sequence[float], n_boot: int = 2000, seed: int = 0) -> Estimate:
    """Median with a percentile bootstrap 95% CI.

    The median, not the mean: every quantity here is a ratio ``||dp|| / ||dt||`` and a ratio
    explodes as its denominator approaches zero. A mean follows those few cells and reports
    the artifact; the median reports the typical case. `n` is carried alongside because an
    estimate over three episodes and one over thirty must not look alike in a table.
    """
    x = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if x.size == 0:
        return Estimate(float("nan"), float("nan"), float("nan"), 0)
    if x.size < 3:
        return Estimate(float(np.median(x)), float("nan"), float("nan"), int(x.size))
    rng = np.random.default_rng(seed)
    draws = np.median(rng.choice(x, size=(n_boot, x.size), replace=True), axis=1)
    return Estimate(float(np.median(x)),
                    float(np.percentile(draws, 2.5)),
                    float(np.percentile(draws, 97.5)),
                    int(x.size))


def prediction_psnr(pred: np.ndarray, true: np.ndarray, data_range: float = 2.0,
                    cap: float = 100.0) -> float:
    """PSNR of a predicted frame against the ACTUAL frame, in dB. `data_range` is 2.0 for the
    [-1,1] convention both harnesses use. An exact prediction would be infinite; capped so it
    can be bootstrapped."""
    mse = float(np.mean((np.asarray(pred, float) - np.asarray(true, float)) ** 2))
    if mse <= 0.0:
        return cap
    return float(min(cap, 10.0 * np.log10(data_range * data_range / mse)))


# ---------------------------------------------------------------------------------- guard --

@dataclass
class Guard:
    """Excludes cells where the TRUE change is too small for a ratio to mean anything.

    Calibrated from the observed divergences, never from an assumed scaling law. An earlier
    version required ``dt > 0.25 * std * sqrt(D)``, which for 64x64x3 Pong frames evaluates to
    ~13.9 while the real divergence is ~3.0 (only the ball and paddles differ). Every cell was
    rejected and the sweep silently measured nothing.

    The count of exclusions is part of the result. A guard that quietly drops most of the data
    is indistinguishable, in a table, from one that drops none.
    """

    fraction_of_median: float = 0.25
    floor: float = field(default=float("nan"), init=False)
    n_kept: int = field(default=0, init=False)
    n_dropped: int = field(default=0, init=False)

    def calibrate(self, true_divergences: Sequence[float]) -> "Guard":
        finite = [d for d in true_divergences if np.isfinite(d)]
        self.floor = (self.fraction_of_median * float(np.median(finite))) if finite else 0.0
        return self

    def keep(self, true_divergence: float) -> bool:
        if not np.isfinite(self.floor):
            raise RuntimeError("Guard.calibrate() must be called before Guard.keep()")
        ok = true_divergence >= self.floor
        self.n_kept += int(ok)
        self.n_dropped += int(not ok)
        return ok

    def as_dict(self) -> dict:
        return {"floor": self.floor, "n_kept": self.n_kept, "n_dropped": self.n_dropped,
                "fraction_of_median": self.fraction_of_median}


# ------------------------------------------------------------------------ the measurement --

@dataclass(frozen=True)
class PairMeasurement:
    """Everything measurable from one (A, B) rollout pair, indexed by delay."""

    delays: List[int]
    consistency: Dict[int, float]
    causal_magnitude: Dict[int, float]
    causal_alignment: Dict[int, float]
    true_divergence: Dict[int, float]
    prediction_psnr: Dict[int, float] = None
    model_divergence: Dict[int, float] = None
    cross_psnr: Dict[int, float] = None
    intervened_psnr: Dict[int, float] = None   # PSNR of the INTERVENED arm against its own truth, PSNR(p', t')


def measure_pair(
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    true_a: np.ndarray,
    true_b: np.ndarray,
    context: np.ndarray,
    intervention_step: int,
    delays: Sequence[int],
    *,
    check_causality_direction: bool = True,
    atol: float = 0.0,
) -> PairMeasurement:
    """Both metrics, one pass, one pair of rollouts.

    All four arrays are ``[T, D]`` over the SAME horizon: `pred_*` are the model's imagined
    futures under action sequences A and B, `true_*` the corresponding ground truth.
    `context` is ``[C, D]``, the observed past both arms share.

    `consistency` never sees the intervention -- it asks only whether the rollout continues the
    observed past, which is what an MBench-style score measures. `causal_magnitude` asks whether
    the prediction moved by the right AMOUNT when the cause flipped; `causal_alignment` whether
    it moved in the right DIRECTION. Both are reported because they fail differently: in high
    dimension an unrelated change of any size has alignment near zero, so alignment alone
    cannot tell a silent model from a misdirected one.

    Raises if the two futures differ BEFORE the intervention. A cause cannot act backwards, so
    any such difference means the harness is leaking (unpaired sampling noise, a stray RNG
    draw, dropout left enabled) and no number computed here would mean anything.
    """
    for name, arr in (("pred_a", pred_a), ("pred_b", pred_b),
                      ("true_a", true_a), ("true_b", true_b)):
        if arr.ndim != 2:
            raise ValueError(f"{name} must be [T, D], got shape {arr.shape}")
    if not (pred_a.shape == pred_b.shape and true_a.shape == true_b.shape):
        raise ValueError("A and B arrays must have matching shapes")
    if pred_a.shape[0] != true_a.shape[0]:
        raise ValueError("prediction and ground truth must span the same horizon")

    if check_causality_direction and intervention_step > 0:
        pre = np.abs(pred_a[:intervention_step] - pred_b[:intervention_step]).max()
        if pre > atol:
            raise AssertionError(
                f"futures differ by {pre:.3g} BEFORE the intervention at step "
                f"{intervention_step}: a cause cannot act backwards, so the harness is leaking. "
                "Check that both arms share seeds/noise (common random numbers) and that the "
                "model is in eval mode."
            )

    horizon = pred_a.shape[0]
    cons: Dict[int, float] = {}
    mag: Dict[int, float] = {}
    align: Dict[int, float] = {}
    tdiv: Dict[int, float] = {}
    psnr: Dict[int, float] = {}
    mdiv: Dict[int, float] = {}
    xpsnr: Dict[int, float] = {}
    ipsnr: Dict[int, float] = {}
    for d in delays:
        i = intervention_step + d
        if i >= horizon:
            continue
        # consistency: same rollouts, blind to the intervention
        cons[d] = 0.5 * (consistency(pred_a[: i + 1], context)
                         + consistency(pred_b[: i + 1], context))
        mag[d] = response_magnitude(pred_a[i], pred_b[i], true_a[i], true_b[i])
        align[d] = causal_response(pred_a[i], pred_b[i], true_a[i], true_b[i])
        tdiv[d] = float(np.linalg.norm(true_b[i] - true_a[i]))
        # PREDICTION ACCURACY, the standard "how good is this world model" number, from arm A
        # only so it never sees the intervention. It exists to answer two objections at once:
        # "your consistency score is home-made" (this one is not) and "your models are just
        # weak" (a game predicted WELL yet causally blind refutes that). PSNR on the [-1,1]
        # pixel range, so 0 dB error-free would be inf; capped at 100 for finite aggregation.
        psnr[d] = prediction_psnr(pred_a[i], true_a[i])
        # ABSOLUTE model divergence ||pred_b - pred_a||. Unlike the magnitude RATIO it is defined
        # when the true divergence is zero, which is exactly the case the hallucination probe
        # needs: does the model move when the emulator has not (yet) moved?
        mdiv[d] = float(np.linalg.norm(pred_b[i] - pred_a[i]))
        # GENIE-STYLE CONTROLLABILITY (arXiv:2402.15391): dPSNR = PSNR(true_a, pred_a) - PSNR(true_a, pred_b).
        # The field's closest existing metric. It asks whether the action MATTERS to the prediction,
        # not whether the response is CORRECT: a model that responds in the wrong direction scores a
        # large dPSNR and zero alignment. Recorded so the two can be contrasted on the same rollouts.
        xpsnr[d] = prediction_psnr(pred_b[i], true_a[i])
        ipsnr[d] = prediction_psnr(pred_b[i], true_b[i])
    return PairMeasurement(list(cons.keys()), cons, mag, align, tdiv, psnr, mdiv, xpsnr, ipsnr)


# ------------------------------------------------------------------- response decomposition --

def _masked_response(dp: np.ndarray, dt: np.ndarray, mask: np.ndarray) -> tuple:
    """Magnitude and alignment restricted to a subset of coordinates."""
    if mask.sum() == 0:
        return float("nan"), float("nan")
    p, t = dp[mask], dt[mask]
    nt = float(np.linalg.norm(t))
    if nt == 0.0:
        return float("nan"), float("nan")
    return float(np.linalg.norm(p)) / nt, float(np.dot(p, t)) / (nt * nt)


def _support(v: np.ndarray, rel_tol: float = 0.01) -> np.ndarray:
    """Coordinates that actually changed: |v| above `rel_tol` of the largest change.

    A QUANTILE cannot be used here and the reason is worth recording. Difference images are
    SPARSE -- in Pong only the paddle moves, so >99% of the 12,288 pixels are exactly zero and
    `np.quantile(d, 0.90)` evaluates to 0.0. The test `d >= 0` is then true everywhere, the
    actuated set swallows the whole frame, and the consequence set is empty at every delay.
    That is exactly what happened on the first run (consequence_fraction = 0.0000 across both
    games), and the synthetic tests missed it because they used DENSE difference arrays.
    """
    m = float(np.abs(v).max())
    return np.abs(v) > rel_tol * m if m > 0 else np.zeros(v.shape, dtype=bool)


def decompose_response(pred_a, pred_b, true_a, true_b, intervention_step, delays,
                       rel_tol: float = 0.01):
    """Split the causal response into IMMEDIATE ACTUATION and DOWNSTREAM CONSEQUENCE.

    This exists to answer the confound that decides the paper. A model can score a high causal
    response for a shallow reason: if flipping the action moves some pixels IMMEDIATELY and
    those pixels then persist, reproducing them is closer to copying than to modelling a cause.
    The interesting quantity is whether the model tracks what the intervention causes LATER,
    in coordinates the action did not touch directly.

    The split is taken from the GROUND TRUTH, never from the model, so it cannot be tuned to
    flatter a prediction:

      ACTUATED    -- coordinates already changed at the FIRST measured delay. The direct,
                     immediate effect of the action.
      CONSEQUENCE -- coordinates that change at delay d but were NOT in the actuated set. The
                     part that required propagating the cause through the system's dynamics.

    It is deliberately substrate-agnostic (no game knowledge, no hand-drawn regions), so the
    same function applies to Atari frames and to DMControl pixels and the two arms stay
    comparable.

    Returns {delay: {"actuated_mag", "actuated_align", "consequence_mag", "consequence_align",
                     "consequence_fraction"}}. `consequence_fraction` is the share of the true
    divergence living outside the actuated set -- if it is ~0 there is nothing downstream to
    track and the other numbers at that delay carry no information.
    """
    i0 = intervention_step + min(delays)
    actuated = _support(true_b[i0] - true_a[i0], rel_tol)
    if not actuated.any():
        return {}

    out = {}
    for d in delays:
        i = intervention_step + d
        if i >= pred_a.shape[0]:
            continue
        dt, dp = true_b[i] - true_a[i], pred_b[i] - pred_a[i]
        live = _support(dt, rel_tol)
        consequence = live & ~actuated
        total = float(np.linalg.norm(dt))
        am, aa = _masked_response(dp, dt, actuated)
        cm, ca = _masked_response(dp, dt, consequence)
        out[d] = {
            "actuated_mag": am, "actuated_align": aa,
            "consequence_mag": cm, "consequence_align": ca,
            "consequence_fraction": (float(np.linalg.norm(dt[consequence])) / total
                                     if total > 0 and consequence.sum() else 0.0),
        }
    return out


# --------------------------------------------------------------------------- aggregation --

def aggregate(pairs: Sequence[PairMeasurement], delays: Sequence[int],
              guard: Guard | None = None) -> dict:
    """Aggregate per-episode measurements into the reported table.

    The guard is calibrated on the pooled true divergences across ALL delays, so the threshold
    does not drift between columns and cannot be tuned per cell to flatter a curve.
    """
    guard = guard or Guard()
    guard.calibrate([p.true_divergence[d] for p in pairs for d in delays
                     if d in p.true_divergence])

    rows = {}
    for d in delays:
        keep = [p for p in pairs if d in p.true_divergence and guard.keep(p.true_divergence[d])]
        rows[str(d)] = {
            # consistency is NOT guarded: it does not divide by the true change, so a small
            # denominator cannot distort it, and dropping those episodes would compare the two
            # curves on different subsets -- destroying the very comparison this measures.
            "consistency": bootstrap_median([p.consistency[d] for p in pairs
                                             if d in p.consistency]).as_dict(),
            "causal_magnitude": bootstrap_median([p.causal_magnitude[d] for p in keep]).as_dict(),
            "causal_alignment": bootstrap_median([p.causal_alignment[d] for p in keep]).as_dict(),
            "true_divergence": bootstrap_median([p.true_divergence[d] for p in pairs
                                                 if d in p.true_divergence]).as_dict(),
            "prediction_psnr": bootstrap_median([p.prediction_psnr[d] for p in pairs
                                                 if p.prediction_psnr and d in p.prediction_psnr]).as_dict(),
            "model_divergence": bootstrap_median([p.model_divergence[d] for p in pairs
                                                  if p.model_divergence and d in p.model_divergence]).as_dict(),
            "intervened_psnr": bootstrap_median([p.intervened_psnr[d] for p in pairs if p.intervened_psnr and d in p.intervened_psnr]).as_dict(),
            "delta_psnr": bootstrap_median([p.prediction_psnr[d] - p.cross_psnr[d] for p in pairs
                                            if p.cross_psnr and d in p.cross_psnr]).as_dict(),
        }
    return {"rows": rows, "guard": guard.as_dict(), "n_pairs": len(pairs)}


def verdict(agg: dict, delays: Sequence[int], consistency_retained: float = 0.8,
            causal_lost: float = 0.3, causal_floor: float = 1e-6) -> dict:
    """Does the separation hold? Stated as a rule, applied identically to every task.

    Requires BOTH: consistency retains most of its short-delay value, AND causal response falls
    to a small fraction of its own. Either alone is uninformative -- if both decay together,
    then a consistency score already detects the failure and there is nothing new to report.

    Three outcomes, not two. If the response is ~0 at the FIRST delay as well, the model never
    tracked the cause at any horizon, and the relative test ``mn < 0.3 * m0`` degenerates to
    ``0 < 0``. That is a different claim from "the trace decays", it is a stronger and simpler
    one, and collapsing it into "no separation" would report a model that is causally blind
    throughout as though it had passed. It gets its own label.
    """
    usable = [d for d in delays
              if str(d) in agg["rows"] and agg["rows"][str(d)]["causal_magnitude"]["n"] >= 5]
    if len(usable) < 2:
        return {"verdict": "unmeasurable", "reason": "fewer than two delays survived the guard"}
    d0, dn = usable[0], usable[-1]
    c0 = agg["rows"][str(d0)]["consistency"]["median"]
    cn = agg["rows"][str(dn)]["consistency"]["median"]
    m0 = agg["rows"][str(d0)]["causal_magnitude"]["median"]
    mn = agg["rows"][str(dn)]["causal_magnitude"]["median"]
    keeps = np.isfinite(cn) and np.isfinite(c0) and cn > consistency_retained * c0
    never_tracked = np.isfinite(m0) and abs(m0) < causal_floor
    loses = (np.isfinite(mn) and np.isfinite(m0) and not never_tracked
             and mn < causal_lost * m0)
    if never_tracked:
        label = "CAUSALLY BLIND THROUGHOUT" if keeps else "no separation"
    else:
        label = "SEPARATION" if (keeps and loses) else "no separation"
    return {
        "verdict": label,
        "never_tracked": bool(never_tracked),
        "first_delay": d0, "last_delay": dn,
        "consistency": [c0, cn], "causal_magnitude": [m0, mn],
        "consistency_retained": bool(keeps), "causal_lost": bool(loses),
    }


def assert_null(rollout: Callable[[Sequence[int]], np.ndarray], actions: Sequence[int]) -> float:
    """Prove the caller's rollout is PAIRED before any causal number is believed.

    Runs the identical action sequence twice and returns the maximum difference, which must be
    exactly zero. A stochastic model that is not seeded identically per arm will return a
    non-zero "response" to an intervention that never happened (an unpaired stochastic model
    can show a 15/255 response to a null intervention, which reads as causality).
    """
    a = np.asarray(rollout(actions), dtype=float)
    b = np.asarray(rollout(actions), dtype=float)
    return float(np.abs(a - b).max())
