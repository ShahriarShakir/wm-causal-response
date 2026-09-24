"""Two scores on the same clips: self-CONSISTENCY vs CAUSAL RESPONSE.

The claim under test: they come apart. A predictor can be smoothly self-consistent with the
past it saw while tracking no cause at all, and a consistency-style benchmark cannot tell it
from one that genuinely tracked the cause.

Definitions, deliberately simple so nothing hides in the metric:

* consistency(pred, past)  -- how well the predicted future continues the OBSERVED past.
  Operationalised as 1 - normalised discontinuity at the seam plus smoothness of the roll-out.
  This is the family MBench-style scores live in: it never looks at the cause.

* causal_response(pred_A, pred_B, truth_A, truth_B) -- when the early CAUSE flips A->B, does
  the prediction move in the RIGHT DIRECTION by the RIGHT AMOUNT?  Measured as the alignment
  between the predicted change and the true change:
      cr = <pred_B - pred_A, truth_B - truth_A> / (||truth_B - truth_A||^2)
  1.0 = tracks the cause exactly; 0.0 = ignores it; negative = moves the wrong way.
  Note it is an ALIGNMENT, not an error: a predictor that is wrong about the absolute future
  but right about the DIFFERENCE still scores 1, which is the point.
"""
from __future__ import annotations
import numpy as np


def _rms(x: np.ndarray) -> float:
    """Root-MEAN-square, not a raw norm.

    The first version used np.linalg.norm, which grows as sqrt(D). On 64x64x3 = 12,288
    pixels the seam term came out ~55 for ANY prediction, so consistency = exp(-55) ~ 1e-22
    for every rollout -- a metric that returned the same number regardless of the input, and
    which then made the decision gate vacuous. The D=4 synthetic gate never exposed it.
    Dividing by sqrt(D) makes the score comparable across dimensionalities.
    """
    x = np.asarray(x, float).ravel()
    return float(np.sqrt(np.mean(x * x)))


def consistency(pred: np.ndarray, past: np.ndarray) -> float:
    """Self-consistency of a predicted roll-out with the observed past. Never sees the cause."""
    pred = np.asarray(pred, float); past = np.asarray(past, float)
    scale = float(np.std(past)) + 1e-9
    seam = _rms(pred[0] - past[-1]) / scale                    # discontinuity at the join
    if len(pred) > 2:
        rough = float(np.mean([_rms(a) for a in np.diff(pred, n=2, axis=0)])) / scale
    else:
        rough = 0.0
    return float(np.exp(-(seam + rough)))


def causal_response(pred_a, pred_b, truth_a, truth_b) -> float:
    """Alignment of the predicted change with the true change when the cause flips."""
    dp = np.asarray(pred_b, float) - np.asarray(pred_a, float)
    dt = np.asarray(truth_b, float) - np.asarray(truth_a, float)
    denom = float(np.sum(dt * dt))
    if denom < 1e-12:
        return float("nan")            # the intervention did nothing; the item is unusable
    return float(np.sum(dp * dt) / denom)


def response_magnitude(pred_a, pred_b, truth_a, truth_b) -> float:
    """||predicted change|| / ||true change||. Dimension-robust companion to the alignment.

    Added after `causal_response` (an alignment) failed its positive control in pixel space.
    In 12,288 dimensions two vectors that are not deliberately aligned are nearly orthogonal,
    so the alignment collapses to ~0 even when the predicted change has 30% of the true
    magnitude. Reporting magnitude AND alignment separates "the model did not react" from
    "the model reacted in a different direction" -- which are completely different failures
    and were indistinguishable before.
    """
    import numpy as _np
    dp = _np.asarray(pred_b, float) - _np.asarray(pred_a, float)
    dt = _np.asarray(truth_b, float) - _np.asarray(truth_a, float)
    nt = float(_np.sqrt(_np.sum(dt * dt)))
    if nt < 1e-12:
        return float("nan")
    return float(_np.sqrt(_np.sum(dp * dp)) / nt)
