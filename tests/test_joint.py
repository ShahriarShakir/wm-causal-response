"""Tests for the joint measurement. Each one encodes a failure mode that produces a plausible-looking
wrong number, so that the measurement cannot return it.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from ccaus.joint import (Estimate, Guard, aggregate, assert_null, bootstrap_median,
                         measure_pair, verdict)

RNG = np.random.default_rng(0)
D, T, C, K = 64, 20, 5, 3
DELAYS = [1, 3, 6, 12]


def _pair(model_tracks: bool, noise: float = 0.01):
    """Build one (A, B) pair. `model_tracks` decides whether the model follows the cause."""
    context = RNG.normal(0, 1, (C, D))
    base = RNG.normal(0, 1, (T, D))
    effect = np.zeros((T, D))
    effect[K:] = np.linspace(0, 1, T - K)[:, None] * RNG.normal(0, 1, (1, D))
    true_a, true_b = base, base + effect
    pred_a = base + RNG.normal(0, noise, (T, D))
    # identical BEFORE the intervention in both cases -- a cause cannot act backwards
    pred_b = pred_a.copy()
    if model_tracks:
        pred_b[K:] += effect[K:]
    else:
        # forgetful: correct immediately, then the trace decays -- the observed shape
        decay = np.exp(-np.arange(T - K) / 1.2)[:, None]
        pred_b[K:] += effect[K:] * decay
    return pred_a, pred_b, true_a, true_b, context


def test_separation_is_detected_when_it_exists():
    """A model that ignores the cause must show flat consistency and collapsed response."""
    pairs = [measure_pair(*_pair(model_tracks=False), K, DELAYS) for _ in range(30)]
    agg = aggregate(pairs, DELAYS)
    v = verdict(agg, DELAYS)
    print("  blind model ->", v["verdict"], "| consistency", [f"{c:.3f}" for c in v["consistency"]],
          "| causal", [f"{m:.4f}" for m in v["causal_magnitude"]])
    assert v["verdict"] == "SEPARATION", v


def test_causally_blind_model_gets_its_own_label():
    """Zero response at EVERY delay is not "no separation" -- it is a stronger, simpler claim.

    The relative rule ``mn < 0.3 * m0`` degenerates to ``0 < 0`` here, so without a distinct
    label a wholly causally-blind model would be reported as having failed the test it most
    emphatically passes.
    """
    def blind_pair():
        pa, pb, ta, tb, ctx = _pair(model_tracks=False)
        pb = pa.copy()                      # literally no response, at any delay
        return pa, pb, ta, tb, ctx
    pairs = [measure_pair(*blind_pair(), K, DELAYS) for _ in range(20)]
    v = verdict(aggregate(pairs, DELAYS), DELAYS)
    print("  fully blind model ->", v["verdict"])
    assert v["verdict"] == "CAUSALLY BLIND THROUGHOUT", v


def test_no_separation_when_the_model_tracks_the_cause():
    """The complement. Without this, 'SEPARATION' could be the only reachable answer."""
    pairs = [measure_pair(*_pair(model_tracks=True), K, DELAYS) for _ in range(30)]
    v = verdict(aggregate(pairs, DELAYS), DELAYS)
    print("  tracking model ->", v["verdict"], "| causal",
          [f"{m:.4f}" for m in v["causal_magnitude"]])
    assert v["verdict"] == "no separation", v


def test_backwards_causation_is_rejected():
    """Futures differing BEFORE the intervention mean a leaking harness, not a result.

    An unpaired stochastic sampler produced exactly this on the RSSM arm (15/255 under a NULL
    intervention) and it read as causality.
    """
    pa, pb, ta, tb, ctx = _pair(model_tracks=False)
    pb[0] += 1.0                                  # difference before the intervention
    try:
        measure_pair(pa, pb, ta, tb, ctx, K, DELAYS)
    except AssertionError as e:
        print("  correctly rejected:", str(e)[:70])
        return
    raise RuntimeError("leaking harness was NOT rejected")


def test_guard_is_calibrated_from_data_not_from_a_formula():
    """The guard must key off the observed divergence.

    A threshold derived from an assumed scaling law (0.25*std*sqrt(D)) evaluated to ~13.9 while
    Pong's true divergence is ~3.0, so it rejected 100% of cells and the sweep measured nothing.
    """
    g = Guard().calibrate([3.0, 3.2, 2.8, 3.1])
    assert 0.6 < g.floor < 0.9, g.floor            # a quarter of the median, ~0.76
    assert g.keep(3.0) and not g.keep(0.1)
    assert g.n_kept == 1 and g.n_dropped == 1      # exclusions are counted, not hidden
    print(f"  floor {g.floor:.3f}, kept {g.n_kept}, dropped {g.n_dropped}")


def test_median_resists_an_exploding_ratio():
    """One near-zero denominator must not move the reported number.

    Observed for real: single cells of 6.3, 12.3 and 19.9 dragged means around while the typical
    value was ~0.3.
    """
    typical = [0.30, 0.31, 0.29, 0.32, 0.30]
    med = bootstrap_median(typical + [19.9]).median
    mean = float(np.mean(typical + [19.9]))
    print(f"  median {med:.3f} vs mean {mean:.3f} with one exploding cell")
    assert abs(med - 0.30) < 0.02 and mean > 3.0


def test_consistency_is_not_guarded():
    """Consistency must be aggregated over ALL episodes.

    It does not divide by the true change, so a small denominator cannot distort it. Guarding it
    would compare the two curves on different subsets of episodes and destroy the comparison
    the paper rests on.
    """
    pairs = [measure_pair(*_pair(model_tracks=False), K, DELAYS) for _ in range(20)]
    agg = aggregate(pairs, DELAYS, Guard(fraction_of_median=10.0))   # drops nearly everything
    n_cons = agg["rows"][str(DELAYS[0])]["consistency"]["n"]
    n_causal = agg["rows"][str(DELAYS[0])]["causal_magnitude"]["n"]
    print(f"  consistency n={n_cons} (all), causal n={n_causal} (guarded)")
    assert n_cons == 20 and n_causal < 20


def test_null_control_helper():
    """A paired sampler returns exactly 0; an unpaired one does not."""
    paired = lambda acts: np.array([[1.0, 2.0], [3.0, 4.0]])
    assert assert_null(paired, [0, 1]) == 0.0
    unpaired = lambda acts: RNG.normal(0, 1, (2, 2))
    assert assert_null(unpaired, [0, 1]) > 0.0
    print("  paired 0.0, unpaired > 0, helper distinguishes them")


def test_decomposition_separates_actuation_from_consequence():
    """A model that copies the directly-actuated coordinates but cannot propagate the cause
    must score HIGH on actuation and ~0 on consequence.

    This is the control for the confound that decides the paper: a high causal response is
    uninformative if it comes entirely from coordinates the action moved immediately.
    """
    from ccaus.joint import decompose_response
    T, Dd, k = 20, 40, 3
    true_a = np.zeros((T, Dd)); true_b = true_a.copy()
    true_b[k:, :4] += 1.0                      # actuated: immediate and persistent
    for j, t in enumerate(range(k + 3, T)):    # consequence: appears strictly later
        true_b[t, 20:24] += 0.5 * (j + 1)
    pred_a = np.zeros((T, Dd))
    pred_b = pred_a.copy(); pred_b[k:, :4] += 1.0     # copies actuation, ignores consequence
    dec = decompose_response(pred_a, pred_b, true_a, true_b, k, [1, 6, 12])
    late = dec[12]
    print(f"  actuated align {late['actuated_align']:.3f}, "
          f"consequence align {late['consequence_align']:.3f}, "
          f"consequence share {late['consequence_fraction']:.3f}")
    assert late["actuated_align"] > 0.9, late
    assert abs(late["consequence_align"]) < 0.05, late
    assert late["consequence_fraction"] > 0.3, late


def test_decomposition_handles_SPARSE_difference_images():
    """Regression: difference images are sparse, and a quantile threshold silently fails.

    In Pong only the paddle changes, so >99% of the 12,288 pixels are exactly 0 and the 90th
    percentile is 0.0 -- making `d >= 0` true everywhere, swallowing the whole frame into the
    actuated set and leaving the consequence set EMPTY at every delay. The first real run
    reported consequence_fraction = 0.0000 on both games because of this. The original tests
    used dense arrays and could not catch it.
    """
    from ccaus.joint import decompose_response
    T, Dd, k = 20, 4096, 3
    true_a = np.zeros((T, Dd)); true_b = true_a.copy()
    true_b[k:, 100:104] += 1.0                      # actuated: 4 of 4096 pixels
    true_b[k + 4:, 2000:2008] += 0.7                # consequence: appears later, elsewhere
    pred_a = np.zeros((T, Dd))
    pred_b = pred_a.copy(); pred_b[k:, 100:104] += 1.0     # copies actuation only
    dec = decompose_response(pred_a, pred_b, true_a, true_b, k, [1, 6, 12])
    late = dec[12]
    print(f"  sparse: consequence share {late['consequence_fraction']:.3f}, "
          f"actuated align {late['actuated_align']:.3f}, "
          f"consequence align {late['consequence_align']:.3f}")
    assert late["consequence_fraction"] > 0.3, f"consequence set collapsed again: {late}"
    assert late["actuated_align"] > 0.9 and abs(late["consequence_align"]) < 0.05, late


if __name__ == "__main__":
    for fn in (test_separation_is_detected_when_it_exists,
               test_causally_blind_model_gets_its_own_label,
               test_no_separation_when_the_model_tracks_the_cause,
               test_backwards_causation_is_rejected,
               test_guard_is_calibrated_from_data_not_from_a_formula,
               test_median_resists_an_exploding_ratio,
               test_consistency_is_not_guarded,
               test_null_control_helper,
               test_decomposition_separates_actuation_from_consequence,
               test_decomposition_handles_SPARSE_difference_images):
        print(f"{fn.__name__}:")
        fn()
    print("\nALL JOINT TESTS PASSED")
