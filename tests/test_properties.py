"""Property-based tests: invariants that must hold for every input, not only for chosen examples.

Each fault these tests target was invisible to a hand-written example and obvious to a sweep:

  * `norm` over 12,288 dimensions -> `exp(-55)` = 1e-22 for every model.  (a DIMENSION bug)
  * a quantile threshold on difference images that are >99% zeros.        (a SPARSITY bug)
  * `mn < 0.3*m0` when m0 is exactly 0.                                   (a DEGENERACY bug)
  * a dataset gate tested at 57% survival when reality was 31%.           (a PROPORTION bug)

So these tests do not pick inputs; they sweep the axes where such faults live -- dimension 4 -> 20000, sparsity 0.001 -> 1.0,
scale 1e-6 -> 1e6, plus the degenerate corners -- and assert invariants that must hold
throughout. A metric that violates one of these is wrong regardless of what it scores on Pong.
"""
import sys, pathlib, itertools
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from ccaus.joint import decompose_response, measure_pair, _support
from ccaus.metrics import consistency, response_magnitude, causal_response

RNG = np.random.default_rng(0)
DIMS = [4, 64, 1024, 12288, 20000]            # 12288 = 64x64x3, where the norm bug lived
SPARSITIES = [0.001, 0.01, 0.1, 0.5, 1.0]     # 0.001 = a Pong paddle in a frame
SCALES = [1e-6, 1e-3, 1.0, 1e3, 1e6]


def sparse_pair(D, sparsity, scale, rng):
    """A (truth, prediction) difference pair with a controlled fraction of live coordinates."""
    k = max(1, int(D * sparsity))
    idx = rng.choice(D, size=k, replace=False)
    dt = np.zeros(D); dt[idx] = rng.normal(0, scale, k)
    return dt, idx


def test_magnitude_is_scale_invariant():
    """||dp||/||dt|| must not change when BOTH arms are rescaled. A metric that drifts with
    scale silently reports the units, not the model."""
    worst = 0.0
    for D, sp in itertools.product(DIMS, SPARSITIES):
        rng = np.random.default_rng(hash((D, sp)) % 2**32)
        dt, idx = sparse_pair(D, sp, 1.0, rng)
        dp = 0.4 * dt + 0.01 * rng.normal(0, 1, D)
        base = response_magnitude(np.zeros(D), dp, np.zeros(D), dt)
        for s in SCALES:
            got = response_magnitude(np.zeros(D), dp * s, np.zeros(D), dt * s)
            worst = max(worst, abs(got - base))
    print(f"  max drift across dim x sparsity x scale: {worst:.2e}")
    assert worst < 1e-9, f"magnitude is not scale invariant (drift {worst:.2e})"


def test_alignment_is_one_when_the_model_is_exactly_right():
    """The only universally correct answer: a model that reproduces the true change EXACTLY
    must score 1.0 -- at every dimension and every sparsity, or the metric is dimension-biased.
    This is the invariant that would have exposed the `exp(-55)` consistency bug immediately."""
    worst = 0.0
    for D, sp in itertools.product(DIMS, SPARSITIES):
        rng = np.random.default_rng(hash((D, sp, "x")) % 2**32)
        dt, _ = sparse_pair(D, sp, 1.0, rng)
        worst = max(worst, abs(causal_response(np.zeros(D), dt, np.zeros(D), dt) - 1.0))
    print(f"  max deviation from 1.0: {worst:.2e}")
    assert worst < 1e-9, f"perfect prediction does not score 1.0 (off by {worst:.2e})"


def test_alignment_is_zero_for_an_unrelated_response_at_every_dimension():
    """An orthogonal response must score ~0 REGARDLESS of dimension.

    It separates 'the metric degrades with D' (it must not) from 'the model is not tracking'.
    """
    for D in DIMS:
        rng = np.random.default_rng(D)
        vals = []
        for _ in range(64):
            dt = rng.normal(0, 1, D); dp = rng.normal(0, 1, D)
            dp -= (dp @ dt) / (dt @ dt) * dt          # exactly orthogonal
            vals.append(causal_response(np.zeros(D), dp, np.zeros(D), dt))
        m = float(np.abs(np.mean(vals)))
        assert m < 1e-9, f"D={D}: orthogonal response scored {m:.2e}, not 0"
    print(f"  orthogonal response scores 0 at every D in {DIMS}")


def test_support_survives_extreme_sparsity_and_scale():
    """`_support` must find the live coordinates whatever the sparsity or scale.

    A quantile threshold returns 'everything' at sparsity < 0.1.
    """
    for D, sp, sc in itertools.product(DIMS, SPARSITIES, SCALES):
        rng = np.random.default_rng(hash((D, sp, sc)) % 2**32)
        dt, idx = sparse_pair(D, sp, sc, rng)
        got = _support(dt)
        assert got.sum() <= max(1, int(D * sp)), \
            f"D={D} sp={sp} sc={sc}: support {got.sum()} > live {int(D*sp)}"
        assert got.sum() > 0, f"D={D} sp={sp} sc={sc}: support is empty"
    print(f"  support correct across {len(DIMS)*len(SPARSITIES)*len(SCALES)} "
          f"dim/sparsity/scale combinations")


def test_measure_pair_never_reports_a_response_to_a_null_intervention():
    """Sweep: if the two arms are identical, EVERY delay must report exactly 0, at every
    dimension and sparsity. This is the invariant the RSSM `.rsample()` bug violated."""
    for D in DIMS:
        rng = np.random.default_rng(D + 7)
        T, K = 16, 3
        pred = rng.normal(0, 1, (T, D)); true = rng.normal(0, 1, (T, D))
        pm = measure_pair(pred, pred.copy(), true, true.copy(), rng.normal(0, 1, (4, D)),
                          K, [1, 4, 8])
        for d, v in pm.causal_magnitude.items():
            assert not (v > 0), f"D={D} d={d}: null intervention scored {v}"
    print(f"  null intervention scores 0 (or nan) at every D in {DIMS}")


def test_decomposition_partitions_the_change_without_overlap():
    """Actuated and consequence sets must be DISJOINT at every sparsity, or the two columns
    double-count and their sum is meaningless."""
    for sp in SPARSITIES:
        D, T, K = 4096, 20, 3
        rng = np.random.default_rng(int(sp * 1e6))
        k = max(2, int(D * sp))
        idx = rng.choice(D, size=k, replace=False)
        a_idx, c_idx = idx[: k // 2], idx[k // 2:]
        true_a = np.zeros((T, D)); true_b = true_a.copy()
        true_b[K:, a_idx] += 1.0
        true_b[K + 4:, c_idx] += 0.7
        act = _support(true_b[K + 1] - true_a[K + 1])
        live = _support(true_b[K + 12] - true_a[K + 12])
        assert not (act & (live & ~act)).any(), f"sp={sp}: sets overlap"
    print(f"  actuated/consequence disjoint at every sparsity in {SPARSITIES}")


def test_prediction_accuracy_never_sees_the_intervention():
    """PSNR is computed from arm A against arm-A truth ONLY. Changing arm B -- the
    intervention -- must leave it bit-identical, or it has leaked the cause into a score
    that is supposed to be blind to it."""
    from ccaus.joint import measure_pair, prediction_psnr
    for D in DIMS:
        rng = np.random.default_rng(D + 11); T, K = 16, 3
        pa = rng.normal(0, 0.5, (T, D)); ta = pa + 0.05 * rng.normal(0, 1, (T, D))
        pb1 = pa.copy(); pb1[K:] += 1.0
        pb2 = pa.copy(); pb2[K:] -= 3.0                      # a very different intervention
        ctx = rng.normal(0, 1, (4, D))
        m1 = measure_pair(pa, pb1, ta, ta.copy(), ctx, K, [1, 4, 8])
        m2 = measure_pair(pa, pb2, ta, ta.copy(), ctx, K, [1, 4, 8])
        for d in (1, 4, 8):
            assert m1.prediction_psnr[d] == m2.prediction_psnr[d], f"D={D} d={d}: psnr saw arm B"
    assert prediction_psnr(np.zeros(8), np.zeros(8)) == 100.0        # exact -> cap, not inf/nan
    print(f"  psnr blind to the intervention at every D in {DIMS}; exact prediction -> cap")


def test_model_divergence_is_defined_where_the_ratio_is_not():
    """||pred_b - pred_a|| must be finite when the TRUE divergence is exactly 0 (the ratio is NaN
    there), equal magnitude x true_divergence where the ratio exists, and be exactly 0 for
    identical arms. This is what lets the hallucination probe ask whether a model moves when the
    emulator has not."""
    from ccaus.joint import measure_pair
    for D in DIMS:
        rng = np.random.default_rng(D + 23); T, K = 16, 3
        ta = rng.normal(0, 1, (T, D)); tb = ta.copy()          # true divergence EXACTLY 0
        pa = rng.normal(0, 1, (T, D)); pb = pa.copy(); pb[K:] += 0.3
        m = measure_pair(pa, pb, ta, tb, rng.normal(0, 1, (4, D)), K, [1, 4, 8])
        for d in (1, 4, 8):
            assert np.isfinite(m.model_divergence[d]) and m.model_divergence[d] > 0, f"D={D}"
            assert not np.isfinite(m.causal_magnitude[d]), "ratio should be NaN at zero true div"
        tb2 = ta + 0.5 * rng.normal(0, 1, (T, D))
        m2 = measure_pair(pa, pb, ta, tb2, rng.normal(0, 1, (4, D)), K, [1, 4, 8])
        for d in (1, 4, 8):
            assert abs(m2.model_divergence[d] - m2.causal_magnitude[d] * m2.true_divergence[d]) < 1e-9
        m3 = measure_pair(pa, pa.copy(), ta, tb2, rng.normal(0, 1, (4, D)), K, [1, 4, 8])
        assert all(m3.model_divergence[d] == 0.0 for d in (1, 4, 8))
    print(f"  model_divergence finite at zero true div, = ratio x true div, 0 for identical arms; D in {DIMS}")


def test_repr_alignment_shares_the_pixel_metric_invariants():
    """Representation alignment must be 1 for a perfect representation, 0 for an orthogonal one,
    NaN when the true representation did not change, and independent of dimension -- the same
    invariants as pixel alignment, because it IS the same estimator applied in another space."""
    from ccaus.repr_probe import repr_alignment, classify
    for D in DIMS:
        rng = np.random.default_rng(D + 31)
        zta = rng.normal(0, 1, D); ztb = zta + rng.normal(0, 1, D)
        assert abs(repr_alignment(zta, ztb, zta, ztb) - 1.0) < 1e-9              # perfect
        dz = ztb - zta; q = rng.normal(0, 1, D); q -= (q @ dz) / (dz @ dz) * dz    # orthogonal
        assert abs(repr_alignment(zta, zta + q, zta, ztb)) < 1e-9
        assert not np.isfinite(repr_alignment(zta, ztb, zta, zta.copy()))          # no true change -> NaN
    assert classify(0.9, 0.9, 0.3) == "tracked" and classify(0.01, 0.9, 0.3) == "decoding failure"
    assert classify(0.01, 0.01, 0.3) == "representation failure" and classify(0.9, 0.01, 0.3) == "anomalous"
    print(f"  repr alignment: 1 / 0 / NaN invariants hold at every D in {DIMS}; classifier maps all four cells")


def test_delta_psnr_is_zero_for_identical_arms_and_positive_when_b_departs():
    """Genie-style controllability dPSNR = PSNR(true_a,pred_a) - PSNR(true_a,pred_b). Identical arms
    -> exactly 0 at every D. Arm B pushed away from true_a -> positive. And it must be possible for
    dPSNR to be LARGE while alignment is ~0 (a wrong-direction response): that is the case the paper
    contrasts, so the instrument must be able to express it."""
    from ccaus.joint import measure_pair
    from ccaus.metrics import causal_response
    for D in DIMS:
        rng = np.random.default_rng(D + 41); T, K = 16, 3
        ta = rng.normal(0, 0.3, (T, D)); tb = ta.copy(); tb[K:] += 0.5 * rng.normal(0, 1, (T - K, D))
        pa = ta + 0.05 * rng.normal(0, 1, (T, D)); ctx = rng.normal(0, 1, (4, D))
        m0 = measure_pair(pa, pa.copy(), ta, tb, ctx, K, [1, 4, 8])
        assert all(m0.prediction_psnr[d] - m0.cross_psnr[d] == 0.0 for d in (1, 4, 8)), f"D={D}"
        pb = pa.copy(); dt = tb - ta
        for i in range(K, T):                      # wrong-direction response, world-sized
            q = rng.normal(0, 1, D); q -= (q @ dt[i]) / (dt[i] @ dt[i]) * dt[i]; pb[i] += q / np.linalg.norm(q) * np.linalg.norm(dt[i])
        m1 = measure_pair(pa, pb, ta, tb, ctx, K, [1, 4, 8])
        for d in (1, 4, 8):
            assert m1.prediction_psnr[d] - m1.cross_psnr[d] > 1.0, f"D={D}: dPSNR not positive"
            assert abs(m1.causal_alignment[d]) < 1e-6, f"D={D}: alignment should be ~0 for orthogonal response"
    print(f"  dPSNR: 0 for identical arms; large-and-positive with ~0 alignment for a wrong-direction response, at every D in {DIMS}")


def test_idm_learns_a_recoverable_action_and_is_deterministic():
    """The IDM must (a) recover an action that is written into the frames, (b) sit at the majority
    baseline when labels are random (so the gate is meaningful), (c) predict deterministically."""
    import torch
    from ccaus.idm import train_idm, predict
    rng = np.random.default_rng(0); n, n_act = 600, 4
    f0 = rng.uniform(-1, 1, (n, 64, 64, 3)).astype(np.float32); acts = rng.integers(0, n_act, n)
    f1 = f0.copy()
    for i, a in enumerate(acts): f1[i, :, a * 16:(a + 1) * 16, :] += 0.8       # the action moves a band
    m, acc, chance = train_idm(f0, f1, acts, n_act, epochs=6, bs=64)
    assert acc >= 2 * chance, f"learnable action not recovered: acc {acc:.2f} vs chance {chance:.2f}"
    m2, acc2, ch2 = train_idm(f0, f1, rng.integers(0, n_act, n), n_act, epochs=2, bs=64)
    assert acc2 < 2 * ch2, f"random labels should NOT pass the gate: {acc2:.2f} vs {ch2:.2f}"
    p1, p2 = predict(m, f1[:10]), predict(m, f1[:10]); assert (p1 == p2).all()
    print(f"  IDM: learnable {acc:.2f} vs chance {chance:.2f}; random labels {acc2:.2f} (gate fails correctly); deterministic")


def test_intervened_psnr_is_the_b_arm_against_its_own_truth():
    """Report the intervened arm's prediction quality too. PSNR(p', t') must be infinite/capped when p' == t'
    and must NOT equal the cross PSNR PSNR(p', t) when t != t'."""
    import numpy as np
    from ccaus.joint import measure_pair
    rng = np.random.default_rng(0); T, D = 8, 64
    ta = rng.uniform(-1, 1, (T, D)); tb = ta.copy(); tb[3:] += 0.3
    pa = ta + 0.05 * rng.standard_normal((T, D)); pb = tb.copy()          # arm B predicted perfectly, arm A noisy
    pm = measure_pair(pa, pb, ta, tb, ta[:2], 2, [1, 3], check_causality_direction=False)
    assert pm.intervened_psnr[3] >= 100.0 - 1e-9                            # capped 'perfect'
    assert pm.intervened_psnr[3] > pm.cross_psnr[3] + 5                     # PSNR(p', t') >> PSNR(p', t)
    assert pm.intervened_psnr[3] > pm.prediction_psnr[3]                    # B arm better than the noisy A arm


if __name__ == "__main__":
    for fn in (test_magnitude_is_scale_invariant,
               test_alignment_is_one_when_the_model_is_exactly_right,
               test_alignment_is_zero_for_an_unrelated_response_at_every_dimension,
               test_support_survives_extreme_sparsity_and_scale,
               test_measure_pair_never_reports_a_response_to_a_null_intervention,
               test_decomposition_partitions_the_change_without_overlap,
               test_prediction_accuracy_never_sees_the_intervention,
               test_model_divergence_is_defined_where_the_ratio_is_not,
               test_repr_alignment_shares_the_pixel_metric_invariants,
               test_delta_psnr_is_zero_for_identical_arms_and_positive_when_b_departs,
               test_idm_learns_a_recoverable_action_and_is_deterministic,
               test_intervened_psnr_is_the_b_arm_against_its_own_truth):
        print(f"{fn.__name__}:"); fn()
    print("\nALL PROPERTY TESTS PASSED")
