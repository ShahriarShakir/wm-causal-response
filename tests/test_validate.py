"""Tests for the analysis gates. Each encodes a failure of an analysis layer that the gates must catch."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from ccaus.validate import (claim, detectable_rho, gate_correlation, gate_dataset,
                            permutation_p, spearman)


def test_degenerate_units_are_caught():
    """The exact failure, at the REAL proportions: 8 of 26 Atari games carried a live
    intervention; the other 18 had median true divergence 0.0000 and their near-zero response
    was read as model blindness.

    The fixture must match the real SHAPE of the data, not a convenient miniature. An earlier
    version of this test used 4 live of 7 (57%) -- which legitimately passes the majority gate,
    so the test failed while the code was correct. Same class of error as testing a sparse
    difference image with a dense array.
    """
    td = {f"live{i}": 2.0 + i for i in range(8)}
    td.update({f"dead{i}": 0.0 for i in range(18)})
    rep = gate_dataset(td)
    assert not rep.ok, "degenerate games slipped through"
    assert "31% survive" in str(rep), str(rep)
    print(f"  {rep.failures()[0].detail[:88]}")


def test_degenerate_units_are_NAMED_even_when_the_gate_passes():
    """An exclusion must always be visible, not only when it is large enough to fail."""
    td = {f"live{i}": 2.0 for i in range(9)}; td["BankHeist"] = 0.0
    rep = gate_dataset(td)
    assert rep.ok and "BankHeist" in str(rep), str(rep)
    print("  passing gate still names the excluded unit")


def test_underpowered_correlation_is_refused():
    """rho=+0.500 at n=8 was reported with p=0.217. At n=8 the smallest detectable rho is
    ~0.71, so that p carries no information about absence."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=8); y = 0.5 * x + rng.normal(size=8)
    rep = gate_correlation(x, y)
    assert not rep.ok
    out = claim("consistency does not predict causal fidelity", rep)
    assert out.startswith("UNSUPPORTED")
    print(f"  n=8 smallest detectable rho = {detectable_rho(8):.3f} -> refused")


def test_a_real_effect_at_adequate_n_passes():
    """The gates must not refuse everything, or they are useless."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=30); y = x + 0.25 * rng.normal(size=30)
    rep = gate_correlation(x, y)
    assert rep.ok, str(rep)
    print(f"  n=30, rho={spearman(x,y):.3f}, p={permutation_p(x,y,2000):.4f} -> supported")


def test_permutation_p_is_calibrated():
    """Under the null the p-value must be roughly uniform, or every gate built on it lies."""
    rng = np.random.default_rng(2)
    ps = [permutation_p(rng.normal(size=12), rng.normal(size=12), 400, seed=i)
          for i in range(60)]
    frac = float(np.mean([p < 0.05 for p in ps]))
    print(f"  false-positive rate at alpha=0.05: {frac:.3f} (expect ~0.05)")
    assert frac < 0.20


if __name__ == "__main__":
    for fn in (test_degenerate_units_are_caught,
               test_degenerate_units_are_NAMED_even_when_the_gate_passes, test_underpowered_correlation_is_refused,
               test_a_real_effect_at_adequate_n_passes, test_permutation_p_is_calibrated):
        print(f"{fn.__name__}:"); fn()
    print("\nALL VALIDATE TESTS PASSED")
