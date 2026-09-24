"""Representation-level causal response: is blindness a REPRESENTATION failure or a DECODING failure?

Pixel-level alignment asks whether the DECODED future moves correctly when the cause changes.
A latent-state model can fail that in two different ways: its representation never encodes the
intervention (a modelling failure), or the representation encodes it but the decoder does not
render it (a decoding failure). Only the second is fixable at the decoder. The two are
distinguished by measuring alignment in representation space with the SAME estimator:

    repr_alignment(d) = <z_b - z_a, z*_b - z*_a> / ||z*_b - z*_a||^2

where z is the model's own representation of its predicted frame and z* is the representation
of the TRUE frame encoded by the same encoder. Using the model's encoder for the ground truth
keeps both sides in one space; the encoder is deterministic, so this introduces no sampling.

Everything is dimension-agnostic and reuses `causal_response`; a perfect representation scores
1, an unrelated one 0, and identical arms give NaN (no true change) exactly as in pixel space.
"""
from __future__ import annotations
import numpy as np
from ccaus.metrics import causal_response, response_magnitude


def repr_alignment(z_a: np.ndarray, z_b: np.ndarray, zt_a: np.ndarray, zt_b: np.ndarray) -> float:
    """Alignment of the model's representation change with the true representation change."""
    return causal_response(z_a.ravel(), z_b.ravel(), zt_a.ravel(), zt_b.ravel())


def repr_magnitude(z_a, z_b, zt_a, zt_b) -> float:
    return response_magnitude(z_a.ravel(), z_b.ravel(), zt_a.ravel(), zt_b.ravel())


def classify(pixel_align: float, repr_align: float, thr: float) -> str:
    """Four outcomes, from the two alignments at the same delay.

    representation ≥ thr and pixels ≥ thr  -> 'tracked'
    representation ≥ thr and pixels <  thr  -> 'decoding failure'   (encoded, not rendered)
    representation <  thr and pixels <  thr  -> 'representation failure' (never encoded)
    representation <  thr and pixels ≥ thr  -> 'anomalous'          (rendered without encoding: must be rare)
    """
    r, p = repr_align >= thr, pixel_align >= thr
    return "tracked" if (r and p) else "decoding failure" if r else "representation failure" if not p else "anomalous"
