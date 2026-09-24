"""Mechanism probe: does a model MOVE when the emulator has not (yet) moved?

The latency law says effects that first appear several steps after the action are lost. One
mechanism would make that inevitable: a model trained on next-frame prediction has no gradient
for an effect absent from the next frame, so (a) it never learns the latent link, and (b) it may
learn a spurious IMMEDIATE response instead. Signature of (b): at delay 1, TRUE divergence ≈ 0
but MODEL divergence > 0 -- a hallucinated immediate effect. Signature of (a): when the real
effect finally appears (delay 8 in Alien/BankHeist), alignment stays ≈ 0.

Uses the ABSOLUTE model divergence ||pred_b - pred_a||, which is defined where the ratio is not.
Reported per game and architecture with per-seed spread; no claim is made in this file.
"""
import json, pathlib, statistics as st, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
OUT = pathlib.Path(__file__).resolve().parents[1] / "outputs"
DELAYS = ["1", "2", "3", "5", "8", "12", "16", "20"]
out = {}
for arch, pfx in (("DIAMOND", "video"), ("IRIS", "iris")):
    for g in ("Alien", "BankHeist", "Pong", "Freeway"):
        f = OUT / f"{pfx}_{g}-v5_dur8.json"
        if not f.exists(): continue
        d = json.load(open(f)); S = [str(x) for x in d["seeds"]]
        r0 = d["results"][S[0]]["aggregate"]["rows"]["1"]
        if "model_divergence" not in r0 or r0["model_divergence"]["n"] == 0: continue
        rows = {dl: [d["results"][s]["aggregate"]["rows"][dl] for s in S] for dl in DELAYS}
        med = lambda dl, k: st.fmean(r[k]["median"] for r in rows[dl] if np.isfinite(r[k]["median"])) \
                            if any(np.isfinite(r[k]["median"]) for r in rows[dl]) else float("nan")
        sd = lambda dl, k: st.pstdev([r[k]["median"] for r in rows[dl] if np.isfinite(r[k]["median"])] or [0.0])
        print(f"\n=== {arch} {g} ===   {'delay':>5}{'TRUE div':>10}{'MODEL div':>11}{'ratio M/T':>10}{'align':>8}")
        rec = {}
        for dl in DELAYS:
            td, mdv, al = med(dl, "true_divergence"), med(dl, "model_divergence"), med(dl, "causal_alignment")
            ratio = mdv / td if td > 1e-9 else float("inf")
            print(f"{'':22}{dl:>5}{td:>10.3f}{mdv:>11.3f}{ratio:>10.2f}{al:>8.3f}")
            rec[dl] = {"true_div": td, "model_div": mdv, "model_div_sd": sd(dl, "model_divergence"), "align": al}
        # the hallucination signature at delay 1
        h = rec["1"]
        print(f"{'':22}d1: model moves {h['model_div']:.3f} ± {h['model_div_sd']:.3f} while emulator moves {h['true_div']:.3f}"
              f"  -> {'HALLUCINATED immediate effect' if h['true_div'] < 0.05 and h['model_div'] > 0.2 else 'no hallucination signature'}")
        out[f"{arch}/{g}"] = rec
json.dump(out, open(OUT / "hallucination_probe.json", "w"), indent=1)
print("\nwrote outputs/hallucination_probe.json")
