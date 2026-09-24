"""Score the PRE-REGISTERED latency predictions against measured causal alignment.

Predictions were written from the emulator alone (preregistered_latency_prediction.json) before
any model ran on the emulator-chosen interventions. Measured files carry the _k{k}_w{w} suffix.
BLIND is confirmed if median alignment at delay 20 < 0.1; TRACKED is confirmed if >= 0.3 AND the
model predicts the game well (PSNR > 30) -- the law only speaks for well-predicted games; a
TRACKED prediction on a badly-predicted game is scored 'not applicable', not wrong.
"""
import json, pathlib, statistics as st, sys
import numpy as np
OUT = pathlib.Path(__file__).resolve().parents[1] / "outputs"
P = json.load(open(OUT / "preregistered_latency_prediction.json"))
card = {}
for arch, pfx in (("DIAMOND", "video"), ("IRIS", "iris")):
    rows = []
    for g, pr in P.items():
        k, w = pr["offset"], pr["warmup"]
        f = OUT / f"{pfx}_{g}-v5_dur8{'_k'+str(k) if k != 3 else ''}{'_w'+str(w) if w else ''}.json"
        if not f.exists(): continue
        d = json.load(open(f)); S = [str(x) for x in d["seeds"]]
        med = lambda key: st.fmean(v for v in (d["results"][s]["aggregate"]["rows"]["20"][key]["median"] for s in S) if np.isfinite(v)) \
                          if any(np.isfinite(d["results"][s]["aggregate"]["rows"]["20"][key]["median"]) for s in S) else float("nan")
        al, ps, td = med("causal_alignment"), med("prediction_psnr"), med("true_divergence")
        pred = pr["prediction"]
        # SCORING RULE, fixed 2026-09-04T16:20 BEFORE any re-run result existed (jobs cKW_* queued, none finished).
        # Absolute alignments differ by architecture (IRIS conditions on ONE frame; its Pong is 0.19 where
        # DIAMOND's is 0.86), so a single TRACKED threshold would refute every IRIS prediction regardless of
        # truth. TRACKED threshold = ~half of each architecture's historical Pong alignment; BLIND = < 0.05.
        TRACK_THR = {"DIAMOND": 0.40, "IRIS": 0.09}[arch]
        if not np.isfinite(al) or td <= 0.1: verdict = "unmeasurable"
        elif pred == "BLIND": verdict = "CONFIRMED" if al < 0.05 else "REFUTED"
        elif pred == "TRACKED": verdict = ("CONFIRMED" if al >= TRACK_THR else "REFUTED") if ps > 30 else "n/a (PSNR<=30)"
        else: verdict = "no prediction"
        rows.append(dict(game=g, latency=pr["latency"], prediction=pred, alignment=al, psnr=ps, true_div=td, verdict=verdict))
    c = {v: sum(1 for r in rows if r["verdict"] == v) for v in ("CONFIRMED", "REFUTED", "n/a (PSNR<=30)", "no prediction", "unmeasurable")}
    card[arch] = {"rows": rows, "counts": c}
    print(f"\n{arch}: {c}")
    for r in sorted(rows, key=lambda r: (r["prediction"], -r["alignment"])):
        print(f"  {r['game']:<15} lat {str(r['latency']):>4}  pred {r['prediction']:<13} align {r['alignment']:.4f}  PSNR {r['psnr']:.1f}  -> {r['verdict']}")
json.dump(card, open(OUT / "prediction_scorecard.json", "w"), indent=1); print("\nwrote outputs/prediction_scorecard.json")
