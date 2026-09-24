"""Natural disagreements and matched PSNR, both from existing per-pair files.
(a) Per (architecture, game) cell at d20 under the chosen interventions: median dPSNR (= PSNR(p_a,t_a) - PSNR(p_b,t_a),
    'does the future move with the action') beside A, R, err ('does it move correctly'). A cell with dPSNR clearly > 0
    and A < 0.05 is a natural failure that dPSNR reports as controllable.
(b) Matched PSNR: Alien episode pairs whose factual-arm PSNR at d20 lies inside Pong's interquartile PSNR band; their alignment.
Writes outputs/dpsnr_cells.json."""
import json, pathlib, sys, numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]; OUT = ROOT / "outputs"; sys.path.insert(0, str(ROOT))
sys.argv = [sys.argv[0]]; exec(open(ROOT / "scripts" / "response_error.py").read().split("res = {}")[0])   # reuse chosen_files(), err_norm, constants
from ccaus.joint import bootstrap_median
def pairs(f):
    d = json.load(open(f)); R = [r for s in d["results"] for r in d["results"][s]["raw"]]
    floor = float(np.mean([d["results"][s]["aggregate"]["guard"]["floor"] for s in d["results"]])); return R, floor
res = {"cells": {}, "matched_psnr": {}}
for arch, prefix in (("DIAMOND", "video"), ("IRIS", "iris"), ("STORM", "storm")):
    res["cells"][arch] = {}
    for g, f in sorted(chosen_files(prefix).items()):
        R, floor = pairs(f); dl = "20"
        keep = [r for r in R if r["true_divergence"].get(dl) is not None and r["true_divergence"][dl] >= floor and r["causal_alignment"].get(dl) is not None]
        if not keep: continue
        A = np.array([r["causal_alignment"][dl] for r in keep], float); Rm = np.array([r["causal_magnitude"][dl] for r in keep], float)
        dp = np.array([r["prediction_psnr"][dl] - r["cross_psnr"][dl] for r in keep if r.get("cross_psnr") and r["cross_psnr"].get(dl) is not None], float)
        ok = np.isfinite(A) & np.isfinite(Rm)
        res["cells"][arch][g] = dict(n=int(ok.sum()), A=float(np.median(A[ok])), R=float(np.median(Rm[ok])), err=float(np.median(Rm[ok] ** 2 + 1 - 2 * A[ok])),
                                     dpsnr=bootstrap_median(dp[np.isfinite(dp)]).as_dict() if dp.size else None)
    cells = res["cells"][arch]
    miss = {g: c for g, c in cells.items() if c["dpsnr"] and c["dpsnr"]["lo"] > 0 and c["A"] < 0.05}
    res["cells"][arch + "_dpsnr_positive_but_blind"] = sorted(miss)
    print(f"{arch}: dPSNR > 0 (CI) yet A < 0.05 at d20 -> {sorted(miss)}")
    for g in sorted(miss): c = cells[g]; print(f"   {g:<12} dPSNR {c['dpsnr']['median']:+.2f} [{c['dpsnr']['lo']:+.2f},{c['dpsnr']['hi']:+.2f}]  A {c['A']:+.3f}  R {c['R']:.2f}  err {c['err']:.2f}")
    # (b) matched PSNR, Alien vs Pong
    for interv, (fa, fp) in (("chosen", (chosen_files(prefix).get("Alien"), chosen_files(prefix).get("Pong"))), ("fixed", (str(OUT / f"{prefix}_Alien-v5_dur8.json"), str(OUT / f"{prefix}_Pong-v5_dur8.json")))):
      if fa and fp and pathlib.Path(fa).exists() and pathlib.Path(fp).exists():
        Ra, fla = pairs(fa); Rp, flp = pairs(fp)
        pp = np.array([r["prediction_psnr"]["20"] for r in Rp]); lo, hi = np.percentile(pp, [25, 75])
        sel = [r for r in Ra if lo <= r["prediction_psnr"]["20"] <= hi and r["true_divergence"]["20"] >= fla and r["causal_alignment"].get("20") is not None]
        allA = [r for r in Ra if r["true_divergence"]["20"] >= fla and r["causal_alignment"].get("20") is not None]
        res["matched_psnr"][f"{arch}_{interv}"] = dict(intervention=interv, pong_psnr_iqr=[float(lo), float(hi)], n_alien_matched=len(sel), n_alien_all=len(allA),
            alien_matched_align=bootstrap_median([r["causal_alignment"]["20"] for r in sel]).as_dict() if sel else None,
            alien_matched_psnr=float(np.median([r["prediction_psnr"]["20"] for r in sel])) if sel else None,
            alien_all_align=bootstrap_median([r["causal_alignment"]["20"] for r in allA]).as_dict())
        m = res["matched_psnr"][f"{arch}_{interv}"]; print(f"{arch} {interv}: Pong IQR [{lo:.1f},{hi:.1f}] dB; Alien pairs inside it {len(sel)}/{len(allA)}, their alignment {m['alien_matched_align']['median'] if sel else float('nan'):.3f} (all Alien {m['alien_all_align']['median']:.3f})")
json.dump(res, open(OUT / "dpsnr_cells.json", "w"), indent=1); print("wrote outputs/dpsnr_cells.json")
