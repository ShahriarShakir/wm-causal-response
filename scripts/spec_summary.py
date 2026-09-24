"""Off-support summary per architecture: median off-support fraction at delay 20 in blind (A_b < 0.05) and tracked (A_b >= T) cells,
with ranges, and the median cosines (T now read from storm_extend.json thresholds).
Writes outputs/spec_summary.json. Runs after spec_analysis.py and storm_extend.py."""
import json, pathlib, os, numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]; os.chdir(ROOT)
S = json.load(open("outputs/spec_analysis.json")); T = json.load(open("outputs/storm_extend.json"))["thresholds"]; summ = {}
for a in ("DIAMOND", "IRIS", "STORM", "TWM"):
    if a not in S or a not in T: continue
    cells = {g: r for g, r in S[a].items() if not g.startswith("_") and r.get("20") and r["20"].get("offsupport")}
    off = lambda g: cells[g]["20"]["offsupport"]["median"]; A = lambda g: cells[g]["20"]["align_B"]["median"]
    blind = [g for g in cells if A(g) < 0.05]; tracked = [g for g in cells if A(g) >= T[a]]
    summ[a] = dict(n=len(cells), T=T[a], n_blind=len(blind), off_blind=float(np.median([off(g) for g in blind])) if blind else None, off_blind_range=[float(min(off(g) for g in blind)), float(max(off(g) for g in blind))] if blind else None,
                   n_tracked=len(tracked), off_tracked=float(np.median([off(g) for g in tracked])) if tracked else None, off_tracked_range=[float(min(off(g) for g in tracked)), float(max(off(g) for g in tracked))] if tracked else None,
                   cos_pBpC_median=float(np.median([cells[g]["20"]["cos_pBpC"]["median"] for g in cells if cells[g]["20"].get("cos_pBpC")])), cos_tBtC_median=float(np.median([cells[g]["20"]["cos_tBtC"]["median"] for g in cells if cells[g]["20"].get("cos_tBtC")])))
    print(a, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in summ[a].items()})
json.dump(summ, open("outputs/spec_summary.json", "w"), indent=1); print("wrote outputs/spec_summary.json")
