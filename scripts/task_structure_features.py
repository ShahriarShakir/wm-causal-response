"""Task-structural features per (architecture, game) under the FIXED intervention (video_*/iris_*_dur8.json, 12+12 points with a true response): latency, immediacy, downstream fraction, plus alignment/PSNR/consistency. feeds predictive_validity.py, make_figures.py and the authoritative file."""
import json, glob, statistics as st, sys
import numpy as np
import pathlib; ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); import os; os.chdir(ROOT)
from ccaus.validate import spearman, permutation_p, detectable_rho
DELAYS = ["1","2","3","5","8","12","16","20"]
def feats(path):
    d = json.load(open(path)); S = [str(x) for x in d["seeds"]]
    rows = {dl: [d["results"][s]["aggregate"]["rows"][dl] for s in S] for dl in DELAYS}
    med = lambda dl, k: st.fmean(r[k]["median"] for r in rows[dl] if np.isfinite(r[k]["median"])) \
                        if any(np.isfinite(r[k]["median"]) for r in rows[dl]) else float("nan")
    td = [med(dl, "true_divergence") for dl in DELAYS]
    td20 = td[-1]
    if not np.isfinite(td20) or td20 <= 0.1: return None
    # LATENCY: first delay at which the true effect reaches 25% of its d20 size
    lat = next((int(DELAYS[i]) for i, v in enumerate(td) if np.isfinite(v) and v >= 0.25*td20), 99)
    # IMMEDIACY: how much of the final effect is already present at delay 1
    imm = (td[0] / td20) if np.isfinite(td[0]) else 0.0
    dec = [d["results"][s]["decomposition"]["20"]["consequence_fraction"] for s in S]
    return dict(align=med("20","causal_alignment"), align1=med("1","causal_alignment"),
                psnr=med("20","prediction_psnr"), cons=med("20","consistency"),
                td20=td20, latency=lat, immediacy=imm, downstream=float(np.nanmean(dec)))
pts = []
for arch, pat in (("DIAMOND","video_*_dur8.json"), ("IRIS","iris_*_dur8.json")):
    for f in sorted(glob.glob(f"outputs/{pat}")):
        g = f.split("_")[1].replace("-v5","")
        x = feats(f)
        if x: pts.append((arch, g, x))
print(f"{len(pts)} (game, arch) points\n")
print(f"{'arch':<8}{'game':<13}{'align d20':>10}{'latency':>8}{'immed':>7}{'downstr':>8}{'PSNR':>6}")
for a,g,x in sorted(pts, key=lambda t: -t[2]['align']):
    print(f"{a:<8}{g:<13}{x['align']:>10.4f}{x['latency']:>8}{x['immediacy']:>7.2f}{x['downstream']:>8.2f}{x['psnr']:>6.1f}")
print("\nDoes a TASK-STRUCTURAL feature predict causal alignment? (pooled, n=%d)" % len(pts))
al = [x['align'] for _,_,x in pts]
for name in ("latency","immediacy","downstream","psnr","td20"):
    v = [x[name] for _,_,x in pts]
    r = spearman(v, al); p = permutation_p(v, al)
    ok = abs(r) >= detectable_rho(len(pts))
    print(f"  rho(align, {name:<10}) = {r:+.3f}  p = {p:.4f}  {'POWERED' if ok else 'underpowered'}  (detectable {detectable_rho(len(pts)):.3f})")
json.dump([dict(arch=a, game=g, **x) for a,g,x in pts], open("outputs/task_structure_features.json","w"), indent=1)
