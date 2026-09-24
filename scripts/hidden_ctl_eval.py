"""Wrong-action control for the IRIS pre-sampling probe (v12): per game and delay, medians with bootstrap CI of
B|B (probe), C|B (model's response to a different action projected on the true response to B), B|C, and cos(true B, true C).
Action-specific if C|B is well below B|B. Note: this control cannot separate visual tracking from an action-token echo;
the frames-only control (v13) does. Writes outputs/hidden_ctl_eval.json."""
import json, pathlib, sys, glob, numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]; OUT = ROOT / "outputs"; sys.path.insert(0, str(ROOT))
from ccaus.joint import bootstrap_median
res = {}
for f in sorted(glob.glob(str(OUT / "irishiddenctl_*.json"))):
    if "DRY" in f: continue
    d = json.load(open(f)); g = d["game"].split("/")[-1].replace("-v5", ""); raws = [r for s in d["results"] for r in d["results"][s]["raw"]]
    res[g] = {"n": len(raws)}
    for dl in ("1", "2", "3", "5", "8"):
        row = {}
        for k in ("B_vs_trueB", "C_vs_trueB", "B_vs_trueC", "cos_trueB_trueC"):
            v = [r[k][dl] for r in raws if r[k].get(dl) is not None and r[k][dl] == r[k][dl]]
            row[k] = bootstrap_median(v).as_dict() if v else None
        bb, cb = [r["B_vs_trueB"][dl] for r in raws if r["B_vs_trueB"].get(dl) is not None], [r["C_vs_trueB"][dl] for r in raws if r["C_vs_trueB"].get(dl) is not None]
        diff = [b - c for b, c in zip(bb, cb) if b == b and c == c]; rng = np.random.default_rng(0)
        obs = float(np.mean(diff)); null = [abs(float(np.mean(np.array(diff) * rng.choice([-1, 1], len(diff))))) for _ in range(5000)]
        row["p_BB_gt_CB"] = float((np.sum(np.array(null) >= abs(obs)) + 1) / (len(null) + 1)); row["diff_BB_minus_CB"] = obs
        res[g][dl] = row
    print(f"{g:<10} n {len(raws)} | " + " ".join(f"d{dl}: B|B {res[g][dl]['B_vs_trueB']['median']:.2f} C|B {res[g][dl]['C_vs_trueB']['median']:.2f} (p {res[g][dl]['p_BB_gt_CB']:.3f})" for dl in ("1", "3", "8")))
json.dump(res, open(OUT / "hidden_ctl_eval.json", "w"), indent=1); print("wrote outputs/hidden_ctl_eval.json")
