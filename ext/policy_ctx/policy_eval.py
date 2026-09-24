"""Policy-context control evaluation (arm D; reading rule in ext/policy_ctx/README.md). Per (arch, game, epsilon): delay-20 alignment
under policy-driven contexts beside the random-context value of the main sweep, PSNR, the median true divergence at delay 20 (zero
means the trained policy makes the intervention inert by delay 20 and the cell is unmeasurable there), and the six-game Spearman
between the two context types at epsilon 0. Writes outputs/policy_ctx_eval.json."""
import json, glob, pathlib, sys, numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT)); OUT = ROOT / "outputs"
from ccaus.validate import spearman
E = json.load(open(OUT / "storm_extend.json")); res = {"cells": {}, "spearman_eps0": {}}
for f in sorted(glob.glob(str(OUT / "policy_*.json"))):
    if "DRY" in f or "policy_ctx_eval" in f: continue   # the glob also matches this script's own output
    d = json.load(open(f)); a = d["arch"].upper(); g = d["game"].split("/")[-1].replace("-v5", ""); Ss = list(d["results"])
    vals = lambda dl, k: [d["results"][s]["aggregate"]["rows"][dl][k]["median"] for s in Ss if dl in d["results"][s]["aggregate"]["rows"]]
    A20 = [v for v in vals("20", "causal_alignment") if v == v]; td = vals("20", "true_divergence")
    res["cells"][f"{a}/{g}/eps{d['epsilon']:g}"] = dict(arch=a, game=g, epsilon=d["epsilon"], A20_policy=float(np.mean(A20)) if A20 else None, A20_policy_sd=float(np.std(A20)) if A20 else None, n_seeds_measurable=len(A20),
        A1_policy=float(np.nanmean(vals("1", "causal_alignment"))), A20_random=E["per_game"][a][g]["20"]["A"], psnr_policy=float(np.mean(vals("20", "prediction_psnr"))), psnr_random=E["per_game"][a][g]["20"]["psnr"],
        true_div20_policy=float(np.median(td)), true_div20_random=E["per_game"][a][g]["20"]["tdiv"] if "tdiv" in E["per_game"][a][g]["20"] else None, guard_kept=int(sum(d["results"][s]["aggregate"]["guard"]["n_kept"] for s in Ss)), guard_dropped=int(sum(d["results"][s]["aggregate"]["guard"]["n_dropped"] for s in Ss)))
for a in ("DIAMOND", "IRIS"):
    cs = [c for c in res["cells"].values() if c["arch"] == a and c["epsilon"] == 0.0 and c["A20_policy"] is not None]
    res["spearman_eps0"][a] = dict(n=len(cs), rho=spearman([c["A20_policy"] for c in cs], [c["A20_random"] for c in cs]), games=[c["game"] for c in cs])
for k, c in res["cells"].items(): print(f"{k:<26} A20 policy {c['A20_policy'] if c['A20_policy'] is None else round(c['A20_policy'], 3)!s:>7} (random {c['A20_random']:+.3f}) A1 {c['A1_policy']:+.3f} psnr {c['psnr_policy']:.1f} true-div20 {c['true_div20_policy']:.2f} kept/dropped {c['guard_kept']}/{c['guard_dropped']}")
print("six-game Spearman (eps 0):", {a: (round(v["rho"], 2), v["n"]) for a, v in res["spearman_eps0"].items()})
json.dump(res, open(OUT / "policy_ctx_eval.json", "w"), indent=1); print("wrote outputs/policy_ctx_eval.json")
