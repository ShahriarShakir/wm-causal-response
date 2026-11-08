"""R7 (rule in rebuttal/README.md, fixed before any job ran): goal reaching against the diagnostic scores, over the 78 cells.
G per cell from ext/outputs_goal/goal_<arch>_<Game>.json (mean chance-corrected success over informative trials, at least 10);
A_h and PSNR_h from the paper's sweep files with R1's estimator (mean of per-seed medians, delays 1, 2, 3, 5, 8, 12, at least 4 of 6);
delay-20 Delta-PSNR from outputs/dpsnr_cells.json where it exists. Writes rebuttal/outputs/r7_goal.json."""
import json, pathlib, numpy as np
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parents[2]; OUT = ROOT / "rebuttal" / "outputs"
S = json.load(open(ROOT / "outputs" / "emulator_intervention_search.json")); DP = json.load(open(ROOT / "outputs" / "dpsnr_cells.json"))["cells"]
GAMES = sorted(S); ARCHS = ("DIAMOND", "IRIS", "STORM"); H_DELAYS = ("1", "2", "3", "5", "8", "12")
PREFIX = {"DIAMOND": ("video", 0), "IRIS": ("iris", 0), "STORM": ("storm", 4)}
def rho(x, y):
    r = spearmanr(x, y).correlation; return float(r) if r == r else float("nan")

# ---- R1's per-delay estimator (rebuttal/scripts/r1_horizon_gain.py, copied) ----
def fname(arch, g, k, w):
    pre, ws = PREFIX[arch]; w = max(0, w - ws); return ROOT / "outputs" / f"{pre}_{g}-v5_dur8{'_k' + str(k) if k != 3 else ''}{'_w' + str(w) if w else ''}.json"
def curve(arch, g):
    d = json.load(open(fname(arch, g, int(S[g]["best_offset"]), int(S[g]["best_warmup"])))); Ss = list(d["results"]); r = {}
    for dl in H_DELAYS:
        A = [d["results"][s]["aggregate"]["rows"][dl]["causal_alignment"]["median"] for s in Ss if dl in d["results"][s]["aggregate"]["rows"]]
        ps = [d["results"][s]["aggregate"]["rows"][dl]["prediction_psnr"]["median"] for s in Ss if dl in d["results"][s]["aggregate"]["rows"]]
        ok = bool(A) and all(v == v for v in A)
        r[dl] = dict(A=float(np.mean(A)) if ok else None, psnr=float(np.mean(ps)) if ps and all(v == v for v in ps) else None)
    return r

cells, missing = [], []
for A in ARCHS:
    for g in GAMES:
        f = ROOT / "ext" / "outputs_goal" / f"goal_{A.lower()}_{g}.json"
        if not f.exists(): missing.append(f"{A}/{g}"); continue
        d = json.load(open(f))
        if not d["complete"]: missing.append(f"{A}/{g} (incomplete)"); continue
        inf = [t for t in d["trials"] if t.get("informative")]
        G = float(np.mean([(t["success"] - t["chance"]) / (1 - t["chance"]) for t in inf])) if len(inf) >= 10 else None
        cv = curve(A, g); ah = [cv[k]["A"] for k in H_DELAYS if cv[k]["A"] is not None]; ph = [cv[k]["psnr"] for k in H_DELAYS if cv[k]["psnr"] is not None]
        dps = ((DP.get(A, {}).get(g) or {}).get("dpsnr") or {}).get("median")
        cells.append(dict(arch=A, game=g, G=G, n_informative=len(inf), n_skipped=d["n_skipped"], success=sum(bool(t.get("success")) for t in inf),
                          mean_chance=float(np.mean([t["chance"] for t in inf])) if inf else None,
                          A_h=float(np.mean(ah)) if len(ah) >= 4 else None, psnr_h=float(np.mean(ph)) if len(ph) >= 4 else None, dpsnr_20=dps))

rng = np.random.default_rng(0)
def use(kx, ky="G", extra=None): return [c for c in cells if c.get(kx) is not None and c.get(ky) is not None and (extra is None or c.get(extra) is not None)]
def pooled(cs, kx, ky="G"): return rho([c[ky] for c in cs], [c[kx] for c in cs])
def perm_p(cs, kx, ky="G", B=10000):
    obs = pooled(cs, kx, ky); gs = sorted({c["game"] for c in cs}); by = {g: [c for c in cs if c["game"] == g] for g in gs}; hits = 0
    if obs != obs: return float("nan")
    for _ in range(B):
        perm = dict(zip(gs, rng.permutation(gs))); xs, ys = [], []
        for g in gs:
            src = {c["arch"]: c[kx] for c in by[perm[g]]}
            for c in by[g]:
                if c["arch"] in src: xs.append(src[c["arch"]]); ys.append(c[ky])
        if len(xs) > 3 and abs(rho(ys, xs)) >= abs(obs) - 1e-12: hits += 1
    return (hits + 1) / (B + 1)
def boot(cs, kx, ky="G", minus=None, B=4000):
    gs = sorted({c["game"] for c in cs}); v = []
    for _ in range(B):
        s = [c for g in rng.choice(gs, len(gs)) for c in cs if c["game"] == g]
        r = pooled(s, kx, ky) - (pooled(s, minus, ky) if minus else 0.0)
        if r == r: v.append(r)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else None

res = {"rule": "rebuttal/README.md R7", "n_cells": len(cells), "missing": missing, "cells": cells}
c1 = use("A_h"); res["T1"] = dict(n=len(c1), rho=pooled(c1, "A_h"), p_perm=perm_p(c1, "A_h"), ci=boot(c1, "A_h"))
res["T1"]["reading"] = "alignment predicts goal reaching" if res["T1"]["rho"] >= 0.41 and res["T1"]["p_perm"] < 0.05 else "not established"
c2 = use("A_h", extra="psnr_h")
res["T2_psnr"] = dict(n=len(c2), rho_psnr=pooled(c2, "psnr_h"), p_psnr=perm_p(c2, "psnr_h"), ci_psnr=boot(c2, "psnr_h"),
                      diff_A_minus_psnr=pooled(c2, "A_h") - pooled(c2, "psnr_h"), diff_ci=boot(c2, "A_h", minus="psnr_h"))
c3 = use("A_h", extra="dpsnr_20")
res["T2_dpsnr"] = dict(n=len(c3), rho_A_h_subset=pooled(c3, "A_h"), rho_dpsnr=pooled(c3, "dpsnr_20"), p_dpsnr=perm_p(c3, "dpsnr_20"), ci_dpsnr=boot(c3, "dpsnr_20"),
                       diff_A_minus_dpsnr=pooled(c3, "A_h") - pooled(c3, "dpsnr_20"), diff_ci=boot(c3, "A_h", minus="dpsnr_20"))
for k, key in (("T2_psnr", "diff_ci"), ("T2_dpsnr", "diff_ci")):
    ci = res[k][key]; res[k]["reading"] = "alignment better" if ci and ci[0] > 0 else ("comparator better" if ci and ci[1] < 0 else "no detectable difference")
res["T3"] = dict(per_arch={A: dict(n=len([c for c in c1 if c["arch"] == A]), rho=rho([c["G"] for c in c1 if c["arch"] == A], [c["A_h"] for c in c1 if c["arch"] == A])) for A in ARCHS},
                 flagship={f"DIAMOND/{g}": next(({k: c[k] for k in ("G", "A_h", "psnr_h", "n_informative", "success")} for c in cells if c["arch"] == "DIAMOND" and c["game"] == g), None) for g in ("Alien", "Pong")})
json.dump(res, open(OUT / "r7_goal.json", "w"), indent=1)
for k in ("T1", "T2_psnr", "T2_dpsnr", "T3"): print(k, res[k])
print("missing:", missing)
