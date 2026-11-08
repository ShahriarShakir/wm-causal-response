"""R1 (rule in rebuttal/README.md): closed-loop gain against alignment averaged over the planner's 12-step horizon.
Reads outputs/closed_loop_eval.json and outputs/storm_extend.json; writes rebuttal/outputs/r1_horizon_gain.json."""
import json, pathlib, numpy as np
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "rebuttal" / "outputs"; OUT.mkdir(parents=True, exist_ok=True)
CL = json.load(open(ROOT / "outputs" / "closed_loop_eval.json"))
E = json.load(open(ROOT / "outputs" / "storm_extend.json"))
T = E["thresholds"]
H_DELAYS = ("1", "2", "3", "5", "8", "12")          # inside the planner's 12-step horizon
ARCH = (("diamond", "DIAMOND"), ("iris", "IRIS"), ("storm", "STORM"))
rng = np.random.default_rng(0)

def rho(x, y): return float(spearmanr(x, y).correlation)

# Per-delay curves from the sweep files with the estimator of scripts/make_base_tables.py (rows(); copied, not imported, because
# importing that script rewrites the paper's base tables). storm_extend.json stores delays 1-8 and 20 only; delay 12 needs the sweep files.
S = json.load(open(ROOT / "outputs" / "emulator_intervention_search.json"))
PREFIX = {"DIAMOND": ("video", 0), "IRIS": ("iris", 0), "STORM": ("storm", 4)}
ALL_D = ("1", "2", "3", "5", "8", "12", "16", "20")
def fname(arch, g, k, w):
    pre, ws = PREFIX[arch]; w = max(0, w - ws); return ROOT / "outputs" / f"{pre}_{g}-v5_dur8{'_k' + str(k) if k != 3 else ''}{'_w' + str(w) if w else ''}.json"
def curve(arch, g):
    f = fname(arch, g, int(S[g]["best_offset"]), int(S[g]["best_warmup"])); d = json.load(open(f)); Ss = list(d["results"]); r = {}
    for dl in ALL_D:
        A = [d["results"][s]["aggregate"]["rows"][dl]["causal_alignment"]["median"] for s in Ss if dl in d["results"][s]["aggregate"]["rows"]]
        ps = [d["results"][s]["aggregate"]["rows"][dl]["prediction_psnr"]["median"] for s in Ss if dl in d["results"][s]["aggregate"]["rows"]]
        ok = bool(A) and all(v == v for v in A)
        r[dl] = dict(A=float(np.mean(A)) if ok else None, psnr=float(np.mean(ps)) if ps else None)
    return r
check = []
cells = []                                             # (game, arch, gain, A_h, A_20, psnr_20)
for a, A in ARCH:
    for g, row in CL["games"].items():
        if a not in row or row[a].get("gain_norm") is None: continue
        pg = curve(A, g); st = E["per_game"].get(A, {}).get(g, {})
        for d in ("1", "2", "3", "5", "8", "20"):       # the copied estimator must reproduce the stored values
            if st.get(d, {}).get("A") is not None: check.append(abs(st[d]["A"] - pg[d]["A"]))
        if pg.get("20", {}).get("A") is None: continue
        ah = [pg[d]["A"] for d in H_DELAYS if pg.get(d, {}).get("A") is not None]
        if len(ah) < 4: continue                       # deviation noted in rebuttal/README.md, R1: delays gated out by the degeneracy gate are skipped (Qbert, delay 1)
        cells.append(dict(game=g, arch=A, gain=row[a]["gain_norm"], A_h=float(np.mean(ah)), A_20=pg["20"]["A"], psnr_20=pg["20"]["psnr"],
                          per_delay={d: pg[d]["A"] for d in H_DELAYS + ("16", "20")}, useful=row[a]["useful"]))

assert check and max(check) < 1e-9, f"copied estimator does not reproduce storm_extend.json (max diff {max(check) if check else None})"
print(f"estimator check: {len(check)} stored values reproduced, max |diff| {max(check):.1e}")
res = {"rule": "rebuttal/README.md R1", "horizon_delays": H_DELAYS, "thresholds": T, "estimator_check_n": len(check), "per_arch": {}, "pooled": {}, "cells": cells}
for A in ("DIAMOND", "IRIS", "STORM"):
    c = [x for x in cells if x["arch"] == A]; gain = [x["gain"] for x in c]
    res["per_arch"][A] = dict(n=len(c), rho_A_h=rho(gain, [x["A_h"] for x in c]), rho_A_20=rho(gain, [x["A_20"] for x in c]),
                              rho_psnr_20=rho(gain, [x["psnr_20"] for x in c]))

games = sorted({x["game"] for x in cells})
def pooled(key, cs): return rho([x["gain"] for x in cs], [x[key] for x in cs])
def perm_p(key, B=10000):
    """game-level block permutation: the predictor values of a game (all its architectures) move together to another game's gains"""
    obs = pooled(key, cells); by = {g: [x for x in cells if x["game"] == g] for g in games}; hits = 0
    for _ in range(B):
        perm = dict(zip(games, rng.permutation(games))); xs, ys = [], []
        for g in games:
            src = {x["arch"]: x[key] for x in by[perm[g]]}
            for x in by[g]:
                if x["arch"] in src: xs.append(src[x["arch"]]); ys.append(x["gain"])
        if len(xs) > 3 and abs(rho(ys, xs)) >= abs(obs) - 1e-12: hits += 1
    return (hits + 1) / (B + 1)
def boot_ci(key, B=4000, diff_with=None):
    vals = []
    for _ in range(B):
        gs = rng.choice(games, len(games)); cs = [x for g in gs for x in cells if x["game"] == g]
        v = pooled(key, cs)
        if diff_with: v -= pooled(diff_with, cs)
        if v == v: vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
for key in ("A_h", "A_20", "psnr_20"):
    res["pooled"][key] = dict(n=len(cells), n_games=len(games), rho=pooled(key, cells), p_perm=perm_p(key), ci=boot_ci(key))
res["pooled"]["diff_A_h_minus_psnr_20"] = dict(value=pooled("A_h", cells) - pooled("psnr_20", cells), ci=boot_ci("A_h", diff_with="psnr_20"))
res["pooled"]["diff_A_h_minus_A_20"] = dict(value=pooled("A_h", cells) - pooled("A_20", cells), ci=boot_ci("A_h", diff_with="A_20"))
P = res["pooled"]["A_h"]
res["reading"] = "tracks closed-loop gain" if (P["rho"] >= 0.41 and P["p_perm"] < 0.05) else "not established"
json.dump(res, open(OUT / "r1_horizon_gain.json", "w"), indent=1)

for A, v in res["per_arch"].items():
    print(f"{A}: n={v['n']} rho(gain, A_h)={v['rho_A_h']:+.2f}  rho(gain, A_20)={v['rho_A_20']:+.2f}  rho(gain, PSNR_20)={v['rho_psnr_20']:+.2f}")
for k, v in res["pooled"].items(): print("pooled", k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()})
print("READING:", res["reading"])
for x in cells:
    if x["game"] == "CrazyClimber": print("CrazyClimber", x["arch"], "A_h", round(x["A_h"], 3), "T", T[x["arch"]], "per-delay", {d: round(v, 2) for d, v in x["per_delay"].items()}, "gain", round(x["gain"], 2), "useful", x["useful"])
