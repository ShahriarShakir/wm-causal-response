"""R6 (rule in rebuttal/README.md, fixed before any job ran): experiment A, the response and the reward ranking at each model's
own planner-visited states. Reads ext/outputs/visited_<arch>_<Game>[_e<a>-<b>].json, outputs/closed_loop_eval.json and
rebuttal/outputs/r1_horizon_gain.json; writes rebuttal/outputs/r6_visited.json."""
import json, glob, os, pathlib, re, numpy as np
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parents[2]; OUT = ROOT / "rebuttal" / "outputs"
SRC = os.environ.get("VISITED_OUTDIR", "outputs")                    # outputs_gpu for the GPU replication
CL = json.load(open(ROOT / "outputs" / "closed_loop_eval.json")); R1 = json.load(open(OUT / "r1_horizon_gain.json")); T = R1["thresholds"]
GAMES = ("Alien", "Asterix", "Boxing", "Breakout", "CrazyClimber", "Frostbite", "Qbert", "Seaquest"); ARCHS = ("DIAMOND", "IRIS", "STORM")
DEL = ("1", "2", "3", "5", "8", "12"); FRAC = 0.25                      # ccaus.joint.Guard default: 0.25 of the median true divergence
rng = np.random.default_rng(0)
def rho(x, y):
    r = spearmanr(x, y).correlation; return float(r) if r == r else float("nan")
def fin(v): return [float(x) for x in v if x is not None and np.isfinite(x)]

def cell_stats(A, g):
    files = [f for f in glob.glob(str(ROOT / "ext" / SRC / f"visited_{A.lower()}_{g}*.json")) if "DRY" not in f]
    if not files: return None
    eps, aggs = [], []
    for f in files:
        d = json.load(open(f)); eps += d["episodes"]; aggs.append((d.get("episode_range"), d.get("aggregate"), d.get("complete"), d.get("determinism_match"), d.get("n_episodes_done")))
    pairs = [p for e in eps for p in e["pairs"]]
    tds = [p["true_divergence"][d] for p in pairs for d in DEL if p["true_divergence"].get(d) is not None and np.isfinite(p["true_divergence"][d])]
    floor = FRAC * float(np.median(tds)) if tds else 0.0
    med_A, med_P, n_A = {}, {}, {}
    for d in DEL:
        # ccaus.joint.bootstrap_median drops non-finite values before the median (a pair with zero true change passes a
        # zero floor with alignment 0/0); the same rule here
        av = fin([p["causal_alignment"].get(d) for p in pairs if p["true_divergence"].get(d) is not None and p["true_divergence"][d] >= floor])
        med_A[d] = float(np.median(av)) if av else None; n_A[d] = len(av)
        ps = fin([p["prediction_psnr"].get(d) for p in pairs])
        med_P[d] = float(np.median(ps)) if ps else None
    # the recomputation must reproduce the library's aggregate wherever one complete file covers the cell:
    # guard floor, and per delay the alignment median, its count, its finiteness, and the PSNR median
    check = None
    if len(aggs) == 1 and aggs[0][1]:
        lib = aggs[0][1]; diffs = [abs(lib["guard"]["floor"] - floor)]
        for d in DEL:
            la, lp = lib["rows"][d]["causal_alignment"], lib["rows"][d]["prediction_psnr"]
            mine = med_A[d] if med_A[d] is not None else float("nan")
            assert la["n"] == n_A[d] and bool(np.isfinite(la["median"])) == bool(np.isfinite(mine)), f"{A}/{g} delay {d}: library {la} vs n {n_A[d]} median {mine}"
            if np.isfinite(mine): diffs.append(abs(la["median"] - mine))
            if med_P[d] is not None: diffs.append(abs(lp["median"] - med_P[d]))
        check = max(diffs)
        assert check < 1e-9, f"{A}/{g}: recomputation differs from the library aggregate by {check}"
    a_vals = [v for v in med_A.values() if v is not None]
    rk, rf, n_inf = [], [], 0
    for e in eps:
        for r in e["ranking"]:
            if len(set(r["native"])) < 2: continue
            n_inf += 1; x = rho(r["model"], r["native"]); y = rho(r["true_frames"], r["native"])
            if x == x: rk.append(x)
            if y == y: rf.append(y)
    return dict(n_episodes=len(eps), n_pairs=len(pairs), guard_floor=floor, A_by_delay=med_A, n_by_delay=n_A, psnr_by_delay=med_P,
                A_vis=float(np.mean(a_vals)) if len(a_vals) >= 4 else None,
                psnr_vis=float(np.mean([v for v in med_P.values() if v is not None])) if any(v is not None for v in med_P.values()) else None,
                rank_vis=float(np.median(rk)) if n_inf >= 5 and rk else None, ref_vis=float(np.median(rf)) if n_inf >= 5 and rf else None, n_informative=n_inf,
                returns=[e["ret"] for e in eps], determinism=[a[3] for a in aggs], complete=all(a[2] for a in aggs), library_check_maxdiff=check,
                skipped=sum(len(e["skipped"]) for e in eps))

ah = {(c["arch"], c["game"]): c["A_h"] for c in R1["cells"]}
cells = []
for A in ARCHS:
    for g in GAMES:
        s = cell_stats(A, g); gain = CL["games"].get(g, {}).get(A.lower(), {}).get("gain_norm")
        if s is None or gain is None: continue
        cells.append(dict(arch=A, game=g, gain=gain, A_h=ah.get((A, g)), T=T[A], **s))
games = sorted({c["game"] for c in cells})
def usable(key): return [c for c in cells if c.get(key) is not None]
def pooled(cs, kx, ky="gain"): return rho([c[ky] for c in cs], [c[kx] for c in cs])
def perm_p(kx, ky="gain", B=10000):
    cs = [c for c in cells if c.get(kx) is not None and c.get(ky) is not None]; obs = pooled(cs, kx, ky); gs = sorted({c["game"] for c in cs})
    by = {g: [c for c in cs if c["game"] == g] for g in gs}; hits = 0
    if obs != obs: return float("nan")                                   # an undefined correlation has no p-value
    for _ in range(B):
        perm = dict(zip(gs, rng.permutation(gs))); xs, ys = [], []
        for g in gs:
            src = {c["arch"]: c[kx] for c in by[perm[g]]}
            for c in by[g]:
                if c["arch"] in src: xs.append(src[c["arch"]]); ys.append(c[ky])
        if len(xs) > 3 and abs(rho(ys, xs)) >= abs(obs) - 1e-12: hits += 1
    return (hits + 1) / (B + 1)
def boot(kx, ky="gain", minus=None, B=4000):
    cs = [c for c in cells if c.get(kx) is not None and c.get(ky) is not None and (minus is None or c.get(minus) is not None)]; gs = sorted({c["game"] for c in cs}); v = []
    for _ in range(B):
        pick = rng.choice(gs, len(gs)); s = [c for g in pick for c in cs if c["game"] == g]
        r = pooled(s, kx, ky) - (pooled(s, minus, ky) if minus else 0.0)
        if r == r: v.append(r)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else None
res = {"rule": "rebuttal/README.md R6", "n_cells": len(cells), "cells": cells}
c1 = usable("A_vis")
res["T1"] = dict(n=len(c1), rho=pooled(c1, "A_vis"), p_perm=perm_p("A_vis"), ci=boot("A_vis"),
                 per_arch={A: rho([c["gain"] for c in c1 if c["arch"] == A], [c["A_vis"] for c in c1 if c["arch"] == A]) for A in ARCHS})
res["T1"]["reading"] = "visited-state alignment tracks closed-loop gain" if (res["T1"]["rho"] >= 0.41 and res["T1"]["p_perm"] < 0.05) else "not established"
c2 = [c for c in c1 if c.get("A_h") is not None]
res["T2"] = dict(rho_Avis_Ah=pooled(c2, "A_vis", "A_h"), n=len(c2),
                 side_changes=[f"{c['arch']}/{c['game']}: diag {c['A_h']:.3f} -> visited {c['A_vis']:.3f} (T {c['T']:.2f})" for c in c2 if (c["A_h"] >= c["T"]) != (c["A_vis"] >= c["T"])])
sc = next((c for c in cells if c["arch"] == "STORM" and c["game"] == "CrazyClimber"), None)
res["T2"]["storm_crazyclimber"] = None if sc is None or sc["A_vis"] is None else dict(A_vis=sc["A_vis"], T=sc["T"], reading="evaluation-condition effect" if sc["A_vis"] >= 0.19 else "reward-readout effect")
c3 = usable("rank_vis")
res["T3"] = dict(n=len(c3), rho_gain_rank=pooled(c3, "rank_vis"), p_gain_rank=perm_p("rank_vis"), ci_gain_rank=boot("rank_vis"),
                 rho_rank_Avis=pooled([c for c in c3 if c.get("A_vis") is not None], "A_vis", "rank_vis"), p_rank_Avis=perm_p("A_vis", "rank_vis"),
                 ci_rank_Avis=boot("A_vis", "rank_vis"), no_rank=[f"{c['arch']}/{c['game']}" for c in cells if c.get("rank_vis") is None])
c4 = [c for c in c1 if c.get("psnr_vis") is not None]
res["T4"] = dict(n=len(c4), rho_gain_psnr=pooled(c4, "psnr_vis"), p_perm=perm_p("psnr_vis"), ci=boot("psnr_vis"),
                 diff_Avis_minus_psnr=pooled(c4, "A_vis") - pooled(c4, "psnr_vis"), diff_ci=boot("A_vis", minus="psnr_vis"))
res["checks"] = dict(library_maxdiff=max([c["library_check_maxdiff"] for c in cells if c["library_check_maxdiff"] is not None], default=None),
                     incomplete=[f"{c['arch']}/{c['game']}" for c in cells if not c["complete"]],
                     determinism={f"{c['arch']}/{c['game']}": c["determinism"] for c in cells})
if SRC != "outputs": res["source"] = "ext/" + SRC
json.dump(res, open(OUT / ("r6_visited.json" if SRC == "outputs" else "r6_visited_%s.json" % SRC), "w"), indent=1)
for k in ("T1", "T2", "T3", "T4", "checks"): print(k, {kk: vv for kk, vv in res[k].items()})
for c in cells: print(f"{c['arch']:8s}{c['game']:13s} gain {c['gain']:+.2f}  A_h {c['A_h'] if c['A_h'] is None else round(c['A_h'], 3)}  A_vis {c['A_vis'] if c['A_vis'] is None else round(c['A_vis'], 3)}  rank {c['rank_vis']}  ref {c['ref_vis']}  inf {c['n_informative']}  pairs {c['n_pairs']}")
