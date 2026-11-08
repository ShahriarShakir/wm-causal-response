"""R6b, post hoc (not part of the R6 rule; the R6 result stays the one in r6_visited.json). Two checks on experiment A:
(1) does the CPU rerun's planner behave like the stored GPU planner (per-cell mean return, seed 42, episodes 0 to 9)?
(2) does the T1 reading depend on the cells where the guard floor is 0 or on which gain is used?
  S1  T1 without the cells whose guard floor is 0;
  S2  T1 with, in those cells only, the floor set to 0.25 of the median of the NONZERO true divergences;
  S3  T1, T3 and T4 against the gain of the CPU rerun itself (10 episodes, same random and oracle baselines).
Reads rebuttal/outputs/r6_visited.json, the visited files, outputs/closed_loop_eval.json and ext/outputs/cl_<arch>_<Game>.json;
writes rebuttal/outputs/r6b_sensitivity.json."""
import json, glob, pathlib, numpy as np
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parents[2]; OUT = ROOT / "rebuttal" / "outputs"
R6 = json.load(open(OUT / "r6_visited.json")); CL = json.load(open(ROOT / "outputs" / "closed_loop_eval.json"))
DEL = ("1", "2", "3", "5", "8", "12"); FRAC = 0.25
def rho(x, y):
    r = spearmanr(x, y).correlation; return float(r) if r == r else float("nan")
def fin(v): return [float(x) for x in v if x is not None and np.isfinite(x)]

def tests(cells, kx, ky="gain", B=10000, Bb=4000):
    """Pooled Spearman, game-level permutation p and game-clustered bootstrap interval; the same procedure as r6_visited.py."""
    rng = np.random.default_rng(0)
    cs = [c for c in cells if c.get(kx) is not None and c.get(ky) is not None]
    obs = rho([c[ky] for c in cs], [c[kx] for c in cs]); gs = sorted({c["game"] for c in cs}); by = {g: [c for c in cs if c["game"] == g] for g in gs}
    hits = 0
    for _ in range(B):
        perm = dict(zip(gs, rng.permutation(gs))); xs, ys = [], []
        for g in gs:
            src = {c["arch"]: c[kx] for c in by[perm[g]]}
            for c in by[g]:
                if c["arch"] in src: xs.append(src[c["arch"]]); ys.append(c[ky])
        if len(xs) > 3 and abs(rho(ys, xs)) >= abs(obs) - 1e-12: hits += 1
    v = []
    for _ in range(Bb):
        s = [c for g in rng.choice(gs, len(gs)) for c in by[g]]; r = rho([c[ky] for c in s], [c[kx] for c in s])
        if r == r: v.append(r)
    return dict(n=len(cs), rho=obs, p_perm=(hits + 1) / (B + 1), ci=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))])

cells = [dict(c) for c in R6["cells"]]
zero = [c for c in cells if c["guard_floor"] == 0.0]
res = {"note": "post hoc; the R6 result is r6_visited.json", "zero_floor_cells": [f"{c['arch']}/{c['game']}" for c in zero]}

# (1) CPU rerun vs stored GPU planner, per cell
cmp = []
for c in cells:
    vis = [e["ret"] for f in glob.glob(str(ROOT / "ext" / "outputs" / f"visited_{c['arch'].lower()}_{c['game']}*.json")) if "DRY" not in f for e in json.load(open(f))["episodes"]]
    st = json.load(open(ROOT / "ext" / "outputs" / f"cl_{c['arch'].lower()}_{c['game']}.json"))
    gpu = st["results"]["42"]["returns"][:10]                             # the stored seed-42 episodes 0 to 9 (GPU)
    b = CL["games"][c["game"]]; rnd, orc = b["random"]["mean"], b["oracle"]["mean"]
    c["gain_cpu"] = (float(np.mean(vis)) - rnd) / (orc - rnd)
    cmp.append(dict(cell=f"{c['arch']}/{c['game']}", cpu_mean=float(np.mean(vis)), gpu_seed42_mean=float(np.mean(gpu)) if gpu else None,
                    gpu_all_mean=b[c["arch"].lower()]["mean"], gain=c["gain"], gain_cpu=c["gain_cpu"]))
res["cpu_vs_gpu"] = dict(cells=cmp, rho_gain_cpu_vs_gain=rho([c["gain"] for c in cells], [c["gain_cpu"] for c in cells]),
                         rho_cpu_mean_vs_gpu_all_mean=rho([x["cpu_mean"] for x in cmp], [x["gpu_all_mean"] for x in cmp]))

# S1: without the zero-floor cells
keep = [c for c in cells if c["guard_floor"] > 0.0]
res["S1_without_zero_floor"] = dict(A_vis=tests(keep, "A_vis"), psnr_vis=tests(keep, "psnr_vis"))

# S2: zero-floor cells with a floor from the nonzero true divergences
s2 = []
for c in cells:
    c2 = dict(c)
    if c["guard_floor"] == 0.0:
        P = [p for f in glob.glob(str(ROOT / "ext" / "outputs" / f"visited_{c['arch'].lower()}_{c['game']}*.json")) if "DRY" not in f
             for e in json.load(open(f))["episodes"] for p in e["pairs"]]
        pos = [t for t in fin([p["true_divergence"].get(d) for p in P for d in DEL]) if t > 0]
        fl = FRAC * float(np.median(pos)); meds, ns = {}, {}
        for d in DEL:
            av = fin([p["causal_alignment"].get(d) for p in P if p["true_divergence"].get(d) is not None and p["true_divergence"][d] >= fl])
            meds[d] = float(np.median(av)) if av else None; ns[d] = len(av)
        vals = [v for v in meds.values() if v is not None]
        c2.update(A_vis=float(np.mean(vals)) if len(vals) >= 4 else None, guard_floor_s2=fl, A_by_delay_s2=meds, n_by_delay_s2=ns)
    s2.append(c2)
res["S2_positive_floor"] = dict(A_vis=tests(s2, "A_vis"), diff_vs_psnr=tests(s2, "A_vis")["rho"] - tests(s2, "psnr_vis")["rho"],
                                cells={f"{c['arch']}/{c['game']}": dict(A_vis=c["A_vis"], floor=c["guard_floor_s2"], n=c["n_by_delay_s2"]) for c in s2 if "guard_floor_s2" in c})

# S3: the CPU rerun's own gain
res["S3_cpu_gain"] = dict(A_vis=tests(cells, "A_vis", "gain_cpu"), psnr_vis=tests(cells, "psnr_vis", "gain_cpu"), rank_vis=tests(cells, "rank_vis", "gain_cpu"))

# D1: Genie's delta-PSNR at the visited states (Reviewer Ya7M, Q4): mean over the R6 delays of the library aggregate's median
for c in cells:
    f = [x for x in glob.glob(str(ROOT / "ext" / "outputs" / f"visited_{c['arch'].lower()}_{c['game']}*.json")) if "DRY" not in x]
    assert len(f) == 1, f                                                # one complete file per cell, so its aggregate covers every pair
    rows = json.load(open(f[0]))["aggregate"]["rows"]; v = [rows[d]["delta_psnr"]["median"] for d in DEL if np.isfinite(rows[d]["delta_psnr"]["median"])]
    c["dpsnr_vis"] = float(np.mean(v)) if len(v) >= 4 else None
res["D1_delta_psnr"] = dict(dpsnr_vis=tests(cells, "dpsnr_vis"), A_vis=tests(cells, "A_vis"), psnr_vis=tests(cells, "psnr_vis"),
                            cells={f"{c['arch']}/{c['game']}": c["dpsnr_vis"] for c in cells})

# D2: the planner's candidate ranking, descriptive: mean Spearman, how often the top candidate is the emulator's best,
# and how often the model scores all 8 candidates the same (informative decisions only)
from scipy.stats import fisher_exact
d2 = {}
for c in cells:
    f = [x for x in glob.glob(str(ROOT / "ext" / "outputs" / f"visited_{c['arch'].lower()}_{c['game']}*.json")) if "DRY" not in x]
    R = [r for e in json.load(open(f[0]))["episodes"] for r in e["ranking"] if len(set(r["native"])) >= 2]
    if not R: d2[f"{c['arch']}/{c['game']}"] = dict(n_informative=0); continue
    best = lambda s, r: r["native"][int(np.argmax(s))] == max(r["native"])
    xs = fin([rho(r["model"], r["native"]) for r in R]); ys = fin([rho(r["true_frames"], r["native"]) for r in R])
    d2[f"{c['arch']}/{c['game']}"] = dict(n_informative=len(R), mean_rho_model=float(np.mean(xs)) if xs else None,
        mean_rho_true_frames=float(np.mean(ys)) if ys else None,
        top1_chance=float(np.mean([sum(v == max(r["native"]) for v in r["native"]) / len(r["native"]) for r in R])),   # a uniformly random pick
        top1_model=float(np.mean([best(r["model"], r) for r in R])), top1_true_frames=float(np.mean([best(r["true_frames"], r) for r in R])),
        frac_constant_model=float(np.mean([len(set(np.round(r["model"], 9))) == 1 for r in R])))
res["D2_ranking"] = d2

# D3: the planning-risk reading given to Reviewer TeeE (W2), recomputed with the visited-state alignment:
# aligned (A_vis >= the architecture's tracked threshold) against a detectable gain (closed_loop_eval.json "useful")
tab = {"aligned": [0, 0], "not_aligned": [0, 0]}                          # [useful, not useful]
for c in cells:
    k = "aligned" if c["A_vis"] >= c["T"] else "not_aligned"; u = CL["games"][c["game"]][c["arch"].lower()]["useful"]
    tab[k][0 if u else 1] += 1
res["D3_planning_risk"] = dict(table=tab, fisher_p=float(fisher_exact([tab["aligned"], tab["not_aligned"]])[1]),
                               not_aligned_cells=[f"{c['arch']}/{c['game']}" for c in cells if c["A_vis"] < c["T"]])
json.dump(res, open(OUT / "r6b_sensitivity.json", "w"), indent=1)
for k, v in res.items():
    if k != "cpu_vs_gpu": print(k, v)
print("cpu_vs_gpu", {k: v for k, v in res["cpu_vs_gpu"].items() if k != "cells"})
for x in cmp: print(f"  {x['cell']:22s} cpu {x['cpu_mean']:7.1f}  gpu42 {x['gpu_seed42_mean']}  gpu_all {x['gpu_all_mean']:7.1f}  gain {x['gain']:+.2f}  gain_cpu {x['gain_cpu']:+.2f}")
