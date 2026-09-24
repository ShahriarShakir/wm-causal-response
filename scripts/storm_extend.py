"""Third arm (STORM): the same per-game quantities as for DIAMOND/IRIS and the cross-architecture analyses over THREE pairs.
Rules fixed 2026-09-09 before any STORM measurement landed:
 - chosen files: storm_<G>-v5_dur8_k{k}_w{max(0,w-4)}.json (warm-up shortened by 4; k==3 written without suffix)
 - per game at d20: guarded medians of A, R, err = R^2+1-2A, PSNR, consistency (mean of per-seed aggregate medians, as in §6m)
 - taxonomy: T_STORM = 0.5 x STORM's Pong alignment at d20 (the rule that gave 0.40 for DIAMOND and 0.09 for IRIS: half the tracked flagship);
   tracked = d20 >= T and d20 >= half the d1-8 peak; aligned-then-forgotten = peak >= T and d20 < half peak; never-aligned = peak < T; partial = missing
 - cross-architecture: Spearman over games present in both arms, bootstrap 95% CI, permutation p (20000), Holm over the three pairs, for A, consistency, PSNR
Writes outputs/storm_extend.json."""
import json, pathlib, sys, numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]; OUT = ROOT / "outputs"; sys.path.insert(0, str(ROOT))
from ccaus.validate import spearman, permutation_p, mde_rho, detectable_rho
from ccaus.joint import bootstrap_median
S = json.load(open(OUT / "emulator_intervention_search.json"))
def chosen(prefix, wshift, d=None):
    fs = {}
    for g, e in S.items():
        k, w = int(e["best_offset"]), max(0, int(e["best_warmup"]) - wshift)
        f = (d or OUT) / f"{prefix}_{g}-v5_dur8{'_k' + str(k) if k != 3 else ''}{'_w' + str(w) if w else ''}.json"
        if f.exists(): fs[g] = f
    return fs
ARCH = {"DIAMOND": ("video", 0, OUT), "IRIS": ("iris", 0, OUT), "STORM": ("storm", 4, OUT)}   # the three architectures of the paper (2026-09-15: a fourth arm was closed by decision and is not read)
def per_game(f):
    d = json.load(open(f)); Ss = list(d["results"]); rows = {}
    for dl in ("1", "2", "3", "5", "8", "20"):
        vals = lambda k: [d["results"][s]["aggregate"]["rows"][dl][k]["median"] for s in Ss if dl in d["results"][s]["aggregate"]["rows"]]
        A_raw = vals("causal_alignment"); A = A_raw if (A_raw and all(v == v for v in A_raw)) else []   # ORIGINAL rule: a nan seed-median at a delay makes the cell's value missing -> 'partial'
        rows[dl] = dict(A=float(np.mean(A)) if A else None, R=float(np.mean(vals("causal_magnitude"))), psnr=float(np.mean(vals("prediction_psnr"))), consistency=float(np.mean(vals("consistency"))), tdiv=float(np.mean(vals("true_divergence"))))
        rows[dl]["err"] = (rows[dl]["R"] ** 2 + 1 - 2 * rows[dl]["A"]) if rows[dl]["A"] is not None else None
    return rows
res = {"per_game": {}, "thresholds": {}, "taxonomy": {}, "cross": {}}
for arch, (prefix, ws, d_) in ARCH.items():
    res["per_game"][arch] = {g: per_game(f) for g, f in sorted(chosen(prefix, ws, d_).items())}
# Thresholds (pre-declared 2026-09-12 before the registration-fix rerun landed): DIAMOND keeps its pre-registered 0.40 (its
# measurement is unaffected by the fix); IRIS and STORM use the rule the paper states for STORM, half of Pong's delay-20 alignment
# under the emulator-chosen intervention, because IRIS's old 0.09 was derived from the mis-registered Pong value.
T = {"DIAMOND": 0.40}
for _a in ("IRIS", "STORM"):
    if "Pong" in res["per_game"].get(_a, {}) and res["per_game"][_a]["Pong"]["20"]["A"] is not None: T[_a] = 0.5 * res["per_game"][_a]["Pong"]["20"]["A"]
res["thresholds"] = T
for arch, games in res["per_game"].items():
    if arch not in T: continue
    tax = {}
    for g, r in games.items():
        early = [r[dl]["A"] for dl in ("1", "2", "3", "5", "8")]; d20 = r["20"]["A"]
        peak = max(early) if all(v is not None for v in early) else float("nan")      # ORIGINAL rule: any missing early delay -> partial
        if d20 is None or peak != peak: cls = "partial"
        elif d20 >= T[arch] and d20 >= 0.5 * peak: cls = "tracked"
        elif peak >= T[arch] and d20 < 0.5 * peak: cls = "aligned-then-forgotten"
        elif peak < T[arch]: cls = "never-aligned"
        else: cls = "partial"
        tax[g] = dict(cls=cls, peak=float(peak), d20=d20)
    res["taxonomy"][arch] = tax
    from collections import Counter; print(arch, "T =", round(T[arch], 3), dict(Counter(v["cls"] for v in tax.values())))
pairs = [("DIAMOND", "IRIS"), ("DIAMOND", "STORM"), ("IRIS", "STORM")]
for q in ("A", "consistency", "psnr"):
    out = {}
    for a, b in pairs:
        gs = sorted(set(res["per_game"].get(a, {})) & set(res["per_game"].get(b, {})))
        gs = [g for g in gs if res["per_game"][a][g]["20"][q] is not None and res["per_game"][b][g]["20"][q] is not None]
        if len(gs) < 6: continue
        x = [res["per_game"][a][g]["20"][q] for g in gs]; y = [res["per_game"][b][g]["20"][q] for g in gs]
        rho = spearman(x, y); p = permutation_p(x, y, n_perm=20000); rng = np.random.default_rng(0)
        boots = [spearman([x[i] for i in idx], [y[i] for i in idx]) for idx in (rng.integers(0, len(x), len(x)) for _ in range(2000))]
        out[f"{a}-{b}"] = dict(n=len(gs), rho=rho, p=p, ci=[float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))], mde=mde_rho(len(gs)))
    ps = sorted(out.items(), key=lambda kv: kv[1]["p"])
    run = 0.0   # Holm step-down with the running maximum
    for i, (k, v) in enumerate(ps):
        run = max(run, min(1.0, v["p"] * (len(ps) - i))); v["p_holm"] = run; v["holm_pass"] = run < 0.05
    res["cross"][q] = out
    print(q, {k: (round(v["rho"], 2), [round(c, 2) for c in v["ci"]], round(v["p_holm"], 4)) for k, v in out.items()})
if "STORM" in res["taxonomy"]:
    agree = {f"{a}-{b}": sum(res["taxonomy"][a][g]["cls"] == res["taxonomy"][b][g]["cls"] for g in set(res["taxonomy"][a]) & set(res["taxonomy"][b])) for a, b in pairs if a in res["taxonomy"] and b in res["taxonomy"]}
    res["mode_agreement"] = agree; print("mode agreement:", agree)
# ---- added 2026-09-12 (panel audit): threshold sensitivity of the taxonomy and within-architecture correlations at n = 26 ----
from collections import Counter as _Ctr
res["taxonomy_sensitivity"] = {}
for arch, games in res["per_game"].items():
    if arch not in T: continue
    row = {}
    for mult in (0.5, 1.0, 2.0):
        Tm = T[arch] * mult; cnt = _Ctr()
        for g, r in games.items():
            early = [r[dl]["A"] for dl in ("1", "2", "3", "5", "8")]; d20 = r["20"]["A"]
            peak = max(early) if all(v is not None for v in early) else float("nan")
            if d20 is None or peak != peak: cls = "partial"
            elif d20 >= Tm and d20 >= 0.5 * peak: cls = "tracked"
            elif peak >= Tm and d20 < 0.5 * peak: cls = "aligned-then-forgotten"
            elif peak < Tm: cls = "never-aligned"
            else: cls = "partial"
            cnt[cls] += 1
        row[f"x{mult:g}"] = dict(T=Tm, **{k: cnt[k] for k in ("tracked", "aligned-then-forgotten", "never-aligned", "partial")})
    res["taxonomy_sensitivity"][arch] = row; print(arch, "taxonomy at T/2, T, 2T:", {k: (v["tracked"], v["aligned-then-forgotten"], v["never-aligned"], v["partial"]) for k, v in row.items()})
res["within"] = {}
for arch, games in res["per_game"].items():
    gs = [g for g in games if games[g]["20"]["A"] is not None]
    A_ = [games[g]["20"]["A"] for g in gs]; out = {}
    for q in ("consistency", "psnr"):
        x = [games[g]["20"][q] for g in gs]; rho = spearman(x, A_); p_ = permutation_p(x, A_, n_perm=20000)
        out[f"{q}_vs_A"] = dict(n=len(gs), rho=rho, p=p_, mde=mde_rho(len(gs)), detectable=detectable_rho(len(gs)), powered=bool(abs(rho) >= detectable_rho(len(gs))), detected=bool(abs(rho) >= mde_rho(len(gs))))
    res["within"][arch] = out; print(arch, "within-architecture at n =", len(gs), {k: (round(v["rho"], 2), round(v["p"], 3)) for k, v in out.items()})
# ---- per-evaluation-seed cross-architecture rho at delay 20 (the seed-stability row of the ablation table) ----
def per_seed_d20(prefix, wshift, d_):
    out = {}
    for g, f in chosen(prefix, wshift, d_).items():
        d = json.load(open(f))
        for sd in d["results"]:
            v = d["results"][sd]["aggregate"]["rows"].get("20", {}).get("causal_alignment", {}).get("median")
            if v is not None and v == v: out.setdefault(sd, {})[g] = v
    return out
_ps = {a: per_seed_d20(*ARCH[a]) for a in ("DIAMOND", "IRIS", "STORM")}
res["per_seed_cross"] = {}
for a, b in [("DIAMOND", "IRIS"), ("DIAMOND", "STORM"), ("IRIS", "STORM")]:
    rhos = []
    for sd in sorted(set(_ps[a]) & set(_ps[b]), key=int):
        gs = sorted(set(_ps[a][sd]) & set(_ps[b][sd]))
        if len(gs) >= 10: rhos.append(spearman([_ps[a][sd][g] for g in gs], [_ps[b][sd][g] for g in gs]))
    res["per_seed_cross"][f"{a}-{b}"] = dict(rhos=rhos, mean=float(np.mean(rhos)), sd=float(np.std(rhos)), min=float(min(rhos)), max=float(max(rhos)))
    print(f"per-seed rho {a}-{b}: {[round(r, 2) for r in rhos]} mean {np.mean(rhos):.2f} sd {np.std(rhos):.2f}")
json.dump(res, open(OUT / "storm_extend.json", "w"), indent=1); print("wrote outputs/storm_extend.json")
