"""Fixed-scorer planning experiment (v14) evaluation, rules fixed before the files landed.
Per (arch, game): scorer gate; median over informative contexts of Spearman(model-scored return, emulator return) and of the CEILING
Spearman(scorer on true frames, emulator return); regret; hit@2. Model quality relative to ceiling = rho_model / rho_ceiling.
Across cells: Spearman of rho_model with causal alignment A(d20), err, PSNR, dPSNR, per architecture and pooled (dependence noted),
with the 80%-power MDE. CLAIM allowed only if tracked games sit near the ceiling, blind games at chance, and the A-correlation's
bootstrap CI excludes 0 while PSNR's and dPSNR's do not. Writes outputs/fsrank_analysis.json."""
import os, json, pathlib, sys, glob, numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]; OUT = ROOT / "outputs"; sys.path.insert(0, str(ROOT))
from ccaus.joint import bootstrap_median
from ccaus.validate import spearman, permutation_p, mde_rho
cells = json.load(open(OUT / "dpsnr_cells.json"))["cells"]; re_ = json.load(open(OUT / "response_error.json"))
SEARCH = json.load(open(OUT / "emulator_intervention_search.json"))
def sustained_A(arch, g):
    """Median guarded alignment at d20 and d40 under the SUSTAINED intervention (DUR = 40), from the dur40 files; None if not landed."""
    k, w = int(SEARCH[g]["best_offset"]), int(SEARCH[g]["best_warmup"]); prefix = {"DIAMOND": "video", "IRIS": "iris", "STORM": "storm"}[arch.replace("_H40", "")]
    if prefix == "storm": w = max(0, w - 4)
    stem = f"{prefix}_{g}-v5_dur40{'_k' + str(k) if k != 3 else ''}{'_w' + str(w) if w else ''}"
    f = next((OUT / n for n in (stem + "_h48.json", stem + ".json") if (OUT / n).exists()), None)   # IRIS (v5) has a fixed horizon 24 and no suffix
    if f is None: return None
    d = json.load(open(f)); R = [r for s_ in d["results"] for r in d["results"][s_]["raw"]]
    floor = float(np.mean([d["results"][s_]["aggregate"]["guard"]["floor"] for s_ in d["results"]])); out = {}
    for dl in ("20", "36", "44"):
        if dl not in R[0]["causal_alignment"]: out[dl] = None; continue
        v = [r["causal_alignment"][dl] for r in R if r["true_divergence"].get(dl) is not None and r["true_divergence"][dl] >= floor and r["causal_alignment"].get(dl) is not None and r["causal_alignment"][dl] == r["causal_alignment"][dl]]
        out[dl] = float(np.median(v)) if len(v) >= 30 else None
    return out
res = {}
for f in sorted(OUT.glob(os.environ.get("FSRANK_GLOB", "fsrank2_*_*_dense.json"))):   # frozen per-game scorer files (switched from fsrank_* on 2026-09-12; FSRANK_GLOB overrides)
    if "DRY" in f.name: continue
    d = json.load(open(f)); arch, g = d["arch"].upper(), d["game"].split("/")[-1].replace("-v5", "")
    r = {"scorer": d["scorer"], "gate": d["scorer"]["gate"]}
    if d["scorer"]["gate"] and d["results"]:
        ctx = [c for s in d["results"] for c in d["results"][s]["raw"]]; inf = [c for c in ctx if c["informative"]]
        r.update(n_ctx=len(ctx), n_informative=len(inf),
                 # per-context correlations are RECOMPUTED from the stored candidate returns with the midrank helper;
                 # the values stored in the files were computed with ordinal ranks and depend on candidate order where returns tie
                 rho_model=bootstrap_median([spearman(c["model_return"], c["emu_return"]) for c in inf]).as_dict() if inf else None,
                 rho_ceiling=bootstrap_median([spearman(c["ceiling_return"], c["emu_return"]) for c in inf]).as_dict() if inf else None,
                 regret=bootstrap_median([c["regret"] for c in inf if c["regret"] is not None]).as_dict() if inf else None,
                 regret_ceiling=bootstrap_median([c["regret_ceiling"] for c in inf if c["regret_ceiling"] is not None]).as_dict() if inf else None,
                 hit2=float(np.mean([c["hit2"] for c in inf])) if inf else None)
    cell = cells.get(arch, {}).get(g); rr = (re_.get(arch, {}).get(g) or {}).get("20")
    if cell: r.update(A=cell["A"], err=cell["err"], dpsnr=cell["dpsnr"]["median"] if cell["dpsnr"] else None)
    if rr: r.update(psnr=rr["psnr"])
    # exploratory secondary predictor, declared 2026-09-07 23:05 before IRIS Frostbite / sustained data landed: the CONSEQUENCE half of the
    # 8-step response (decompose_response; coordinates that changed only later), median over seeds at d20 from the chosen file
    try:
        k_, w_ = int(SEARCH[g]["best_offset"]), int(SEARCH[g]["best_warmup"]); pre = {"DIAMOND": "video", "IRIS": "iris", "STORM": "storm"}[arch.replace("_H40", "")]
        if pre == "storm": w_ = max(0, w_ - 4)
        fc = OUT / f"{pre}_{g}-v5_dur8{'_k' + str(k_) if k_ != 3 else ''}{'_w' + str(w_) if w_ else ''}.json"
        dc = json.load(open(fc)); vals = [dc["results"][s_]["decomposition"]["20"]["consequence_align"] for s_ in dc["results"] if dc["results"][s_].get("decomposition", {}).get("20", {}).get("consequence_align") is not None]
        vals = [v for v in vals if v == v]
        if vals: r["A_cons"] = float(np.median(vals))
    except Exception: pass
    sa = sustained_A(arch, g)
    if sa: r.update(A_sus20=sa["20"], A_sus44=sa["44"], A_sus=(sa["44"] if sa["44"] is not None else sa["20"]))
    res.setdefault(arch, {})[g] = r
    if r["gate"] and r.get("rho_model"):
        print(f"{arch:<8} {g:<10} scorer bal-acc {d['scorer']['balanced_acc']:.2f} | informative {r['n_informative']}/{r['n_ctx']} | rho model {r['rho_model']['median']:+.2f} [{r['rho_model']['lo']:+.2f},{r['rho_model']['hi']:+.2f}]  ceiling {r['rho_ceiling']['median']:+.2f} | regret {r['regret']['median']:.2f} (ceiling {r['regret_ceiling']['median'] if r.get('regret_ceiling') else float('nan'):.2f}) | A {r.get('A', float('nan')):+.2f} err {r.get('err', float('nan')):.2f}")
    else:
        print(f"{arch:<8} {g:<10} scorer gate {'PASS' if r['gate'] else 'FAIL'} (bal-acc {d['scorer']['balanced_acc']:.2f}, classes {d['scorer']['classes']}) | no ranking")
def across(items, label):
    """Cells may be keyed 'ARCH/game' (pooled) or 'game'. Bootstrap CIs resample GAMES (clusters), keeping every architecture's cell of a
    resampled game together, so pooled intervals respect the within-game dependence."""
    ok = {k: v for k, v in items.items() if v.get("rho_model") and v["n_informative"] >= 10 and "A" in v and v.get("rho_ceiling")}
    if len(ok) < 4: return None
    keys = list(ok); q = [ok[k]["rho_model"]["median"] for k in keys]; out = {"n": len(ok), "mde_rho_80": mde_rho(len(ok))}
    games = sorted({k.split("/")[-1] for k in keys}); out["n_games"] = len(games); out["bootstrap"] = "game-clustered"; out["permutation"] = "game-clustered block permutation, two-sided"
    rng = np.random.default_rng(0)
    for key in ("A_sus", "A", "A_cons", "err", "psnr", "dpsnr"):
        x = [ok[k].get(key) for k in keys]
        if any(v is None for v in x): continue
        rho = spearman(x, q)
        # game-clustered permutation: a random bijection of the games is drawn and every cell takes the ranking quality of the
        # same-position cell of the game it is mapped to, so whole games move together; two-sided on |rho|
        by_game = {g: [i for i, k in enumerate(keys) if k.split("/")[-1] == g] for g in games}; prng = np.random.default_rng(1); cnt = 0; NP = 2000
        assert len({len(v) for v in by_game.values()}) == 1, "block permutation needs equal block sizes (every game has every architecture); stratify otherwise"
        for _ in range(NP):
            mp = dict(zip(games, prng.permutation(games))); qp = list(q)
            for g in games:
                src = by_game[mp[g]]
                for j, i in enumerate(by_game[g]): qp[i] = q[src[j]]   # same-position (same-architecture: keys are sorted ARCH/game) cell of the mapped game
            cnt += abs(spearman(x, qp)) >= abs(rho) - 1e-12
        p = float((cnt + 1) / (NP + 1))
        boots = []
        for _ in range(1000):
            gs = rng.choice(games, len(games)); idx = [i for g in gs for i, k in enumerate(keys) if k.split("/")[-1] == g]
            if len(set(idx)) >= 4: boots.append(spearman([x[i] for i in idx], [q[i] for i in idx]))
        out[key] = dict(rho=rho, p=p, ci=[float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))])
    print(f"{label}: n {out['n']} MDE {out['mde_rho_80']:.2f} | " + " ".join(f"{k}: {v['rho']:+.2f} [{v['ci'][0]:+.2f},{v['ci'][1]:+.2f}] p {v['p']:.3f}" for k, v in out.items() if isinstance(v, dict)))
    return out
for arch in list(res): res[arch]["_across"] = across({g: v for g, v in res[arch].items() if not g.startswith("_")}, arch)
pooled = {f"{a}/{g}": v for a in res for g, v in res[a].items() if not g.startswith("_")}
res["_pooled_across"] = across(pooled, "POOLED (games shared across architectures; descriptive)")
# ---- grouped test (pre-declared alternative to the correlation, which is underpowered at n < 10):
#      blind cells (A < 0.05) vs tracked cells (A >= 0.40): relative ranking quality rho_model / rho_ceiling; exact permutation test on the difference of medians;
#      counts: blind cells at chance (rho_model CI includes 0) and tracked cells near the ceiling (rho_model >= 0.5 * rho_ceiling)
def grouped(items, label):
    ok = {k: v for k, v in items.items() if v.get("rho_model") and v["n_informative"] >= 10 and "A" in v and v.get("rho_ceiling") and v["rho_ceiling"]["median"] > 0.2}
    blind = {k: v for k, v in ok.items() if v["A"] < 0.05}; tracked = {k: v for k, v in ok.items() if v["A"] >= 0.40}
    if not blind or not tracked: return None
    rel = lambda v: v["rho_model"]["median"]                    # the model's own ranking correlation (the true-frame scorer value is a REFERENCE, not a ceiling)
    b, t_ = [rel(v) for v in blind.values()], [rel(v) for v in tracked.values()]
    obs = float(np.median(t_) - np.median(b)); pool = b + t_; rng = np.random.default_rng(0); cnt = 0; n_perm = 5000
    for _ in range(n_perm):
        perm = rng.permutation(pool); cnt += (float(np.median(perm[len(b):]) - np.median(perm[:len(b)])) >= obs)
    out = dict(n_blind=len(b), n_tracked=len(t_), rel_blind=b, rel_tracked=t_, median_rel_blind=float(np.median(b)), median_rel_tracked=float(np.median(t_)), diff=obs, p_perm=(cnt + 1) / (n_perm + 1),
               blind_at_chance=sum(1 for v in blind.values() if v["rho_model"]["lo"] <= 0 <= v["rho_model"]["hi"] or v["rho_model"]["median"] < 0.1),
               tracked_near_ceiling=sum(1 for v in tracked.values() if rel(v) >= 0.5 * v["rho_ceiling"]["median"]), blind_cells=sorted(blind), tracked_cells=sorted(tracked))
    print(f"{label} GROUPED: blind {out['n_blind']} cells rel-quality median {out['median_rel_blind']:+.2f} ({out['blind_at_chance']} at chance) | tracked {out['n_tracked']} cells median {out['median_rel_tracked']:+.2f} ({out['tracked_near_ceiling']} near ceiling) | diff {obs:+.2f} p {out['p_perm']:.4f}")
    return out
for arch in [a for a in res if not a.startswith("_")]: res[arch]["_grouped"] = grouped({g: v for g, v in res[arch].items() if not g.startswith("_")}, arch)
res["_pooled_grouped"] = grouped(pooled, "POOLED")
# ---- within-game contrast (pre-declared 2026-09-07 21:40): same game, same scorer, same emulator, only the model differs.
#      For games where BOTH architectures pass the gate: relative quality difference (DIAMOND - IRIS) against alignment difference.
wg = {}
for g in set(res.get("DIAMOND", {})) & set(res.get("IRIS", {})):
    d, i = res["DIAMOND"].get(g), res["IRIS"].get(g)
    if g.startswith("_") or not (d and i and d.get("rho_model") and i.get("rho_model") and d.get("rho_ceiling") and i.get("rho_ceiling")): continue
    rel = lambda v: v["rho_model"]["median"] / v["rho_ceiling"]["median"] if v["rho_ceiling"]["median"] > 0.2 else None
    if rel(d) is None or rel(i) is None: continue
    wg[g] = dict(rel_D=rel(d), rel_I=rel(i), A_D=d.get("A"), A_I=i.get("A"), diff_rel=rel(d) - rel(i), diff_A=(d.get("A") or 0) - (i.get("A") or 0))
    print(f"WITHIN-GAME {g:<10} DIAMOND rel {rel(d):+.2f} (A {d.get('A', float('nan')):+.2f}) | IRIS rel {rel(i):+.2f} (A {i.get('A', float('nan')):+.2f}) | diff rel {wg[g]['diff_rel']:+.2f} vs diff A {wg[g]['diff_A']:+.2f}")
if len(wg) >= 3:
    x = [v["diff_A"] for v in wg.values()]; y = [v["diff_rel"] for v in wg.values()]
    res["_within_game"] = dict(cells=wg, rho=spearman(x, y), p=permutation_p(x, y, n_perm=5000), n=len(wg)); print(f"WITHIN-GAME across {len(wg)} games: rho(diff A, diff rel) {res['_within_game']['rho']:+.2f} p {res['_within_game']['p']:.3f}")
else: res["_within_game"] = dict(cells=wg)
# DIAMOND and IRIS share contexts, candidates and true futures exactly (same episode seeds, same emulator preprocessing); STORM's
# emulator replays use its frame pooling and warm-up shift, so its cells are matched within STORM only. Pooled correlations on the matched pair:
_mc = [(a, g, r) for a in ("DIAMOND", "IRIS") for g, r in res[a].items() if isinstance(r, dict) and r.get("gate") and r.get("rho_model") and r.get("A") is not None]
if len(_mc) >= 10:
    _x = [r["rho_model"]["median"] for _, _, r in _mc]; _games = sorted({g for _, g, _ in _mc}); _out = {"n": len(_mc), "n_games": len(_games), "permutation": "game-clustered block permutation, two-sided", "bootstrap": "game-clustered"}
    _bg = {g: [i for i, (_, gg, _) in enumerate(_mc) if gg == g] for g in _games}; _prng = np.random.default_rng(1)
    for k in ("A", "err", "psnr"):
        _y = [r.get(k) for _, _, r in _mc]
        if all(v is not None for v in _y):
            _rho = spearman(_y, _x); _cnt = 0; _NP = 2000
            for _ in range(_NP):   # game-clustered permutation, as in across()
                _mp = dict(zip(_games, _prng.permutation(_games))); _xp = list(_x)
                for g in _games:
                    _src = _bg[_mp[g]]
                    for j, i in enumerate(_bg[g]): _xp[i] = _x[_src[min(j, len(_src) - 1)]]
                _cnt += abs(spearman(_y, _xp)) >= abs(_rho) - 1e-12
            _boots = []
            for _ in range(1000):
                _gs = _prng.choice(_games, len(_games)); _idx = [i for g in _gs for i in _bg[g]]
                if len(set(_idx)) >= 4: _boots.append(spearman([_y[i] for i in _idx], [_x[i] for i in _idx]))
            _out[k] = dict(rho=_rho, p=float((_cnt + 1) / (_NP + 1)), ci=[float(np.nanpercentile(_boots, 2.5)), float(np.nanpercentile(_boots, 97.5))])
    res["_pooled_matched_DI"] = _out; print("matched DIAMOND+IRIS pooled:", {k: (round(v["rho"], 2), round(v["p"], 3)) for k, v in _out.items() if isinstance(v, dict)})
json.dump(res, open(OUT / "fsrank_analysis.json", "w"), indent=1); print("wrote outputs/fsrank_analysis.json")
