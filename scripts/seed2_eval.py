"""STORM training-seed variability (ext arm A; rule fixed before running): per game, d20 causal alignment (mean of per-seed evaluation medians, as in
§6m) for the seed-1 checkpoint (outputs/storm_*) and the seed-2 checkpoint (ext/outputs/storm_*). Reports Spearman between the two rankings over
26 games with bootstrap CI and permutation p; the taxonomy class under both seeds (same rule, T = half of each seed's Pong alignment); flagship
(Alien blind < 0.05 and Pong tracked) under seed 2; and the cross-architecture transfer recomputed with seed 2 in place of seed 1.
Writes outputs/seed2_eval.json."""
import json, pathlib, sys, glob, numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]; OUT = ROOT / "outputs"; EXT = ROOT / "ext" / "outputs"; sys.path.insert(0, str(ROOT))
from ccaus.validate import spearman, permutation_p
S = json.load(open(OUT / "emulator_intervention_search.json")); E = json.load(open(OUT / "storm_extend.json"))
def per_game(f):
    d = json.load(open(f)); Ss = list(d["results"]); rows = {}
    for dl in ("1", "2", "3", "5", "8", "20"):
        A = [d["results"][s]["aggregate"]["rows"][dl]["causal_alignment"]["median"] for s in Ss if dl in d["results"][s]["aggregate"]["rows"]]
        rows[dl] = float(np.mean(A)) if A and all(v == v for v in A) else None
    return rows
def files(base):
    fs = {}
    for g, e in S.items():
        k, w = int(e["best_offset"]), max(0, int(e["best_warmup"]) - 4)
        f = base / f"storm_{g}-v5_dur8{'_k' + str(k) if k != 3 else ''}{'_w' + str(w) if w else ''}.json"
        if f.exists(): fs[g] = f
    return fs
s1 = {g: per_game(f) for g, f in files(OUT).items()}; s2 = {g: per_game(f) for g, f in files(EXT).items()}
common = sorted(g for g in s1 if g in s2 and s1[g]["20"] is not None and s2[g]["20"] is not None)
res = {"n_games": len(common), "seed1": {g: s1[g]["20"] for g in common}, "seed2": {g: s2[g]["20"] for g in common}}
if len(common) >= 6:
    x = [s1[g]["20"] for g in common]; y = [s2[g]["20"] for g in common]; rho = spearman(x, y); p = permutation_p(x, y, n_perm=20000)
    rng = np.random.default_rng(0); boots = [spearman([x[i] for i in idx], [y[i] for i in idx]) for idx in (rng.integers(0, len(x), len(x)) for _ in range(2000))]
    res["seed1_vs_seed2"] = dict(rho=rho, p=p, ci=[float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))])
    print(f"seed1 vs seed2 alignment ranking over {len(common)} games: rho {rho:+.2f} [{res['seed1_vs_seed2']['ci'][0]:+.2f}, {res['seed1_vs_seed2']['ci'][1]:+.2f}] p {p:.4f}")
    def tax(rows, T):
        early = [rows[dl] for dl in ("1", "2", "3", "5", "8")]; d20 = rows["20"]
        if any(v is None for v in early) or d20 is None: return "partial"
        pk = max(early)
        if d20 >= T and d20 >= 0.5 * pk: return "tracked"
        if pk >= T and d20 < 0.5 * pk: return "aligned-then-forgotten"
        if pk < T: return "never-aligned"
        return "partial"
    T1 = E["thresholds"]["STORM"]; T2 = 0.5 * s2["Pong"]["20"] if "Pong" in s2 and s2["Pong"]["20"] else T1
    c1 = {g: tax(s1[g], T1) for g in common}; c2 = {g: tax(s2[g], T2) for g in common}
    from collections import Counter
    res["taxonomy_seed1"] = dict(Counter(c1.values())); res["taxonomy_seed2"] = dict(Counter(c2.values())); res["mode_agreement"] = sum(c1[g] == c2[g] for g in common); res["T_seed2"] = T2
    print(f"taxonomy seed1 {res['taxonomy_seed1']} | seed2 (T {T2:.3f}) {res['taxonomy_seed2']} | same class {res['mode_agreement']}/{len(common)}")
    res["flagship_seed2"] = dict(Alien=s2.get("Alien", {}).get("20"), Pong=s2.get("Pong", {}).get("20")); print("flagship seed2: Alien", res["flagship_seed2"]["Alien"], "Pong", res["flagship_seed2"]["Pong"])
    for other in ("DIAMOND", "IRIS"):
        gs = [g for g in common if E["per_game"][other].get(g, {}).get("20", {}).get("A") is not None]
        xo = [E["per_game"][other][g]["20"]["A"] for g in gs]; y2 = [s2[g]["20"] for g in gs]; r2 = spearman(xo, y2); p2 = permutation_p(xo, y2, n_perm=20000)
        res[f"{other}-STORMseed2"] = dict(n=len(gs), rho=r2, p=p2); print(f"transfer {other}-STORM(seed2): rho {r2:+.2f} p {p2:.4f} (seed1 was {E['cross']['A'][f'{other}-STORM']['rho']:+.2f})")
json.dump(res, open(OUT / "seed2_eval.json", "w"), indent=1); print("wrote outputs/seed2_eval.json")
