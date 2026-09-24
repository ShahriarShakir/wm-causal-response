"""Final audit on existing files (2026-09-10): (1) consensus difficulty ranking across the three architectures and its reliability
(Kendall's W); (2) do the causal HORIZONS transfer (Spearman of per-game horizon across pairs)? (3) is the models' action-dependence equal
to the world's per game (cos(dp_b,dp_c) vs cos(dt_b,dt_c) across games, slope and rho)? (4) does an emulator-only property (true response
magnitude at d20, latency) predict consensus difficulty? (5) does PSNR predict off-support energy? Writes outputs/final_audit.json."""
import json, pathlib, sys, numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]; OUT = ROOT / "outputs"; sys.path.insert(0, str(ROOT))
from ccaus.validate import spearman, permutation_p, mde_rho
E = json.load(open(OUT / "storm_extend.json")); SP = json.load(open(OUT / "spec_analysis.json")); RE = json.load(open(OUT / "response_error.json"))
arches = ["DIAMOND", "IRIS", "STORM"]; games = sorted(set.intersection(*[set(E["per_game"][a]) for a in arches]))
A = {a: {g: E["per_game"][a][g]["20"]["A"] for g in games if E["per_game"][a][g]["20"]["A"] is not None} for a in arches}
common = [g for g in games if all(g in A[a] for a in arches)]
res = {"n_games": len(common)}
# (1) Kendall's W over the three rankings + consensus rank
from scipy.stats import rankdata
R = np.array([rankdata([A[a][g] for g in common]) for a in arches]); m, n = R.shape
S = ((R.sum(0) - m * (n + 1) / 2) ** 2).sum(); W = 12 * S / (m ** 2 * (n ** 3 - n))
cons = {g: float(R[:, i].mean()) for i, g in enumerate(common)}
res["kendall_W"] = float(W); res["consensus_rank_by_alignment"] = dict(sorted(cons.items(), key=lambda kv: -kv[1]))
print(f"(1) Kendall's W of the three difficulty rankings over {n} games: {W:.2f}  (1 = identical rankings); hardest (lowest alignment) consensus: {[g for g,_ in sorted(cons.items(), key=lambda kv: kv[1])[:6]]}; easiest: {[g for g,_ in sorted(cons.items(), key=lambda kv: -kv[1])[:6]]}")
# (2) horizon transfer (causal horizon = first delay where A falls below half its d1-8 peak), from per-game rows
def horizon(r):
    early = [r[dl]["A"] for dl in ("1", "2", "3", "5", "8")]; 
    if any(v is None for v in early): return None
    pk = max(early); 
    for dl in ("1", "2", "3", "5", "8", "20"):
        v = r[dl]["A"]
        if v is None: return None
        if pk >= 0.05 and v < 0.5 * pk: return int(dl)
    return 24
H = {a: {g: horizon(E["per_game"][a][g]) for g in common} for a in arches}
res["horizon"] = {}
for a, b in (("DIAMOND", "IRIS"), ("DIAMOND", "STORM"), ("IRIS", "STORM")):
    gs = [g for g in common if H[a][g] is not None and H[b][g] is not None]
    x = [H[a][g] for g in gs]; y = [H[b][g] for g in gs]; rho = spearman(x, y); p = permutation_p(x, y, n_perm=5000)
    res["horizon"][f"{a}-{b}"] = dict(n=len(gs), rho=rho, p=p); print(f"(2) horizon transfer {a}-{b}: rho {rho:+.2f} p {p:.3f} n {len(gs)}")
# (3) action-dependence of the model vs the world, per game
res["action_dependence"] = {}
for a in arches:
    cells = {g: r for g, r in SP[a].items() if not g.startswith("_") and r.get("20") and r["20"].get("cos_pBpC") and r["20"].get("cos_tBtC")}
    x = [cells[g]["20"]["cos_tBtC"]["median"] for g in cells]; y = [cells[g]["20"]["cos_pBpC"]["median"] for g in cells]
    rho = spearman(x, y); slope = float(np.polyfit(x, y, 1)[0]); res["action_dependence"][a] = dict(n=len(x), rho=rho, slope=slope, p=permutation_p(x, y, n_perm=5000))
    print(f"(3) {a}: cos(model responses) vs cos(world responses) across {len(x)} games: rho {rho:+.2f} (p {res['action_dependence'][a]['p']:.3f}), slope {slope:.2f}")
# (4) emulator-only predictors of consensus difficulty
td = {g: RE["DIAMOND"][g]["20"]["true_div"] for g in common if RE["DIAMOND"].get(g, {}).get("20")}
gs = [g for g in common if g in td]; x = [td[g] for g in gs]; y = [cons[g] for g in gs]
res["true_div_vs_consensus"] = dict(n=len(gs), rho=spearman(x, y), p=permutation_p(x, y, n_perm=5000), mde=mde_rho(len(gs)))
print(f"(4) true response magnitude ||dt|| at d20 vs consensus difficulty rank: rho {res['true_div_vs_consensus']['rho']:+.2f} p {res['true_div_vs_consensus']['p']:.3f} (n {len(gs)}, MDE {res['true_div_vs_consensus']['mde']:.2f})")
# (5) does accuracy predict misplacement?
res["psnr_vs_offsupport"] = {}
for a in arches:
    cells = {g: r for g, r in SP[a].items() if not g.startswith("_") and r.get("20") and r["20"].get("offsupport") and r["20"].get("psnr_B")}
    x = [cells[g]["20"]["psnr_B"]["median"] for g in cells]; y = [cells[g]["20"]["offsupport"]["median"] for g in cells]
    res["psnr_vs_offsupport"][a] = dict(n=len(x), rho=spearman(x, y), p=permutation_p(x, y, n_perm=5000)); print(f"(5) {a}: PSNR vs off-support energy: rho {res['psnr_vs_offsupport'][a]['rho']:+.2f} p {res['psnr_vs_offsupport'][a]['p']:.3f}")
json.dump(res, open(OUT / "final_audit.json", "w"), indent=1); print("wrote outputs/final_audit.json")
