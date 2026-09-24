"""Decoupling index (audit 2026-09-08, existing per-pair files only): does the model's change depend on whether the world changes?
Per game and delay: median ||dp|| on cells where the true response is below the guard (the action was inert in the emulator) divided by
median ||dp|| on cells where it was not. Near 0: the model is silent when the action is inert (its change is coupled to the consequence).
Near 1 or above: the model changes the same amount whether or not the world does (its change is triggered by the action input alone).
Requires >= 10 cells of each kind. Correlated with alignment across games. Writes outputs/decoupling.json."""
import json, numpy as np, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]; OUT = ROOT / "outputs"; sys.path.insert(0, str(ROOT))
from ccaus.validate import spearman, permutation_p, mde_rho
from ccaus.joint import bootstrap_median
S = json.load(open(OUT / "emulator_intervention_search.json")); RE = json.load(open(OUT / "response_error.json"))
def chosen(prefix):
    fs = {}
    for g, e in S.items():
        k, w = int(e["best_offset"]), int(e["best_warmup"])
        if prefix == "storm": w = max(0, w - 4)
        f = OUT / f"{prefix}_{g}-v5_dur8{'_k' + str(k) if k != 3 else ''}{'_w' + str(w) if w else ''}.json"
        if f.exists(): fs[g] = f
    return fs
res = {}
for arch, prefix in (("DIAMOND", "video"), ("IRIS", "iris"), ("STORM", "storm")):
    res[arch] = {}
    for g, f in sorted(chosen(prefix).items()):
        d = json.load(open(f)); R = [r for s in d["results"] for r in d["results"][s]["raw"]]; floor = float(np.mean([d["results"][s]["aggregate"]["guard"]["floor"] for s in d["results"]]))
        row = {}
        for dl in ("8", "20"):
            null = [r["model_divergence"][dl] for r in R if r["true_divergence"].get(dl) is not None and r["true_divergence"][dl] == 0.0 and r["model_divergence"].get(dl) is not None]   # EXACT no-change cells only
            small = [r["model_divergence"][dl] for r in R if r["true_divergence"].get(dl) is not None and 0.0 < r["true_divergence"][dl] < floor and r["model_divergence"].get(dl) is not None]
            kept = [r["model_divergence"][dl] for r in R if r["true_divergence"].get(dl) is not None and r["true_divergence"][dl] >= floor and r["true_divergence"][dl] > 0.0 and r["model_divergence"].get(dl) is not None]   # active = above the guard AND a nonzero true change
            assert len(null) + len(small) + len(kept) == sum(1 for r in R if r["true_divergence"].get(dl) is not None and r["model_divergence"].get(dl) is not None), "inert, below-guard and active pairs must partition the pairs"
            if len(null) >= 10 and len(kept) >= 10 and np.median(kept) > 0:
                # bootstrap CI of the ratio of medians
                rng = np.random.default_rng(0); boots = [np.median(rng.choice(null, len(null))) / max(np.median(rng.choice(kept, len(kept))), 1e-9) for _ in range(1000)]
                row[dl] = dict(n_null=len(null), n_small=len(small), n_kept=len(kept), dp_null=float(np.median(null)), dp_small=(float(np.median(small)) if small else None), dp_kept=float(np.median(kept)), index=float(np.median(null) / np.median(kept)), ci=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], exact_zero=True)
        if row:
            row["A"] = (RE[arch].get(g, {}).get("20") or {}).get("align"); row["R"] = (RE[arch].get(g, {}).get("20") or {}).get("magnitude"); res[arch][g] = row
    ok = {g: r for g, r in res[arch].items() if r.get("20") and r.get("A") is not None}
    x = [ok[g]["A"] for g in ok]; y = [ok[g]["20"]["index"] for g in ok]
    res[arch]["_across"] = dict(n=len(ok), rho=spearman(x, y), p=permutation_p(x, y, n_perm=5000), mde=mde_rho(len(ok)))
    print(f"== {arch}: decoupling index at d20 vs alignment: n {len(ok)} rho {res[arch]['_across']['rho']:+.2f} p {res[arch]['_across']['p']:.4f} (MDE {res[arch]['_across']['mde']:.2f})")
    for g in sorted(ok, key=lambda g: ok[g]["A"]): print(f"   {g:<14} A {ok[g]['A']:+.2f} R {ok[g]['R']:.2f} | index {ok[g]['20']['index']:.2f} [{ok[g]['20']['ci'][0]:.2f}, {ok[g]['20']['ci'][1]:.2f}]  (||dp|| inert {ok[g]['20']['dp_null']:.2f} vs active {ok[g]['20']['dp_kept']:.2f}; n {ok[g]['20']['n_null']}/{ok[g]['20']['n_kept']})")
json.dump(res, open(OUT / "decoupling.json", "w"), indent=1); print("wrote outputs/decoupling.json")
