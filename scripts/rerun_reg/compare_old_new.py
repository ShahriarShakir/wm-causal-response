"""Old (mis-registered, archived) vs new (fixed) IRIS/STORM per-game numbers under the emulator-chosen intervention.
Writes outputs/reg_fix_comparison.json and prints the summary that goes into the authoritative file and the paper's appendix."""
import json, pathlib, sys, numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT)); OUT = ROOT / "outputs"; ARC = OUT / "_reg_bug_2026-09-12"
from ccaus.validate import spearman
S = json.load(open(OUT / "emulator_intervention_search.json"))
def fname(d, pre, g, ws):
    e = S[g]; k, w = int(e["best_offset"]), max(0, int(e["best_warmup"]) - ws); return d / f"{pre}_{g}-v5_dur8{'_k' + str(k) if k != 3 else ''}{'_w' + str(w) if w else ''}.json"
def rows(f):
    d = json.load(open(f)); Ss = list(d["results"]); r = {}
    for dl in ("1", "2", "3", "5", "8", "12", "16", "20"):
        A = [d["results"][s]["aggregate"]["rows"][dl]["causal_alignment"]["median"] for s in Ss if dl in d["results"][s]["aggregate"]["rows"]]
        r[dl] = float(np.mean(A)) if A and all(v == v for v in A) else None
    return r
def cls(r, T):
    early = [r[dl] for dl in ("1", "2", "3", "5", "8")]; d20 = r["20"]
    if d20 is None or any(v is None for v in early): return "partial"
    peak = max(early)
    if d20 >= T and d20 >= 0.5 * peak: return "tracked"
    if peak >= T and d20 < 0.5 * peak: return "aligned-then-forgotten"
    if peak < T: return "never-aligned"
    return "partial"
res = {}
for arch, pre, ws in (("IRIS", "iris", 0), ("STORM", "storm", 4)):
    old = {g: rows(fname(ARC, pre, g, ws)) for g in S if fname(ARC, pre, g, ws).exists()}
    new = {g: rows(fname(OUT, pre, g, ws)) for g in S if fname(OUT, pre, g, ws).exists()}
    gs = sorted(set(old) & set(new)); T_old = {"IRIS": 0.09, "STORM": 0.126}[arch]
    T_new = 0.5 * new["Pong"]["20"] if "Pong" in new and new["Pong"]["20"] is not None else T_old
    per = {g: dict(old={dl: old[g][dl] for dl in ("1", "2", "5", "20")}, new={dl: new[g][dl] for dl in ("1", "2", "5", "20")}, cls_old=cls(old[g], T_old), cls_new=cls(new[g], T_new)) for g in gs}
    x = [old[g]["20"] for g in gs if old[g]["20"] is not None and new[g]["20"] is not None]; y = [new[g]["20"] for g in gs if old[g]["20"] is not None and new[g]["20"] is not None]
    x1 = [old[g]["1"] for g in gs if old[g]["1"] is not None and new[g]["1"] is not None]; y1 = [new[g]["1"] for g in gs if old[g]["1"] is not None and new[g]["1"] is not None]
    res[arch] = dict(n=len(gs), T_old=T_old, T_new=T_new, rho_d20=spearman(x, y) if len(x) > 5 else None, rho_d1=spearman(x1, y1) if len(x1) > 5 else None,
                     mean_abs_change_d20=float(np.mean(np.abs(np.array(x) - np.array(y)))) if x else None, mean_abs_change_d1=float(np.mean(np.abs(np.array(x1) - np.array(y1)))) if x1 else None,
                     class_changes=[(g, per[g]["cls_old"], per[g]["cls_new"]) for g in gs if per[g]["cls_old"] != per[g]["cls_new"]], per_game=per)
    print(f"{arch}: n={len(gs)} | d20 old-vs-new rho {res[arch]['rho_d20']} mean|change| {res[arch]['mean_abs_change_d20']} | d1 rho {res[arch]['rho_d1']} mean|change| {res[arch]['mean_abs_change_d1']} | T {T_old:.3f} -> {T_new:.3f} | class changes {len(res[arch]['class_changes'])}: {res[arch]['class_changes']}")
    for g in ("Alien", "Pong"):
        if g in per: print(f"   {g}: d1 {per[g]['old']['1']} -> {per[g]['new']['1']} | d20 {per[g]['old']['20']} -> {per[g]['new']['20']}")
json.dump(res, open(OUT / "reg_fix_comparison.json", "w"), indent=1); print("wrote outputs/reg_fix_comparison.json")
