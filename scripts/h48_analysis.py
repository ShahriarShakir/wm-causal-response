"""Pre-declared analysis for the DIAMOND horizon-48 ablation.
Step 1 identity check (d<=20 must reproduce the h24 file: common random numbers), step 2 guarded per-delay
medians with bootstrap CIs over the extended delays, step 3 Alien-throughout / Pong causal horizon / Cliff's delta.
Writes outputs/h48_analysis.json. Never edits its rules after seeing the data."""
import json, pathlib, sys, numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]; OUT = ROOT / "outputs"; sys.path.insert(0, str(ROOT))
from ccaus.joint import bootstrap_median
PAIRS = {"Alien": ("video_Alien-v5_dur8_k14_w60.json", "video_Alien-v5_dur8_k14_w60_h48.json"),
         "Pong": ("video_Pong-v5_dur8_k2.json", "video_Pong-v5_dur8_k2_h48.json")}
BLIND, TRACKED = 0.05, 0.40
def raws(d): return [(s, r) for s in d["results"] for r in d["results"][s]["raw"]]
def cliffs(a, b):
    a, b = np.asarray(a), np.asarray(b); return float(((a[:, None] > b[None, :]).mean() - (a[:, None] < b[None, :]).mean()))
res = {}
for g, (f24, f48) in PAIRS.items():
    d24, d48 = json.load(open(OUT / f24)), json.load(open(OUT / f48))
    D48 = [str(x) for x in d48["delays"]]; D24 = [str(x) for x in d24["delays"]]
    # --- step 1: identity check, per seed and delay <= 20, on the per-episode alignment (same seed, same episode order)
    maxdiff = 0.0; n_cmp = 0
    for s in d48["results"]:
        r24, r48 = d24["results"][s]["raw"], d48["results"][s]["raw"]
        for e24, e48 in zip(r24, r48):
            for dl in D24:
                if int(dl) > 20 or dl not in e24["causal_alignment"] or dl not in e48["causal_alignment"]: continue
                a, b = e24["causal_alignment"][dl], e48["causal_alignment"][dl]
                if a is None or b is None: continue
                maxdiff = max(maxdiff, abs(a - b)); n_cmp += 1
    # --- step 2: the SAME guard as every reported number: the per-seed `aggregate()` rows stored in the file
    #     (per-pair data-calibrated guard, 0.25 x median true divergence); pooled as in the generator (mean over
    #     the 7 per-seed medians), with the per-seed range carried alongside.
    S = list(d48["results"]); rows = {}
    for dl in D48:
        per = [d48["results"][s]["aggregate"]["rows"][dl]["causal_alignment"]["median"] for s in S]
        per = [x for x in per if x == x]
        n = [d48["results"][s]["aggregate"]["rows"][dl]["causal_alignment"]["n"] for s in S]
        cons = [d48["results"][s]["aggregate"]["rows"][dl]["consistency"]["median"] for s in S]
        rows[dl] = dict(valid=bool(per and min(n) >= 30), n_min=min(n), align_mean_of_seed_medians=(float(np.mean(per)) if per else None),
                        align_seed_min=(min(per) if per else None), align_seed_max=(max(per) if per else None),
                        consistency=float(np.mean(cons)), true_div=float(np.mean([d48["results"][s]["aggregate"]["rows"][dl]["true_divergence"]["median"] for s in S])))
    floor = float(np.mean([d48["results"][s]["aggregate"]["guard"]["floor"] for s in S]))
    valid = [dl for dl in D48 if rows[dl]["valid"]]
    med = {dl: rows[dl]["align_mean_of_seed_medians"] for dl in valid}
    below_tracked = [dl for dl in valid if med[dl] < TRACKED]; below_blind = [dl for dl in valid if med[dl] < BLIND]
    res[g] = dict(files=[f24, f48], identity_check=dict(pairs_compared=n_cmp, max_abs_diff=maxdiff, passed=bool(maxdiff < 1e-6)),
                  floor=floor, rows=rows, valid_delays=valid, max_align_over_valid=(max(med.values()) if med else None),
                  first_delay_below_tracked=(below_tracked[0] if below_tracked else None), first_delay_below_blind=(below_blind[0] if below_blind else None),
                  blind_throughout=bool(med and max(med.values()) < BLIND), tracked_through_20=bool(all(med[dl] >= TRACKED for dl in valid if int(dl) <= 20)))
# --- step 3: Cliff's delta Alien vs Pong at d20 and the last valid delay of both
def vals(g, dl): return [r["causal_alignment"][dl] for _, r in raws(json.load(open(OUT / PAIRS[g][1]))) if r["causal_alignment"].get(dl) is not None]
common = [dl for dl in res["Alien"]["valid_delays"] if dl in res["Pong"]["valid_delays"]]
res["cliffs_delta_Pong_minus_Alien"] = {dl: cliffs(vals("Pong", dl), vals("Alien", dl)) for dl in ["20", common[-1]] if dl in common}
json.dump(res, open(OUT / "h48_analysis.json", "w"), indent=1)
for g in PAIRS:
    r = res[g]; print(f"{g}: identity check {'PASSED' if r['identity_check']['passed'] else 'FAILED'} (max |diff| {r['identity_check']['max_abs_diff']:.2e} over {r['identity_check']['pairs_compared']} pairs); floor {r['floor']:.3g}")
    print("   " + "  ".join(f"d{dl}:{'%.2f' % v['align_mean_of_seed_medians']}[{v['align_seed_min']:.2f},{v['align_seed_max']:.2f}]{'' if v['valid'] else '(n/a)'}" for dl, v in r["rows"].items()))
    print(f"   blind_throughout={r['blind_throughout']} tracked_through_20={r['tracked_through_20']} first<0.40: d{r['first_delay_below_tracked']} first<0.05: d{r['first_delay_below_blind']}")
print("Cliff's delta Pong-Alien:", {k: round(v, 3) for k, v in res["cliffs_delta_Pong_minus_Alien"].items()}); print("wrote outputs/h48_analysis.json")
