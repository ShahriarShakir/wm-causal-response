"""Pre-declared evaluation of the IRIS pre-sampling hidden-state probe. Steps, in order:
1. guard (repr_guarded.py, key IRIS_hidden)  2. classify on valid d1-d8 medians with T = 0.09  3. paired per-episode
permutation test hidden - pixel (Holm over games) + bootstrap CI of the hidden median.  Decision rules are in §8a and
are applied verbatim here. Writes outputs/hidden_probe_eval.json. Runs on whatever files have landed and says how many."""
import json, pathlib, sys, subprocess, numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]; OUT = ROOT / "outputs"; sys.path.insert(0, str(ROOT))
from ccaus.repr_probe import classify
from ccaus.joint import bootstrap_median
from ccaus.validate import permutation_p
subprocess.run([sys.executable, str(ROOT / "scripts" / "repr_guarded.py")], check=True, capture_output=True)
G = json.load(open(OUT / "repr_guarded.json"))["IRIS_hidden"]
T, BLIND, EARLY = 0.09, 0.05, ["1", "2", "3", "5", "8"]
files = {f.stem.split("_")[1]: f for f in OUT.glob("irishidden_*.json") if "DRY" not in f.name}
res = {"n_landed": len(files), "games": {}}
pvals = {}
for g, f in sorted(files.items()):
    d = json.load(open(f)); raws = [r for s in d["results"] for r in d["results"][s]["raw"]]
    valid = [dl for dl in EARLY if G[g]["delays"][dl]["valid"]]
    hid = float(np.nanmedian([G[g]["delays"][dl]["repr_align"] for dl in valid])) if valid else float("nan")
    pix = float(np.nanmedian([G[g]["delays"][dl]["pixel_align"] for dl in valid])) if valid else float("nan")
    # per-episode paired difference, averaged over the valid early delays
    diff = [np.nanmean([r["repr_align"][dl] - r["pixel_align"][dl] for dl in valid if r["repr_align"].get(dl) is not None and r["pixel_align"].get(dl) is not None]) for r in raws]
    diff = [x for x in diff if x == x]
    hid_ep = [np.nanmean([r["repr_align"][dl] for dl in valid if r["repr_align"].get(dl) is not None]) for r in raws]; hid_ep = [x for x in hid_ep if x == x]
    ci = bootstrap_median(hid_ep).as_dict() if hid_ep else None
    hid = ci["median"] if ci else hid                      # point estimate and CI from the same per-episode quantity
    # exact sign-flip permutation on the paired differences
    rng = np.random.default_rng(0); obs = float(np.mean(diff)) if diff else float("nan")
    null = [abs(float(np.mean(np.array(diff) * rng.choice([-1, 1], len(diff))))) for _ in range(20000)] if diff else []
    p = float((np.sum(np.array(null) >= abs(obs)) + 1) / (len(null) + 1)) if null else float("nan")
    pvals[g] = p
    res["games"][g] = dict(valid_early_delays=valid, hidden_median=hid, pixel_median=pix, hidden_ci=ci, n_pairs=len(diff), paired_mean_diff=obs, p_raw=p,
                           pixel_blind=bool(pix < BLIND), cls=classify(pix, hid, T) if valid else "no valid delay")
# Holm over landed games
order = sorted(pvals, key=lambda g: pvals[g]); m = len(order)
for i, g in enumerate(order):
    res["games"][g]["p_holm"] = min(1.0, pvals[g] * (m - i)); res["games"][g]["holm_pass"] = bool(res["games"][g]["p_holm"] < 0.05)
blind = {g: r for g, r in res["games"].items() if r["pixel_blind"]}
rep = [g for g, r in blind.items() if r["hidden_median"] >= T and r["hidden_ci"] and r["hidden_ci"]["lo"] > BLIND and r["holm_pass"]]
res["pixel_blind_games"] = sorted(blind); res["replicating_games"] = sorted(rep)
res["decision"] = ("INCOMPLETE (%d/6 landed)" % len(files)) if len(files) < 6 else ("REPLICATES" if len(rep) >= 2 else ("PARTIAL" if len(rep) == 1 else "DOES NOT REPLICATE"))
json.dump(res, open(OUT / "hidden_probe_eval.json", "w"), indent=1)
for g, r in sorted(res["games"].items()):
    print(f"{g:<10} valid d{','.join(r['valid_early_delays'])}: hidden {r['hidden_median']:.3f} CI [{r['hidden_ci']['lo']:.3f},{r['hidden_ci']['hi']:.3f}] pixel {r['pixel_median']:.3f} -> {r['cls']:<24} paired diff {r['paired_mean_diff']:+.3f} p_holm {r['p_holm']:.4f} {'PIXEL-BLIND' if r['pixel_blind'] else ''}")
print("pixel-blind:", res["pixel_blind_games"], "| replicating:", res["replicating_games"], "| decision:", res["decision"]); print("wrote outputs/hidden_probe_eval.json")
