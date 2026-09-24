"""STORM latent probe evaluation (v17 files, controls built in): per game and delay, medians + CI of pixel A, full A, frames-only A,
actions-only A, share; frames-only quotable where share >= 0.25; POSITIVE where frames-only A >= T_STORM (0.5 x Pong d20 from storm_extend)
with CI > 0.05 at a quotable delay <= 8. Writes outputs/storm_hidden_eval.json."""
import json, pathlib, sys, glob, numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]; OUT = ROOT / "outputs"; sys.path.insert(0, str(ROOT))
from ccaus.joint import bootstrap_median
T = (json.load(open(OUT / "storm_extend.json")).get("thresholds", {}).get("STORM", 0.09)) if (OUT / "storm_extend.json").exists() else 0.09
res = {"T": T}
for f in sorted(glob.glob(str(OUT / "stormhidden_*.json"))):
    if "DRY" in f: continue
    d = json.load(open(f)); g = d["game"].split("/")[-1].replace("-v5", ""); raws = [r for s in d["results"] for r in d["results"][s]["raw"]]
    row = {"n": len(raws), "delays": {}}
    for dl in [str(x) for x in d["delays"]]:
        x = {}
        for k in ("pixel_align", "full_align", "frames_only_align", "actions_only_align", "share"):
            v = [r[k][dl] for r in raws if r[k].get(dl) is not None and r[k][dl] == r[k][dl]]
            x[k] = bootstrap_median(v).as_dict() if v else None
        x["quotable"] = bool(x["share"] and x["share"]["median"] >= 0.25); row["delays"][dl] = x
    pix = float(np.nanmedian([row["delays"][dl]["pixel_align"]["median"] for dl in ("1", "2", "3", "5", "8") if row["delays"].get(dl) and row["delays"][dl]["pixel_align"]]))
    pos = [dl for dl in ("1", "2", "3", "5", "8") if row["delays"].get(dl) and row["delays"][dl]["quotable"] and row["delays"][dl]["frames_only_align"] and row["delays"][dl]["frames_only_align"]["median"] >= T and row["delays"][dl]["frames_only_align"]["lo"] > 0.05]
    row["pixel_early"] = pix; row["pixel_blind"] = bool(pix < 0.05); row["positive_delays"] = pos; res[g] = row
    print(f"{g:<10} n {len(raws)} pixel(d1-8) {pix:.3f} {'BLIND' if pix < 0.05 else 'tracked'} | " + " ".join(f"d{dl}: full {row['delays'][dl]['full_align']['median']:.2f} fo {row['delays'][dl]['frames_only_align']['median'] if row['delays'][dl]['frames_only_align'] else float('nan'):.2f}{'' if row['delays'][dl]['quotable'] else '*'} ao {row['delays'][dl]['actions_only_align']['median'] if row['delays'][dl]['actions_only_align'] else float('nan'):.2f} share {row['delays'][dl]['share']['median']:.2f}" for dl in ("1", "3", "8", "20") if row["delays"].get(dl)) + f" -> {'POSITIVE d' + ','.join(pos) if pos else 'no visual tracking at quotable early delays'}")
json.dump(res, open(OUT / "storm_hidden_eval.json", "w"), indent=1); print("wrote outputs/storm_hidden_eval.json")
