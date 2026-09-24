"""Frames-only control for the IRIS pre-sampling probe (v13), the DECIDING control (rule fixed before the files landed):
per game and delay, pooled over seeds and episodes: full alignment (the probe), frames-only alignment (true hidden change with the
ACTION tokens held at arm a), actions-only alignment, share = ||frames-only change|| / ||full change||, cos(full, frames-only).
frames-only alignment is quotable only where share >= 0.25. POSITIVE for a game if at some quotable early delay (d <= 8)
frames-only A >= 0.09 with bootstrap CI lower bound > 0.05. Writes outputs/hidden_fo_eval.json."""
import json, pathlib, sys, glob, numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]; OUT = ROOT / "outputs"; sys.path.insert(0, str(ROOT))
from ccaus.joint import bootstrap_median
PIX = json.load(open(OUT / "hidden_probe_eval.json"))["games"]
res = {}
for f in sorted(glob.glob(str(OUT / "irishiddenfo_*.json"))):
    if "DRY" in f: continue
    d = json.load(open(f)); g = d["game"].split("/")[-1].replace("-v5", ""); raws = [r for s in d["results"] for r in d["results"][s]["raw"]]
    res[g] = {"n": len(raws), "pixel_blind": PIX.get(g, {}).get("pixel_blind"), "pixel_A": PIX.get(g, {}).get("pixel_median"), "delays": {}}
    for dl in [str(x) for x in d["delays"]]:
        row = {}
        for k in ("full_align", "frames_only_align", "actions_only_align", "share", "cos_full_vs_frames_only"):
            v = [r[k][dl] for r in raws if r[k].get(dl) is not None and r[k][dl] == r[k][dl]]
            row[k] = bootstrap_median(v).as_dict() if v else None
        row["quotable"] = bool(row["share"] and row["share"]["median"] >= 0.25)
        res[g]["delays"][dl] = row
    early = [dl for dl in res[g]["delays"] if int(dl) <= 8 and res[g]["delays"][dl]["quotable"]]
    pos = [dl for dl in early if res[g]["delays"][dl]["frames_only_align"] and res[g]["delays"][dl]["frames_only_align"]["median"] >= 0.09 and res[g]["delays"][dl]["frames_only_align"]["lo"] > 0.05]
    res[g]["quotable_early_delays"] = early; res[g]["positive_delays"] = pos; res[g]["positive"] = bool(pos)
    print(f"{g:<10} n {len(raws)} pixel A {res[g]['pixel_A']:.3f} {'BLIND' if res[g]['pixel_blind'] else 'tracked'} | " + " ".join(
        f"d{dl}: full {r['full_align']['median']:.2f} fo {r['frames_only_align']['median'] if r['frames_only_align'] else float('nan'):.2f}{'' if r['quotable'] else '*'} ao {r['actions_only_align']['median'] if r['actions_only_align'] else float('nan'):.2f} share {r['share']['median']:.2f}" for dl, r in res[g]["delays"].items() if dl in ("1", "2", "3", "5", "8", "20")) + f" -> {'POSITIVE at d' + ','.join(pos) if pos else 'no visual tracking at quotable early delays'}")
blind_pos = [g for g, r in res.items() if r["pixel_blind"] and r["positive"]]; tracked_pos = [g for g, r in res.items() if not r["pixel_blind"] and r["positive"]]
res["_summary"] = {"blind_games_positive": blind_pos, "tracked_games_positive": tracked_pos,
                   "verdict": ("REPLICATES with visual evidence" if len(blind_pos) >= 2 else ("PARTIAL" if blind_pos else "the pre-sampling 'decoding failure' is the action token's echo")) }
print("blind games positive:", blind_pos, "| tracked games positive:", tracked_pos, "| verdict:", res["_summary"]["verdict"])
json.dump(res, open(OUT / "hidden_fo_eval.json", "w"), indent=1); print("wrote outputs/hidden_fo_eval.json")
