"""R7b, post hoc and descriptive (not part of the R7 rule; the R7 result is r7_goal.json): goal reaching by the paper's taxonomy class, and in the
cells where Delta-PSNR's interval lies above zero while delay-20 alignment is below 0.05 (the lists in outputs/dpsnr_cells.json).
Reads rebuttal/outputs/r7_goal.json, rebuttal/outputs/r2_shift_tolerance.json (the classes of Table 7) and outputs/dpsnr_cells.json;
writes rebuttal/outputs/r7b_goal_posthoc.json."""
import json, pathlib, numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]; OUT = ROOT / "rebuttal" / "outputs"
R7 = json.load(open(OUT / "r7_goal.json")); CLS = {k: v["cls"] for k, v in json.load(open(OUT / "r2_shift_tolerance.json"))["cells"].items()}
DP = json.load(open(ROOT / "outputs" / "dpsnr_cells.json"))["cells"]
G = {f"{c['arch']}/{c['game']}": c["G"] for c in R7["cells"] if c["G"] is not None}
by = {}
for k, g in G.items(): by.setdefault(CLS.get(k, "unknown"), []).append(g)
dis = {f"{a}/{g}": G.get(f"{a}/{g}") for a in ("DIAMOND", "IRIS", "STORM") for g in (DP.get(f"{a}_dpsnr_positive_but_blind") or [])}
measurable = {k: v for k, v in dis.items() if k != "STORM/CrazyClimber"}      # the paper: CrazyClimber's Delta-PSNR is 0.00 dB, the other seven are measurable
res = {"note": "post hoc, descriptive", "by_class": {k: dict(n=len(v), median_G=float(np.median(v))) for k, v in by.items()},
       "dissent_cells": dis, "dissent_measurable_range": [min(measurable.values()), max(measurable.values())]}
json.dump(res, open(OUT / "r7b_goal_posthoc.json", "w"), indent=1)
print(res)
