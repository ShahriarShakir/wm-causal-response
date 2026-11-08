"""R6, GPU replication against the CPU run, per cell (post hoc comparison; the R6 tests on the GPU run follow once all 24 cells are in).
For every cell with a complete GPU file in ext/outputs_gpu/: the cell statistics of rule R6 on both devices (A_vis, PSNR_vis, RANK_vis, mean return),
whether A_vis lies on the same side of the architecture's tracked threshold, and how many GPU episodes end with exactly the stored return.
Reads ext/outputs/visited_*.json, ext/outputs_gpu/visited_*.json and rebuttal/outputs/r1_horizon_gain.json; writes rebuttal/outputs/r6_gpu_vs_cpu.json."""
import json, pathlib, numpy as np
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parents[2]; OUT = ROOT / "rebuttal" / "outputs"
T = json.load(open(OUT / "r1_horizon_gain.json"))["thresholds"]
GAMES = ("Alien", "Asterix", "Boxing", "Breakout", "CrazyClimber", "Frostbite", "Qbert", "Seaquest"); ARCHS = ("DIAMOND", "IRIS", "STORM")
DEL = ("1", "2", "3", "5", "8", "12")

def cell(f):
    """Rule R6 cell statistics from one complete file; the library aggregate is the estimator (r6_visited.py checks it equals the recomputation)."""
    d = json.load(open(f)); rows = d["aggregate"]["rows"]
    a = [rows[k]["causal_alignment"]["median"] for k in DEL if np.isfinite(rows[k]["causal_alignment"]["median"])]
    p = [rows[k]["prediction_psnr"]["median"] for k in DEL if np.isfinite(rows[k]["prediction_psnr"]["median"])]
    inf = [r for e in d["episodes"] for r in e["ranking"] if len(set(r["native"])) >= 2]
    rk = [x for x in (spearmanr(r["model"], r["native"]).correlation for r in inf) if x == x]
    return dict(A_vis=float(np.mean(a)) if len(a) >= 4 else None, psnr_vis=float(np.mean(p)) if p else None,
                rank_vis=float(np.median(rk)) if len(inf) >= 5 and rk else None, mean_return=float(np.mean(d["returns"])),
                n_episodes=d["n_episodes_done"], determinism_match=d["determinism_match"], complete=d["complete"], device=d.get("device", "cpu"))

cells = []
for A in ARCHS:
    for g in GAMES:
        fg = ROOT / "ext" / "outputs_gpu" / f"visited_{A.lower()}_{g}.json"; fc = ROOT / "ext" / "outputs" / f"visited_{A.lower()}_{g}.json"
        if not fg.exists(): continue
        gpu, cpu = cell(fg), cell(fc)
        if not gpu["complete"]: continue
        same = None if gpu["A_vis"] is None or cpu["A_vis"] is None else (gpu["A_vis"] >= T[A]) == (cpu["A_vis"] >= T[A])
        cells.append(dict(arch=A, game=g, T=T[A], gpu=gpu, cpu=cpu, same_side=same))
res = {"note": "post hoc comparison of the GPU replication with the CPU run; the R6 tests on the GPU run come once all 24 cells are complete",
       "n_cells": len(cells), "cells": cells,
       "episodes": sum(c["gpu"]["n_episodes"] for c in cells), "episodes_with_stored_return": sum(c["gpu"]["determinism_match"] for c in cells),
       "same_side_of_T": sum(bool(c["same_side"]) for c in cells), "cells_with_A_vis": sum(c["same_side"] is not None for c in cells),
       "max_abs_A_vis_difference": max((abs(c["gpu"]["A_vis"] - c["cpu"]["A_vis"]) for c in cells if c["same_side"] is not None), default=None)}
json.dump(res, open(OUT / "r6_gpu_vs_cpu.json", "w"), indent=1)
print({k: v for k, v in res.items() if k != "cells"})
for c in cells:
    f = lambda x: "-" if x is None else f"{x:.3f}"
    print(f"{c['arch']:8s}{c['game']:13s} A_vis cpu {f(c['cpu']['A_vis'])} gpu {f(c['gpu']['A_vis'])} (T {c['T']:.2f}, same side {c['same_side']}) | stored-return match {c['gpu']['determinism_match']}/{c['gpu']['n_episodes']}")
