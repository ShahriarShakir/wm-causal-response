"""PSNR-threshold sweep of the latency law (reads task_structure_features.json)."""
import json, sys, numpy as np
import pathlib, os; ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); os.chdir(ROOT); from ccaus.validate import spearman, permutation_p, detectable_rho
P = json.load(open("outputs/task_structure_features.json"))
print("Is PSNR>30 cherry-picked? Sweep the threshold; the law must hold across a RANGE, not at one value.\n")
print(f"{'PSNR >':>7}{'n':>4}{'rho':>8}{'p':>8}{'detect':>8}  powered?   within-arch rho (D / I)")
rows = []
for thr in (0, 24, 26, 28, 30, 32, 34):
    W = [p for p in P if p["psnr"] > thr]
    if len(W) < 6: continue
    a = [q["align"] for q in W]; l = [q["latency"] for q in W]
    r, pv, dr = spearman(l, a), permutation_p(l, a), detectable_rho(len(W))
    # game-clustered block permutation: the points are architecture-game cells that share games
    import numpy as _np; games = sorted({q["game"] for q in W}); bg = {g: [i for i, q in enumerate(W) if q["game"] == g] for g in games}; prng = _np.random.default_rng(1); cnt = 0; NP = 20000
    for _ in range(NP):
        mp = dict(zip(games, prng.permutation(games))); ap = list(a)
        for g in games:
            src = bg[mp[g]]
            for j, i in enumerate(bg[g]): ap[i] = a[src[min(j, len(src) - 1)]]
        cnt += abs(spearman(l, ap)) >= abs(r) - 1e-12
    pv_cl = float((cnt + 1) / (NP + 1)); n_games = len(games)
    wa = []
    for arch in ("DIAMOND", "IRIS"):
        Q = [q for q in W if q["arch"] == arch]
        wa.append(spearman([q["latency"] for q in Q], [q["align"] for q in Q]) if len(Q) >= 4 else float("nan"))
    rows.append(dict(thr=thr, n=len(W), n_games=n_games, rho=r, p=pv, p_game_clustered=pv_cl, detectable=dr, powered=abs(r) >= dr, within=wa))
    print(f"{thr:>7}{len(W):>4}{r:>8.3f}{pv:>8.4f}{dr:>8.3f}  {'YES' if abs(r)>=dr else 'no ':>8}   {wa[0]:+.3f} / {wa[1]:+.3f}")
print("\nIMMEDIACY (continuous version of latency: fraction of the d20 effect already present at d1):")
for thr in (0, 30):
    W = [p for p in P if p["psnr"] > thr]; a = [q["align"] for q in W]; i = [q["immediacy"] for q in W]
    print(f"  PSNR>{thr:<3} n={len(W)}: rho(align, immediacy) = {spearman(i,a):+.3f}  p = {permutation_p(i,a):.4f}  detectable {detectable_rho(len(W)):.3f}")
json.dump(rows, open("outputs/latency_law_threshold_sweep.json", "w"), indent=1)
print("\nwrote outputs/latency_law_threshold_sweep.json")
