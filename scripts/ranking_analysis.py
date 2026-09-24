"""Evaluation of the action ranking (rules fixed before the jobs landed): per (arch, game) the median over informative contexts
of Spearman(model return, emulator return), top-1 regret and hit@2, pooled over seeds with bootstrap CI; then across games
Spearman of ranking quality vs causal alignment A(d20), PSNR, dPSNR, with the powered MDE at that n stated. Writes
outputs/ranking_analysis.json."""
import json, pathlib, sys, glob, numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]; OUT = ROOT / "outputs"; sys.path.insert(0, str(ROOT))
from ccaus.joint import bootstrap_median
from ccaus.validate import spearman, permutation_p, mde_rho
DRY = "--dry" in sys.argv
cells = json.load(open(OUT / "dpsnr_cells.json"))["cells"]
res = {}
for f in sorted(OUT.glob("ranking_*_*.json")):
    if ("DRY" in f.name) != DRY: continue
    d = json.load(open(f)); arch, g = d["arch"].upper() + ("_H40" if "_h40" in f.name else ""), d["game"].split("/")[-1].replace("-v5", "")
    ctx = [c for s in d["results"] for c in d["results"][s]["raw"]]; inf = [c for c in ctx if c["informative"]]
    r = dict(n_ctx=len(ctx), n_informative=len(inf),
             spearman=bootstrap_median([spearman(c["model_return"], c["emu_return"]) for c in inf if c["spearman"] is not None]).as_dict() if inf else None,   # recomputed with midranks
             regret=bootstrap_median([c["regret"] for c in inf if c["regret"] is not None]).as_dict() if inf else None,
             hit2=float(np.mean([c["hit2"] for c in inf if c["hit2"] is not None])) if inf else None)
    cell = cells.get(arch.replace("_H40", ""), {}).get(g)
    if cell: r.update(A=cell["A"], err=cell["err"], dpsnr=cell["dpsnr"]["median"] if cell["dpsnr"] else None)
    res.setdefault(arch, {})[g] = r
    print(f"{arch:<8} {g:<12} informative {len(inf)}/{len(ctx)}  rho {r['spearman']['median'] if r['spearman'] else float('nan'):+.2f}  regret {r['regret']['median'] if r['regret'] else float('nan'):.2f}  hit2 {r['hit2'] if r['hit2'] is not None else float('nan'):.2f} | A {r.get('A', float('nan')):+.2f}")
for arch, games in res.items():
    ok = {g: r for g, r in games.items() if r["spearman"] and r["n_informative"] >= 10 and "A" in r}
    if len(ok) >= 4:
        q = [ok[g]["spearman"]["median"] for g in ok]
        out = {}
        for k in ("A", "err", "dpsnr"):
            x = [ok[g][k] for g in ok]
            if any(v is None for v in x): continue
            rho = spearman(x, q); p = permutation_p(x, q, n_perm=5000); out[k] = dict(rho=rho, p=p, n=len(ok))
        out["mde_rho_80"] = mde_rho(len(ok)); res[arch]["_across_games"] = out
        print(f"{arch}: ranking quality vs " + ", ".join(f"{k}: rho {v['rho']:+.2f} (p {v['p']:.3f})" for k, v in out.items() if k != "mde_rho_80") + f" | n {len(ok)}, MDE(80%) {out['mde_rho_80']:.2f}")
json.dump(res, open(OUT / ("ranking_analysis_DRY.json" if DRY else "ranking_analysis.json"), "w"), indent=1); print("wrote ranking_analysis")
