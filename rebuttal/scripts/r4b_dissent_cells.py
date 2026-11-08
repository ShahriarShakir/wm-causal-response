"""R4b, post hoc extension of R4 (rule in rebuttal/README.md): the same worked-example statistics for EVERY cell in which Delta-PSNR's
interval lies above zero while delay-20 alignment is below 0.05 (the lists in outputs/dpsnr_cells.json, as in the paper's appendix), not only the
two closed-loop ones. Per cell, over its delay-20 audit panels: median alignment, median magnitude (model change over world change) and the median
share of the model's change energy outside the pixels where the world changes; one example episode rendered per cell (closest to the median).
Reads ext/object_audit/panels30.tar.gz, ext/outputs/objaudit30_*.json and outputs/dpsnr_cells.json; writes rebuttal/outputs/r4b_dissent_cells.json
and rebuttal/figs/dissent_<arch>_<Game>_ep<k>.png."""
import glob, io, json, pathlib, tarfile
import numpy as np, cv2
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[2]; OUT = ROOT / "rebuttal" / "outputs"; FIG = ROOT / "rebuttal" / "figs"; FIG.mkdir(exist_ok=True)
DP = json.load(open(ROOT / "outputs" / "dpsnr_cells.json"))["cells"]
want = {(a.lower(), g) for a in ("DIAMOND", "IRIS", "STORM") for g in (DP.get(f"{a}_dpsnr_positive_but_blind") or [])}
assert want, "no dissent cells listed"
# ---- the R4 computation of r1b_r1c_r4_r5.py, unchanged ----
IMG = 64; idx = np.arange(IMG * IMG, dtype=np.int32).reshape(IMG, IMG); up_idx = cv2.resize(idx, (160, 210), interpolation=cv2.INTER_NEAREST); pos = {}
for y in range(210):
    for x in range(160): pos.setdefault(int(up_idx[y, x]), (y, x))
PY = np.array([pos[i][0] for i in range(IMG * IMG)]).reshape(IMG, IMG); PX = np.array([pos[i][1] for i in range(IMG * IMG)]).reshape(IMG, IMG)
KEY = {}
for f in glob.glob(str(ROOT / "ext" / "outputs" / "objaudit30_*.json")):
    if "DRY" in f: continue
    for pid, k in json.load(open(f))["panel_key"].items():
        if (k["arch"], k["game"]) in want: KEY[pid] = k
recs = []
with tarfile.open(ROOT / "ext" / "object_audit" / "panels30.tar.gz") as tf:
    for m in tf:
        pid = pathlib.Path(m.name).stem
        if pid not in KEY: continue
        im = np.array(Image.open(io.BytesIO(tf.extractfile(m).read())).convert("RGB")); t = [im[:, i * 480:(i + 1) * 480][::3, ::3] for i in range(4)]
        ta, tb = (cv2.resize(x, (IMG, IMG), interpolation=cv2.INTER_AREA).astype(float) for x in t[:2]); pa, pb = (x[PY, PX].astype(float) for x in t[2:])
        dt, dp = tb - ta, pb - pa; sup = np.abs(dt).sum(-1) > 10
        if sup.sum() < 4: continue
        den = (dt * dt).sum(); recs.append(dict(cell=f"{KEY[pid]['arch']}/{KEY[pid]['game']}", ep=KEY[pid]["episode"], A=float((dp * dt).sum() / den),
                                                R=float(np.sqrt((dp * dp).sum() / den)), off=float((dp[~sup] ** 2).sum() / max((dp ** 2).sum(), 1e-9)), fr=(ta, tb, pa, pb, dt, dp)))
res = {"note": "post hoc extension of R4 to every Delta-PSNR-dissent cell", "cells": {}}
for cell in sorted({r["cell"] for r in recs}):
    rs = [r for r in recs if r["cell"] == cell]; mA, mR = np.median([r["A"] for r in rs]), np.median([r["R"] for r in rs])
    ex = min(rs, key=lambda r: abs(r["A"] - mA) + 0.05 * abs(r["R"] - mR)); ta, tb, pa, pb, dt, dp = ex["fr"]
    def mag(d): m = np.abs(d).sum(-1); return np.repeat((255 * m / max(m.max(), 1e-9))[..., None], 3, -1)
    img = np.concatenate([cv2.resize(np.clip(x, 0, 255).astype(np.uint8), (192, 192), interpolation=cv2.INTER_NEAREST) for x in (ta, tb, pa, pb, mag(dt), mag(dp))], 1)
    a, g = cell.split("/"); fn = FIG / f"dissent_{a}_{g}_ep{ex['ep']}.png"; cv2.imwrite(str(fn), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    dps = (DP.get(a.upper(), {}).get(g, {}) or {})
    res["cells"][cell] = dict(n=len(rs), median_A=float(mA), median_R=float(mR), median_off_support=float(np.median([r["off"] for r in rs])),
                              example=dict(episode=ex["ep"], A=ex["A"], R=ex["R"], off_support=ex["off"], figure=str(fn.relative_to(ROOT))),
                              paper_dpsnr=dps.get("dpsnr"), paper_A20=dps.get("A"))
res["summary"] = dict(n_cells=len(res["cells"]), median_R_range=[min(v["median_R"] for v in res["cells"].values()), max(v["median_R"] for v in res["cells"].values())],
                      median_off_range=[min(v["median_off_support"] for v in res["cells"].values()), max(v["median_off_support"] for v in res["cells"].values())],
                      median_A_range=[min(v["median_A"] for v in res["cells"].values()), max(v["median_A"] for v in res["cells"].values())])
json.dump(res, open(OUT / "r4b_dissent_cells.json", "w"), indent=1)
print(res["summary"])
for c, v in res["cells"].items():
    d = v["paper_dpsnr"] or {}
    print(f"{c:22s} n {v['n']:2d}  A {v['median_A']:+.3f}  R {v['median_R']:.2f}  off {v['median_off_support']:.2f}  dPSNR {d.get('median', float('nan')):+.2f} [{d.get('lo', float('nan')):+.2f}, {d.get('hi', float('nan')):+.2f}]  fig {v['example']['figure']}")
