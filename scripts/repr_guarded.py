"""Guarded representation-probe analysis, for the RSSM file (rssmrepr.json) and IRIS files (irisrepr_*.json).

Representation alignment is a ratio whose denominator is the TRUE representation divergence
||z*_b - z*_a||. Where that is ~0 the ratio is meaningless, so every delay is guarded with the
same data-calibrated rule as pixel space (floor = 0.25 x median over delays) and reported as
valid / invalid. Writes outputs/repr_guarded.json.
"""
import json, glob, pathlib, statistics as st
import numpy as np
OUT = pathlib.Path(__file__).resolve().parents[1] / "outputs"
DL = ["1", "2", "3", "5", "8", "12", "16", "20"]
def guarded(raws, dl_list):
    rtd = {dl: float(np.median([r["repr_true_div"][dl] for r in raws if dl in r["repr_true_div"]])) for dl in dl_list}
    floor = 0.25 * float(np.median(list(rtd.values())))
    out = {}
    for dl in dl_list:
        ok = rtd[dl] >= floor
        ra = float(np.nanmedian([r["repr_align"][dl] for r in raws if dl in r["repr_align"]])) if ok else None
        pa = [r["pixel_align"][dl] for r in raws if dl in r["pixel_align"] and r["pixel_align"][dl] is not None]
        out[dl] = dict(repr_true_div=rtd[dl], valid=bool(ok), repr_align=ra, pixel_align=float(np.nanmedian(pa)) if pa else None)
    return {"floor": floor, "delays": out}
res = {"RSSM": {}, "IRIS": {}, "IRIS_hidden": {}}
f = OUT / "rssmrepr.json"
if f.exists():
    d = json.load(open(f))
    for t, ps in d["results"].items():
        raws = [r for s in ps for r in ps[s]["raw"]]; res["RSSM"][t] = guarded(raws, [dl for dl in DL if dl in raws[0]["repr_align"]])
for f in sorted(OUT.glob("irisrepr_*.json")):
    d = json.load(open(f)); g = d["game"].split("/")[-1].replace("-v5", "")
    raws = [r for s in d["results"] for r in d["results"][s]["raw"]]; res["IRIS"][g] = guarded(raws, DL)
for f in sorted(OUT.glob("irishidden_*.json")):
    if "DRY" in f.name: continue
    d = json.load(open(f)); g = d["game"].split("/")[-1].replace("-v5", "")
    raws = [r for s in d["results"] for r in d["results"][s]["raw"]]; res["IRIS_hidden"][g] = guarded(raws, DL)
json.dump(res, open(OUT / "repr_guarded.json", "w"), indent=1)
for arch, items in res.items():
    for name, g in items.items():
        cells = "  ".join(f"d{dl}:{('%.2f' % v['repr_align']) if v['valid'] else 'n/a'}/{('%.3f' % v['pixel_align']) if v['pixel_align'] is not None else 'nan'}" for dl, v in g["delays"].items() if dl in ("1","5","8","12","20"))
        print(f"{arch:<5} {name:<18} repr/pixel  {cells}")
print("wrote outputs/repr_guarded.json")
