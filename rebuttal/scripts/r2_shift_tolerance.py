"""R2 (rule in rebuttal/README.md): does a few-pixel position tolerance remove the low alignment?
Reads the delay-20 audit panels straight from ext/object_audit/panels30.tar.gz (no extraction), the panel keys in
ext/outputs/objaudit30_*.json and the sweep files for the per-cell classes; writes rebuttal/outputs/r2_shift_tolerance.json."""
import io, json, glob, tarfile, pathlib, numpy as np, cv2
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parents[2]; OUT = ROOT / "rebuttal" / "outputs"; OUT.mkdir(parents=True, exist_ok=True)
SIGMAS = (0, 1, 2, 4); MIN_CHANGED_PX = 4; MIN_EPS = 5; IMG = 64
E = json.load(open(ROOT / "outputs" / "storm_extend.json")); T = E["thresholds"]
S = json.load(open(ROOT / "outputs" / "emulator_intervention_search.json"))
PREFIX = {"DIAMOND": ("video", 0), "IRIS": ("iris", 0), "STORM": ("storm", 4)}
ALL_D = ("1", "2", "3", "5", "8", "12", "16", "20")

# ---- per-cell curves and classes: estimator and rules of scripts/make_base_tables.py (copied, not imported) ----
def fname(arch, g, k, w):
    pre, ws = PREFIX[arch]; w = max(0, w - ws); return ROOT / "outputs" / f"{pre}_{g}-v5_dur8{'_k' + str(k) if k != 3 else ''}{'_w' + str(w) if w else ''}.json"
def curve(arch, g):
    d = json.load(open(fname(arch, g, int(S[g]["best_offset"]), int(S[g]["best_warmup"])))); Ss = list(d["results"]); r = {}
    for dl in ALL_D:
        A = [d["results"][s]["aggregate"]["rows"][dl]["causal_alignment"]["median"] for s in Ss if dl in d["results"][s]["aggregate"]["rows"]]
        r[dl] = float(np.mean(A)) if (A and all(v == v for v in A)) else None
    return r
def classify(r, t):
    early = [r[dl] for dl in ("1", "2", "3", "5", "8")]; d20 = r["20"]
    peak = max(early) if all(v is not None for v in early) else float("nan")
    if d20 is None or peak != peak: return "partial"
    if d20 >= t and d20 >= 0.5 * peak: return "tracked"
    if peak >= t and d20 < 0.5 * peak: return "aligned-then-forgotten"
    if peak < t: return "never-aligned"
    return "partial"
CLS, A20 = {}, {}
for A in ("DIAMOND", "IRIS", "STORM"):
    for g in S:
        r = curve(A, g); CLS[(A, g)] = classify(r, T[A]); A20[(A, g)] = r["20"]
from collections import Counter
counts = {A: dict(Counter(CLS[(a, g)] for (a, g) in CLS if a == A)) for A in ("DIAMOND", "IRIS", "STORM")}
print("class counts (must match Table 7):", counts)

# ---- exact inverse of the harness's nearest upsampling 64x64 -> 210x160 (v23_object_audit.up) ----
idx = np.arange(IMG * IMG, dtype=np.int32).reshape(IMG, IMG)
up_idx = cv2.resize(idx, (160, 210), interpolation=cv2.INTER_NEAREST)
pos = {}
for y in range(210):
    for x in range(160): pos.setdefault(int(up_idx[y, x]), (y, x))
assert len(pos) == IMG * IMG, "nearest upsampling does not cover every source pixel"
PY = np.array([pos[i][0] for i in range(IMG * IMG)]).reshape(IMG, IMG); PX = np.array([pos[i][1] for i in range(IMG * IMG)]).reshape(IMG, IMG)
def down64(up_img): return up_img[PY, PX]
def to64(native): return cv2.resize(native, (IMG, IMG), interpolation=cv2.INTER_AREA).astype(np.float64)

KEY = {}
for f in glob.glob(str(ROOT / "ext" / "outputs" / "objaudit30_*.json")):
    if "DRY" in f: continue
    for pid, k in json.load(open(f))["panel_key"].items(): KEY[pid] = k
ARCH = {"diamond": "DIAMOND", "iris": "IRIS", "storm": "STORM"}

def align(dp, dt, s):
    if s: dp, dt = (np.stack([gaussian_filter(x[..., c], s, mode="constant") for c in range(3)], -1) for x in (dp, dt))
    den = float((dt * dt).sum()); return float((dp * dt).sum() / den) if den > 0 else None

per_cell, n_roundtrip = {}, 0
with tarfile.open(ROOT / "ext" / "object_audit" / "panels30.tar.gz") as tf:
    for m in tf:
        if not m.name.endswith(".png"): continue
        pid = pathlib.Path(m.name).stem; k = KEY.get(pid)
        if k is None: continue
        im = np.array(Image.open(io.BytesIO(tf.extractfile(m).read())).convert("RGB"))
        tiles = [im[:, i * 480:(i + 1) * 480][::3, ::3] for i in range(4)]          # 3x nearest -> exact 210x160 tiles
        ta, tb = to64(tiles[0]), to64(tiles[1])
        pa64, pb64 = down64(tiles[2]), down64(tiles[3])
        for p64, t in ((pa64, tiles[2]), (pb64, tiles[3])):                           # round trip must be exact
            assert np.array_equal(cv2.resize(p64, (160, 210), interpolation=cv2.INTER_NEAREST), t); n_roundtrip += 1
        dt = tb - ta; dp = pb64.astype(np.float64) - pa64.astype(np.float64)
        if int((np.abs(dt).sum(-1) > 10).sum()) < MIN_CHANGED_PX: continue
        cell = (ARCH[k["arch"]], k["game"])
        per_cell.setdefault(cell, []).append({s: align(dp, dt, s) for s in SIGMAS})

res = {"rule": "rebuttal/README.md R2", "sigmas": SIGMAS, "thresholds": T, "class_counts": counts, "roundtrip_checked": n_roundtrip, "cells": {}}
for (A, g), eps in per_cell.items():
    if len(eps) < MIN_EPS: continue
    res["cells"][f"{A}/{g}"] = dict(cls=CLS[(A, g)], n=len(eps), paper_A20=A20[(A, g)], T=T[A],
                                    med={str(s): float(np.median([e[s] for e in eps if e[s] is not None])) for s in SIGMAS})
C = res["cells"]
keys = [c for c in C if C[c]["paper_A20"] is not None]
gate = float(spearmanr([C[c]["med"]["0"] for c in keys], [C[c]["paper_A20"] for c in keys]).correlation)
res["sanity_rho"] = gate; res["sanity_pass"] = gate >= 0.70
res["sanity_rho_per_arch"] = {A: float(spearmanr([C[c]["med"]["0"] for c in keys if c.startswith(A + "/")], [C[c]["paper_A20"] for c in keys if c.startswith(A + "/")]).correlation) for A in ("DIAMOND", "IRIS", "STORM")}
summ = {}
for cls in ("tracked", "aligned-then-forgotten", "never-aligned", "partial"):
    cs = [c for c in C if C[c]["cls"] == cls]
    summ[cls] = dict(n_cells=len(cs), **{f"median_sigma{s}": float(np.median([C[c]["med"][str(s)] for c in cs])) if cs else None for s in SIGMAS},
                     **{f"reach_T_sigma{s}": int(sum(C[c]["med"][str(s)] >= C[c]["T"] for c in cs)) for s in SIGMAS})
res["by_class"] = summ
na = summ["never-aligned"]; n = na["n_cells"]
if not res["sanity_pass"]: res["reading"] = "failed reconstruction: supports no claim"
elif na["reach_T_sigma2"] >= n / 2: res["reading"] = "SUPPORTED: low alignment is mostly a few-pixel misplacement"
elif na["reach_T_sigma4"] < n / 4: res["reading"] = "REJECTED: low alignment is not a few-pixel misplacement"
else: res["reading"] = "PARTIAL"
json.dump(res, open(OUT / "r2_shift_tolerance.json", "w"), indent=1)
print(f"round trips exact: {n_roundtrip}; usable cells: {len(C)}; sanity rho {gate:.2f} per arch {res['sanity_rho_per_arch']}")
for cls, v in summ.items(): print(cls, v)
print("never-aligned cells reaching T at sigma 4:", [c for c in C if C[c]["cls"] == "never-aligned" and C[c]["med"]["4"] >= C[c]["T"]])
print("READING:", res["reading"])
