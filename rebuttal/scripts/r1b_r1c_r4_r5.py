"""R1b, R1c, R4, R5 (rules in rebuttal/README.md). Reads rebuttal/outputs/r1_horizon_gain.json, r2_shift_tolerance.json,
outputs/dpsnr_cells.json, the sweep files and the audit panels; writes rebuttal/outputs/r1b_r1c_r4_r5.json and two example figures."""
import io, json, glob, tarfile, pathlib, numpy as np, cv2
from PIL import Image
from scipy.stats import spearmanr, fisher_exact

ROOT = pathlib.Path(__file__).resolve().parents[2]; OUT = ROOT / "rebuttal" / "outputs"; FIG = ROOT / "rebuttal" / "figs"; FIG.mkdir(exist_ok=True)
R1 = json.load(open(OUT / "r1_horizon_gain.json")); R2 = json.load(open(OUT / "r2_shift_tolerance.json")); T = R1["thresholds"]
DP = json.load(open(ROOT / "outputs" / "dpsnr_cells.json"))["cells"]
rng = np.random.default_rng(0); res = {"rules": "rebuttal/README.md R1b, R1c, R4, R5"}
def rho(x, y): return float(spearmanr(x, y).correlation)

# ---- R1b: horizon-aligned vs useful over the 24 closed-loop cells ----
cells = R1["cells"]; tab = [[0, 0], [0, 0]]
for c in cells: tab[0 if c["A_h"] >= T[c["arch"]] else 1][0 if c["useful"] else 1] += 1
odds, p = fisher_exact(tab)
res["R1b"] = dict(table={"aligned_useful": tab[0][0], "aligned_not": tab[0][1], "not_aligned_useful": tab[1][0], "not_aligned_not": tab[1][1]},
                  fisher_p=float(p), reading="separates" if p < 0.05 else "counts only",
                  not_aligned_but_useful=[f"{c['arch']}/{c['game']}" for c in cells if c["A_h"] < T[c["arch"]] and c["useful"]],
                  aligned_not_useful=[f"{c['arch']}/{c['game']}" for c in cells if c["A_h"] >= T[c["arch"]] and not c["useful"]])

# ---- R1c: delay-20 Delta-PSNR vs gain ----
dc = []
for c in cells:
    v = DP.get(c["arch"], {}).get(c["game"], {}) or {}
    if v.get("dpsnr"): dc.append(dict(c, dpsnr=v["dpsnr"]["median"]))
games = sorted({c["game"] for c in dc})
def pooled(cs, key): return rho([x["gain"] for x in cs], [x[key] for x in cs])
def perm_p(key, B=10000):
    obs = pooled(dc, key); by = {g: [x for x in dc if x["game"] == g] for g in games}; hits = 0
    for _ in range(B):
        perm = dict(zip(games, rng.permutation(games))); xs, ys = [], []
        for g in games:
            src = {x["arch"]: x[key] for x in by[perm[g]]}
            for x in by[g]:
                if x["arch"] in src: xs.append(src[x["arch"]]); ys.append(x["gain"])
        if len(xs) > 3 and abs(rho(ys, xs)) >= abs(obs) - 1e-12: hits += 1
    return (hits + 1) / (B + 1)
def boot(key, B=4000):
    v = []
    for _ in range(B):
        gs = rng.choice(games, len(games)); cs = [x for g in gs for x in dc if x["game"] == g]; r = pooled(cs, key)
        if r == r: v.append(r)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
res["R1c"] = dict(n=len(dc), n_games=len(games), per_arch={A: dict(n=len([x for x in dc if x["arch"] == A]),
                  rho_dpsnr=rho([x["gain"] for x in dc if x["arch"] == A], [x["dpsnr"] for x in dc if x["arch"] == A]) if len([x for x in dc if x["arch"] == A]) > 3 else None)
                  for A in ("DIAMOND", "IRIS", "STORM")})
for key in ("dpsnr", "A_h", "A_20", "psnr_20"):
    res["R1c"][f"pooled_{key}"] = dict(rho=pooled(dc, key), p_perm=perm_p(key), ci=boot(key))

# ---- R5: never-aligned cells, maximum alignment over all eight delays ----
S = json.load(open(ROOT / "outputs" / "emulator_intervention_search.json")); PREFIX = {"DIAMOND": ("video", 0), "IRIS": ("iris", 0), "STORM": ("storm", 4)}
ALL_D = ("1", "2", "3", "5", "8", "12", "16", "20")
def curve(arch, g):
    pre, ws = PREFIX[arch]; k, w = int(S[g]["best_offset"]), max(0, int(S[g]["best_warmup"]) - ws)
    d = json.load(open(ROOT / "outputs" / f"{pre}_{g}-v5_dur8{'_k' + str(k) if k != 3 else ''}{'_w' + str(w) if w else ''}.json")); r = {}
    for dl in ALL_D:
        A = [d["results"][s]["aggregate"]["rows"][dl]["causal_alignment"]["median"] for s in d["results"] if dl in d["results"][s]["aggregate"]["rows"]]
        r[dl] = float(np.mean(A)) if (A and all(v == v for v in A)) else None
    return r
na = [c for c, v in R2["cells"].items() if v["cls"] == "never-aligned"]; mx = {}
for c in na:
    A, g = c.split("/"); r = curve(A, g); vals = [v for v in r.values() if v is not None]; mx[c] = dict(max_any_delay=max(vals), T=T[A], argmax=max((v, d) for d, v in r.items() if v is not None)[1])
res["R5"] = dict(n=len(na), n_reaching_T_any_delay=int(sum(v["max_any_delay"] >= v["T"] for v in mx.values())),
                 reaching=[c for c, v in mx.items() if v["max_any_delay"] >= v["T"]], median_max=float(np.median([v["max_any_delay"] for v in mx.values()])), cells=mx)

# ---- R4: worked examples from the audit panels ----
IMG = 64; idx = np.arange(IMG * IMG, dtype=np.int32).reshape(IMG, IMG); up_idx = cv2.resize(idx, (160, 210), interpolation=cv2.INTER_NEAREST); pos = {}
for y in range(210):
    for x in range(160): pos.setdefault(int(up_idx[y, x]), (y, x))
PY = np.array([pos[i][0] for i in range(IMG * IMG)]).reshape(IMG, IMG); PX = np.array([pos[i][1] for i in range(IMG * IMG)]).reshape(IMG, IMG)
want = {("iris", "Asterix"), ("storm", "Boxing")}; KEY = {}
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
res["R4"] = {}
for cell in sorted({r["cell"] for r in recs}):
    rs = [r for r in recs if r["cell"] == cell]; mA, mR = np.median([r["A"] for r in rs]), np.median([r["R"] for r in rs])
    ex = min(rs, key=lambda r: abs(r["A"] - mA) + 0.05 * abs(r["R"] - mR)); ta, tb, pa, pb, dt, dp = ex["fr"]
    def mag(d): m = np.abs(d).sum(-1); return np.repeat((255 * m / max(m.max(), 1e-9))[..., None], 3, -1)
    img = np.concatenate([cv2.resize(np.clip(x, 0, 255).astype(np.uint8), (192, 192), interpolation=cv2.INTER_NEAREST) for x in (ta, tb, pa, pb, mag(dt), mag(dp))], 1)
    fn = FIG / f"example_{cell.replace('/', '_')}_ep{ex['ep']}.png"; cv2.imwrite(str(fn), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    res["R4"][cell] = dict(n=len(rs), median_A=float(mA), median_R=float(mR), median_off_support=float(np.median([r["off"] for r in rs])),
                           example=dict(episode=ex["ep"], A=ex["A"], R=ex["R"], off_support=ex["off"], figure=str(fn.relative_to(ROOT))),
                           paper_dpsnr=(DP.get(cell.split("/")[0].upper(), {}).get(cell.split("/")[1], {}) or {}).get("dpsnr"),
                           paper_A20=(DP.get(cell.split("/")[0].upper(), {}).get(cell.split("/")[1], {}) or {}).get("A"))
json.dump(res, open(OUT / "r1b_r1c_r4_r5.json", "w"), indent=1)
print("R1b", res["R1b"]); print("R1c", {k: v for k, v in res["R1c"].items()}); print("R5", {k: v for k, v in res["R5"].items() if k != "cells"})
for c, v in res["R4"].items(): print("R4", c, {k: v[k] for k in ("n", "median_A", "median_R", "median_off_support", "example", "paper_A20")}, "dpsnr", v["paper_dpsnr"])
