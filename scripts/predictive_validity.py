"""The 'so what': does causal alignment predict whether an agent trained INSIDE the world model
succeeds in the real game, better than prediction accuracy or consistency do?

Agent returns are the PUBLISHED Atari-100k tables of DIAMOND (arXiv:2405.12399, Table 1) and
IRIS (arXiv:2209.00588), extracted from the PDFs with a cross-check: the IRIS column inside
DIAMOND's table must equal the IRIS column of IRIS's own table for every game, or the column
indexing is wrong and the script stops. Human-normalised score = (agent-random)/(human-random).

Three tests, all gated:
  B1  within each architecture, rho(metric, HNS) for alignment / PSNR / consistency, n=12
  B2  DIFFERENCE test: Delta(DIAMOND-IRIS) of HNS vs Delta of each metric -- game difficulty
      cancels, leaving 'which model is better on this game and does the metric know it'
  B3  pooled 24 points (reported, but game features are shared across archs)
"""
import json, re, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from ccaus.validate import spearman, permutation_p, detectable_rho, claim, gate_correlation
try: from pypdf import PdfReader
except ImportError: from PyPDF2 import PdfReader
SP = pathlib.Path(sys.argv[1]); OUT = pathlib.Path(__file__).resolve().parents[1] / "outputs"
GAMES = ["Alien","Amidar","Assault","Asterix","BankHeist","BattleZone","Boxing","Breakout","ChopperCommand",
         "CrazyClimber","DemonAttack","Freeway","Frostbite","Gopher","Hero","Jamesbond","Kangaroo","Krull",
         "KungFuMaster","MsPacman","Pong","PrivateEye","Qbert","RoadRunner","Seaquest","UpNDown"]
def table(pdf):
    t = "\n".join((p.extract_text() or "") for p in PdfReader(pdf).pages); rows = {}
    for g in GAMES:
        pat = re.compile(r"^\s*" + re.sub(r"(?<!^)(?=[A-Z])", r"\\s?", g) + r"\s+([-\d.,]+(?:\s+[-\d.,]+)*)", re.M)
        m = pat.search(t); rows[g] = [float(x.replace(",", "")) for x in m.group(1).split()]
    return rows
CACHE = SP / "published_tables.json"   # the rows parsed from the two papers, kept so the analysis runs without the PDFs
if (SP / "diamond.pdf").exists() and (SP / "iris.pdf").exists():
    D, I = table(SP / "diamond.pdf"), table(SP / "iris.pdf")
    json.dump({"diamond": D, "iris": I}, open(CACHE, "w"), indent=1)
else:
    c = json.load(open(CACHE)); D, I = c["diamond"], c["iris"]
    print(f"published tables read from {CACHE.name} (the PDFs are not present)\n")
for g in GAMES:   # cross-check column indexing
    assert abs(D[g][4] - I[g][-1]) < 1e-6, f"column mismatch on {g}: {D[g][4]} vs {I[g][-1]}"
hns = {g: {"DIAMOND": (D[g][7]-D[g][0])/(D[g][1]-D[g][0]), "IRIS": (D[g][4]-D[g][0])/(D[g][1]-D[g][0])} for g in GAMES}
print("column cross-check passed on all 26 games (IRIS column identical in both papers)\n")
P = json.load(open(OUT / "task_structure_features.json"))
live = sorted({p["game"] for p in P if p["arch"]=="DIAMOND"} & {p["game"] for p in P if p["arch"]=="IRIS"})
M = {(p["arch"], p["game"]): p for p in P}
print(f"{'game':<13}{'HNS D':>8}{'HNS I':>8} | {'align D':>8}{'align I':>8} | {'PSNR D':>7}{'PSNR I':>7}")
for g in live:
    print(f"{g:<13}{hns[g]['DIAMOND']:>8.2f}{hns[g]['IRIS']:>8.2f} | {M[('DIAMOND',g)]['align']:>8.3f}{M[('IRIS',g)]['align']:>8.3f}"
          f" | {M[('DIAMOND',g)]['psnr']:>7.1f}{M[('IRIS',g)]['psnr']:>7.1f}")
res = {"live_games": live, "hns": {g: hns[g] for g in live}}
def block(label, x, y, n_note=""):
    r, p, d = spearman(x, y), permutation_p(x, y), detectable_rho(len(x))
    print(f"  {label:<38} rho = {r:+.3f}  p = {p:.4f}  n = {len(x)}  detectable {d:.3f}  {'POWERED' if abs(r)>=d else 'underpowered'}")
    return {"rho": r, "p": p, "n": len(x), "detectable": d, "powered": bool(abs(r) >= d)}
print("\nB1  within-architecture: which metric predicts the agent's real return?")
for arch in ("DIAMOND", "IRIS"):
    y = [hns[g][arch] for g in live]; res[f"B1_{arch}"] = {}
    for k in ("align", "psnr", "cons"):
        res[f"B1_{arch}"][k] = block(f"{arch}: HNS ~ {k}", [M[(arch,g)][k] for g in live], y)
print("\nB2  DIFFERENCE (DIAMOND - IRIS) per game: game difficulty cancels")
dy = [hns[g]["DIAMOND"] - hns[g]["IRIS"] for g in live]; res["B2"] = {}
for k in ("align", "psnr", "cons"):
    dx = [M[("DIAMOND",g)][k] - M[("IRIS",g)][k] for g in live]
    res["B2"][k] = block(f"Delta HNS ~ Delta {k}", dx, dy)
print("\nB3  pooled 24 points (game features shared across archs -- inflated independence)")
y = [hns[g][a] for a in ("DIAMOND","IRIS") for g in live]; res["B3"] = {}
for k in ("align", "psnr", "cons"):
    res["B3"][k] = block(f"pooled: HNS ~ {k}", [M[(a,g)][k] for a in ("DIAMOND","IRIS") for g in live], y)
json.dump(res, open(OUT / "predictive_validity.json", "w"), indent=1); print("\nwrote outputs/predictive_validity.json")
