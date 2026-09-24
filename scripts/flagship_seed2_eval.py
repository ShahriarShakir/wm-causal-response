"""Second training seed on the flagship pair (arms E and F, READMEs in ext/diamond_seed2 and ext/iris_seed2).
Compares the released checkpoint (outputs/) with our seed-2 checkpoint (ext/outputs/) on Alien and Pong under the chosen and the
fixed intervention, with the estimator of Table 1 (mean over evaluation seeds of the per-seed median delay-20 alignment; PSNR too).
Applies the pre-declared reading written in the READMEs before training started: the replicate 'reproduces the flagship' if seed-2
Alien is below 0.05 and seed-2 Pong is above the architecture's tracked threshold (DIAMOND 0.40 under both interventions; IRIS 0.108
under the chosen one). Either outcome is reported. Writes outputs/flagship_seed2_eval.json."""
import json, pathlib, numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]; OUT = ROOT / "outputs"; EXT = ROOT / "ext" / "outputs"
T = {"DIAMOND": 0.40, "IRIS": 0.108}; PRE = {"DIAMOND": "video", "IRIS": "iris"}
FILES = {"Alien": {"chosen": "_k14_w60", "fixed": ""}, "Pong": {"chosen": "_k2", "fixed": ""}}
def agg(f, dl="20"):
    d = json.load(open(f)); Ss = list(d["results"])
    med = lambda k: [d["results"][s]["aggregate"]["rows"][dl][k]["median"] for s in Ss if dl in d["results"][s]["aggregate"]["rows"]]
    A = med("causal_alignment"); return dict(A=float(np.mean(A)), A_sd=float(np.std(A)), psnr=float(np.mean(med("prediction_psnr"))), n_seeds=len(Ss))
res = {}
for arch, pre in PRE.items():
    res[arch] = {}
    for g, ivs in FILES.items():
        for iv, suf in ivs.items():
            f1 = OUT / f"{pre}_{g}-v5_dur8{suf}.json"; f2 = EXT / f"{pre}_{g}-v5_dur8{suf}.json"
            if not f2.exists(): res[arch][f"{g}/{iv}"] = {"released": agg(f1) if f1.exists() else None, "seed2": None}; continue
            res[arch][f"{g}/{iv}"] = {"released": agg(f1), "seed2": agg(f2)}
    # pre-declared reading
    ivs = ("chosen", "fixed") if arch == "DIAMOND" else ("chosen",)
    have = all(res[arch][f"{g}/{iv}"]["seed2"] for g in FILES for iv in ivs)
    if have:
        alien_blind = all(res[arch][f"Alien/{iv}"]["seed2"]["A"] < 0.05 for iv in ivs)
        pong_tracked = all(res[arch][f"Pong/{iv}"]["seed2"]["A"] > T[arch] for iv in ivs)
        res[arch]["reading"] = dict(alien_below_005=alien_blind, pong_above_T=pong_tracked, reproduces=bool(alien_blind and pong_tracked), interventions=list(ivs), threshold=T[arch])
    else: res[arch]["reading"] = None
json.dump(res, open(OUT / "flagship_seed2_eval.json", "w"), indent=1)
for arch in res:
    for k, v in res[arch].items():
        if k == "reading": print(f"{arch} pre-declared reading: {v}"); continue
        s2 = v["seed2"]; r1 = v["released"]
        print(f"{arch:8s} {k:13s} released A20 {r1['A']:.3f} (PSNR {r1['psnr']:.1f}) | seed 2 " + (f"A20 {s2['A']:.3f} sd {s2['A_sd']:.3f} (PSNR {s2['psnr']:.1f})" if s2 else "not yet run"))
