"""Latency correlate on the full 26-game chosen set (writes outputs/latency_correlate_chosen.json)."""
import json, sys, numpy as np; sys.path.insert(0,".")
from ccaus.validate import spearman, permutation_p, detectable_rho
P = json.load(open("outputs/preregistered_latency_prediction.json"))
X = json.load(open("outputs/psnr_crossarch_chosen.json"))
out = {}
print("LATENCY (from the emulator, chosen interventions) vs CAUSAL ALIGNMENT, full suite n=26\n")
for arch in ("DIAMOND","IRIS"):
    G = X[arch]["games"]; games = [g for g in G if g in P and P[g]["latency"] is not None]
    lat = [P[g]["latency"] for g in games]; al = [G[g]["alignment"] for g in games]; ps = [G[g]["psnr"] for g in games]
    r, p, d = spearman(lat, al), permutation_p(lat, al), detectable_rho(len(games))
    W = [i for i,g in enumerate(games) if ps[i] > 30]
    rw, pw, dw = (spearman([lat[i] for i in W],[al[i] for i in W]), permutation_p([lat[i] for i in W],[al[i] for i in W]), detectable_rho(len(W))) if len(W) >= 6 else (float("nan"),float("nan"),float("nan"))
    out[arch] = dict(n=len(games), rho=r, p=p, detectable=d, powered=bool(abs(r)>=d),
                     n_well=len(W), rho_well=rw, p_well=pw, detectable_well=dw, powered_well=bool(abs(rw)>=dw) if np.isfinite(rw) else False)
    print(f"  {arch:<8} all n={len(games)}: rho = {r:+.3f} p = {p:.4f} detect {d:.3f} {'POWERED' if abs(r)>=d else 'underpowered'} | "
          f"PSNR>30 n={len(W)}: rho = {rw:+.3f} p = {pw:.4f} detect {dw:.3f} {'POWERED' if np.isfinite(rw) and abs(rw)>=dw else 'underpowered'}")
json.dump(out, open("outputs/latency_correlate_chosen.json","w"), indent=1); print("\nwrote outputs/latency_correlate_chosen.json")
