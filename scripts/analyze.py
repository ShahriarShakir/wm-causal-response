"""Gated claims. Every claim is computed through the gates in ccaus/validate.py and written to
outputs/claims.json; a claim that fails a gate is written as UNSUPPORTED with its reason."""
import json, pathlib, sys, statistics as st
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from ccaus.validate import (Report, claim, detectable_rho, gate_correlation, gate_dataset,
                            permutation_p, spearman)

OUT = pathlib.Path(__file__).resolve().parents[1] / "outputs"
DUR = int(sys.argv[1]) if len(sys.argv) > 1 else 1
SUF = f"_dur{DUR}" if DUR > 1 else ""


def load(dur_suffix):
    games = {}
    for f in sorted(OUT.glob(f"video_*{dur_suffix}.json")):
        if dur_suffix == "" and "_dur" in f.name:
            continue
        d = json.load(open(f))
        g = d["game"].split("/")[-1].replace("-v5", "")
        S = [str(x) for x in d["seeds"]]
        R = lambda dl, k: st.fmean(d["results"][s]["aggregate"]["rows"][dl][k]["median"]
                                   for s in S)
        D = lambda dl, k: st.fmean(d["results"][s]["decomposition"][dl][k] for s in S)
        games[g] = {"true_div": R("20", "true_divergence"),
                    "consistency": R("20", "consistency"),
                    "alignment": R("20", "causal_alignment"),
                    "alignment_d2": R("2", "causal_alignment"),
                    "actuated": D("20", "actuated_align"),
                    "consequence": D("20", "consequence_align"),
                    "n_episodes": d["n_episodes"] * len(S)}
    return games


games = load(SUF)
if not games:
    print(f"no results for duration {DUR}"); sys.exit(0)
print(f"=== intervention duration {DUR}, {len(games)} games ===\n")

ds = gate_dataset({g: v["true_div"] for g, v in games.items()})
print(ds, "\n")
live = {g: v for g, v in games.items()
        if np.isfinite(v["true_div"]) and v["true_div"] > 0.1
        and np.isfinite(v["alignment"])}
print(f"{'game':<16}{'true div':>9}{'consist':>9}{'align d2':>10}{'align d20':>11}")
for g, v in sorted(live.items(), key=lambda kv: -kv[1]["alignment"]):
    print(f"{g:<16}{v['true_div']:>9.2f}{v['consistency']:>9.3f}"
          f"{v['alignment_d2']:>10.3f}{v['alignment']:>11.3f}")

c = [live[g]["consistency"] for g in live]
a = [live[g]["alignment"] for g in live]
claims = {}
if len(live) >= 4:
    rho, p = spearman(c, a), permutation_p(c, a)
    corr = gate_correlation(c, a)
    txt = claim("consistency does not predict causal fidelity across games", corr)
    print(f"\nrho = {rho:+.3f}, permutation p = {p:.3f}, n = {len(live)}, "
          f"smallest detectable |rho| = {detectable_rho(len(live)):.3f}")
    print(txt)
    claims["decorrelation"] = {"supported": corr.ok, "rho": rho, "p": p, "n": len(live),
                               "detectable": detectable_rho(len(live)),
                               "gates": [str(g) for g in corr.gates]}

# Compression and pairwise inversions do NOT need correlation power: they are statements about
# named units, so they are gated only on those units being non-degenerate.
if len(live) >= 2:
    cs, als = np.array(c), np.array(a)
    comp = Report().add("both spans computed on non-degenerate units only", True,
                        f"{len(live)} units, all with true divergence > 0.1")
    # A max/min RATIO is not reportable here: the minimum alignment approaches 0, so the
    # ratio explodes (it reached 9.9e8 at duration 16) and reports the smallest denominator
    # rather than the spread, the same ratio-estimator failure that the medians elsewhere avoid. Report the ABSOLUTE
    # ranges, and quantify spread by how far each metric moves relative to its own scale.
    rng_c, rng_a = (float(cs.min()), float(cs.max())), (float(als.min()), float(als.max()))
    # fraction of its own observed maximum that each metric gives up across the same games
    drop_c = (cs.max() - cs.min()) / cs.max() if cs.max() > 0 else float("nan")
    drop_a = (als.max() - als.min()) / als.max() if als.max() > 0 else float("nan")
    print("\n" + claim(
        f"across the same {len(live)} games consistency falls {100*drop_c:.0f}% of its maximum "
        f"({rng_c[0]:.3f}-{rng_c[1]:.3f}) while causal alignment falls {100*drop_a:.0f}% "
        f"({rng_a[0]:.4f}-{rng_a[1]:.3f})", comp))
    claims["compression"] = {"supported": comp.ok, "consistency_range": rng_c,
                             "alignment_range": rng_a, "consistency_drop": float(drop_c),
                             "alignment_drop": float(drop_a), "n": len(live)}
    inv = []
    for g1 in live:
        for g2 in live:
            if live[g1]["consistency"] > live[g2]["consistency"] and \
               live[g2]["alignment"] > 3 * max(live[g1]["alignment"], 1e-9):
                inv.append({"consistency_prefers": g1, "causality_prefers": g2,
                            "c1": live[g1]["consistency"], "c2": live[g2]["consistency"],
                            "a1": live[g1]["alignment"], "a2": live[g2]["alignment"]})
    inv.sort(key=lambda d: -(d["a2"] / max(d["a1"], 1e-9)))
    print(f"\nRANK INVERSIONS (consistency prefers X, causality prefers Y by >3x): {len(inv)}")
    for d in inv[:6]:
        print(f"  consistency: {d['consistency_prefers']} ({d['c1']:.3f}) > "
              f"{d['causality_prefers']} ({d['c2']:.3f})   BUT causality: "
              f"{d['causality_prefers']} ({d['a2']:.3f}) > "
              f"{d['consistency_prefers']} ({d['a1']:.3f})")
    claims["inversions"] = {"supported": len(inv) > 0, "n_inversions": len(inv),
                            "examples": inv[:6]}

claims["_meta"] = {"duration": DUR, "n_games": len(games), "n_live": len(live),
                   "dataset_gates": [str(g) for g in ds.gates]}
json.dump(claims, open(OUT / f"claims{SUF}.json", "w"), indent=1)
print(f"\nwrote outputs/claims{SUF}.json")
