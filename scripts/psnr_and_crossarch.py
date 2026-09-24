"""Two questions the audit said a top-venue a reader will ask, answered through the gates.

  A. "Your models are just weak."  Does PREDICTION ACCURACY (PSNR of the predicted future
     against the actual one) predict CAUSAL ALIGNMENT across games? If a game is predicted
     well yet causally blind, prediction quality is not the explanation.
  B. Cross-architecture agreement at the larger n: do DIAMOND and IRIS still rank games the
     same way by causal alignment, and still NOT by consistency, and how does PSNR behave?

Every correlation goes through `gate_correlation`; every game set through `gate_dataset`.
Nothing here is reportable unless its gate line says PASS.
"""
import json, glob, itertools, pathlib, statistics as st, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from ccaus.validate import (claim, detectable_rho, gate_correlation, gate_dataset,
                            permutation_p, spearman)

OUT = pathlib.Path(__file__).resolve().parents[1] / "outputs"
DL = "20"


CHOSEN = "--chosen" in sys.argv          # use the emulator-chosen per-game (offset, warm-up) files
def chosen_files(prefix):
    prefix = {"DIAMOND": "video", "IRIS": "iris"}.get(prefix, prefix)   # load() receives the ARCH label
    S = json.load(open(OUT / "emulator_intervention_search.json"))
    for g, r in S.items():
        k, w = r["best_offset"], r["best_warmup"]
        f = OUT / f"{prefix}_{g}-v5_dur8{'_k'+str(k) if k != 3 else ''}{'_w'+str(w) if w else ''}.json"
        if f.exists(): yield f
def load(pattern, prefix):
    games = {}
    for f in (sorted(chosen_files(prefix)) if CHOSEN else sorted(OUT.glob(pattern))):
        d = json.load(open(f))
        g = d["game"].split("/")[-1].replace("-v5", "")
        S = [str(x) for x in d["seeds"]]
        rows = [d["results"][s]["aggregate"]["rows"][DL] for s in S]
        if "prediction_psnr" not in rows[0] or rows[0]["prediction_psnr"]["n"] == 0:
            continue                                  # pre-PSNR file: not comparable
        def med(k):
            # a degenerate game (true divergence 0) has NaN alignment on EVERY seed; that is a
            # fact for the dataset gate to report, not a reason for the loader to crash
            v = [r[k]["median"] for r in rows if np.isfinite(r[k]["median"])]
            return st.fmean(v) if v else float("nan")
        games[g] = {"true_div": med("true_divergence"), "consistency": med("consistency"),
                    "alignment": med("causal_alignment"), "psnr": med("prediction_psnr")}
    return games


def exact_or_perm_p(x, y):
    n = len(x)
    if n <= 9:                                        # enumerate every ordering
        r = abs(spearman(x, y)); c = t = 0
        for perm in itertools.permutations(range(n)):
            t += 1; c += abs(spearman(x, [y[i] for i in perm])) >= r - 1e-12
        return c / t, "exact"
    return permutation_p(x, y), "permutation(20000)"


def corr_block(name, x, y, xname, yname):
    rho = spearman(x, y); p, kind = exact_or_perm_p(x, y); n = len(x)
    g = gate_correlation(x, y)
    print(f"\n{name}: rho({xname}, {yname}) = {rho:+.3f}, p = {p:.4f} [{kind}], n = {n}, "
          f"smallest detectable |rho| = {detectable_rho(n):.3f}")
    print(claim(f"{xname} predicts {yname}", g))
    return {"rho": rho, "p": p, "test": kind, "n": n, "detectable": detectable_rho(n),
            "supported": g.ok, "gates": [str(t) for t in g.gates]}


report = {}
for arch, pat in (("DIAMOND", "video_*_dur8.json"), ("IRIS", "iris_*_dur8.json")):
    G = load(pat, arch)
    if not G:
        print(f"{arch}: no PSNR-bearing results yet"); continue
    ds = gate_dataset({g: v["true_div"] for g, v in G.items()})
    live = {g: v for g, v in G.items() if v["true_div"] > 0.1 and np.isfinite(v["alignment"])}
    print(f"\n===== {arch}: {len(G)} games with PSNR, {len(live)} live =====\n{ds}")
    print(f"\n{'game':<15}{'PSNR dB':>9}{'consist':>9}{'align':>9}{'true div':>10}")
    for g, v in sorted(live.items(), key=lambda kv: -kv[1]["alignment"]):
        print(f"{g:<15}{v['psnr']:>9.2f}{v['consistency']:>9.3f}{v['alignment']:>9.4f}{v['true_div']:>10.2f}")
    ps = [live[g]["psnr"] for g in live]; al = [live[g]["alignment"] for g in live]
    cs = [live[g]["consistency"] for g in live]
    rep = {"n_games": len(G), "n_live": len(live), "games": live,
           "dataset_gates": [str(t) for t in ds.gates]}
    if len(live) >= 4:
        rep["psnr_vs_alignment"] = corr_block(f"{arch} Q-A", ps, al, "PSNR", "causal alignment")
        rep["psnr_vs_consistency"] = corr_block(f"{arch}", ps, cs, "PSNR", "consistency")
    if "BankHeist" in live:
        rank = sorted(live, key=lambda g: -live[g]["psnr"]).index("BankHeist") + 1
        print(f"\n  BankHeist PSNR rank: {rank} of {len(live)} (1 = best predicted)")
        rep["bankheist_psnr_rank"] = [rank, len(live)]
    report[arch] = rep

if "DIAMOND" in report and "IRIS" in report:
    both = sorted(set(report["DIAMOND"]["games"]) & set(report["IRIS"]["games"]))
    print(f"\n===== CROSS-ARCHITECTURE, games live in BOTH: n = {len(both)} =====")
    xa = [report["DIAMOND"]["games"][g]["alignment"] for g in both]
    ya = [report["IRIS"]["games"][g]["alignment"] for g in both]
    xc = [report["DIAMOND"]["games"][g]["consistency"] for g in both]
    yc = [report["IRIS"]["games"][g]["consistency"] for g in both]
    xp = [report["DIAMOND"]["games"][g]["psnr"] for g in both]
    yp = [report["IRIS"]["games"][g]["psnr"] for g in both]
    report["cross"] = {"games": both,
        "alignment": corr_block("cross-arch", xa, ya, "DIAMOND align", "IRIS align"),
        "consistency": corr_block("cross-arch", xc, yc, "DIAMOND cons", "IRIS cons"),
        "psnr": corr_block("cross-arch", xp, yp, "DIAMOND PSNR", "IRIS PSNR")}
json.dump(report, open(OUT / ("psnr_crossarch_chosen.json" if CHOSEN else "psnr_crossarch.json"), "w"), indent=1)
print("\nwrote outputs/" + ("psnr_crossarch_chosen.json" if CHOSEN else "psnr_crossarch.json"))
