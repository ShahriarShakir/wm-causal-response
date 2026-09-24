"""Action-consequence CORRESPONDENCE test. Uses the three-arm files (spec_*):
per episode, the model's response to action b is projected on the world's response to b (align_B, the correct pairing) and on the world's
response to c (the WRONG pairing; by symmetry of the stored quantities we use align_CB = model response to c projected on the world's
response to b). If the model encodes which action produced which consequence, the correct pairing must beat the wrong one:
  per game: paired difference D = align_B - align_CB over episodes at delays 8 and 20, sign-flip permutation p (20,000), Holm over games per
  architecture; suite-level: fraction of games with D > 0 and a binomial p.
A game passes the CORRESPONDENCE test if D > 0 with Holm p < 0.05. Writes outputs/correspondence.json."""
import json, pathlib, sys, glob, numpy as np
from math import comb
ROOT = pathlib.Path(__file__).resolve().parents[1]; OUT = ROOT / "outputs"; sys.path.insert(0, str(ROOT))
res = {}
for arch, prefix in (("DIAMOND", "diamond"), ("IRIS", "iris"), ("STORM", "storm")):
    rows = {}
    for f in sorted(glob.glob(str(OUT / f"spec_{prefix}_*.json"))):
        if "DRY" in f: continue
        d = json.load(open(f)); g = d["game"].split("/")[-1].replace("-v5", ""); raws = [r for s in d["results"] for r in d["results"][s]["raw"]]
        rows[g] = {}
        for dl in ("8", "20"):
            D = [r["align_B"][dl] - r["align_CB"][dl] for r in raws if r["align_B"].get(dl) is not None and r["align_CB"].get(dl) is not None and r["align_B"][dl] == r["align_B"][dl] and r["align_CB"][dl] == r["align_CB"][dl]]
            if len(D) < 10: rows[g][dl] = None; continue
            D = np.array(D); obs = float(D.mean()); rng = np.random.default_rng(0)
            null = np.array([float((D * rng.choice([-1, 1], len(D))).mean()) for _ in range(20000)])
            p = float((np.sum(null >= obs) + 1) / (len(null) + 1))                       # one-sided: correct pairing > wrong pairing
            rows[g][dl] = dict(n=len(D), mean_diff=obs, median_align_B=float(np.median([r["align_B"][dl] for r in raws if r["align_B"].get(dl) is not None])), median_align_CB=float(np.median([r["align_CB"][dl] for r in raws if r["align_CB"].get(dl) is not None])), p=p)
    for dl in ("8", "20"):
        gs = [g for g in rows if rows[g].get(dl)]; ps = sorted(((rows[g][dl]["p"], g) for g in gs))
        run = 0.0   # Holm step-down: adjusted p is the running maximum of (m - i) p_(i), so it never decreases with the raw p
        for i, (p, g) in enumerate(ps):
            run = max(run, min(1.0, p * (len(ps) - i))); rows[g][dl]["p_holm"] = run; rows[g][dl]["pass"] = bool(run < 0.05 and rows[g][dl]["mean_diff"] > 0)
        npos = sum(1 for g in gs if rows[g][dl]["mean_diff"] > 0); n = len(gs); p_binom = sum(comb(n, k) for k in range(npos, n + 1)) / 2 ** n
        npass = sum(1 for g in gs if rows[g][dl]["pass"])
        res.setdefault(arch, {})[f"suite_d{dl}"] = dict(n_games=n, positive=npos, holm_pass=npass, p_binomial=p_binom)
        print(f"{arch} d{dl}: correct pairing > wrong pairing in {npos}/{n} games (binomial p {p_binom:.4f}); Holm-significant in {npass}/{n}")
    res[arch]["games"] = rows
    # blind vs tracked split (A_b at d20 < 0.05 vs >= T)
    T = {"DIAMOND": 0.40, "IRIS": 0.09, "STORM": 0.126}[arch]
    blind = [g for g in rows if rows[g].get("20") and rows[g]["20"]["median_align_B"] < 0.05]; tracked = [g for g in rows if rows[g].get("20") and rows[g]["20"]["median_align_B"] >= T]
    res[arch]["blind_pass_d20"] = [g for g in blind if rows[g]["20"]["pass"]]; res[arch]["tracked_pass_d20"] = [g for g in tracked if rows[g]["20"]["pass"]]
    print(f"   at d20: tracked games passing {len(res[arch]['tracked_pass_d20'])}/{len(tracked)}; blind games passing {len(res[arch]['blind_pass_d20'])}/{len(blind)} ({res[arch]['blind_pass_d20']})")
json.dump(res, open(OUT / "correspondence.json", "w"), indent=1); print("wrote outputs/correspondence.json")
