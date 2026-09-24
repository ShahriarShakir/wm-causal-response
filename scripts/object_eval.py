"""Object-level evaluation (v19 files), rule fixed 2026-09-11: per (arch, game) at delay 20, the tolerant hit rate at 8 px (object present in both
generated arms AND model displacement within 8 px of the true one), the direction error, the magnitude ratio, presence, and the swap statistic.
Classes from the taxonomy (same rule as the paper). Decision: LABEL VALIDATED if the median hit@8 of never-aligned cells is far below that of
tracked cells (permutation p < 0.05) and the same holds at hit@4 and hit@2 (the pattern does not depend on tolerance); NARROW the title if
never-aligned cells reach hit@8 comparable to tracked ones. Also reports 'absent' (object not drawn) vs 'misplaced' (drawn, wrong displacement)
among blind cells, and the swap statistic by class. Writes outputs/object_eval.json."""
import json, glob, pathlib, sys, numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]; OUT = ROOT / "outputs"; sys.path.insert(0, str(ROOT))
E = json.load(open(OUT / "storm_extend.json"))
res = {}
def perm_p(a, b, n=20000):
    a, b = np.array(a, float), np.array(b, float); obs = np.median(b) - np.median(a); pool = np.concatenate([a, b]); rng = np.random.default_rng(0); c = 0
    for _ in range(n):
        p = rng.permutation(pool); c += (np.median(p[len(a):]) - np.median(p[:len(a)])) >= obs
    return (c + 1) / (n + 1)
for arch in ("DIAMOND", "IRIS", "STORM"):
    tax = E["taxonomy"].get(arch, {}); cells = {}
    for f in sorted(glob.glob(str(OUT / f"obj_{arch.lower()}_*.json"))):
        if "DRY" in f: continue
        d = json.load(open(f)); g = d["game"].split("/")[-1].replace("-v5", ""); a = d["aggregate"]["20"]
        if a["n"] < 10: continue
        # locator validity (added 2026-09-12): fraction of episodes in which the colour locator finds the actuated object in the TRUE frames
        # at delay 20 (true_count > 0); the model-side presence rate is only meaningful where this is near one
        tp = [bool(r["true_count"].get("20", 0) > 0) for r in d["raw"] if isinstance(r.get("true_count"), dict)]
        cells[g] = dict(cls=tax.get(g, {}).get("cls", "?"), n=a["n"], present=a["present"], hit2=a["hit"]["2"], hit4=a["hit"]["4"], hit8=a["hit"]["8"],
                        dir_err=(a["dir_err_deg"] or {}).get("median"), mag_ratio=(a["mag_ratio"] or {}).get("median"), swap=(a["swap"] or {}).get("median"), true_disp=a["true_disp_norm"],
                        true_present=float(np.mean(tp)) if tp else None)
    if not cells: continue
    by = {c: [g for g in cells if cells[g]["cls"] == c] for c in ("tracked", "aligned-then-forgotten", "never-aligned", "partial")}
    summ = {}
    for c, gs in by.items():
        if not gs: continue
        summ[c] = dict(n=len(gs), hit2=float(np.median([cells[g]["hit2"] for g in gs])), hit4=float(np.median([cells[g]["hit4"] for g in gs])), hit8=float(np.median([cells[g]["hit8"] for g in gs])),
                       present=float(np.median([cells[g]["present"] for g in gs])), swap=float(np.median([cells[g]["swap"] for g in gs if cells[g]["swap"] is not None])) if any(cells[g]["swap"] is not None for g in gs) else None,
                       dir_err=float(np.median([cells[g]["dir_err"] for g in gs if cells[g]["dir_err"] is not None])) if any(cells[g]["dir_err"] is not None for g in gs) else None)
    tests = {}
    if by["tracked"] and by["never-aligned"]:
        for k in ("hit2", "hit4", "hit8", "present"):
            tests[k] = dict(tracked=float(np.median([cells[g][k] for g in by["tracked"]])), blind=float(np.median([cells[g][k] for g in by["never-aligned"]])), p=perm_p([cells[g][k] for g in by["never-aligned"]], [cells[g][k] for g in by["tracked"]]))
    blind = by["never-aligned"]; absent = [g for g in blind if cells[g]["present"] < 0.5]; misplaced = [g for g in blind if cells[g]["present"] >= 0.5 and cells[g]["hit8"] < 0.5]; tolerant = [g for g in blind if cells[g]["hit8"] >= 0.5]
    verdict = "LABEL VALIDATED" if tests and all(tests[k]["p"] < 0.05 for k in ("hit2", "hit4", "hit8")) else ("NARROW" if tests and tests["hit8"]["p"] >= 0.2 else "MIXED")
    res[arch] = dict(cells=cells, by_class=summ, tests=tests, blind_absent=absent, blind_misplaced=misplaced, blind_tolerant_hit=tolerant, verdict=verdict)
    print(f"== {arch}: " + " | ".join(f"{c}: n {v['n']} hit8 {v['hit8']:.2f} hit4 {v['hit4']:.2f} present {v['present']:.2f} swap {v['swap'] if v['swap'] is not None else float('nan'):+.2f}" for c, v in summ.items()))
    if tests: print("   tracked vs never-aligned: " + ", ".join(f"{k} {v['tracked']:.2f} vs {v['blind']:.2f} (p {v['p']:.4f})" for k, v in tests.items()))
    print(f"   blind cells: absent {absent} | drawn but misplaced {misplaced} | tolerant hit >= 0.5 at 8 px {tolerant} -> {verdict}")
# ---- POOLED over architectures with GAME-CLUSTERED permutation (games shared across archs): tracked vs never-aligned cells ----
allc = {f"{a}/{g}": c for a, r in res.items() for g, c in r["cells"].items()}
tr = {k: v for k, v in allc.items() if v["cls"] == "tracked"}; bl = {k: v for k, v in allc.items() if v["cls"] == "never-aligned"}
if tr and bl:
    games = sorted({k.split("/")[1] for k in allc}); rng = np.random.default_rng(1); pooled = {}
    for key in ("hit2", "hit4", "hit8", "present"):
        obs = np.median([v[key] for v in tr.values()]) - np.median([v[key] for v in bl.values()]); c = 0; n = 20000
        # permute the CLASS label at the GAME level (a game's cells keep their labels together) to respect clustering
        lab = {k: v["cls"] for k, v in allc.items()}
        for _ in range(n):
            perm = dict(zip(games, rng.permutation(games)))          # relabel by permuting games' identities among cells
            t2 = [allc[k][key] for k in allc if lab[f"{k.split('/')[0]}/{perm[k.split('/')[1]]}"] == "tracked"] if all(f"{k.split('/')[0]}/{perm[k.split('/')[1]]}" in lab for k in allc) else None
            if t2 is None: break
            b2 = [allc[k][key] for k in allc if lab[f"{k.split('/')[0]}/{perm[k.split('/')[1]]}"] == "never-aligned"]
            if t2 and b2: c += (np.median(t2) - np.median(b2)) >= obs
        pooled[key] = dict(tracked=float(np.median([v[key] for v in tr.values()])), blind=float(np.median([v[key] for v in bl.values()])), n_tracked=len(tr), n_blind=len(bl), p=(c + 1) / (n + 1))
    absent = [k for k, v in bl.items() if v["present"] < 0.5]; tolerant = [k for k, v in bl.items() if v["hit8"] >= 0.5]
    res["_pooled"] = dict(tests=pooled, blind_absent=absent, blind_tolerant_hit=tolerant, swap_tracked=float(np.median([v["swap"] for v in tr.values() if v["swap"] is not None])), swap_blind=float(np.median([v["swap"] for v in bl.values() if v["swap"] is not None])))
    print("== POOLED (game-clustered permutation): " + ", ".join(f"{k} tracked {v['tracked']:.2f} vs blind {v['blind']:.2f} (p {v['p']:.4f})" for k, v in pooled.items()) + f" | n {len(tr)} vs {len(bl)} | swap {res['_pooled']['swap_tracked']:+.2f} vs {res['_pooled']['swap_blind']:+.2f}")
    print(f"   blind cells absent: {len(absent)}/{len(bl)} {absent} | blind cells with hit@8 >= 0.5: {len(tolerant)}/{len(bl)} {tolerant}")
_tp = [c["true_present"] for a_ in ("DIAMOND", "IRIS", "STORM") if a_ in res for c in res[a_]["cells"].values() if c.get("true_present") is not None]
if _tp and "_pooled" in res: res["_pooled"]["locator_true_present"] = dict(median=float(np.median(_tp)), min=float(np.min(_tp)), n_cells=len(_tp)); print(f"locator validity on TRUE frames at d20: median {np.median(_tp):.3f}, min {np.min(_tp):.3f} over {len(_tp)} cells")
json.dump(res, open(OUT / "object_eval.json", "w"), indent=1); print("wrote outputs/object_eval.json")
