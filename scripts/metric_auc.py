"""Which score predicts whether the actuated object is drawn? (added 2026-09-12 after the panel audit)
Ground truth: object-level presence at delay 20 (object_eval.json, a readout none of the pixel scores saw). Cell = (architecture, game)
under the emulator-chosen intervention. For every score computed on the same rollouts (consistency, PSNR, dPSNR where available,
alignment A, magnitude R, error eps, off-support fraction O) we report the AUC for separating ABSENT cells (presence < 0.5, the rule of
object_eval) from PRESENT ones, pooled over architectures, with a game-clustered bootstrap CI (resample games, keep all
architectures of a game together) and a permutation p against AUC = 0.5 that permutes labels within architecture.
Writes outputs/metric_auc.json and the LaTeX tables tab_metric_auc.tex and tab_metric_auc_full.tex in PAPER_DIR."""
import json, os, pathlib, sys, numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]; OUT = ROOT / "outputs"; sys.path.insert(0, str(ROOT)); P = pathlib.Path(os.environ.get("PAPER_DIR", ROOT / "paper"))
E = json.load(open(OUT / "storm_extend.json")); O = json.load(open(OUT / "object_eval.json")); SP = json.load(open(OUT / "spec_analysis.json")); RE = json.load(open(OUT / "response_error.json")); DP = json.load(open(OUT / "dpsnr_cells.json"))["cells"]; AU = json.load(open(OUT / "object_audit_eval.json"))["cells"] if (OUT / "object_audit_eval.json").exists() else {}
rows = []
for arch in ("DIAMOND", "IRIS", "STORM"):
    for g, c in O.get(arch, {}).get("cells", {}).items():
        pg = E["per_game"].get(arch, {}).get(g); 
        if not pg or pg["20"]["A"] is None: continue
        re_ = (RE.get(arch, {}).get(g) or {}).get("20") or {}; sp = (SP.get(arch, {}).get(g) or {}).get("20") or {}
        dp = (DP.get(arch, {}).get(g) or {}); dp = dp.get("dpsnr") if isinstance(dp, dict) else None; dp = dp.get("median") if isinstance(dp, dict) else dp
        au = AU.get(f"{arch}/{g}") or {}
        rows.append(dict(arch=arch, game=g, absent=int(c["present"] < 0.5), misplaced=int(c["hit8"] < 0.5), absent_t=int(au["template_present"] < 0.5) if au else None, misplaced_t=int(au["hit8_template"] < 0.5) if au else None, wrong_t=int(au["hit8_template_strict"] < 0.5) if au else None, worse_t=(int(au["frac_better"] < 0.5) if (au and au.get("n_pairs", 0) >= 3) else None), consistency=pg["20"]["consistency"], psnr=pg["20"]["psnr"], A=pg["20"]["A"], R=pg["20"]["R"], eps=pg["20"]["err"],
                         dpsnr=dp, offsupport=(sp.get("offsupport") or {}).get("median")))
def auc(x, y):
    x = np.asarray(x, float); y = np.asarray(y, int); pos = x[y == 1]; neg = x[y == 0]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    return float((np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])))
# sign convention: score is oriented so that HIGHER = more likely ABSENT (blind): low A, low PSNR? We report AUC of (-score) for A, R, consistency, PSNR, dpsnr and of (+score) for eps, offsupport, and give the direction.
SCORES = {"consistency (one rollout)": ("consistency", -1), "PSNR (reference, one arm)": ("psnr", -1), "dPSNR (two rollouts, factual reference)": ("dpsnr", -1),
          "alignment A (this work)": ("A", -1), "magnitude R (this work)": ("R", -1), "error eps (this work)": ("eps", +1), "off-support fraction O (this work)": ("offsupport", +1)}
res = {"n_cells": len(rows), "n_absent": int(sum(r["absent"] for r in rows)), "n_misplaced": int(sum(r["misplaced"] for r in rows)), "scores": {}, "scores_misplaced": {}}
rng = np.random.default_rng(0); games = sorted({r["game"] for r in rows})
if not os.environ.get("AUC_TABLES_ONLY"):
    for target, store in (("absent", "scores"), ("misplaced", "scores_misplaced")):
      print(f"== target: {target}")
      for lab, (k, sgn) in SCORES.items():
          sub = [r for r in rows if r.get(k) is not None and r[k] == r[k]]
          if len(sub) < 10: res[store][lab] = None; continue
          x = [sgn * r[k] for r in sub]; y = [r[target] for r in sub]; a = auc(x, y)
          boots = []
          for _ in range(2000):
              gsel = rng.choice(games, len(games), replace=True); bs = [r for g in gsel for r in sub if r["game"] == g]
              boots.append(auc([sgn * r[k] for r in bs], [r[target] for r in bs]))
          perm = []
          for _ in range(5000):
              yp = np.array(y).copy()
              for arch in ("DIAMOND", "IRIS", "STORM"):
                  idx = [i for i, r in enumerate(sub) if r["arch"] == arch]; yp[idx] = rng.permutation(yp[idx])
              perm.append(auc(x, yp))
          res[store][lab] = dict(key=k, direction="higher score = present" if sgn < 0 else "higher score = absent", n=len(sub), auc=a, ci=[float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))], p=float((np.sum(np.asarray(perm) >= a) + 1) / (len(perm) + 1)))
          print(f"{lab:<42} n {len(sub):3d} AUC {a:.2f} [{res[store][lab]['ci'][0]:.2f}, {res[store][lab]['ci'][1]:.2f}] p {res[store][lab]['p']:.4f}")
    # robustness: the same AUCs on the cells whose true divergence at delay 20 is above the median across cells
    _td = {(r["arch"], r["game"]): E["per_game"][r["arch"]][r["game"]]["20"].get("tdiv") for r in rows}
    _med = float(np.median([v for v in _td.values() if v is not None])); hi = [r for r in rows if (_td.get((r["arch"], r["game"])) or 0) > _med]
    res["high_tdiv_subset"] = {"n": len(hi), "tdiv_median_cut": _med, "scores": {}}
    for lab, (k, sgn) in SCORES.items():
        sub = [r for r in hi if r.get(k) is not None and r[k] == r[k]]
        if len(sub) >= 10: res["high_tdiv_subset"]["scores"][lab] = dict(auc_absent=auc([sgn * r[k] for r in sub], [r["absent"] for r in sub]), auc_misplaced=auc([sgn * r[k] for r in sub], [r["misplaced"] for r in sub]), n=len(sub))
    print("high-true-divergence subset (n=%d):" % len(hi), {lab: (round(v["auc_absent"], 2), round(v["auc_misplaced"], 2)) for lab, v in res["high_tdiv_subset"]["scores"].items()})
    # ---- (2026-09-15): (i) common cells only (every score defined), (ii) paired game-clustered bootstrap of the AUC
    # DIFFERENCE off-support minus each other score, (iii) a single clearly specified permutation: labels are permuted across GAMES, every
    # architecture's cell of a game moving together, (iv) sensitivity to excluding Jamesbond (locator validity 0.32-0.35 there) ----
    common = [r for r in rows if all(r.get(k) is not None and r[k] == r[k] for k, _ in SCORES.values())]
    def game_boot(sub, target, B=2000):
        gs = sorted({r["game"] for r in sub}); out = {lab: [] for lab in SCORES}
        for _ in range(B):
            gsel = rng.choice(gs, len(gs), replace=True); bs = [r for g in gsel for r in sub if r["game"] == g]; y = [r[target] for r in bs]
            for lab, (k, sgn) in SCORES.items(): out[lab].append(auc([sgn * r[k] for r in bs], y))
        return out
    def game_perm(sub, target, lab, B=5000):
        """Stratified game-block permutation (2026-09-21; replaces a scheme that reused a mapped game's last label when the
        games had different numbers of cells). Games are grouped by the SET of architectures present; within a stratum a random bijection of
        the games is drawn and every cell takes the label of the SAME architecture in the game it is mapped to. The permuted labels are then
        a permutation of the observed labels within every architecture, and games are exchangeable within a stratum under the null."""
        k, sgn = SCORES[lab]; x = [sgn * r[k] for r in sub]; obs = auc(x, [r[target] for r in sub]); cnt = 0
        lab_by = {(r["game"], r["arch"]): r[target] for r in sub}
        archs_of = {}
        for r in sub: archs_of.setdefault(r["game"], set()).add(r["arch"])
        strata = {}
        for g, a in archs_of.items(): strata.setdefault(frozenset(a), []).append(g)
        strata = {kk: sorted(v) for kk, v in strata.items()}
        for _ in range(B):
            mapping = {}
            for gs in strata.values():
                for g, g2 in zip(gs, rng.permutation(gs)): mapping[g] = g2
            yp = [lab_by[(mapping[r["game"]], r["arch"])] for r in sub]
            cnt += abs(auc(x, yp) - 0.5) >= abs(obs - 0.5) - 1e-12     # two-sided: permuted AUC at least as far from one half as the observed
        return float((cnt + 1) / (B + 1))
    res["common_cells"] = {"n": len(common), "n_absent": int(sum(r["absent"] for r in common)), "n_misplaced": int(sum(r["misplaced"] for r in common)), "scores": {}, "paired_diff_vs_offsupport": {}}
    for target, key in (("absent", "auc_absent"), ("misplaced", "auc_misplaced")):
        boots = game_boot(common, target); y = [r[target] for r in common]
        for lab, (k, sgn) in SCORES.items():
            a = auc([sgn * r[k] for r in common], y)
            res["common_cells"]["scores"].setdefault(lab, {})[key] = dict(auc=a, ci=[float(np.nanpercentile(boots[lab], 2.5)), float(np.nanpercentile(boots[lab], 97.5))], p_game_perm=game_perm(common, target, lab))
        ref = "off-support fraction O (this work)"
        for lab in SCORES:
            if lab == ref: continue
            d = [bo - bl for bo, bl in zip(boots[ref], boots[lab])]
            res["common_cells"]["paired_diff_vs_offsupport"].setdefault(lab, {})[key] = dict(diff=res["common_cells"]["scores"][ref][key]["auc"] - res["common_cells"]["scores"][lab][key]["auc"], ci=[float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))])
    res["audited"] = {}
    for target, key in (("absent_t", "auc_absent_t"), ("worse_t", "auc_worse_t"), ("wrong_t", "auc_wrong_t")):
        sub = [r for r in common if r.get(target) is not None]
        if len(sub) >= 10 and 0 < sum(r[target] for r in sub) < len(sub):
            boots = game_boot(sub, target); y = [r[target] for r in sub]; res["audited"][key] = {"n": len(sub), "n_pos": int(sum(y)), "scores": {}}
            for lab, (k, sgn) in SCORES.items():
                res["audited"][key]["scores"][lab] = dict(auc=auc([sgn * r[k] for r in sub], y), ci=[float(np.nanpercentile(boots[lab], 2.5)), float(np.nanpercentile(boots[lab], 97.5))], p_game_perm=game_perm(sub, target, lab))
            ref = "off-support fraction O (this work)"
            for lab in SCORES:
                if lab != ref: d = [bo - bl for bo, bl in zip(boots[ref], boots[lab])]; res["audited"][key]["scores"][lab]["diff_O_minus"] = dict(diff=res["audited"][key]["scores"][ref]["auc"] - res["audited"][key]["scores"][lab]["auc"], ci=[float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))])
            print(f"audited target {target} (n={len(sub)}, positives {int(sum(y))}):", {lab[:12]: (round(v["auc"], 2), round(v["p_game_perm"], 3)) for lab, v in res["audited"][key]["scores"].items()})
    nojb = [r for r in common if r["game"] != "Jamesbond"]
    res["common_cells"]["without_jamesbond"] = {lab: {key: auc([sgn * r[k] for r in nojb], [r[t] for r in nojb]) for t, key in (("absent", "auc_absent"), ("misplaced", "auc_misplaced"))} for lab, (k, sgn) in SCORES.items()}
    print("common cells n=%d:" % len(common))
    for lab in SCORES:
        c = res["common_cells"]["scores"][lab]; d = res["common_cells"]["paired_diff_vs_offsupport"].get(lab)
        print(f"  {lab:<42} absent {c['auc_absent']['auc']:.2f} [{c['auc_absent']['ci'][0]:.2f},{c['auc_absent']['ci'][1]:.2f}] p {c['auc_absent']['p_game_perm']:.4f} | misplaced {c['auc_misplaced']['auc']:.2f} [{c['auc_misplaced']['ci'][0]:.2f},{c['auc_misplaced']['ci'][1]:.2f}] p {c['auc_misplaced']['p_game_perm']:.4f}" + (f" | O minus this: {d['auc_absent']['diff']:+.2f} [{d['auc_absent']['ci'][0]:+.2f},{d['auc_absent']['ci'][1]:+.2f}] / {d['auc_misplaced']['diff']:+.2f} [{d['auc_misplaced']['ci'][0]:+.2f},{d['auc_misplaced']['ci'][1]:+.2f}]" if d else ""))
    print("without Jamesbond:", {lab[:14]: (round(v["auc_absent"], 2), round(v["auc_misplaced"], 2)) for lab, v in res["common_cells"]["without_jamesbond"].items()})
if os.environ.get("AUC_TABLES_ONLY"): res = json.load(open(OUT / "metric_auc.json"))   # reuse cached permutation results; rewrite the tables only
else: json.dump(res, open(OUT / "metric_auc.json", "w"), indent=1)
L = ["\\begin{table}[t]\\centering\\small", f"\\caption{{Which score predicts the object-level outcome? AUC, pooled over the three architectures, for separating cells in which the actuated object is not drawn (presence below one half at delay 20, {res['n_absent']} of {res['n_cells']} cells) and cells in which it is not placed within 8 pixels (hit@8 below one half, {res['n_misplaced']} of {res['n_cells']}) from the rest; each score is oriented so that an AUC above one half points the right way; game-clustered bootstrap intervals and within-architecture label-permutation $p$. The object-level readout is one none of these scores saw.}}", "\\label{tab:metric_auc}", "\\begin{tabular}{lrllll}\\toprule", "score & $n$ & AUC absent [95\\% CI] & $p$ & AUC misplaced [95\\% CI] & $p$ \\\\ \\midrule"]
fp = lambda v: '$<0.001$' if v['p'] < 1e-3 else f"{v['p']:.3f}"
cc = res["common_cells"]; AT = res["audited"].get("auc_absent_t"); WT = res["audited"].get("auc_worse_t"); WR = res["audited"].get("auc_wrong_t")
fp2 = lambda p: '$<0.001$' if p < 1e-3 else f"{p:.3f}"
L = ["\\begin{table}[t]\\centering\\scriptsize\\setlength{\\tabcolsep}{4pt}", f"\\caption{{Which score predicts the locator's labels? AUC, pooled over three architectures, for separating cells in which the template locator does not find the actuated object in both generated frames at delay 20 (presence below one half; {AT['n_pos']} of {AT['n']} cells where every score is defined) and cells in which the model's displacement of the object fails to improve on a zero displacement in a majority of the episodes with the object found in both arms, ties counting as no improvement ({WT['n_pos']} of {WT['n']} cells with at least three such episodes); every score is oriented so that an AUC above one half points the right way. Intervals: game-clustered bootstrap (games resampled, every architecture's cell of a game together); $p$: two-sided game-level permutation test of the labels (the fraction of 5000 game-permuted labelings whose AUC lies at least as far from one half as the observed AUC), uncorrected across the seven rows (protocol and paired differences between rows: Appendix~\\ref{{app:object}}).}}", "\\label{tab:metric_auc}", "\\begin{tabular}{p{3.6cm}p{3.6cm}p{3.6cm}}\\toprule", "score & AUC, object not found [95\\% CI], $p$ & AUC, no improvement on zero displacement [95\\% CI], $p$ \\\\ \\midrule"]
for lab in SCORES:
    v = AT["scores"][lab]; w = WT["scores"][lab]
    L.append(f"{lab} & {v['auc']:.2f} [{v['ci'][0]:.2f}, {v['ci'][1]:.2f}], {fp2(v['p_game_perm'])} & {w['auc']:.2f} [{w['ci'][0]:.2f}, {w['ci'][1]:.2f}], {fp2(w['p_game_perm'])} \\\\")
L += ["\\bottomrule\\end{tabular}\\end{table}"]
(P / "tab_metric_auc.tex").write_text("\n".join(L) + "\n")
# appendix version: both audited labels plus the superseded strict label and the paired differences against the off-support row
L = ["\\begin{table}[h]\\centering\\scriptsize\\setlength{\\tabcolsep}{2.2pt}", f"\\caption{{Which score predicts the locator's labels? AUC, pooled over three architectures, for separating cells by two automated labels derived from the template locator (Appendix~\\ref{{app:object}}; these labels are locator outputs, which the human audit of Table~\\ref{{tab:human_compact}} does not validate as facts about the models): column 2, the object found in both generated frames at delay 20 in fewer than half of the episodes ({AT['n_pos']} of {AT['n']} cells where every score is defined); column 3, the estimated displacement error strictly smaller than a zero displacement's in fewer than half of the episodes with the object found in both arms, ties counting as no improvement ({WT['n_pos']} of {WT['n']} cells with at least three such episodes); every score is oriented so that an AUC above one half points the right way. Intervals: game-clustered bootstrap (games resampled, every architecture's cell of a game together); $p$: two-sided game-level permutation test (the fraction of 5000 game-permuted labelings whose AUC lies at least as far from one half as the observed AUC), uncorrected; the two labels are treated as separate seven-row families for Holm correction. Column 4 uses the stricter strict-hit label (a hit within 8 pixels that a zero displacement would not have passed, below one half of all episodes; {WR['n_pos']} of {WR['n']} cells), which counts ineligible episodes as failures and is listed for comparison; the last two columns give the paired game-clustered difference between the off-support fraction's AUC and the row's for the two automated labels.}}", "\\label{tab:metric_auc_full}", "\\begin{tabular}{p{2.1cm}p{2.2cm}p{2.2cm}p{2.2cm}p{1.9cm}p{1.9cm}}\\toprule", "score & not found [CI], $p$ & no improvement on zero [CI], $p$ & strict hit [CI], $p$ & $O$ minus row, not found & $O$ minus row, no improvement \\\\ \\midrule"]
for lab in SCORES:
    v = AT["scores"][lab]; w = WT["scores"][lab]; x = WR["scores"][lab]; d1 = v.get("diff_O_minus"); d2 = w.get("diff_O_minus")
    L.append(f"{lab} & {v['auc']:.2f} [{v['ci'][0]:.2f}, {v['ci'][1]:.2f}], {fp2(v['p_game_perm'])} & {w['auc']:.2f} [{w['ci'][0]:.2f}, {w['ci'][1]:.2f}], {fp2(w['p_game_perm'])} & {x['auc']:.2f} [{x['ci'][0]:.2f}, {x['ci'][1]:.2f}], {fp2(x['p_game_perm'])} & " + (f"{d1['diff']:+.2f} [{d1['ci'][0]:+.2f}, {d1['ci'][1]:+.2f}]" if d1 else "--") + " & " + (f"{d2['diff']:+.2f} [{d2['ci'][0]:+.2f}, {d2['ci'][1]:+.2f}]" if d2 else "--") + " \\\\")
L += ["\\bottomrule\\end{tabular}\\end{table}"]
(P / "tab_metric_auc_full.tex").write_text("\n".join(L) + "\n"); print("wrote outputs/metric_auc.json,", P / "tab_metric_auc.tex", "and tab_metric_auc_full.tex")
