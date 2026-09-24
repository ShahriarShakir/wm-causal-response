"""Object-readout audit evaluation (rules in ext/object_audit/README.md). Reads ext/outputs/objaudit_*.json (all 78 cells, 6 episodes) and
outputs/object_eval.json (taxonomy classes), writes outputs/object_audit_eval.json, PAPER_DIR/tab_object_audit.tex, and the blinded
human-annotation sample (ext/object_audit/human_sample.csv + human_key.json). Usage: audit_eval.py [--with-human ext/object_audit/human_labels.csv]"""
import json, glob, os, pathlib, sys, csv, numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT)); OUT = ROOT / "outputs"; EXT = ROOT / "ext" / "outputs"; P = pathlib.Path(os.environ.get("PAPER_DIR", ROOT / "paper")); HD = ROOT / "ext" / "object_audit"
O = json.load(open(OUT / "object_eval.json")); cls = {(a, g): c["cls"] for a in ("DIAMOND", "IRIS", "STORM") for g, c in O[a]["cells"].items()}
cells = {}
TAG = os.environ.get("OA_TAG", "")
for f in sorted(glob.glob(str(EXT / f"objaudit{TAG}_*.json"))):
    if "DRY" in f: continue
    d = json.load(open(f)); a = d["arch"].upper(); g = d["game"].split("/")[-1].replace("-v5", "")
    eps = [r["delays"]["20"] for r in d["raw"] if "20" in r["delays"] and r["delays"]["20"].get("template")]
    if not eps: continue
    col = [r["colour"]["present"] for r in eps]; tem = [bool(r["template"].get("present")) for r in eps]; selfok = [r["template"].get("self_score", 0) >= 0.9 for r in eps]
    zero8 = [r["zero_disp_passes"]["8"] for r in eps]
    hit_c = [bool(r["colour"].get("hit", {}).get("8")) for r in eps]; hit_t = [bool(r["template"].get("hit", {}).get("8")) for r in eps]
    hit_c_strict = [bool(r["colour"].get("hit", {}).get("8")) and not r["zero_disp_passes"]["8"] for r in eps]     # a hit that a zero displacement would NOT have passed
    hit_t_strict = [bool(r["template"].get("hit", {}).get("8")) and not r["zero_disp_passes"]["8"] for r in eps]
    # ---- corrected displacement endpoint: eligibility and paired error against the zero baseline ----
    elig = [not r["zero_disp_passes"]["8"] for r in eps]                                   # true displacement > 8 px: a zero prediction would fail
    found_elig = [bool(r["template"].get("present")) and e for r, e in zip(eps, elig)]
    hit_elig = [bool(r["template"].get("hit", {}).get("8")) and e for r, e in zip(eps, elig)]
    pairs = [(float(np.linalg.norm(np.array(r["template"]["model_disp"]) - np.array(r["true_disp"]))), float(np.linalg.norm(np.array(r["true_disp"])))) for r in eps if r["template"].get("present") and "model_disp" in r["template"]]
    corr = dict(n_elig=int(sum(elig)), n_found_elig=int(sum(found_elig)), n_hit_elig=int(sum(hit_elig)), acc_elig=(sum(hit_elig) / sum(found_elig)) if sum(found_elig) else None,
                n_pairs=len(pairs), model_err=float(np.median([x for x, _ in pairs])) if pairs else None, zero_err=float(np.median([y for _, y in pairs])) if pairs else None,
                frac_better=float(np.mean([x < y for x, y in pairs])) if pairs else None, err_diff=float(np.median([x - y for x, y in pairs])) if pairs else None)
    cells[(a, g)] = dict(arch=a, game=g, cls=cls.get((a, g), "?"), n=len(eps), colour_present=float(np.mean(col)), template_present=float(np.mean(tem)), self_ok=float(np.mean(selfok)),
                         zero_disp_passes_8=float(np.mean(zero8)), true_disp_norm=float(np.median([r["true_disp_norm"] for r in eps])),
                         hit8_colour=float(np.mean(hit_c)), hit8_template=float(np.mean(hit_t)), hit8_colour_strict=float(np.mean(hit_c_strict)), hit8_template_strict=float(np.mean(hit_t_strict)), **corr,
                         agree=float(np.mean([c == t for c, t in zip(col, tem)])), colour_fn=int(sum(1 for c, t in zip(col, tem) if t and not c)), colour_fp=int(sum(1 for c, t in zip(col, tem) if c and not t)), tem_pos=int(sum(tem)), tem_neg=int(len(tem) - sum(tem)))
res = {"cells": {f"{a}/{g}": v for (a, g), v in cells.items()}, "per_arch": {}, "never_aligned": {}}
for a in ("DIAMOND", "IRIS", "STORM"):
    cs = [v for (aa, g), v in cells.items() if aa == a]
    tp = sum(1 for v in cs for _ in range(v["tem_pos"])) ; fn = sum(v["colour_fn"] for v in cs); fp = sum(v["colour_fp"] for v in cs); tn = sum(v["tem_neg"] for v in cs) - fp
    res["per_arch"][a] = dict(n_cells=len(cs), colour_recall_vs_template=(tp - fn) / tp if tp else None, colour_precision_vs_template=(tp - fn) / max(tp - fn + fp, 1), episode_agreement=float(np.mean([v["agree"] for v in cs])), self_ok=float(np.mean([v["self_ok"] for v in cs])), zero_disp_passes_8=float(np.mean([v["zero_disp_passes_8"] for v in cs])))
    na = [v for v in cs if v["cls"] == "never-aligned"]
    def split(key_present, key_hit):
        absent = [v for v in na if v[key_present] < 0.5]; rest = [v for v in na if v[key_present] >= 0.5]
        tol = [v for v in rest if v[key_hit] >= 0.5]; mis = [v for v in rest if v[key_hit] < 0.5]; return dict(n=len(na), absent=len(absent), misplaced=len(mis), tolerant=len(tol), absent_games=[v["game"] for v in absent], tolerant_games=[v["game"] for v in tol])
    res["never_aligned"][a] = dict(colour=split("colour_present", "hit8_colour"), template=split("template_present", "hit8_template"), template_strict=split("template_present", "hit8_template_strict"))
    print(f"{a}: cells {len(cs)} | colour vs template: recall {res['per_arch'][a]['colour_recall_vs_template']}, precision {res['per_arch'][a]['colour_precision_vs_template']:.2f}, episode agreement {res['per_arch'][a]['episode_agreement']:.2f} | self-ok {res['per_arch'][a]['self_ok']:.2f} | zero-disp passes hit@8 in {res['per_arch'][a]['zero_disp_passes_8']:.2f} of episodes | never-aligned colour {res['never_aligned'][a]['colour']} | template {res['never_aligned'][a]['template']} | template strict {res['never_aligned'][a]['template_strict']}")
for g in ("Alien", "Pong", "Breakout", "Asterix", "BankHeist"):
    print(f"  {g}: " + " | ".join(f"{a} colour {cells[(a,g)]['colour_present']:.2f} template {cells[(a,g)]['template_present']:.2f} zero8 {cells[(a,g)]['zero_disp_passes_8']:.2f} hit8 t/strict {cells[(a,g)]['hit8_template']:.2f}/{cells[(a,g)]['hit8_template_strict']:.2f}" for a in ("DIAMOND", "IRIS", "STORM") if (a, g) in cells))
# ---- human sample: 60 panels, stratified (arch x class x colour outcome) + the named games, seeded ----
keys = {}
for f in sorted(glob.glob(str(EXT / "objaudit_*.json"))):
    if "DRY" in f: continue
    keys.update(json.load(open(f))["panel_key"])
rng = np.random.default_rng(7); by = {}
for pid, k in keys.items(): by.setdefault((k["arch"], cls.get((k["arch"].upper(), k["game"]), "?")), []).append(pid)
sample = set()
for pid, k in keys.items():
    if k["game"] in ("Alien", "Pong", "Breakout", "Asterix", "BankHeist") and k["episode"] == 0: sample.add(pid)
for (a, c), pids in sorted(by.items()):
    pool = [p for p in pids if p not in sample]; rng.shuffle(pool); sample.update(pool[:4])
sample = sorted(sample)[:60] if len(sample) > 60 else sorted(sample)
if (HD / "human_sample.csv").exists() and (HD / "human_key.json").exists():   # FROZEN 2026-09-17: the 60 panels were handed to the annotator; never resample
    sample = [row["panel"] for row in csv.DictReader(open(HD / "human_sample.csv"))]
    print("human sample frozen:", len(sample), "panels (existing human_sample.csv kept)")
else:
  with open(HD / "human_sample.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["panel", "present_generated_a (yes/no/unsure)", "present_generated_b (yes/no/unsure)", "moved_between_arms_in_true_direction (yes/no/unsure)", "notes"])
    for pid in sample: w.writerow([pid, "", "", "", ""])
  json.dump({pid: keys[pid] for pid in sample}, open(HD / "human_key.json", "w"), indent=1)
res["human_sample"] = {"n": len(sample), "panels": sample}
# ---- optional: human labels ----
hl = None
if "--with-human" in sys.argv:
    # Human labels (blinded). CSV columns: panel, B present a, C present b, D moved in the true direction, E moved about as far, notes.
    # Rules fixed before the first labels were read (2026-09-18): a panel enters the presence comparison when B and C are both yes/no
    # ("unsure" excluded and counted); human "present in both arms" = B yes and C yes. A panel enters the displacement comparison when its
    # labels yield a verdict: D no = does not improve; D yes with E in {yes, "no - too little"} = improves (right direction, not overshooting);
    # D unsure, or D yes with E "no - too much" / unsure = undecided (excluded, counted). The human rule is a qualitative proxy for the
    # pixel endpoint (an overshoot can still beat zero). The locator's per-episode verdicts come from the same episode of the 30-episode
    # files (episodes 0 to 5 are identical to the run the panels were drawn from). The example panel used as the worked example is excluded.
    # Several annotators (2026-09-19, rule fixed before the second sheet was integrated): --with-human a.csv,b.csv[,c.csv]. Each sheet is
    # evaluated on its own; pairwise Cohen's kappa is computed on the derived labels over the panels both decided; a CONSENSUS label exists
    # when at least two annotators decided and every annotator who decided agrees (any disagreement = undecided); res["human"] is the
    # consensus block, with the per-annotator blocks under "annotators" and the agreement under "agreement".
    paths = sys.argv[sys.argv.index("--with-human") + 1].split(","); EXCL = set(os.environ.get("HUMAN_EXCLUDE", "").split(",")) - {""}
    def read_labels(path):
        hl = {}; n_unsure_presence = 0; n_undecided_disp = 0
        for row in csv.DictReader(open(path)):
            v = list(row.values()); pid = v[0].strip(); a_, b_, d_, e_ = [(x or "").strip().lower() for x in v[1:5]]; note = (v[5] or "").strip() if len(v) > 5 else ""
            if pid not in keys or pid in EXCL: continue
            rec = dict(note=note, overshoot=(d_ == "yes" and e_ == "no - too much"))
            if a_ in ("yes", "no") and b_ in ("yes", "no"): rec["present"] = (a_ == "yes" and b_ == "yes")
            else: n_unsure_presence += 1
            if d_ == "no": rec["improves"] = False
            elif d_ == "yes" and e_ in ("yes", "no - too little"): rec["improves"] = True
            elif d_ == "yes" or d_ == "unsure": n_undecided_disp += 1
            hl[pid] = rec
        return hl, n_unsure_presence, n_undecided_disp
    def locator(pid):
        k = keys[pid]; d = json.load(open(EXT / f"objaudit{TAG}_{k['arch']}_{k['game']}.json")); r = [x for x in d["raw"] if x["episode"] == k["episode"]][0]["delays"]["20"]; t = r["template"]
        impr = None
        if t.get("present") and "model_disp" in t: impr = float(np.linalg.norm(np.array(t["model_disp"]) - np.array(r["true_disp"]))) < float(np.linalg.norm(np.array(r["true_disp"])))
        return r["colour"]["present"], bool(t.get("present")), impr
    LOC = {pid: locator(pid) for pid in keys if pid not in EXCL}
    def kappa_of(pairs):
        n = len(pairs)
        if n == 0: return float("nan")
        po = sum(x == y for x, y in pairs) / n; cx = {}; cy = {}
        for x, y in pairs: cx[x] = cx.get(x, 0) + 1; cy[y] = cy.get(y, 0) + 1
        pe = sum(cx.get(k, 0) * cy.get(k, 0) for k in set(cx) | set(cy)) / n / n
        return (po - pe) / (1 - pe) if pe < 1 else float("nan")
    def kappa_ci(pairs_by_game, n_boot=2000, seed=0):
        """percentile interval of Cohen's kappa under a game-clustered bootstrap (games resampled with replacement, every panel of a game together)"""
        games = sorted(g for g in pairs_by_game if pairs_by_game[g]); rng = np.random.default_rng(seed); vals = []
        for _ in range(n_boot):
            draw = rng.choice(len(games), len(games), replace=True); pairs = [pr for i in draw for pr in pairs_by_game[games[i]]]
            k_ = kappa_of(pairs)
            if k_ == k_: vals.append(k_)
        return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if len(vals) >= 100 else [float("nan"), float("nan")]
    def by_game(pairs_with_pid):
        d = {}
        for pid, pr in pairs_with_pid: d.setdefault(keys[pid]["game"], []).append(pr)
        return d
    def evaluate(hl, n_unsure_presence, n_undecided_disp):
        conf = {"colour": [0, 0, 0, 0], "template": [0, 0, 0, 0]}   # tp, fn, fp, tn against the human presence label
        disp = {"tp": 0, "fn": 0, "fp": 0, "tn": 0, "n_locator_missing": 0}    # locator "improves on zero" (model err < zero err) vs human
        ov = {"n": 0, "locator_found": 0, "locator_improves": 0}
        for pid, h in hl.items():
            cpres, tpres, impr = LOC[pid]
            if "present" in h:
                for loc, pres in (("colour", cpres), ("template", tpres)):
                    idx = 0 if (h["present"] and pres) else 1 if (h["present"] and not pres) else 2 if (not h["present"] and pres) else 3; conf[loc][idx] += 1
            if "improves" in h:
                if impr is not None:
                    key_ = "tp" if (h["improves"] and impr) else "fn" if (h["improves"] and not impr) else "fp" if (not h["improves"] and impr) else "tn"; disp[key_] += 1
                else: disp["n_locator_missing"] += 1
            if h.get("overshoot"):
                ov["n"] += 1
                if impr is not None: ov["locator_found"] += 1; ov["locator_improves"] += int(impr)
        pr = lambda c: dict(recall=c[0] / max(c[0] + c[1], 1), precision=c[0] / max(c[0] + c[2], 1), accuracy=(c[0] + c[3]) / max(sum(c), 1), n=sum(c))
        nd = disp["tp"] + disp["fn"] + disp["fp"] + disp["tn"]
        def kap(pairs):
            n = len(pairs)
            if n == 0: return float("nan")
            po = sum(x == y for x, y in pairs) / n; cx = {}; cy = {}
            for x, y in pairs: cx[x] = cx.get(x, 0) + 1; cy[y] = cy.get(y, 0) + 1
            pe = sum(cx.get(k, 0) * cy.get(k, 0) for k in set(cx) | set(cy)) / n / n
            return (po - pe) / (1 - pe) if pe < 1 else float("nan")
        res_h = {"n_panels": len(hl), "n_excluded_example": len(EXCL),
                 "presence": {"n_compared": sum(conf["template"]), "n_unsure": n_unsure_presence, "confusion_tp_fn_fp_tn": conf, "colour": pr(conf["colour"]), "template": pr(conf["template"]),
                              "kappa_template": kap([(h["present"], LOC[pid][1]) for pid, h in hl.items() if "present" in h]),
                              "kappa_template_ci": kappa_ci(by_game([(pid, (h["present"], LOC[pid][1])) for pid, h in hl.items() if "present" in h]))},
                 "displacement": {"n_compared": nd, "n_undecided_human": n_undecided_disp, "n_locator_missing": disp["n_locator_missing"], "confusion": disp, "agreement": (disp["tp"] + disp["tn"]) / max(nd, 1),
                                  "kappa": kap([(h["improves"], LOC[pid][2]) for pid, h in hl.items() if "improves" in h and LOC[pid][2] is not None]),
                                  "kappa_ci": kappa_ci(by_game([(pid, (h["improves"], LOC[pid][2])) for pid, h in hl.items() if "improves" in h and LOC[pid][2] is not None])),
                                  "human_improves_rate": (disp["tp"] + disp["fn"]) / max(nd, 1), "locator_improves_rate": (disp["tp"] + disp["fp"]) / max(nd, 1)},
                 "overshoot": ov, "notes_unclear": sorted(pid for pid, h in hl.items() if "unclear" in h["note"].lower())}
        # per-class breakdown (taxonomy class of the panel's cell)
        byc = {}
        for pid, h in hl.items():
            k = keys[pid]; c = cls.get((k["arch"].upper(), k["game"]), "?"); B = byc.setdefault(c, dict(n=0, unsure_presence=0, human_present=0, human_absent=0, template_found_of_present=0, disp_compared=0, human_improves=0, locator_improves=0, agree=0, locator_missing=0, improves_locator_missing=0, unclear=0, tp=0, fn=0, fp=0, tn=0))
            B["n"] += 1; B["unclear"] += ("unclear" in h["note"].lower()); cpres, tpres, impr = LOC[pid]
            if "present" in h:
                if h["present"]: B["human_present"] += 1; B["template_found_of_present"] += int(tpres)
                else: B["human_absent"] += 1
            else: B["unsure_presence"] += 1
            if "improves" in h:
                if impr is not None:
                    B["disp_compared"] += 1; B["human_improves"] += int(h["improves"]); B["locator_improves"] += int(impr); B["agree"] += int(impr == h["improves"])
                    B["tp" if (h["improves"] and impr) else "fn" if (h["improves"] and not impr) else "fp" if (not h["improves"] and impr) else "tn"] += 1
                else: B["locator_missing"] += 1; B["improves_locator_missing"] += int(h["improves"])
        for c, B in byc.items(): B["template_recall"] = B["template_found_of_present"] / max(B["human_present"], 1)
        res_h["by_class"] = byc
        return res_h
    sets = [read_labels(pth) for pth in paths]
    blocks = [evaluate(*s_) for s_ in sets]
    if len(sets) == 1:
        hl = sets[0][0]; res["human"] = blocks[0]
    else:
        # pairwise agreement on the derived labels (panels both decided) and the consensus labels
        def kappa2(pairs):
            n = len(pairs); po = sum(x == y for x, y in pairs) / max(n, 1); cx = {}; cy = {}
            for x, y in pairs: cx[x] = cx.get(x, 0) + 1; cy[y] = cy.get(y, 0) + 1
            pe = sum(cx.get(k, 0) * cy.get(k, 0) for k in set(cx) | set(cy)) / max(n, 1) ** 2
            return {"n": n, "agreement": po, "kappa": ((po - pe) / (1 - pe) if pe < 1 else float("nan"))}
        agreement = {}
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                a_, b_ = sets[i][0], sets[j][0]
                agreement[f"{i + 1}-{j + 1}"] = {}
                for key in ("present", "improves"):
                    pw = [(pid, (a_[pid][key], b_[pid][key])) for pid in a_ if key in a_[pid] and pid in b_ and key in b_[pid]]
                    agreement[f"{i + 1}-{j + 1}"][key] = kappa2([pr for _, pr in pw])
                    agreement[f"{i + 1}-{j + 1}"][key]["kappa_ci"] = kappa_ci(by_game(pw))
                    agreement[f"{i + 1}-{j + 1}"][key]["table"] = {"both_yes": sum(1 for _, (x, y) in pw if x and y), "first_yes_second_no": sum(1 for _, (x, y) in pw if x and not y), "first_no_second_yes": sum(1 for _, (x, y) in pw if (not x) and y), "both_no": sum(1 for _, (x, y) in pw if not x and not y)}
        hl = {}; n_unsure_presence = 0; n_undecided_disp = 0; n_disagree = {"present": 0, "improves": 0}
        for pid in sets[0][0]:
            rec = dict(note=" | ".join(s_[0][pid]["note"] for s_ in sets if pid in s_[0] and s_[0][pid]["note"]), overshoot=all(s_[0][pid].get("overshoot") for s_ in sets if pid in s_[0]))
            for key in ("present", "improves"):
                vals = [s_[0][pid][key] for s_ in sets if pid in s_[0] and key in s_[0][pid]]
                if len(vals) >= 2 and len(set(vals)) == 1: rec[key] = vals[0]
                else:
                    if len(set(vals)) > 1: n_disagree[key] += 1
                    if key == "present": n_unsure_presence += 1
                    else: n_undecided_disp += 1
            hl[pid] = rec
        res["human"] = evaluate(hl, n_unsure_presence, n_undecided_disp)
        res["human"]["sample"] = {"n_panels": len(hl), "n_games": len({keys[p]["game"] for p in hl}), "n_cells": len({(keys[p]["arch"], keys[p]["game"]) for p in hl}), "per_arch": {a: sum(1 for p in hl if keys[p]["arch"] == a) for a in ("diamond", "iris", "storm")},
                                "panel": "five tiles: true frame a, true frame b, generated frame a, generated frame b, true frame at the first difference with the actuated object boxed; upscaled; no text; file name = first 10 hex digits of md5(architecture-game-episode)",
                                "draw": "seeded stratified draw: up to 4 panels per (architecture, taxonomy class) plus episode 0 of Alien, Pong, Breakout, Asterix and BankHeist for every architecture; presented in the order of the hashed names"}
        res["human"].update({"n_annotators": len(sets), "consensus_rule": "at least two annotators decided and all who decided agree; any disagreement = undecided", "n_disagree": n_disagree, "agreement": agreement, "annotators": blocks, "label_files": [os.path.basename(x) for x in paths]})
    if hl:
        order = [c for c in ("tracked", "aligned-then-forgotten", "never-aligned", "partial") if c in res["human"]["by_class"]]
        if len(sets) == 1:
            byc = res["human"]["by_class"]
            T = ["\\begin{table}[h]\\centering\\scriptsize\\setlength{\\tabcolsep}{4pt}", "\\caption{Blinded human check of the object readout, %d panels (one example panel excluded), by taxonomy class of the panel's cell. \\emph{present / absent / unsure}: the annotator's presence label (present = seen in both generated arms); \\emph{recall}: fraction of the panels the annotator marked present in which the template locator found the object; \\emph{compared}: panels on which both the annotator (direction no; or direction yes with magnitude about right or too little; other label combinations are undecided) and the locator (object found with a displacement) could judge whether the predicted displacement improves on zero, the annotator's rule being a qualitative proxy for the pixel endpoint; \\emph{human} and \\emph{locator}: how many of those each judged as improving; \\emph{agree}: panels on which the two verdicts coincide; \\emph{locator missing}: panels the annotator could judge but the locator could not; \\emph{unclear}: panels the annotator flagged as having an unclear target.}" % len(hl), "\\label{tab:human}", "\\begin{tabular}{lrrrrrrrrr}\\toprule", "class & panels & present / absent / unsure & recall & compared & human & locator & agree & locator missing & unclear \\\\ \\midrule"]
            for c in order:
                B = byc[c]; T.append("%s & %d & %d / %d / %d & %.2f & %d & %d & %d & %d & %d & %d \\\\" % (c, B["n"], B["human_present"], B["human_absent"], B["unsure_presence"], B["template_recall"], B["disp_compared"], B["human_improves"], B["locator_improves"], B["agree"], B["locator_missing"], B["unclear"]))
        else:
            H = res["human"]; ag = H["agreement"]; pairs_txt = "; ".join("annotators %s: presence %.2f over %d panels, improvement %.2f over %d" % (k, v["present"]["kappa"], v["present"]["n"], v["improves"]["kappa"], v["improves"]["n"]) for k, v in ag.items())
            T = ["\\begin{table}[h]\\centering\\scriptsize\\setlength{\\tabcolsep}{2pt}", "\\caption{Blinded human check of the object readout by %d annotators on the same %d panels (one example panel excluded), by taxonomy class of the panel's cell; each annotator evaluated on their own labels, then on the consensus labels (a label exists when at least two annotators decided and all who decided agree; any disagreement counts as undecided). Cohen's kappa between annotators on the derived labels over the panels both decided: %s. \\emph{present / absent / unsure}: presence label (present = seen in both generated arms); \\emph{recall}: fraction of the panels labelled present in which the template locator found the object; \\emph{compared}: panels on which both the annotator (direction no; or direction yes with magnitude about right or too little; other combinations undecided) and the locator (object found with a displacement) could judge whether the predicted displacement improves on zero, the human rule being a qualitative proxy for the pixel endpoint; \\emph{human} and \\emph{locator}: how many of those each judged as improving; \\emph{agree}: panels on which the two verdicts coincide; \\emph{missing}: panels the annotator could judge but the locator could not (in brackets, how many of those the annotator judged as improving).}" % (H["n_annotators"], H["n_panels"], pairs_txt), "\\label{tab:human}", "\\begin{tabular}{llrrrrrrrr}\\toprule", "labels & class & panels & present / absent / unsure & recall & compared & human & locator & agree & missing (improving) \\\\ \\midrule"]
            groups = [("annotator %d" % (i + 1), b) for i, b in enumerate(H["annotators"])] + [("consensus", H)]
            for gi, (gname, blk) in enumerate(groups):
                if gi: T.append("\\midrule")
                for ci_, c in enumerate(order):
                    B = blk["by_class"][c]; T.append("%s & %s & %d & %d / %d / %d & %.2f & %d & %d & %d & %d & %d (%d) \\\\" % (gname if ci_ == 0 else "", c, B["n"], B["human_present"], B["human_absent"], B["unsure_presence"], B["template_recall"], B["disp_compared"], B["human_improves"], B["locator_improves"], B["agree"], B["locator_missing"], B["improves_locator_missing"]))
        T += ["\\bottomrule\\end{tabular}\\end{table}"]; (P / "tab_human.tex").write_text("\n".join(T) + "\n"); print("wrote", P / "tab_human.tex")
        if len(sets) > 1:
            H = res["human"]; AG = H["agreement"]; sm = H["sample"]; NA = len(H["annotators"]); blocks_ = [("annotator %d" % (i + 1), b) for i, b in enumerate(H["annotators"])] + [("consensus", H)]
            f2 = lambda v: ("$%.2f$" % v) if v < 0 else ("%.2f" % v); ci = lambda c: "[%s, %s]" % (f2(c[0]), f2(c[1]))
            dec = lambda b, c: "%d/%d" % (b["by_class"][c]["human_present"], b["by_class"][c]["human_present"] + b["by_class"][c]["human_absent"])
            WORD = {2: "two", 3: "three", 4: "four"}.get(NA, str(NA)); pk = sorted(AG)
            if NA == 2:
                ag = AG["1-2"]; kap_txt = "Between the annotators, Cohen's kappa is %.2f %s on presence (%d panels both decided) and %.2f %s on improvement (%d)." % (ag["present"]["kappa"], ci(ag["present"]["kappa_ci"]), ag["present"]["n"], ag["improves"]["kappa"], ci(ag["improves"]["kappa_ci"]), ag["improves"]["n"]); rule = "both decided and agree; disagreement is undecided"
            else:
                kap_txt = "Pairwise Cohen's kappa between annotators: %.2f to %.2f on presence and %.2f to %.2f on improvement (per pair in Appendix~\\ref{app:object})." % (min(AG[k]["present"]["kappa"] for k in pk), max(AG[k]["present"]["kappa"] for k in pk), min(AG[k]["improves"]["kappa"] for k in pk), max(AG[k]["improves"]["kappa"] for k in pk)); rule = "at least two decided and all who decided agree"
            C = ["\\begin{table}[t]\\centering\\scriptsize\\setlength{\\tabcolsep}{%s}" % ("3pt" if NA == 2 else "2pt"),
                 "\\caption{Blinded human audit of the object-level readout: %d panels (%d games, %d cells) annotated independently by %s authors, and the consensus labels (%s). %s \\emph{No verdict}: the locator did not find the object; intervals for every kappa, the protocol, decision rules, contingency tables and the breakdown by class are in Appendix~\\ref{app:object}.}" % (sm["n_panels"], sm["n_games"], sm["n_cells"], WORD, rule, kap_txt),
                 "\\label{tab:human_compact}", "\\begin{tabular}{p{%scm}%s}\\toprule" % ("5.9" if NA == 2 else "5.2", "c" * (NA + 1)), " & " + " & ".join(n for n, _ in blocks_) + " \\\\ \\midrule"]
            rows = [("presence: panels decided / unsure", lambda b: "%d / %d" % (b["presence"]["n_compared"], b["presence"]["n_unsure"])),
                    ("locator recall / precision vs annotator", lambda b: "%.2f / %.2f" % (b["presence"]["template"]["recall"], b["presence"]["template"]["precision"])),
                    ("kappa, locator presence vs annotator", lambda b: ("%s %s" % (f2(b["presence"]["kappa_template"]), ci(b["presence"]["kappa_template_ci"]))) if NA == 2 else f2(b["presence"]["kappa_template"])),
                    ("sees object: tracked, never-aligned (of decided)", lambda b: "%s, %s" % (dec(b, "tracked"), dec(b, "never-aligned"))),
                    ("displacement compared (undecided; no verdict)", lambda b: "%d (%d; %d)" % (b["displacement"]["n_compared"], b["displacement"]["n_undecided_human"], b["displacement"]["n_locator_missing"])),
                    ("agreement / kappa, locator vs annotator", lambda b: ("%.2f / %s %s" % (b["displacement"]["agreement"], f2(b["displacement"]["kappa"]), ci(b["displacement"]["kappa_ci"]))) if NA == 2 else ("%.2f / %s" % (b["displacement"]["agreement"], f2(b["displacement"]["kappa"])))),
                    ("locator improvement calls agreed", lambda b: "%d of %d" % (b["displacement"]["confusion"]["tp"], b["displacement"]["confusion"]["tp"] + b["displacement"]["confusion"]["fp"])),
                    ("locator no-improvement calls agreed", lambda b: "%d of %d" % (b["displacement"]["confusion"]["tn"], b["displacement"]["confusion"]["tn"] + b["displacement"]["confusion"]["fn"])),
                    ("no locator verdict (annotator sees improvement)", lambda b: "%d (%d)" % (b["displacement"]["n_locator_missing"], sum(B["improves_locator_missing"] for B in b["by_class"].values())))]
            for name, fn in rows: C.append(name + " & " + " & ".join(fn(b) for _, b in blocks_) + " \\\\")
            C += ["\\bottomrule\\end{tabular}\\end{table}"]; (P / "tab_human_compact.tex").write_text("\n".join(C) + "\n"); print("wrote", P / "tab_human_compact.tex")
        print("HUMAN:", json.dumps({k: v for k, v in res["human"].items() if k not in ("annotators", "by_class")}))
# ---- class summaries of the corrected endpoint ----
MIN_PAIRS = 3
res["corrected"] = {"min_pairs": MIN_PAIRS, "classes": {}}
for c in ("tracked", "aligned-then-forgotten", "never-aligned"):
    cs = [v for v in cells.values() if v["cls"] == c]; pe = [v for v in cs if v["n_pairs"] >= MIN_PAIRS]; ae = [v for v in cs if v["n_found_elig"] >= 1]
    res["corrected"]["classes"][c] = dict(n_cells=len(cs), n_episodes=int(sum(v["n"] for v in cs)), present_median=float(np.median([v["template_present"] for v in cs])),
        n_cells_paired=len(pe), n_pairs_total=int(sum(v["n_pairs"] for v in pe)), frac_better_median=float(np.median([v["frac_better"] for v in pe])) if pe else None,
        model_err_median=float(np.median([v["model_err"] for v in pe])) if pe else None, zero_err_median=float(np.median([v["zero_err"] for v in pe])) if pe else None,
        n_cells_worse_than_zero=int(sum(1 for v in pe if v["frac_better"] < 0.5)), n_cells_assessable_elig=len(ae), n_elig_total=int(sum(v["n_elig"] for v in cs)),
        acc_elig_median=float(np.median([v["acc_elig"] for v in ae])) if ae else None, n_cells_unassessable=int(sum(1 for v in cs if v["n_pairs"] < MIN_PAIRS)))
    print("corrected", c, res["corrected"]["classes"][c])
res["corrected"]["n_cells_unassessable_total"] = int(sum(1 for v in cells.values() if v["n_pairs"] < MIN_PAIRS)); res["corrected"]["n_episodes_total"] = int(sum(v["n"] for v in cells.values())); res["corrected"]["n_elig_total"] = int(sum(v["n_elig"] for v in cells.values()))
json.dump(res, open(OUT / "object_audit_eval.json", "w"), indent=1); print("wrote outputs/object_audit_eval.json; human sample", len(sample), "panels ->", HD / "human_sample.csv")
