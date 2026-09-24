"""Closed-loop MPC evaluation (arm C; rule fixed before running, ext/README.md): per game and condition, the emulator return over
T=30 decision steps pooled over 3 seeds x 20 episodes; mean with a bootstrap CI over episodes; the model-minus-random difference with
its own bootstrap CI (independent episode resampling in each condition); a model is 'useful' in a game only if that difference's CI
lies entirely above zero (lower bound > 0). Normalised gain = (model - random) / (oracle - random) where the oracle's gain is positive. Across games (n = 12 per
architecture) the gain is correlated with delay-20 alignment and with PSNR, with the minimum detectable rho stated.
Writes outputs/closed_loop_eval.json and PAPER_DIR/tab_closed_loop.tex."""
import json, os, glob, pathlib, sys, numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT)); OUT = ROOT / "outputs"; EXT = ROOT / "ext" / "outputs"; P = pathlib.Path(os.environ.get("PAPER_DIR", ROOT / "paper"))
from ccaus.validate import spearman, permutation_p, mde_rho
E = json.load(open(OUT / "storm_extend.json")); rng = np.random.default_rng(0)
def load(arch, g):
    f = EXT / f"cl_{arch}_{g}.json"
    if not f.exists(): return None
    d = json.load(open(f)); r = [x for s in d["results"] for x in d["results"][s]["returns"]]; return np.asarray(r, float)
def ci_mean(x, B=4000): return [float(np.percentile([np.mean(rng.choice(x, len(x))) for _ in range(B)], q)) for q in (2.5, 97.5)]
games = sorted({pathlib.Path(f).stem.split("_", 2)[2] for f in glob.glob(str(EXT / "cl_*_*.json")) if "DRY" not in f})
res = {"games": {}, "summary": {}}
for g in games:
    R = {a: load(a, g) for a in ("diamond", "iris", "storm", "random", "oracle")}
    if R["random"] is None: continue
    row = {}
    for a in ("diamond", "iris", "storm", "oracle"):
        if R[a] is None: continue
        diff = [np.mean(rng.choice(R[a], len(R[a]))) - np.mean(rng.choice(R["random"], len(R["random"]))) for _ in range(4000)]
        row[a] = dict(n=len(R[a]), mean=float(np.mean(R[a])), ci=ci_mean(R[a]), diff_vs_random=float(np.mean(R[a]) - np.mean(R["random"])), diff_ci=[float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))])
        row[a]["useful"] = bool(row[a]["diff_ci"][0] > 0)
    row["random"] = dict(n=len(R["random"]), mean=float(np.mean(R["random"])), ci=ci_mean(R["random"]))
    orc = row.get("oracle", {}).get("diff_vs_random"); orc_ok = bool(row.get("oracle", {}).get("useful"))   # normalise only where the oracle itself beats random (CI excludes 0); elsewhere the game is uninformative for planning at this horizon
    row["oracle_informative"] = orc_ok
    for a in ("diamond", "iris", "storm"):
        if a in row: row[a]["gain_norm"] = (row[a]["diff_vs_random"] / orc) if (orc_ok and orc and orc > 0) else None
    res["games"][g] = row
for a, A in (("diamond", "DIAMOND"), ("iris", "IRIS"), ("storm", "STORM")):
    gs = [g for g in res["games"] if a in res["games"][g] and res["games"][g][a].get("gain_norm") is not None and E["per_game"].get(A, {}).get(g, {}).get("20", {}).get("A") is not None]
    n_all = sum(1 for g in res["games"] if a in res["games"][g])
    if len(gs) < 6: continue
    gain = [res["games"][g][a]["gain_norm"] for g in gs]; al = [E["per_game"][A][g]["20"]["A"] for g in gs]; ps = [E["per_game"][A][g]["20"]["psnr"] for g in gs]
    res["summary"][A] = dict(n=len(gs), n_games_evaluated=n_all, n_useful=int(sum(res["games"][g][a]["useful"] for g in gs)), n_useful_all=int(sum(res["games"][g][a]["useful"] for g in res["games"] if a in res["games"][g])), useful_games=[g for g in gs if res["games"][g][a]["useful"]], median_gain_norm=float(np.median(gain)), gains={g: res["games"][g][a]["gain_norm"] for g in gs},
                            rho_gain_vs_A=spearman(gain, al), p_A=permutation_p(gain, al), rho_gain_vs_psnr=spearman(gain, ps), p_psnr=permutation_p(gain, ps), mde=mde_rho(len(gs)))
    print(f"{A}: {res['summary'][A]['n_useful']}/{len(gs)} oracle-informative games with model > random (CI excludes 0; {res['summary'][A]['n_useful_all']}/{n_all} over all games): {res['summary'][A]['useful_games']} | median normalised gain {res['summary'][A]['median_gain_norm']:.2f} | rho(gain, A) {res['summary'][A]['rho_gain_vs_A']:+.2f} p {res['summary'][A]['p_A']:.3f} | rho(gain, PSNR) {res['summary'][A]['rho_gain_vs_psnr']:+.2f} p {res['summary'][A]['p_psnr']:.3f} | MDE {res['summary'][A]['mde']:.2f}")
n_or = sum(1 for g in res["games"] if res["games"][g].get("oracle", {}).get("useful")); print(f"oracle > random in {n_or}/{len(res['games'])} games")
json.dump(res, open(OUT / "closed_loop_eval.json", "w"), indent=1)
L = ["\\begin{table}[t]\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}", "\\caption{Closed-loop planning inside the emulator: random-shooting model-predictive control (MPC; $K=8$ candidates, horizon $12$, $30$ decision steps, $60$ episodes per cell) with each world model imagining the candidates and one frozen per-game scorer reading their returns. Cells give the mean emulator return with a bootstrap interval; an asterisk marks a model whose bootstrap interval of the difference from the random policy (episodes resampled independently in each condition) lies entirely above zero (a positive gain; intervals entirely below zero are not marked). The oracle is an emulator-rollout oracle: it plans with the emulator itself under the same frozen scorer, not an optimal-return oracle.}", "\\label{tab:closed_loop}", "\\begin{tabular}{lrrrrr}\\toprule", "game & random & oracle & DIAMOND & IRIS & STORM \\\\ \\midrule"]
fmt = lambda r: ("--" if r is None else f"{r['mean']:.1f} [{r['ci'][0]:.1f}, {r['ci'][1]:.1f}]" + ("$^*$" if r.get("useful") else ""))
for g, row in res["games"].items(): L.append(f"{g} & {fmt(row.get('random'))} & {fmt(row.get('oracle'))} & {fmt(row.get('diamond'))} & {fmt(row.get('iris'))} & {fmt(row.get('storm'))} \\\\")
L += ["\\bottomrule\\end{tabular}\\end{table}"]; (P / "tab_closed_loop.tex").write_text("\n".join(L) + "\n"); print("wrote outputs/closed_loop_eval.json,", P / "tab_closed_loop.tex")
# compact excerpt for the main text: the two games that carry the two-sided claim
C = ["\\begin{table}[t]\\centering\\scriptsize\\setlength{\\tabcolsep}{4pt}", "\\caption{Closed-loop planning, excerpt (all twelve games in Appendix~\\ref{app:planning}): mean emulator return over 60 episodes, bootstrap 95\\% interval; an asterisk marks a model whose interval of the difference from the random policy lies entirely above zero; the oracle is the emulator planning under the same frozen scorer. Every model is blind on Alien; CrazyClimber's alignments are " + ", ".join("$%.2f$" % E["per_game"][_a]["CrazyClimber"]["20"]["A"] for _a in ("DIAMOND", "IRIS")) + " and $%.2f$" % E["per_game"]["STORM"]["CrazyClimber"]["20"]["A"] + " (mean-of-seed-medians estimator, Tables~\\ref{tab:allgames} and~\\ref{tab:allgames_storm}; Table~\\ref{tab:planning} uses pooled medians).}", "\\label{tab:closed_loop_compact}", "\\begin{tabular}{lrrrrr}\\toprule", "game & random & oracle & DIAMOND & IRIS & STORM \\\\ \\midrule"]
for g in ("Alien", "CrazyClimber"):
    row = res["games"][g]; C.append(f"{g} & {fmt(row.get('random'))} & {fmt(row.get('oracle'))} & {fmt(row.get('diamond'))} & {fmt(row.get('iris'))} & {fmt(row.get('storm'))} \\\\")
C += ["\\bottomrule\\end{tabular}\\end{table}"]; (P / "tab_closed_loop_compact.tex").write_text("\n".join(C) + "\n"); print("wrote", P / "tab_closed_loop_compact.tex")
