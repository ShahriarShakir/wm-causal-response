"""Figure: which score tracks usefulness, for reward-ranked planning and for goal reaching.

Reads rebuttal/outputs/r6_visited.json (CPU run), r6_visited_outputs_gpu.json (GPU repeat) and r7_goal.json, and writes
fig_usefulness.{pdf,svg,png} to FIG_DIR (default rebuttal/figs/). Each dot is a Spearman correlation across cells, each bar
its game-clustered bootstrap 95% interval, as reported in the tables of the planner-state and goal-reaching appendices."""
import json, os, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = pathlib.Path(__file__).resolve().parents[2]
O = ROOT / "rebuttal" / "outputs"
OUT = pathlib.Path(os.environ.get("FIG_DIR", ROOT / "rebuttal" / "figs")); OUT.mkdir(parents=True, exist_ok=True)
J = lambda n: json.load(open(O / n))
cpu, gpu, goal = J("r6_visited.json"), J("r6_visited_outputs_gpu.json"), J("r7_goal.json")
THRESHOLD = 0.41                                   # primary tests of both rules: rho >= 0.41 and p < 0.05

# Okabe-Ito; alignment keeps the orange of the two-rollout score in Figure 1
COL = {"alignment": "#E69F00", "psnr": "#0072B2", "ranking": "#009E73", "dpsnr": "#CC79A7"}
PLAN = [("alignment", "alignment", [(r["T1"]["rho"], r["T1"]["ci"]) for r in (cpu, gpu)]),
        ("psnr", "PSNR", [(r["T4"]["rho_gain_psnr"], r["T4"]["ci"]) for r in (cpu, gpu)]),
        ("ranking", "reward\nranking", [(r["T3"]["rho_gain_rank"], r["T3"]["ci_gain_rank"]) for r in (cpu, gpu)])]
GOAL = [("alignment", "alignment", [(goal["T1"]["rho"], goal["T1"]["ci"])]),
        ("psnr", "PSNR", [(goal["T2_psnr"]["rho_psnr"], goal["T2_psnr"]["ci_psnr"])]),
        ("dpsnr", "$\\Delta$PSNR", [(goal["T2_dpsnr"]["rho_dpsnr"], goal["T2_dpsnr"]["ci_dpsnr"])])]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "ytick.major.size": 2.5,
                     "axes.edgecolor": "#444444", "xtick.color": "#222222", "ytick.color": "#222222",
                     "svg.fonttype": "none", "pdf.fonttype": 42})
fig, axes = plt.subplots(1, 2, figsize=(4.95, 1.9), sharey=True, layout="constrained")
for ax, rows, title in ((axes[0], PLAN, "(a) gain of reward-ranked planning"), (axes[1], GOAL, "(b) goal-reaching success")):
    ax.axhline(0, color="#9a9a9a", lw=0.6, zorder=0)
    ax.axhline(THRESHOLD, color="#6f6f6f", lw=0.8, ls=(0, (3, 2)), zorder=0)
    for x, (key, _, ests) in enumerate(rows):
        offsets = (-0.15, 0.15) if len(ests) == 2 else (0.0,)
        for (rho, ci), dx, filled in zip(ests, offsets, (True, False)):
            ax.plot([x + dx] * 2, ci, color=COL[key], lw=1.5, solid_capstyle="round", zorder=2)
            ax.plot(x + dx, rho, "o", ms=5.5, color=COL[key], mfc=COL[key] if filled else "white", mew=1.4, zorder=3)
    ax.set_xticks(range(len(rows)), [r[1] for r in rows])
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_xlim(-0.55, len(rows) - 0.45)
    ax.set_title(title, fontsize=8.5, loc="left", pad=4)
axes[0].set_ylim(-0.3, 1.0); axes[0].set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.50", "0.75", "1"])
axes[0].set_ylabel("Spearman $\\rho$")
axes[1].text(len(GOAL) - 0.5, THRESHOLD - 0.07, "threshold fixed before the runs", ha="right", va="top", fontsize=7, color="#555555")
axes[0].legend(handles=[Line2D([], [], ls="", marker="o", ms=5, color="#555555", mfc="#555555", mew=1.2, label="CPU run"),
                        Line2D([], [], ls="", marker="o", ms=5, color="#555555", mfc="white", mew=1.2, label="GPU repeat")],
               loc="lower right", frameon=False, fontsize=7, handletextpad=0.2, borderaxespad=0.1, labelspacing=0.3)
META = {"Creator": None, "Producer": None, "CreationDate": None}
fig.savefig(OUT / "fig_usefulness.pdf", metadata=META)
fig.savefig(OUT / "fig_usefulness.svg", metadata={"Creator": None, "Date": None})
fig.savefig(OUT / "fig_usefulness.png", dpi=220)
print("fig_usefulness written to", OUT.name)
