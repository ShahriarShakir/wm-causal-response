"""Sensitivity analysis: does the flagship conclusion depend on the analysis settings?

Every threshold was fixed once: guard floor 0.25 of the median, the delay grid, the median-vs-mean
statistic, 60 episodes, 7 evaluation seeds. A result that moves when these move is a result about
the settings, not about world models.

This runs entirely off the RAW per-episode dump, so a full sweep costs no GPU time. The
conclusion under test is the flagship inversion:

    consistency ranks BankHeist ABOVE Pong, while causal alignment ranks Pong far above
    BankHeist.

It is reported as SURVIVES or BREAKS under every setting.
"""
import json, pathlib, sys, itertools, statistics as st
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np

OUT = pathlib.Path(__file__).resolve().parents[1] / "outputs"
DUR = sys.argv[1] if len(sys.argv) > 1 else "8"
BLIND = sys.argv[2] if len(sys.argv) > 2 else "BankHeist"   # the causally blind game
TRACK = sys.argv[3] if len(sys.argv) > 3 else "Pong"        # the causally tracking game
PREFIX = sys.argv[4] if len(sys.argv) > 4 else "video"      # "video" = DIAMOND, "iris" = IRIS
# optional explicit per-game file suffixes (e.g. "_k14_w60" and "_k2") for the emulator-chosen interventions
SFX_B = sys.argv[5] if len(sys.argv) > 5 else ""
SFX_T = sys.argv[6] if len(sys.argv) > 6 else ""
SUF = f"_dur{DUR}" if DUR != "1" else ""


def load_raw(game, sfx=""):
    f = OUT / f"{PREFIX}_{game}-v5{SUF}{sfx}.json"
    if not f.exists():
        return None
    d = json.load(open(f))
    if "raw" not in d["results"][str(d["seeds"][0])]:
        return None                       # pre-dump run
    return d


def recompute(d, delay, guard_frac, stat, n_ep=None, seeds=None):
    """Re-aggregate from raw episodes under a different set of choices."""
    S = [str(s) for s in (seeds or d["seeds"])]
    cons, align, tdiv = [], [], []
    for s in S:
        raw = d["results"][s]["raw"]
        raw = raw[:n_ep] if n_ep else raw
        for r in raw:
            td = r["true_divergence"].get(str(delay))
            if td is None or not np.isfinite(td):
                continue
            tdiv.append(td)
            cons.append(r["consistency"].get(str(delay)))
            align.append((td, r["causal_alignment"].get(str(delay))))
    if not tdiv:
        return None, None
    floor = guard_frac * float(np.median(tdiv))
    kept = [a for td, a in align if td >= floor and a is not None and np.isfinite(a)]
    agg = np.median if stat == "median" else np.mean
    c = [x for x in cons if x is not None and np.isfinite(x)]
    return (float(agg(c)) if c else float("nan"),
            float(agg(kept)) if kept else float("nan"))


def main():
    A, B = load_raw(BLIND, SFX_B), load_raw(TRACK, SFX_T)
    if A is None or B is None:
        print("raw dumps not present yet (runs still in flight)"); return
    print(f"Flagship conclusion under test (duration {DUR}):")
    print(f"  consistency ranks {BLIND} ABOVE {TRACK}, causal alignment ranks {TRACK} ABOVE {BLIND}\n")
    rows, broke = [], []
    GRID = dict(delay=[8, 12, 16, 20], guard=[0.10, 0.25, 0.50, 1.00],
                stat=["median", "mean"], n_ep=[15, 30, 60])
    for delay, guard, stat, n_ep in itertools.product(*GRID.values()):
        ca, aa = recompute(A, delay, guard, stat, n_ep)
        cb, ab = recompute(B, delay, guard, stat, n_ep)
        if None in (ca, aa, cb, ab) or not all(np.isfinite(x) for x in (ca, aa, cb, ab)):
            continue
        holds = (ca > cb) and (ab > aa)
        rows.append((delay, guard, stat, n_ep, ca, cb, aa, ab, holds))
        if not holds:
            broke.append((delay, guard, stat, n_ep, ca, cb, aa, ab))
    n = len(rows); k = sum(r[-1] for r in rows)
    print(f"{'delay':>6}{'guard':>7}{'stat':>8}{'n_ep':>6}"
          f"{'cons BH':>9}{'cons Pong':>10}{'align BH':>10}{'align Pong':>11}  verdict")
    for r in rows[::max(1, n // 24)]:
        print(f"{r[0]:>6}{r[1]:>7.2f}{r[2]:>8}{r[3]:>6}{r[4]:>9.3f}{r[5]:>10.3f}"
              f"{r[6]:>10.4f}{r[7]:>11.4f}  {'holds' if r[8] else 'BREAKS'}")
    print(f"\nconclusion holds in {k}/{n} of {n} setting combinations "
          f"({100*k/max(n,1):.1f}%)")
    if broke:
        print("BREAKS under:")
        for b in broke[:8]:
            print(f"  delay={b[0]} guard={b[1]} stat={b[2]} n_ep={b[3]}: "
                  f"cons {b[4]:.3f}/{b[5]:.3f}  align {b[6]:.4f}/{b[7]:.4f}")

    # per-seed spread: is the flagship stable across evaluation seeds?
    print("\nPER-SEED stability at delay 20, guard 0.25, median:")
    for name, d in ((BLIND, A), (TRACK, B)):
        vals = [recompute(d, 20, 0.25, "median", seeds=[s]) for s in d["seeds"]]
        al = [v[1] for v in vals if v[1] is not None and np.isfinite(v[1])]
        cs = [v[0] for v in vals if v[0] is not None and np.isfinite(v[0])]
        print(f"  {name:<10} consistency {np.mean(cs):.3f} +/- {np.std(cs):.4f} | "
              f"alignment {np.mean(al):.4f} +/- {np.std(al):.4f}  (n={len(al)} seeds)")
    n_causal = sum(1 for r in rows if r[7] > r[6]); n_cons = sum(1 for r in rows if r[4] > r[5])   # added 2026-09-12: the two halves separately
    json.dump({"duration": DUR, "n_settings": n, "n_holds": k, "n_causal_holds": n_causal, "n_consistency_inversion_holds": n_cons,
               "fraction_holds": k / max(n, 1), "breaks": broke[:20], "causal_breaks": [dict(delay=r[0], guard=r[1], stat=r[2], n_ep=r[3], align_blind=r[6], align_track=r[7]) for r in rows if not r[7] > r[6]]},
              open(OUT / f"sensitivity_{PREFIX}_{BLIND}{SFX_B}_vs_{TRACK}{SFX_T}{SUF}.json", "w"), indent=1)
    print(f"\nwrote outputs/sensitivity_{PREFIX}_{BLIND}_vs_{TRACK}{SUF}.json")


if __name__ == "__main__":
    main()
