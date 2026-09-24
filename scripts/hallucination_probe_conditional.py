"""Hallucination probe restricted to episodes whose true frames are identical."""
import pathlib as _pl, os as _os; _os.chdir(_pl.Path(__file__).resolve().parents[1])
import json, numpy as np
print("PER-EPISODE conditional test: model divergence ONLY in episodes whose emulator frames are bit-identical\n")
print(f"{'arch/game':<20}{'delay':>6}{'n identical':>12}{'MODEL div in those (median, IQR)':>34}{'n moved':>9}{'   vs n differing':>18}{'MODEL div there':>17}")
res = {}
for pfx, arch in (("video","DIAMOND"),("iris","IRIS")):
    for g in ("Alien","BankHeist"):
        d = json.load(open(f"outputs/{pfx}_{g}-v5_dur8.json"))
        raws = [r for s in d["results"] for r in d["results"][s]["raw"]]
        for dl in ("1","3","5","8"):
            ident = [r["model_divergence"][dl] for r in raws if r["true_divergence"][dl] == 0.0]
            diff  = [r["model_divergence"][dl] for r in raws if r["true_divergence"][dl] > 0.0]
            key = f"{arch}/{g}/d{dl}"
            res[key] = dict(n_identical=len(ident), model_div_identical_median=float(np.median(ident)) if ident else None,
                            model_div_identical_q25=float(np.percentile(ident,25)) if ident else None,
                            model_div_identical_q75=float(np.percentile(ident,75)) if ident else None,
                            n_moved_when_identical=int(sum(1 for v in ident if v > 0)),
                            n_differing=len(diff), model_div_differing_median=float(np.median(diff)) if diff else None)
            if ident:
                print(f"{arch+'/'+g:<20}{dl:>6}{len(ident):>12}{np.median(ident):>16.3f} [{np.percentile(ident,25):.2f},{np.percentile(ident,75):.2f}]"
                      f"{sum(1 for v in ident if v>0):>9}{len(diff):>18}{(np.median(diff) if diff else float('nan')):>17.3f}")
json.dump(res, open("outputs/hallucination_probe_conditional.json","w"), indent=1)
print("\nwrote outputs/hallucination_probe_conditional.json")
