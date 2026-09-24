"""Evaluation of the RSSM controls (outputs/rssm_controls.json, scripts/v10_rssm_controls.py), rules fixed in the
memory state block before the job landed: per task and delay, pooled over seeds and episodes (median, bootstrap CI):
  latent A / R / err; frames-only alignment (posterior change with the ACTION input held at arm a: visual evidence only);
  actions-only alignment (frames held, actions differ); cos(dz*, dz*_frames_only); norm ratio frames-only / full;
  dec-vs-true (can the decoder render dt from posterior states); img-vs-dec (does the imagined pixel change match it).
Decision: for d >= 2, if frames-only alignment stays >= 0.5 while pixel ~ 0 -> the encoder reads the consequence from
pixels and 'encoded but not rendered' STANDS; if it collapses while actions-only stays high -> RETRACT as a shared-action artefact."""
import json, pathlib, sys, numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]; OUT = ROOT / "outputs"; sys.path.insert(0, str(ROOT))
from ccaus.joint import bootstrap_median
f = OUT / ("rssm_controls_DRY.json" if "--dry" in sys.argv else "rssm_controls.json")
d = json.load(open(f)); D = [str(x) for x in d["delays"]]
KEYS = ["pixel_align", "latent_A", "latent_R", "latent_err", "frames_only_align", "actions_only_align", "cos_post_vs_frames_only", "dec_vs_true_align", "dec_vs_true_mag", "img_vs_dec_align", "repr_true_div", "repr_true_div_frames_only"]
res = {}
for task, ps in d["results"].items():
    raws = [r for s in ps for r in ps[s]["raw"]]; res[task] = {}
    for dl in D:
        row = {}
        for k in KEYS:
            v = [r[k].get(dl) for r in raws if r[k].get(dl) is not None and r[k].get(dl) == r[k].get(dl)]
            row[k] = bootstrap_median(v).as_dict() if v else None
        fr = [r["repr_true_div_frames_only"].get(dl) / r["repr_true_div"].get(dl) for r in raws if r["repr_true_div"].get(dl) and r["repr_true_div_frames_only"].get(dl) is not None]
        row["frames_only_share_of_norm"] = bootstrap_median(fr).as_dict() if fr else None
        res[task][dl] = row
    # the frames-only alignment is a ratio whose denominator is the frames-only posterior change; it is meaningless where that
    # change is a small fraction of the full posterior change (same data-calibrated rule as everywhere: share >= 0.25)
    for dl in D:
        sh = res[task][dl]["frames_only_share_of_norm"]; res[task][dl]["frames_only_valid"] = bool(sh and sh["median"] >= 0.25)
    valid = [dl for dl in D if res[task][dl]["frames_only_valid"]]
    stand = [dl for dl in valid if res[task][dl]["frames_only_align"]["median"] >= 0.5 and res[task][dl]["latent_A"]["median"] >= 0.5]
    res[task]["frames_only_valid_delays"] = valid
    res[task]["verdict"] = ("encoder reads the consequence from pixels and the imagined latent tracks it at d" + ",".join(stand)) if stand else \
        (f"RETRACT: where the frames-only posterior change is measurable (d{','.join(valid)}) the imagined latent does not track it (latent A " + ", ".join(f"{res[task][dl]['latent_A']['median']:.2f}" for dl in valid) + "); at earlier delays the posterior change is the ACTION input's echo (actions-only alignment " + ", ".join(f"{res[task][dl]['actions_only_align']['median']:.2f}" for dl in D if not res[task][dl]['frames_only_valid']) + ")")
    print(f"== {task}: {res[task]['verdict']}")
    print("   d    pix   latA  latR  err | frames-only actions-only cos  norm-share | dec-vs-true(mag) img-vs-dec")
    for dl in D:
        r = res[task][dl]; g = lambda k, f="%.2f": (f % r[k]["median"]) if r.get(k) else " n/a"
        print(f"   {dl:<4} {g('pixel_align','%.3f')} {g('latent_A')} {g('latent_R')} {g('latent_err')} | {g('frames_only_align'):>8}{'' if r['frames_only_valid'] else '*'} {g('actions_only_align'):>10} {g('cos_post_vs_frames_only'):>5} {g('frames_only_share_of_norm'):>8} | {g('dec_vs_true_align')}({g('dec_vs_true_mag')}) {g('img_vs_dec_align')}")
json.dump(res, open(OUT / "rssm_controls_eval.json", "w"), indent=1); print("wrote outputs/rssm_controls_eval.json")
