"""Pixel consequence-specificity (v15) evaluation; reading pre-declared on 2026-09-08 before any file landed.
Per (arch, game), pooled over seeds and episodes, medians with bootstrap CI at d8 and d20 of: align_B (probe), align_CB (wrong action on B's truth),
cos(dp_B, dp_C) (same change for two different actions?), off-support energy of dp_B, cos(dt_B, dt_C), PSNR(p', t').
Classification at d20 (rules): TRACKED-SPECIFIC = align_B >= T and align_CB <= align_B/2; BLIND-AGNOSTIC = align_B < 0.05 and cos(dp_B,dp_C) >= 0.8
(the model's change does not depend on which action); BLIND-MISPLACED = align_B < 0.05 and off-support >= 0.9 and cos < 0.8 (a different change per
action, but where the world did not change); BLIND-SILENT = align_B < 0.05 and magnitude small (from response_error R < 0.1); OTHER. T = 0.40 DIAMOND / 0.09 IRIS.
Writes outputs/spec_analysis.json."""
import json, pathlib, sys, glob, numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]; OUT = ROOT / "outputs"; sys.path.insert(0, str(ROOT))
from ccaus.joint import bootstrap_median
RE = json.load(open(OUT / "response_error.json")); T = {"DIAMOND": 0.40, "IRIS": 0.09}
_se = json.load(open(OUT / "storm_extend.json")) if (OUT / "storm_extend.json").exists() else {}
if _se.get("thresholds", {}).get("STORM"): T["STORM"] = _se["thresholds"]["STORM"]
KEYS = ("align_B", "align_CB", "cos_pBpC", "offsupport", "cos_tBtC", "psnr_B", "true_div_B")
res = {"DIAMOND": {}, "IRIS": {}, "STORM": {}}
for f in sorted(OUT.glob("spec_*_*.json")):
    if "DRY" in f.name: continue
    d = json.load(open(f)); arch = d["arch"].upper(); g = d["game"].split("/")[-1].replace("-v5", ""); raws = [r for s in d["results"] for r in d["results"][s]["raw"]]
    if not raws: res[arch][g] = {"n": 0}; continue
    row = {"n": len(raws)}
    for dl in ("8", "20"):
        row[dl] = {}
        for k in KEYS:
            v = [r[k][dl] for r in raws if r[k].get(dl) is not None and r[k][dl] == r[k][dl]]
            row[dl][k] = bootstrap_median(v).as_dict() if v else None
    x = row["20"]; m = lambda k: (x[k]["median"] if x.get(k) else float("nan")); R = (RE[arch].get(g, {}).get("20") or {}).get("magnitude", float("nan"))
    aB, aCB, c, off = m("align_B"), m("align_CB"), m("cos_pBpC"), m("offsupport")
    if aB >= T[arch] and aCB <= aB / 2: cls = "tracked-specific"
    elif aB >= T[arch]: cls = "tracked-unspecific"
    elif aB < 0.05 and R < 0.1: cls = "blind-silent"
    elif aB < 0.05 and c >= 0.8: cls = "blind-agnostic"
    elif aB < 0.05 and off >= 0.9: cls = "blind-misplaced"
    elif aB < 0.05: cls = "blind-other"
    else: cls = "partial"
    row["class"] = cls; row["R"] = R; res[arch][g] = row
    print(f"{arch:<8} {g:<14} n {len(raws):>3} | d20 align_B {aB:+.2f} align_CB {aCB:+.2f} cos(pB,pC) {c:+.2f} offsupport {off:.2f} cos(tB,tC) {m('cos_tBtC'):+.2f} PSNR' {m('psnr_B'):.1f} R {R:.2f} -> {cls}")
from ccaus.validate import spearman, permutation_p, mde_rho
for arch in res:
    ok = {g: r for g, r in res[arch].items() if not g.startswith("_") and r.get("20") and r["20"].get("align_B") and r["20"].get("offsupport")}
    if len(ok) >= 5:
        A = [ok[g]["20"]["align_B"]["median"] for g in ok]; off = [ok[g]["20"]["offsupport"]["median"] for g in ok]
        spec = [(ok[g]["20"]["align_CB"]["median"] / ok[g]["20"]["align_B"]["median"], ok[g]["20"]["cos_tBtC"]["median"]) for g in ok if ok[g]["20"]["align_B"]["median"] >= 0.4 and ok[g]["20"].get("align_CB") and ok[g]["20"].get("cos_tBtC")]
        res[arch]["_across"] = dict(n=len(ok), rho_offsupport_vs_A=spearman(off, A), p=permutation_p(off, A, n_perm=5000), mde=mde_rho(len(ok)),
                                    specificity_ratio_tracked=[x[0] for x in spec], cos_true_tracked=[x[1] for x in spec])
        print(f"{arch} across {len(ok)} games: off-support vs alignment rho {res[arch]['_across']['rho_offsupport_vs_A']:+.2f} p {res[arch]['_across']['p']:.4f} (MDE {res[arch]['_across']['mde']:.2f}); tracked games: wrong-action/right-action alignment ratio median {np.median([x[0] for x in spec]) if spec else float('nan'):.2f} vs cos(true B, true C) median {np.median([x[1] for x in spec]) if spec else float('nan'):.2f}")
    from collections import Counter
    cnt = Counter(r.get("class", "no data") for g, r in res[arch].items() if not g.startswith("_")); res[arch]["_counts"] = dict(cnt); print(arch, dict(cnt))
json.dump(res, open(OUT / "spec_analysis.json", "w"), indent=1); print("wrote outputs/spec_analysis.json")
