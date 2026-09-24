"""The response triple and the PSNR bound, computed from the existing per-pair files (no new compute).
P2 corollary: a reference-based score bounds the response error only when 2||e|| < ||dt||, where ||e|| is the
prediction-error norm implied by the PSNR (data range 2, D = 64*64*3). Reports 2||e||/||dt|| per game, per arch.
P3: normalized squared response error  err = R^2 + 1 - 2A  per pair (R = causal_magnitude, A = causal_alignment),
median over guarded pairs, for the flagship and all 26 games at d1 and d20. Writes outputs/response_error.json."""
import json, glob, pathlib, re, sys, numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]; OUT = ROOT / "outputs"; sys.path.insert(0, str(ROOT))
D, RANGE = 64 * 64 * 3, 2.0
def err_norm(psnr): return np.sqrt(D) * RANGE / (10 ** (psnr / 20))
SEARCH = json.load(open(OUT / "emulator_intervention_search.json"))
def chosen_files(prefix):
    """Identical rule to scripts/psnr_and_crossarch.py --chosen (fixed offset is k=3, written without a suffix). STORM files carry w-4 (8-frame context)."""
    fs = {}
    for g, e in SEARCH.items():
        k, w = int(e["best_offset"]), int(e["best_warmup"])
        if prefix == "storm": w = max(0, w - 4)
        f = OUT / f"{prefix}_{g}-v5_dur8{'_k'+str(k) if k != 3 else ''}{'_w'+str(w) if w else ''}.json"
        if f.exists(): fs[g] = str(f)
    return fs
res = {}
for arch, prefix in (("DIAMOND", "video"), ("IRIS", "iris"), ("STORM", "storm")):
    res[arch] = {}
    for g, f in sorted(chosen_files(prefix).items()):
        d = json.load(open(f)); R = [r for s in d["results"] for r in d["results"][s]["raw"]]
        floor = float(np.mean([d["results"][s]["aggregate"]["guard"]["floor"] for s in d["results"]]))
        row = {}
        for dl in ("1", "20"):
            keep = [r for r in R if r["true_divergence"].get(dl) is not None and r["true_divergence"][dl] >= floor
                    and r["causal_alignment"].get(dl) is not None and r["causal_magnitude"].get(dl) is not None]
            if not keep: row[dl] = None; continue
            A = np.array([r["causal_alignment"][dl] for r in keep], float); Rm = np.array([r["causal_magnitude"][dl] for r in keep], float)
            ok = np.isfinite(A) & np.isfinite(Rm); n_nan = int((~ok).sum()); A, Rm = A[ok], Rm[ok]
            err = Rm ** 2 + 1 - 2 * A
            psnr = np.array([r["prediction_psnr"][dl] for r in R if r["prediction_psnr"].get(dl) is not None])
            dt = np.array([r["true_divergence"][dl] for r in keep], float)[ok]
            # exact bound when the intervened arm's PSNR is stored (files re-run after 2026-09-08): (||p-t|| + ||p'-t'||) / ||dt|| per pair, median
            ip = [r["intervened_psnr"][dl] for r in keep if r.get("intervened_psnr") and r["intervened_psnr"].get(dl) is not None]
            exact = None
            if len(ip) == len(keep) and ip:
                pa_ = np.array([r["prediction_psnr"][dl] for r in keep], float)[ok]; pb_ = np.array(ip, float)[ok]
                exact = float(np.median((err_norm(pa_) + err_norm(pb_)) / dt))
            row[dl] = dict(n=int(ok.sum()), n_nan=n_nan, align=float(np.median(A)), magnitude=float(np.median(Rm)), error=float(np.median(err)),
                           psnr=float(np.median(psnr)), err_norm=float(err_norm(np.median(psnr))), true_div=float(np.median(dt)),
                           bound_ratio=float(2 * err_norm(np.median(psnr)) / np.median(dt)), bound_ratio_exact=exact,
                           intervened_psnr=(float(np.median(ip)) if ip else None))
        res[arch][g] = row
# FIXED_FLAGSHIP: the same triple for the flagship pair under the FIXED intervention (files without a k/w suffix)
res["fixed"] = {}
for arch, prefix in (("DIAMOND", "video"), ("IRIS", "iris")):
    res["fixed"][arch] = {}
    for g in ("Alien", "Pong"):
        f = OUT / f"{prefix}_{g}-v5_dur8.json"
        if not f.exists(): continue
        d = json.load(open(f)); R = [r for s_ in d["results"] for r in d["results"][s_]["raw"]]
        floor = float(np.mean([d["results"][s_]["aggregate"]["guard"]["floor"] for s_ in d["results"]]))
        row = {}
        for dl in ("1", "20"):
            keep = [r for r in R if r["true_divergence"].get(dl) is not None and r["true_divergence"][dl] >= floor and r["causal_alignment"].get(dl) is not None and r["causal_magnitude"].get(dl) is not None]
            A = np.array([r["causal_alignment"][dl] for r in keep], float); Rm = np.array([r["causal_magnitude"][dl] for r in keep], float); ok = np.isfinite(A) & np.isfinite(Rm)
            row[dl] = dict(n=int(ok.sum()), align=float(np.median(A[ok])), magnitude=float(np.median(Rm[ok])), error=float(np.median(Rm[ok] ** 2 + 1 - 2 * A[ok]))) if ok.any() else None
        res["fixed"][arch][g] = row
json.dump(res, open(OUT / "response_error.json", "w"), indent=1)
for arch in ("DIAMOND", "IRIS", "STORM"):
    if not res.get(arch): continue
    print(f"== {arch}: game  d20 align / magnitude / error | PSNR -> 2||e||/||dt|| (d1, d20)")
    for g, row in res[arch].items():
        r1, r20 = row.get("1"), row.get("20")
        if r20: print(f"  {g:<14} {r20['align']:+.2f} / {r20['magnitude']:.2f} / {r20['error']:.2f} | {r20['psnr']:.1f} dB -> {r1['bound_ratio'] if r1 else float('nan'):.1f}, {r20['bound_ratio']:.1f}")
    br = [row["20"]["bound_ratio"] for row in res[arch].values() if row.get("20")]
    print(f"  bound ratio 2||e||/||dt|| at d20: min {min(br):.1f}, median {np.median(br):.1f}, max {max(br):.1f}  (>1 everywhere = PSNR cannot constrain the response)")
print("wrote outputs/response_error.json")
