"""R3 (rule in rebuttal/README.md; post hoc, the counts are printed in Table 13): the annotators' own displacement labels by
taxonomy class, independent of the locator. Label rule copied from ext/object_audit/audit_eval.py (fixed 2026-09-18): direction no = no
improvement; direction yes with magnitude yes or 'no - too little' = improvement; anything else undecided. Consensus: at least two decided and
all who decided agree. Writes rebuttal/outputs/r3_human_by_class.json."""
import csv, json, pathlib, numpy as np
from scipy.stats import fisher_exact

ROOT = pathlib.Path(__file__).resolve().parents[2]; OUT = ROOT / "rebuttal" / "outputs"
OA = ROOT / "ext" / "object_audit"; EXCL = {"08fcf5d307"}                 # the example panel, as in scripts/rerun_reg/analyze_all.sh
SHEETS = ["human_labels_2026-09-18.csv", "human_labels_annotator2_2026-09-19.csv", "human_labels_annotator3_2026-09-22.csv"]
KEY = json.load(open(OA / "human_key.json"))
CLS = {k: v["cls"] for k, v in json.load(open(OUT / "r2_shift_tolerance.json"))["cells"].items()}   # Table 7 classes (checked in R2)
ARCH = {"diamond": "DIAMOND", "iris": "IRIS", "storm": "STORM"}
rng = np.random.default_rng(0)

def read(path):
    lab = {}
    for row in csv.DictReader(open(OA / path)):
        v = list(row.values()); pid = v[0].strip(); d_, e_ = [(x or "").strip().lower() for x in v[3:5]]
        if pid not in KEY or pid in EXCL: continue
        if d_ == "no": lab[pid] = False
        elif d_ == "yes" and e_ in ("yes", "no - too little"): lab[pid] = True
    return lab
ann = [read(s) for s in SHEETS]
cons = {}
for pid in KEY:
    if pid in EXCL: continue
    dec = [a[pid] for a in ann if pid in a]
    if len(dec) >= 2 and len(set(dec)) == 1: cons[pid] = dec[0]
def cls_of(pid): k = KEY[pid]; return CLS[f"{ARCH[k['arch']]}/{k['game']}"]
def game_of(pid): return KEY[pid]["game"]

res = {"rule": "rebuttal/README.md R3 (post hoc)", "labels": {}}
for name, lab in [(f"annotator {i + 1}", a) for i, a in enumerate(ann)] + [("consensus", cons)]:
    by = {c: [lab[p] for p in lab if cls_of(p) == c] for c in ("tracked", "aligned-then-forgotten", "never-aligned", "partial")}
    tr, na = by["tracked"], by["never-aligned"]
    games = sorted({game_of(p) for p in lab if cls_of(p) in ("tracked", "never-aligned")}); diffs = []
    for _ in range(4000):                                                    # game-clustered bootstrap of the difference
        gs = rng.choice(games, len(games)); t_, n_ = [], []
        for g in gs:
            for p in lab:
                if game_of(p) != g: continue
                if cls_of(p) == "tracked": t_.append(lab[p])
                elif cls_of(p) == "never-aligned": n_.append(lab[p])
        if t_ and n_: diffs.append(np.mean(t_) - np.mean(n_))
    odds, p = fisher_exact([[sum(tr), len(tr) - sum(tr)], [sum(na), len(na) - sum(na)]])
    res["labels"][name] = dict(counts={c: [int(sum(v)), len(v)] for c, v in by.items()}, diff_tracked_minus_never=float(np.mean(tr) - np.mean(na)),
                               diff_ci=[float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))], fisher_p=float(p), n_games=len(games))
c = res["labels"]["consensus"]; res["reading"] = "supports a difference" if c["diff_ci"][0] > 0 else "no detectable difference"
json.dump(res, open(OUT / "r3_human_by_class.json", "w"), indent=1)
for k, v in res["labels"].items(): print(k, v)
print("READING:", res["reading"])
