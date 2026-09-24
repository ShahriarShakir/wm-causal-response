"""Correspondence pass rates per taxonomy class. Reads correspondence.json and
storm_extend.json; writes outputs/correspondence_by_class.json. Runs after correspondence.py."""
import json, pathlib, os
ROOT = pathlib.Path(__file__).resolve().parents[1]; os.chdir(ROOT)
C = json.load(open("outputs/correspondence.json")); E = json.load(open("outputs/storm_extend.json"))
out = {}
for a in ("DIAMOND", "IRIS", "STORM", "TWM"):
    if a not in C or a not in E["taxonomy"]: continue
    tax = E["taxonomy"][a]; g20 = C[a]["games"]
    for cls in ("tracked", "aligned-then-forgotten", "never-aligned", "partial"):
        gs = [g for g, r in tax.items() if r["cls"] == cls and g20.get(g, {}).get("20")]
        out.setdefault(a, {})[cls] = dict(n=len(gs), passing=sum(1 for g in gs if g20[g]["20"]["pass"]), positive=sum(1 for g in gs if g20[g]["20"]["mean_diff"] > 0))
    print(a, {k: f"{v['passing']}/{v['n']} pass, {v['positive']}/{v['n']} positive" for k, v in out[a].items()})
json.dump(out, open("outputs/correspondence_by_class.json", "w"), indent=1); print("wrote outputs/correspondence_by_class.json")
