"""GPU-hours of the core sweep per architecture, summed from the PBS resource-usage footers of the job logs (V100, one GPU per job).
DIAMOND: outputs/d8_*.o (26 games, the emulator-chosen dur-8 sweep, 7 seeds x 60 episodes); IRIS: outputs/rerun_reg/rr_v5I*.o (60 tasks:
core + sensitivity + sustained); STORM: outputs/rerun_reg/rr_v16S*.o (36 tasks). Writes outputs/compute_cost.json."""
import glob, re, json, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
def hours(pattern):
    tot, n = 0.0, 0
    for f in glob.glob(str(ROOT / pattern)):
        m = re.search(r"Walltime Used:\s*(\d+):(\d+):(\d+)", open(f, errors="ignore").read())
        if m: tot += int(m.group(1)) + int(m.group(2)) / 60 + int(m.group(3)) / 3600; n += 1
    return round(tot, 1), n
res = {"DIAMOND_core_26_games": hours("outputs/d8_*.o"), "IRIS_core_sensitivity_sustained_60_tasks": hours("outputs/rerun_reg/rr_v5I*.o"), "STORM_core_sustained_36_tasks": hours("outputs/rerun_reg/rr_v16S*.o"),
       "STORM_training_26_games_seed1": hours("ext/storm_seed2/../../outputs/storm_train_*.o") if glob.glob(str(ROOT / "outputs/storm_train_*.o")) else hours("ext/storm_seed2/t_*.o"), "IRIS_planning_15_games": hours("outputs/rerun_reg/rr_v14fsi*.o"), "closed_loop_diamond_12_games": hours("ext/closed_loop/pbs/cCL_d_*.o")}
for k, v in res.items(): print(f"{k:<44} {v[0]:7.1f} GPU-h over {v[1]} jobs")
json.dump({k: dict(gpu_hours=v[0], n_jobs=v[1]) for k, v in res.items()}, open(ROOT / "outputs" / "compute_cost.json", "w"), indent=1); print("wrote outputs/compute_cost.json")
