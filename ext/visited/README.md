# Visited states

Each model's own closed-loop planner (`ext/closed_loop/v20_closed_loop.py`, unchanged) is rerun for seed 42, episodes 0 to 9, in the 8 games where the emulator oracle gains. Two measurements are added at the states the planner visits: the paper's paired intervention, scored at delays 1 to 12 against the emulator's exact counterfactual, and the model's ranking of the planner's 8 candidate actions against the emulator's own 12-step returns.

The design, the rule written before the runs and the deviations from it are in `rebuttal/README.md` (R6). Result files: `ext/outputs/visited_<model>_<game>.json`. Analysis: `rebuttal/scripts/r6_visited.py`, with post hoc checks in `rebuttal/scripts/r6b_sensitivity.py`. Job scripts of the 17 runs: `pbs/`. The same experiment repeated on GPUs writes to `ext/outputs_gpu/` (job scripts in `pbs_gpu/`), and `rebuttal/scripts/r6_gpu_vs_cpu.py` compares the two runs cell by cell.
