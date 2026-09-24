# Interventional evaluation of action-conditioned world models

Code and per-episode results for the paper. The code measures how a world model's predicted future changes when one action changes, and compares that change with the change produced by an emulator replaying the same two action sequences.

## What is here

- `ccaus/`: the measurement library. `joint.py` computes the paired response (alignment, magnitude and normalised error), the guard and the aggregation. `diamond_harness.py`, `iris_harness.py` and `storm_harness.py` run the three Atari models under common random numbers. `validate.py` holds the statistics: Spearman correlation with average ranks, permutation tests, the gates and the power calculations.
- `scripts/`: the sweeps that produced the result files and the analyses that read them. `scripts/rerun_reg/analyze_all.sh` runs every analysis in order.
- `ext/`: the extension experiments: closed-loop planning, contexts from the agents' own policies, the object-readout audit and the second training seeds. Each folder has a README with its design and the reading rule fixed before it was run.
- `outputs/` and `ext/outputs/`: one JSON file per measured cell with every episode pair, the analysis outputs and the pre-registration files.
- `annotation/`: the 60 panels shown to the three annotators, their instructions, the blank sheet and the three completed sheets.
- `tests/`: four test suites, run by `tests/run_all.py`.

## Rerunning the analysis

No GPU is needed.

    pip install -r requirements.txt
    bash scripts/rerun_reg/analyze_all.sh

This recomputes the main analysis outputs in `outputs/` from the per-episode files and then runs the test suites; it takes under an hour on a CPU. Each extension experiment in `ext/` has its own evaluation script, named in its README. A few analyses also write a LaTeX table, which goes to `generated_tables/`.

## Rerunning the measurements

This needs a GPU and the model checkpoints. `scripts/v4_video_sweep.py` (DIAMOND), `scripts/v5_iris_sweep.py` (IRIS) and `scripts/v16_storm_sweep.py` (STORM) take a game, the number of episodes, the intervention length, its offset and the warm-up, and write one JSON file per cell. The job scripts in `scripts/` and `ext/` record every run behind the paper.

DIAMOND and IRIS use their public checkpoints, placed in `models/` and `models_iris/`. The STORM checkpoints, the second-seed checkpoints and the frozen reward scorers are too large for this package and will be released on acceptance, together with the code that runs the DMControl RSSM; that model's result files are included here. The emulator is the Arcade Learning Environment (ale-py 0.9.0) with sticky actions off and a frame skip of 4.

The registration test (`tests/test_registration.py`) needs the checkpoints and the emulator, so it runs separately from the four suites.

## Reading the code

Start with `measure_pair()` in `ccaus/joint.py`. It takes the two true futures and the two predicted futures of one episode and returns, for each delay, the alignment, magnitude and normalised error of the response, with the consistency score, PSNR and the PSNR difference between arms. `aggregate()` applies the guard and takes the per-seed medians.

A sweep script drives one game and one intervention through a harness and writes the episode pairs. The analysis scripts read those files:

- `storm_extend.py`: per-game summaries, the taxonomy and the cross-architecture correlations.
- `response_error.py`: the response triple and the PSNR bound.
- `spec_analysis.py` and `correspondence.py`: the third arm and the off-support fraction.
- `fsrank_analysis.py` and `ext/closed_loop/closed_loop_eval.py`: the planning results.
- `ext/object_audit/audit_eval.py`: the object readout and its human audit.

## Paths

Paths from the cluster where the experiments ran are replaced by placeholders such as `<ROOT>` (this folder), `<SCRATCH>` and `<STORM_SRC>`. The job scripts read these locations from environment variables and stop with a message when one is unset. Their scheduler headers keep the placeholders and need editing for another cluster.
