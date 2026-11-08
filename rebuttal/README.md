# Analyses added during the discussion

These analyses answer the reviews of the submission. The rule of each was written down before it was run and is copied below, with the reviewer questions it answers. R3 is labelled post hoc, because the counts it uses were already printed in the paper, and R6b holds post hoc checks of R6. Every outcome is reported in the discussion, negatives included. The rules and the scripts cite tables and appendices of the submitted paper; the revised paper adds a summary of the findings as Table 1, so each of those table numbers is one higher there, while the appendix letters are unchanged.

## What is here

| rule | question | script | output |
|---|---|---|---|
| R1 | alignment over the planner's 12-step horizon against closed-loop gain | `rebuttal/scripts/r1_horizon_gain.py` | `rebuttal/outputs/r1_horizon_gain.json` |
| R2 | whether a few-pixel position tolerance lifts the low alignments | `rebuttal/scripts/r2_shift_tolerance.py` | `rebuttal/outputs/r2_shift_tolerance.json` |
| R3 | the annotators' own labels by class, independent of the locator (post hoc) | `rebuttal/scripts/r3_human_by_class.py` | `rebuttal/outputs/r3_human_by_class.json` |
| R1b, R1c, R4, R5 | the diagnostic as a screen, Delta-PSNR as a comparator, two worked examples, and the maximum alignment over all delays | `rebuttal/scripts/r1b_r1c_r4_r5.py` | `rebuttal/outputs/r1b_r1c_r4_r5.json` and `rebuttal/figs/` |
| R6 | the response and the reward ranking at the states each model's own planner visits | `ext/visited/v24_visited_states.py` (experiment) and `rebuttal/scripts/r6_visited.py` (analysis) | `ext/outputs/visited_<model>_<game>.json` and `rebuttal/outputs/r6_visited.json` |
| R6b | post hoc checks of R6: the rerun against the stored planner runs, the cells with a zero guard floor, Delta-PSNR, and the candidate ranking in detail | `rebuttal/scripts/r6b_sensitivity.py` | `rebuttal/outputs/r6b_sensitivity.json` |
| R6, GPU | the R6 experiment repeated on GPUs under the same rule, and compared cell by cell with the CPU run | `ext/visited/v24_visited_states.py` and `rebuttal/scripts/r6_visited.py`, both with `VISITED_OUTDIR=outputs_gpu`, and `rebuttal/scripts/r6_gpu_vs_cpu.py` | `ext/outputs_gpu/visited_<model>_<game>.json`, `rebuttal/outputs/r6_visited_outputs_gpu.json` and `rebuttal/outputs/r6_gpu_vs_cpu.json` |
| R4b | the R4 statistics for every cell in which Delta-PSNR and alignment disagree (post hoc) | `rebuttal/scripts/r4b_dissent_cells.py` | `rebuttal/outputs/r4b_dissent_cells.json` and `rebuttal/figs/dissent_*` |
| R7 | goal reaching: whether the diagnostic scores predict success when the predicted consequence itself is used | `ext/goal/v25_goal_reaching.py` (experiment) and `rebuttal/scripts/r7_goal.py` (analysis) | `ext/outputs_goal/goal_<model>_<game>.json` and `rebuttal/outputs/r7_goal.json` |
| R7b | goal reaching by taxonomy class and in the Delta-PSNR-dissent cells (post hoc) | `rebuttal/scripts/r7b_goal_posthoc.py` | `rebuttal/outputs/r7b_goal_posthoc.json` |
| R6 and R7, figure | which score predicts usefulness: the correlations of the R6 tests, in the CPU run and the GPU repeat, and of the R7 tests, drawn as Figure 4 of the revised paper | `rebuttal/scripts/fig_usefulness.py` | `rebuttal/figs/fig_usefulness.pdf` and `rebuttal/figs/fig_usefulness.png` |

## Rerunning the analyses

No GPU is needed. From the repository root, with the packages in `requirements.txt` installed:

    cat ext/object_audit/panels30.tar.gz.part-* > ext/object_audit/panels30.tar.gz
    python rebuttal/scripts/r1_horizon_gain.py
    python rebuttal/scripts/r2_shift_tolerance.py
    python rebuttal/scripts/r3_human_by_class.py
    python rebuttal/scripts/r1b_r1c_r4_r5.py
    python rebuttal/scripts/r6_visited.py
    python rebuttal/scripts/r6b_sensitivity.py
    VISITED_OUTDIR=outputs_gpu python rebuttal/scripts/r6_visited.py
    python rebuttal/scripts/r6_gpu_vs_cpu.py
    python rebuttal/scripts/r4b_dissent_cells.py
    python rebuttal/scripts/r7_goal.py
    python rebuttal/scripts/r7b_goal_posthoc.py
    python rebuttal/scripts/fig_usefulness.py

Run them in this order; `r7b_goal_posthoc.py` reads the outputs of R7 and R2, and `fig_usefulness.py` those of R6, its GPU repeat and R7. `r3_human_by_class.py` reads the classes that `r2_shift_tolerance.py` writes, `r1b_r1c_r4_r5.py` reads the outputs of R1 and R2, `r6_visited.py` and `r6_gpu_vs_cpu.py` read R1, and `r6b_sensitivity.py` reads R6. The archive `ext/object_audit/panels30.tar.gz` holds the 1,955 delay-20 panels of the object audit and is split into three parts to fit the repository's file-size limit. The scripts take about 4 minutes on a CPU.

## Rerunning the visited-state experiment

`ext/visited/v24_visited_states.py` runs the closed-loop planner of `ext/closed_loop/v20_closed_loop.py` unchanged and adds two measurements at the states it visits (rule R6 below). It needs the model checkpoints and the frozen scorers described in the main README, and writes one file per model and game to `ext/outputs/`:

    python ext/visited/v24_visited_states.py diamond ALE/CrazyClimber-v5

Source locations are placeholders such as `<DIAMOND_SRC>` and `<STORM_SRC>`, as in the rest of the package (see the main README). The job scripts in `ext/visited/pbs/` record the 17 runs behind the results, on CPU nodes with 12 cores each; they ran in parallel and finished within 100 minutes. The job scripts in `ext/visited/pbs_gpu/` record the GPU replication, one GPU per job, with the output in `ext/outputs_gpu/`.

## Rerunning the goal-reaching experiment

`ext/goal/v25_goal_reaching.py` uses the same model calls as the visited-state experiment and runs the trials of rule R7 below at the paper's diagnostic states, writing one file per model and game to `ext/outputs_goal/`:

    python ext/goal/v25_goal_reaching.py diamond ALE/Pong-v5

The job scripts in `ext/goal/pbs/` record the 28 runs behind the results, on CPU nodes with 12 cores each, and the two reruns described under the deviations (`gR_dF_*.sh`).

## The rules

The rules below are copied from the record written before each analysis was run. Only details of the computing site were removed.

### R1. Alignment over the planner's own horizon against closed-loop gain (TeeE Q3, ZP5t Q2, Ya7M W3/Q4)

- Reason fixed by the design, not by the data: the closed-loop planner imagines 12 steps (Appendix N, Table 18), while the paper correlates its gain with delay-20 alignment. The delays inside the planner's horizon are 1, 2, 3, 5, 8 and 12.
- Statistic: Spearman rho between the oracle-normalised gain G (`outputs/closed_loop_eval.json`, the 8 oracle-informative games per architecture) and the mean alignment over delays 1, 2, 3, 5, 8, 12 (`outputs/storm_extend.json` per_game, the mean-of-seed-medians estimator of Tables 12 and 23). Per architecture (n = 8, descriptive; minimum detectable rho about 0.85) and pooled over the 24 cells with a game-level permutation p (games permuted as blocks, 10,000 permutations) and a game-clustered bootstrap 95% interval (4,000 resamples).
- Comparators computed the same way: delay-20 alignment (the paper's 0.26, 0.76, 0.71 must be reproduced, else the script is wrong) and delay-20 PSNR.
- Reading: horizon-matched alignment "tracks closed-loop gain" if the pooled rho is at least 0.41 (the paper's pre-declared gate) AND its permutation p < 0.05; otherwise "not established". Whether it beats PSNR is stated only if both intervals are reported; no claim of superiority unless the pooled difference's game-clustered interval excludes zero.
- Also reported, descriptive: each CrazyClimber cell's horizon-matched alignment against its architecture's tracked threshold T.

### R2. Does a few-pixel position tolerance remove the low alignment? (TeeE W1/Q1, Ya7M W1/Q1)

- Data: the 1,955 delay-20 panels of the object audit (`ext/object_audit/panels30.tar.gz`, 30 episodes per cell, all 78 cells). Each panel stores the native true frames of both arms and the model's 64x64 frames of both arms (nearest upsampling). The 64x64 true frames are rebuilt with the harness's own `cv2.resize(..., INTER_AREA)`; the generated 64x64 frames are recovered by inverting the nearest upsampling, checked exactly by a round trip before use.
- Statistic: per episode, A_sigma = <G_sigma * dp, G_sigma * dt> / ||G_sigma * dt||^2, with dt = true b - true a, dp = generated b - generated a, and G_sigma a Gaussian blur of sigma = 0 (none), 1, 2 and 4 pixels on the 64x64 grid, per colour channel. Blurring both changes lets a consequence drawn a few pixels away from the true one overlap it. Guard: episodes whose true change at delay 20 has at least 4 changed pixels. Per cell: median over guarded episodes, cells with at least 5 such episodes.
- Sanity gate: the sigma = 0 cell medians must correlate with the paper's delay-20 cell alignment (Tables 12 and 23) at Spearman rho >= 0.70 over the usable cells. If the gate fails, the analysis is reported as a failed reconstruction and supports no claim.
- Reading, on the never-aligned cells with enough episodes: the explanation "low alignment is a few-pixel misplacement of a correct consequence" is SUPPORTED if at sigma = 2 at least half of them reach their architecture's tracked threshold T (0.40 DIAMOND, 0.11 IRIS, 0.19 STORM, Table 7); REJECTED if fewer than a quarter reach T at sigma = 4; otherwise PARTIAL. Control: the number of tracked cells at or above T at each sigma. Median alignment by class at each sigma is reported in full.

### R1b, R1c, R4 and R5 (written after R1 to R3 had run, before these were computed)

- **R1b, the diagnostic as a screen (TeeE W2).** Over the 24 oracle-informative closed-loop cells: a 2x2 table of "aligned over the planner's horizon" (R1's horizon mean at or above the architecture's T) against "useful" (the model-minus-random interval above zero, Table 18). Reported as counts with Fisher's exact p; descriptive. Reading: "the screen separates useful from not useful cells" only if Fisher p < 0.05; otherwise counts only.
- **R1c, Delta-PSNR as a third comparator (Ya7M Q4).** Spearman rho between the closed-loop gain and the delay-20 Delta-PSNR median of `outputs/dpsnr_cells.json`, over the closed-loop cells where Delta-PSNR exists, per architecture and pooled, with the same game-level permutation p and game-clustered bootstrap interval as R1. Descriptive; no superiority claim unless a difference's interval excludes zero.
- **R4, worked examples (TeeE Q2, Ya7M Q3, ZP5t Q1).** For the two Delta-PSNR-dissent cells that are also closed-loop games (IRIS Asterix, STORM Boxing): on the 30 delay-20 audit panels, the median alignment, magnitude and the share of the model's change energy outside the pixels where the world changes; one example episode per cell (the episode closest to the cell's median alignment and magnitude) rendered for the response. Descriptive.
- **R5, never-aligned at every delay (TeeE Q1 timing).** For the 28 never-aligned cells, the maximum alignment over all eight measured delays (1, 2, 3, 5, 8, 12, 16, 20) against T. Descriptive.

### R3. Human labels by class, independent of the locator (TeeE Q1, Ya7M Q1)

- Not blind: the per-class counts are already printed in Table 13 (columns "compared/human" and "missing (improving)"), and they were seen while writing this triage (consensus: right-direction improvement in 9 of 11 judged tracked panels and 2 of 13 judged never-aligned panels). The computation below only adds an interval and is labelled post hoc.
- Statistic: for each annotator and the consensus, the share of judged panels labelled "improves on zero displacement" (direction yes with magnitude about right or too little) in tracked against never-aligned panels, with a game-clustered bootstrap 95% interval of the difference (4,000 resamples) and Fisher's exact p at panel level.
- Reading: the human labels "support a difference" if the consensus difference's interval excludes zero.

### R6. The response and the reward ranking at the planner's own visited states (TeeE Q3, ZP5t Q2, Ya7M Q4)

Written before any job was submitted.

- **Visited states.** Each model's own closed-loop planner, `ext/closed_loop/v20_closed_loop.py` copied unchanged into `ext/visited/v24_visited_states.py` (K = 8 candidates, horizon 12, 30 decisions, the same frozen scorer, warm-up and seeds), rerun for seed 42, episodes 0 to 9, in the 8 oracle-informative games (Alien, Asterix, Boxing, Breakout, CrazyClimber, Frostbite, Qbert, Seaquest) for DIAMOND, IRIS and STORM. The executed actions are logged, so every visited state can be replayed exactly on CPU later. The measurement uses its own random streams and every harness reseeds per step, so the planner's choices are unaffected. Determinism check: each cell's 10 returns are compared with the stored seed-42 returns of episodes 0 to 9; the match count is reported. A mismatch means GPU non-determinism; the states are then still the planner's own.
- **Measurement 1, pixel response at visited states.** At decisions t = 0, 3, ..., 27: the paper's paired intervention from that state (context = the planner's last 4 frames, 8 for STORM; random base actions from a separate stream; window start K = 3, duration 8, the game's emulator-chosen action offset; true arms by clone and restore of the emulator; model arms under common random numbers), scored with the library's `measure_pair` and `aggregate` (guard calibrated per cell, 0.25 of the median true divergence), at delays 1, 2, 3, 5, 8, 12. Cell statistic A_vis = mean over these delays of the median alignment (at least 4 of the 6 delays with guarded pairs). PSNR_vis = mean over the same delays of the median prediction PSNR.
- **Measurement 2, reward ranking at visited states.** At every decision, for the planner's own 8 candidates: the model's scored return (what the planner used), the emulator's native 12-step game return, and the frozen scorer applied to the true frames. RANK_vis = median over informative decisions (native returns not all equal) of Spearman(model scores, native returns); REF_vis the same for the scorer on true frames. Cells with fewer than 5 informative decisions have no RANK_vis (counted).
- **Tests**, over the 24 cells, with game-level permutation p (10,000) and game-clustered bootstrap 95% intervals (4,000):
  - T1 (primary): rho(closed-loop gain, A_vis). Reading "visited-state alignment tracks closed-loop gain" if rho >= 0.41 and p < 0.05; otherwise "not established".
  - T2: agreement of A_vis with the diagnostic horizon alignment A_h of R1 (Spearman over 24 cells) and the number of cells that change side of their architecture's T. Named cell, STORM CrazyClimber: A_vis >= 0.19 reads its planning gain as an evaluation-condition effect; A_vis < 0.19 as a reward-readout effect.
  - T3: rho(gain, RANK_vis) and rho(RANK_vis, A_vis); descriptive.
  - T4: rho(gain, PSNR_vis) and the difference rho(gain, A_vis) - rho(gain, PSNR_vis) with its interval; no superiority claim unless the interval excludes zero.
- Every outcome is reported, negatives included. Delays, guard, cells and statistics do not change after results.

### R7. Goal reaching: does alignment predict success when the predicted consequence itself is used? (TeeE Q2/W2, Ya7M W3/Q4, ZP5t Q2)

Written before any job was submitted.

- **Reason fixed by the design.** R6 found that reward-ranked planning gain follows PSNR and the reward ranking, which needs only the order of the candidates' returns. A use that reads the predicted consequence itself needs that consequence to be right. Reaching a goal image, by choosing the action sequence whose predicted outcome matches a goal frame as in visual model-predictive control, is the standard such use.
- **Cells and states.** All 78 model-game cells (26 games; DIAMOND, IRIS, STORM). States follow the paper's diagnostic protocol: reset with the trial's seed, the game's emulator-chosen warm-up of random actions (`best_warmup` in `outputs/emulator_intervention_search.json`), then a context of the model's length (4 frames, 8 for STORM) filled with random actions, as in v20 and v24; sticky actions off. Models, harnesses and checkpoints as in v24, on CPU nodes.
- **Trials.** 50 per cell (seeds 0 to 49). Each trial draws 8 candidate action sequences of length 12 uniformly at random from a trial-specific stream, and one of them, uniformly from the same stream, sets the goal: its true frame after 12 steps (emulator clone and restore), at the models' 64x64 resolution. The model predicts the frame after 12 steps for every candidate from the same context, with one model seed per trial (common random numbers), and chooses the candidate whose predicted frame is nearest the goal in squared error at 64x64. Success: the chosen candidate's true frame equals the goal (maximum absolute difference below 1e-6 at 64x64). Chance for the trial: the share of the 8 candidates whose true frame equals the goal. A trial is skipped if any candidate's episode ends within the 12 steps, and uninformative if all 8 true frames are equal; both are counted.
- **Cell statistic.** G = mean over informative trials of (success - chance) / (1 - chance): 0 is chance, 1 is always right. A cell needs at least 10 informative trials.
- **Comparators,** from the paper's stored sweep files with R1's estimator (mean of the per-seed medians): A_h = mean alignment over delays 1, 2, 3, 5, 8, 12 (at least 4 of the 6); PSNR_h = mean prediction PSNR over the same delays. Delta-PSNR exists only at delay 20 and only for some cells (`outputs/dpsnr_cells.json`), so it is compared on that subset.
- **Tests** over the cells with G and A_h, with game-level permutation p (10,000) and game-clustered bootstrap 95% intervals (4,000), as in R1 and R6:
  - T1 (primary): rho(G, A_h). Reading "alignment predicts goal reaching" if rho >= 0.41 and p < 0.05; otherwise "not established".
  - T2: rho(G, PSNR_h) and the difference rho(G, A_h) - rho(G, PSNR_h) with its interval; on the Delta-PSNR subset, rho(G, Delta-PSNR) and rho(G, A_h) - rho(G, Delta-PSNR) with its interval. "Alignment predicts goal reaching better than X" only if that interval excludes zero.
  - T3 (descriptive): per architecture (26 cells each), and the flagship pair, DIAMOND on Alien and on Pong.
- Every outcome is reported, negatives included. Cells, trials, horizon, success rule, comparators and statistics do not change after results.

## Deviations from the rules

- R1. Qbert's delay 1 is gated out in every seed, because the world has not changed yet at that delay. The horizon mean therefore skips gated delays and needs at least 4 of the 6.
- R6, device. The experiment ran on CPU nodes instead of GPUs, a change recorded before any measurement. CPU and GPU draw different random numbers, so the rerun is a new run of the same planner and not a replay of the stored GPU episodes, and the determinism check was not expected to pass. 139 of the 240 episodes end with exactly the stored return.
- R6, analysis code. In seven cells more than half of the true changes at the visited states are exactly zero, so the guard floor is 0 and pairs with zero true change enter with alignment 0/0. The library's estimator drops such non-finite values before the median. The first version of `r6_visited.py` did not, and its output was discarded. The script now drops them and checks its recomputation against the library's aggregate in every cell and delay. The rule did not change.
- R6, GPU replication. After the CPU run, the experiment was repeated on GPUs as first planned, with the same script and rule and the output in a separate folder. All 24 cells are complete: 240 of the 240 episodes end with exactly the stored GPU return, and the visited-state alignment is on the same side of the threshold as in the CPU run in 23 of 24 cells.
- R7, two DIAMOND cells. The job that ran DIAMOND on Qbert and RoadRunner hit its one-hour limit: on Qbert the model ran about 100 times slower than on the other games, because subnormal floating-point values slow the CPU. Both cells were rerun with subnormals flushed to zero (`FLUSH_DENORMAL=1`), which changes only values below about 1e-38. The first four Qbert trials of the slow run, kept in `ext/outputs_goal/partial/`, made the same choices as the rerun. The rule, cells, trials and statistics did not change.
