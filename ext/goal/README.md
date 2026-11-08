# Goal reaching

Each model must choose which of 8 random action sequences of 12 steps reaches a goal frame: the true frame, 12 steps on, of one of the 8 sequences. The model predicts the frame after 12 steps for every sequence from the same context and picks the one nearest the goal. A use that reads the predicted consequence itself, such as visual model-predictive control towards a goal image, works this way. The trials start at the paper's diagnostic states (the game's warm-up, then a random-action context), 50 per model and game, in all 78 cells.

The design, the rule written before the runs and the reading are in `rebuttal/README.md` (R7). Result files: `ext/outputs_goal/goal_<model>_<game>.json`. Analysis: `rebuttal/scripts/r7_goal.py`. Job scripts of the 28 runs: `pbs/`.
