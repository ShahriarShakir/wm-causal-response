# Extension arms

Each folder here holds one additional experiment that the main sweeps do not cover. Every arm was designed, and its reading rule written down, before it was run; the README of each arm states that rule. Result files are written to ext/outputs/ and are read by the analysis scripts named in the paper.

- object_audit: the object-level readout (two locators for the actuated object) and the blinded human audit of it (Section 4.3, Appendix K).
- policy_ctx: the control that replaces random-action contexts by contexts produced by each released agent's own policy (Section 4.1, Appendix D).
- closed_loop: random-shooting model-predictive control inside the emulator with each world model, against a random policy and an emulator-rollout oracle (Section 4.6, Appendix N).
- storm_seed2: the second training seed of STORM on all 26 games (Section 4.2, Appendix E).
- diamond_seed2 and iris_seed2: the second training seed of the two released models on the flagship pair (Section 4.2, Appendix D).
- seed2_flagship: the evaluation of those two replicates with the paper's sweep and the script that scores them against the released checkpoints.
- visited: each model's own closed-loop planner, rerun with the paired intervention measured at the states it visits and its ranking of the candidate actions compared with the emulator's returns, on CPU nodes and again on GPUs (added during the discussion; design and rule in rebuttal/README.md, R6).
- goal: goal reaching at the diagnostic states, where the model must pick the action sequence that reaches a goal frame (added during the discussion; design and rule in rebuttal/README.md, R7).

A fourth architecture was trained with its authors' code but did not produce functioning agents in our environment, so it is not reported; the paper discloses this in Appendix C.
