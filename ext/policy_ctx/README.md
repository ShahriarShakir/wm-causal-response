# Policy-context control

Question. Is causal blindness an artefact of random-action contexts that lie outside the agent's own state distribution?

Design, fixed before running. For six games, the warm-up, the context and the base action sequence are produced by the released agent's own actor acting in the real emulator (a categorical sample from its policy; pure policy on all six games, and additionally with a ten percent random-action mixture on Alien and Pong). The intervention is the per-game emulator-chosen offset on the same eight-step window; both arms are replayed exactly, and the model imagines both under common random numbers with the same estimators and guard as the main sweep.

Reading rule, fixed before running. Per game, the delay-20 alignment under policy contexts is placed beside the random-context value of the main sweep. Blindness persists if Alien stays below 0.05 and Pong stays tracked in both architectures; otherwise the paper reports that the evaluation distribution changes the verdict, which is itself a result. Either outcome is reported.

Outputs. outputs/policy_{architecture}_{game}-v5_dur8_k{k}_w{w}_eps{eps}.json, with the same structure as the main sweep files; policy_eval.py summarises them into outputs/policy_ctx_eval.json.
