# Evaluation of the second training seeds

eval_diamond_seed2.sh and eval_iris_seed2.sh run the paper's sweep on the second-seed checkpoints for Alien and Pong under the emulator-chosen and the fixed intervention (60 episodes, 8-step window, the same delays and guard as the main sweep), writing to ext/outputs/. Each script first checks that its training reached its final epoch and skips a game whose training is incomplete.

scripts/flagship_seed2_eval.py then compares each replicate with the released checkpoint using the estimator of Table 1 (mean over evaluation seeds of the per-seed median delay-20 alignment) and applies the reading rules written in ext/diamond_seed2/README.md and ext/iris_seed2/README.md, writing outputs/flagship_seed2_eval.json.
