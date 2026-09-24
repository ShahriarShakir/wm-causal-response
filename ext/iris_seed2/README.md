# IRIS, second training seed on the flagship pair

Purpose. Training-seed variation for the second released model, on Alien and Pong.

Design, fixed before launch. The authors' code and default configuration (600 epochs, 100k environment steps) in the authors' own pinned environment, seed 2, one GPU per game, chained jobs using the code's own resume. The training scripts in this folder record the exact invocation.

Reading rule, fixed before launch. The replicate reproduces the flagship if seed-2 Alien is below 0.05 and seed-2 Pong is above IRIS's tracked threshold (0.108) at delay 20 under the emulator-chosen intervention. Either outcome is reported.

Evaluation. The replicate is measured with the paper's sweep, IRIS_CKPT pointing at the last checkpoint and IRIS_OUT=ext/outputs (see ext/seed2_flagship).
