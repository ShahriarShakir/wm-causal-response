# DIAMOND, second training seed on the flagship pair

Purpose. Training-seed variation for a released model, on Alien and Pong.

Design, fixed before launch. The authors' code and default Atari-100k configuration (100k environment steps; the trainer runs 950 collection epochs and 50 final epochs), seed 2, one GPU per game, chained jobs with the code's own resume. The training scripts in this folder record the exact invocation.

Reading rule, fixed before launch. The replicate reproduces the flagship if seed-2 Alien is below 0.05 and seed-2 Pong is above DIAMOND's tracked threshold of 0.40 at delay 20 under both interventions. Either outcome is reported.

Evaluation. The replicate is measured with the paper's sweep, DIAMOND_CKPT pointing at the last checkpoint and DIAMOND_OUT=ext/outputs (see ext/seed2_flagship).
