# STORM, second training seed

STORM releases no Atari checkpoints, so the paper trains one model per game with the authors' code and configuration (100k environment steps, a two-layer transformer of width 512), on the same GPU class as the rest of the paper. This folder holds the training scripts of a second training seed for all 26 games. The measurement of the second seed uses the same sweep script as the first (scripts/v16_storm_sweep.py) and is summarised by scripts/seed2_eval.py, which compares the two seeds' game rankings by delay-20 alignment and their taxonomy classes.
