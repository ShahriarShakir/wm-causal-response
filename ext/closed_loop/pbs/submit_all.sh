#!/bin/bash
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
cd ${ROOT}
qsub ext/closed_loop/pbs/cCL_d_Alien.sh
qsub ext/closed_loop/pbs/cCL_i_Alien.sh
qsub ext/closed_loop/pbs/cCL_d_Amidar.sh
qsub ext/closed_loop/pbs/cCL_i_Amidar.sh
qsub ext/closed_loop/pbs/cCL_d_Asterix.sh
qsub ext/closed_loop/pbs/cCL_i_Asterix.sh
qsub ext/closed_loop/pbs/cCL_d_Boxing.sh
qsub ext/closed_loop/pbs/cCL_i_Boxing.sh
qsub ext/closed_loop/pbs/cCL_d_Breakout.sh
qsub ext/closed_loop/pbs/cCL_i_Breakout.sh
qsub ext/closed_loop/pbs/cCL_d_CrazyClimber.sh
qsub ext/closed_loop/pbs/cCL_i_CrazyClimber.sh
qsub ext/closed_loop/pbs/cCL_d_Freeway.sh
qsub ext/closed_loop/pbs/cCL_i_Freeway.sh
qsub ext/closed_loop/pbs/cCL_d_Frostbite.sh
qsub ext/closed_loop/pbs/cCL_i_Frostbite.sh
qsub ext/closed_loop/pbs/cCL_d_MsPacman.sh
qsub ext/closed_loop/pbs/cCL_i_MsPacman.sh
qsub ext/closed_loop/pbs/cCL_d_Qbert.sh
qsub ext/closed_loop/pbs/cCL_i_Qbert.sh
qsub ext/closed_loop/pbs/cCL_d_Seaquest.sh
qsub ext/closed_loop/pbs/cCL_i_Seaquest.sh
qsub ext/closed_loop/pbs/cCL_d_UpNDown.sh
qsub ext/closed_loop/pbs/cCL_i_UpNDown.sh
qsub ext/closed_loop/pbs/cCL_s0.sh
qsub ext/closed_loop/pbs/cCL_s1.sh
qsub ext/closed_loop/pbs/cCL_ctl.sh
