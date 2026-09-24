#!/bin/bash
# Locations are read from the environment (see the README); the script stops when one is not set.
ROOT=${ROOT:?set ROOT to the directory of this package}
cd ${ROOT}
qsub ext/policy_ctx/cPC_d_Alien.sh
qsub ext/policy_ctx/cPC_d_Pong.sh
qsub ext/policy_ctx/cPC_d_BankHeist.sh
qsub ext/policy_ctx/cPC_d_Freeway.sh
qsub ext/policy_ctx/cPC_d_Breakout.sh
qsub ext/policy_ctx/cPC_d_Boxing.sh
qsub ext/policy_ctx/cPC_i0.sh
qsub ext/policy_ctx/cPC_i1.sh
