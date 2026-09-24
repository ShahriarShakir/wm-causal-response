#!/bin/bash
# Runs every analysis on the per-episode result files, then the test suites. Run it from the repository root.
# Order matters: storm_extend.py writes the taxonomy that most later steps read.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD
PY="env -u PYTHONHOME python3"
# analyses that also write a LaTeX table put it in generated_tables/
export PAPER_DIR=${PAPER_DIR:-$PWD/generated_tables}; mkdir -p "$PAPER_DIR"
run() { echo "### $*"; $PY "$@" 2>&1 | grep -vE "Warning|warn" | tail -4; }
run scripts/storm_extend.py
run scripts/response_error.py
run scripts/dpsnr_cells.py
run scripts/psnr_and_crossarch.py
run scripts/psnr_and_crossarch.py --chosen
run scripts/latency_correlate_chosen.py
run scripts/score_predictions.py
run scripts/task_structure_features.py
run scripts/latency_law_threshold_sweep.py
run scripts/hallucination_probe.py
run scripts/hallucination_probe_conditional.py
# predictive_validity.py reads the published Atari-100k returns of the two released models from outputs/published_returns
run scripts/predictive_validity.py outputs/published_returns
run scripts/sensitivity.py 8 Alien Pong iris _k14_w60 _k2
run scripts/sensitivity.py 8 Alien Pong iris
run scripts/hidden_probe_eval.py
run scripts/hidden_fo_eval.py
run scripts/hidden_ctl_eval.py
run scripts/repr_guarded.py
run scripts/storm_hidden_eval.py
run scripts/spec_analysis.py
run scripts/spec_summary.py
run scripts/correspondence.py
run scripts/correspondence_by_class.py
run scripts/decoupling.py
run scripts/object_eval.py
run scripts/seed2_eval.py
run scripts/fsrank_analysis.py
run scripts/metric_auc.py
run scripts/final_audit.py
run ext/closed_loop/closed_loop_eval.py   # closed-loop returns -> outputs/closed_loop_eval.json and both closed-loop tables
OA_TAG=30 HUMAN_EXCLUDE=08fcf5d307 $PY ext/object_audit/audit_eval.py --with-human ext/object_audit/human_labels_2026-09-18.csv,ext/object_audit/human_labels_annotator2_2026-09-19.csv,ext/object_audit/human_labels_annotator3_2026-09-22.csv
$PY tests/run_all.py | tail -3
