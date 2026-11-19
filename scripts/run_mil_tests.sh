#!/usr/bin/env bash
# MIL evaluation of one configuration: fit the four MIL models, score them at bag level, refit them
# under shuffled bag labels, test spatial coherence (real vs shuffled) and key-instance utility.
#
# Needs the classifiers and the CVAE of the configuration (README, step 2) and the TR/TR models of
# exp_501.
#
# Usage:   bash scripts/run_mil_tests.sh exp_608
# Environment:
#   PY                 Python interpreter (default: python)
#   MISVM_BACKEND      local | modal: where the mi-SVM fits run (default: local; modal runs the five
#                      folds in parallel CPU containers)
#   SKIP_CVAE_CONTROL  set to 1 to skip the shuffled-label CVAE control, which retrains the CVAE on a
#                      Modal GPU (without it, the CVAE rows of the spatial table have no control)
set -euo pipefail
cd "$(dirname "$0")/.."

EXP=${1:?usage: bash scripts/run_mil_tests.sh <configuration, e.g. exp_608>}
PY=${PY:-python}
BACKEND=${MISVM_BACKEND:-local}

echo "=== MIL fits"
"$PY" -m pipeline.mil_fit_stage --config "$EXP" --methods misvm cbr --backend "$BACKEND"
"$PY" -m pipeline.mil_attention_stage --config "$EXP"

echo "=== Bag-level metrics"
"$PY" scripts/bag_level_eval.py --config "$EXP" --method misvm --variant mi_svm
"$PY" scripts/bag_level_eval.py --config "$EXP" --method cbr --variant reconstruction
"$PY" scripts/bag_level_eval.py --config "$EXP" --method cbr --variant encoder_head
"$PY" scripts/attention_bag_level_eval.py --config "$EXP"

echo "=== Shuffled-label controls"
"$PY" scripts/shuffled_label_control.py --config "$EXP" --method misvm --backend "$BACKEND"
"$PY" scripts/shuffled_label_control.py --config "$EXP" --method attention
if [ "${SKIP_CVAE_CONTROL:-0}" != "1" ]; then
    "$PY" scripts/shuffled_label_control.py --config "$EXP" --method cvae   # retrains the CVAE (Modal GPU)
    "$PY" scripts/shuffled_label_control.py --config "$EXP" --method cbr    # scores the retrained CVAE
fi

echo "=== Spatial coherence, real vs shuffled"
"$PY" scripts/spatial_paired_real_vs_shuffled.py --config "$EXP"

echo "=== Key-instance utility"
for tier in misvm:mi_svm attention:attention cbr:reconstruction cbr:encoder_head; do
    "$PY" scripts/cvae_weighted_rf_vote.py --config "$EXP" --cvae-config "$EXP" \
        --method "${tier%%:*}" --variant "${tier##*:}"
done
"$PY" scripts/key_instance_paired_bootstrap.py --config "$EXP"

echo "Done: $EXP. Patient-level versions of the paired tests: python scripts/patient_bootstrap.py --stage all"
