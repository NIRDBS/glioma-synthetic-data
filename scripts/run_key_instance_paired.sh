#!/usr/bin/env bash
# Key-instance utility with per-slide predictions for every MIL tier of the
# appendix experiments, then the paired per-slide bootstrap and the paper tables.
# Resumable: a cell whose per-slide file exists is skipped.
set -u
cd "$(dirname "$0")/.."
PY=/c/Users/Andrei/anaconda3/envs/glioma/python.exe
LOGS=results/mil_missing_tests_logs
mkdir -p "$LOGS"

EXPS="exp_607 exp_608 exp_609 exp_2.607 exp_2.608 exp_4.608"
TIERS="misvm:mi_svm attention:attention cbr:reconstruction cbr:encoder_head"

for exp in $EXPS; do
  for t in $TIERS; do
    m=${t%%:*}; v=${t##*:}
    out=results/$exp/mil_eval/key_instance_per_slide/${m}_${v}.json
    if [ -f "$out" ]; then echo "skip $exp $m/$v (done)"; continue; fi
    echo "=== $(date +%H:%M:%S) $exp $m/$v"
    "$PY" -u -W ignore scripts/cvae_weighted_rf_vote.py --config "$exp" --cvae-config "$exp" \
        --method "$m" --variant "$v" > "$LOGS/ki_paired_${exp}_${m}_${v}.log" 2>&1
    echo "    exit $? -> $(grep -h 'key_instance_train (NEW RF' "$LOGS/ki_paired_${exp}_${m}_${v}.log" | sed 's/^ *//')"
  done
  "$PY" scripts/key_instance_paired_bootstrap.py --config "$exp" | grep -v '^$'
done

(cd paper && "$PY" gen_appendix_controlled_experiments.py --results-tables-only)
echo "=== $(date +%H:%M:%S) all done"
