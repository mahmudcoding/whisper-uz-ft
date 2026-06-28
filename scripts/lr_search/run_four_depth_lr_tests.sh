#!/usr/bin/env bash
set -uo pipefail

cd /home/mahmud/whisper-uz-ft
mkdir -p reports/lr_search

declare -a RUNS=(
  "phase4x_encoder_bcd_decoder_2e5_bs4_fast|configs/lr_search/blockwise/encoder_bcd_decoder_2e5_bs4_fast.yaml|reports/lr_search/phase4_encoder_bcd_decoder_2e5_bs4_fast.log"
  "phase4x_encoder_bcd_decoder_1e5_bs4_fast|configs/lr_search/blockwise/encoder_bcd_decoder_1e5_bs4_fast.yaml|reports/lr_search/phase4_encoder_bcd_decoder_1e5_bs4_fast.log"
  "phase4x_full_encoder_decoder_2e5_bs1_safe|configs/lr_search/blockwise/full_encoder_decoder_2e5_bs1_safe.yaml|reports/lr_search/phase4_full_encoder_decoder_2e5_bs1_safe.log"
  "phase4x_full_encoder_decoder_1e5_bs1_safe|configs/lr_search/blockwise/full_encoder_decoder_1e5_bs1_safe.yaml|reports/lr_search/phase4_full_encoder_decoder_1e5_bs1_safe.log"
)

summary="reports/lr_search/four_depth_lr_tests_sequence.log"
{
  echo "Sequence started: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  echo "Runs:"
  printf '  %s\n' "${RUNS[@]}"
} | tee -a "$summary"

for item in "${RUNS[@]}"; do
  IFS='|' read -r experiment_id config_path log_path <<< "$item"
  echo "===== START ${experiment_id}: $(date -u '+%Y-%m-%d %H:%M:%S UTC') =====" | tee -a "$summary"
  .venv/bin/python scripts/lr_search/run_experiment.py \
    --config "$config_path" \
    --experiment-id "$experiment_id" \
    2>&1 | tee "$log_path"
  rc=${PIPESTATUS[0]}
  echo "===== END ${experiment_id}: rc=${rc}: $(date -u '+%Y-%m-%d %H:%M:%S UTC') =====" | tee -a "$summary"
done

echo "Sequence finished: $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee -a "$summary"
