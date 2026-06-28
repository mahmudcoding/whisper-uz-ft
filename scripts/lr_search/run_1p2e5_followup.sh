#!/usr/bin/env bash
set -euo pipefail

# Insert two requested 300-step screens between completed Block-C candidates.
# The Phase 4 controller is stopped only after its current child has written
# metrics, so no active training subprocess is interrupted.

cd /home/mahmud/whisper-uz-ft

CONTROLLER_PID="${CONTROLLER_PID:-2106991}"
CURRENT_METRICS="outputs_lr_search/blockwise/phase4b_screen_block_c_1em06_d_8em06/metrics.json"
LOG_DIR="reports/lr_search"
LOG_FILE="$LOG_DIR/phase4_1p2e5_followup.log"
DECODER_ID="phase4x_screen_decoder_1p2em05"
BLOCK_D_ID="phase4x_screen_block_d_1p2em05"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

resume_controller() {
  if kill -0 "$CONTROLLER_PID" 2>/dev/null; then
    kill -CONT "$CONTROLLER_PID" || true
  fi
}
trap resume_controller EXIT INT TERM

if ! kill -0 "$CONTROLLER_PID" 2>/dev/null; then
  echo "[$(date -Is)] Controller PID $CONTROLLER_PID is not running; refusing to reorder the queue." >&2
  exit 1
fi

echo "[$(date -Is)] Waiting for current Block-C screen metrics: $CURRENT_METRICS"
until [[ -f "$CURRENT_METRICS" ]]; do
  sleep 2
done

# The runner writes metrics before it exits. Stop the waiting parent while the
# child completes cleanup, preventing launch of the next Phase 4 candidate.
kill -STOP "$CONTROLLER_PID"
echo "[$(date -Is)] Paused Phase 4 controller $CONTROLLER_PID after current metrics were written."

while pgrep -af "run_experiment.py.*phase4b_screen_block_c_1em06_d_8em06" >/dev/null; do
  sleep 2
done

source .venv/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "[$(date -Is)] Starting decoder-only 1.2e-5 screen."
.venv/bin/python scripts/lr_search/run_experiment.py \
  --config configs/lr_search/phase1a_decoder_lr_8e6.yaml \
  --experiment-id "$DECODER_ID" \
  --set experiment_name=phase4_decoder_1p2e5_screen \
  --set decoder_learning_rate=1.2e-5 \
  --set learning_rate=1.2e-5

DECODER_STABLE="$(python - <<'PY'
import json
from pathlib import Path
p = Path('outputs_lr_search/phase4x_screen_decoder_1p2em05/metrics.json')
if not p.exists():
    print('false')
else:
    m = json.loads(p.read_text())
    b = m.get('best_validation_metrics') or {}
    stable = (
        m.get('status') == 'completed'
        and m.get('stable') is True
        and float(b.get('eval_wer', float('inf'))) <= 2.0
        and float(b.get('eval_hallucination_rate', 0.0)) <= 0.05
        and float(b.get('eval_language_confusion_rate', 0.0)) <= 0.05
    )
    print(str(stable).lower())
PY
)"

is_strict_regime_winner() {
  local candidate_id="$1"
  shift
  python - "$candidate_id" "$@" <<'PY'
import json
import sys
from pathlib import Path

root = Path('outputs_lr_search')
candidate_id, *controls = sys.argv[1:]

def load(experiment_id):
    path = root / experiment_id / 'metrics.json'
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    best = payload.get('best_validation_metrics') or {}
    if payload.get('status') != 'completed' or not payload.get('stable'):
        return None
    if best.get('eval_wer') is None or best.get('eval_cer') is None:
        return None
    return float(best['eval_wer']), float(best['eval_cer'])

candidate = load(candidate_id)
comparators = [item for item in (load(control) for control in controls) if item is not None]
if candidate is not None and (not comparators or candidate < min(comparators)):
    print('true')
else:
    print('false')
PY
}

if [[ "$DECODER_STABLE" == "true" ]]; then
  echo "[$(date -Is)] Decoder-only 1.2e-5 passed screening; starting its 30h validation run."
  .venv/bin/python scripts/lr_search/run_experiment.py \
    --config configs/lr_search/decoder_best_main.yaml \
    --experiment-id phase4x_main_decoder_1p2em05 \
    --set experiment_name=phase4_decoder_1p2e5_main \
    --set decoder_learning_rate=1.2e-5 \
    --set learning_rate=1.2e-5

  if [[ "$(is_strict_regime_winner phase4x_main_decoder_1p2em05 phase2_decoder_8em06 phase2_decoder_2em05)" == "true" ]]; then
    echo "[$(date -Is)] Decoder-only 1.2e-5 is the best stable 30h decoder-only result; testing 1.6e-5."
    .venv/bin/python scripts/lr_search/run_experiment.py \
      --config configs/lr_search/phase1a_decoder_lr_8e6.yaml \
      --experiment-id phase4x_screen_decoder_1p6em05 \
      --set experiment_name=phase4_decoder_1p6e5_screen \
      --set decoder_learning_rate=1.6e-5 \
      --set learning_rate=1.6e-5
    DECODER_16_STABLE="$(python - <<'PY'
import json
from pathlib import Path
p = Path('outputs_lr_search/phase4x_screen_decoder_1p6em05/metrics.json')
m = json.loads(p.read_text()) if p.exists() else {}
b = m.get('best_validation_metrics') or {}
print(str(m.get('status') == 'completed' and m.get('stable') is True and float(b.get('eval_wer', float('inf'))) <= 2.0 and float(b.get('eval_hallucination_rate', 0.0)) <= 0.05 and float(b.get('eval_language_confusion_rate', 0.0)) <= 0.05).lower())
PY
)"
    if [[ "$DECODER_16_STABLE" == "true" ]]; then
      echo "[$(date -Is)] Decoder-only 1.6e-5 passed screening; starting its 30h validation run."
      .venv/bin/python scripts/lr_search/run_experiment.py \
        --config configs/lr_search/decoder_best_main.yaml \
        --experiment-id phase4x_main_decoder_1p6em05 \
        --set experiment_name=phase4_decoder_1p6e5_main \
        --set decoder_learning_rate=1.6e-5 \
        --set learning_rate=1.6e-5
    else
      echo "[$(date -Is)] Decoder-only 1.6e-5 failed screening; its 30h run will not start."
    fi
  else
    echo "[$(date -Is)] Decoder-only 1.2e-5 was not the best stable 30h decoder-only result; 1.6e-5 skipped."
  fi

  echo "[$(date -Is)] Starting Block-D 1.2e-5 screen with decoder fixed at 8e-6."
  .venv/bin/python scripts/lr_search/run_experiment.py \
    --config configs/lr_search/phase4_block_d_1p2e5_screen.yaml \
    --experiment-id "$BLOCK_D_ID"

  BLOCK_D_STABLE="$(python - <<'PY'
import json
from pathlib import Path
p = Path('outputs_lr_search/phase4x_screen_block_d_1p2em05/metrics.json')
if not p.exists():
    print('false')
else:
    m = json.loads(p.read_text())
    b = m.get('best_validation_metrics') or {}
    stable = (
        m.get('status') == 'completed'
        and m.get('stable') is True
        and float(b.get('eval_wer', float('inf'))) <= 2.0
        and float(b.get('eval_hallucination_rate', 0.0)) <= 0.05
        and float(b.get('eval_language_confusion_rate', 0.0)) <= 0.05
    )
    print(str(stable).lower())
PY
)"
  if [[ "$BLOCK_D_STABLE" == "true" ]]; then
    echo "[$(date -Is)] Block-D 1.2e-5 passed screening; starting its 30h validation run."
    .venv/bin/python scripts/lr_search/run_experiment.py \
      --config configs/lr_search/phase4_block_d_1p2e5_main.yaml \
      --experiment-id phase4x_main_block_d_1p2em05

    if [[ "$(is_strict_regime_winner phase4x_main_block_d_1p2em05 phase2_upper_encoder_5em07 phase2_upper_encoder_1em06 phase2_upper_encoder_2em06 phase2_upper_encoder_5em06 phase2_upper_encoder_8em06)" == "true" ]]; then
      echo "[$(date -Is)] Block-D 1.2e-5 is the best stable 30h upper-encoder result; testing 1.6e-5."
      .venv/bin/python scripts/lr_search/run_experiment.py \
        --config configs/lr_search/phase4_block_d_1p6e5_screen.yaml \
        --experiment-id phase4x_screen_block_d_1p6em05
      BLOCK_D_16_STABLE="$(python - <<'PY'
import json
from pathlib import Path
p = Path('outputs_lr_search/phase4x_screen_block_d_1p6em05/metrics.json')
m = json.loads(p.read_text()) if p.exists() else {}
b = m.get('best_validation_metrics') or {}
print(str(m.get('status') == 'completed' and m.get('stable') is True and float(b.get('eval_wer', float('inf'))) <= 2.0 and float(b.get('eval_hallucination_rate', 0.0)) <= 0.05 and float(b.get('eval_language_confusion_rate', 0.0)) <= 0.05).lower())
PY
)"
      if [[ "$BLOCK_D_16_STABLE" == "true" ]]; then
        echo "[$(date -Is)] Block-D 1.6e-5 passed screening; starting its 30h validation run."
        .venv/bin/python scripts/lr_search/run_experiment.py \
          --config configs/lr_search/phase4_block_d_1p6e5_main.yaml \
          --experiment-id phase4x_main_block_d_1p6em05
      else
        echo "[$(date -Is)] Block-D 1.6e-5 failed screening; its 30h run will not start."
      fi
    else
      echo "[$(date -Is)] Block-D 1.2e-5 was not the best stable 30h upper-encoder result; 1.6e-5 skipped."
    fi
  else
    echo "[$(date -Is)] Block-D 1.2e-5 failed screening; its 30h run will not start."
  fi
else
  echo "[$(date -Is)] Decoder-only 1.2e-5 failed screening; Block-D 1.2e-5 will not run."
fi

echo "[$(date -Is)] Follow-up complete; resuming Phase 4 controller."
resume_controller
trap - EXIT INT TERM
