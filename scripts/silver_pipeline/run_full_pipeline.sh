#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_XET=1

echo "[$(date -u +%FT%TZ)] Waiting for acquisition to finish."
while pgrep -f "scripts/silver_pipeline/acquire_silver.py" >/dev/null; do
  sleep 60
done

python - <<'PY'
import json
from pathlib import Path
import yaml
cfg=yaml.safe_load(Path("configs/silver_datasets.yaml").read_text())
for name in cfg["datasets"]:
    path=Path(cfg["output_root"])/name/"acquisition.json"
    if not path.exists():
        raise SystemExit(f"Missing acquisition record: {path}")
    record=json.loads(path.read_text())
    if record["returncode"] != 0:
        raise SystemExit(f"Acquisition failed: {record}")
print("All pinned source acquisitions completed.")
PY

scripts/silver_pipeline/convert_teacher_to_ct2.sh
python scripts/silver_pipeline/export_silver.py
python scripts/silver_pipeline/build_prefilter.py
python scripts/silver_pipeline/score_teacher.py
python scripts/silver_pipeline/finalize_silver.py
python scripts/silver_pipeline/update_silver_docs.py
python scripts/update_docs.py --check
echo "[$(date -u +%FT%TZ)] SILVER pipeline complete."
