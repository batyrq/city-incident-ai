#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
VENV_DIR="${VENV_DIR:-.venv}"

echo "[bootstrap] creating virtualenv at ${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

install_optional() {
  local label="$1"
  local extra="$2"
  if python -m pip install ".[${extra}]"; then
    echo "[bootstrap] installed optional extra: ${label}"
  else
    echo "[bootstrap] skipped optional extra: ${label}"
  fi
}

install_optional "ultralytics" "ultralytics"
install_optional "opencv" "dev"
install_optional "tracking deps" "tracking"
install_optional "anomalib" "anomalib"
install_optional "sam2" "sam2"

echo "[bootstrap] summary"
python - <<'PY'
from src.utils.env import package_summary
print(package_summary(["yaml", "pandas", "cv2", "ultralytics", "anomalib", "sam2"]))
PY
