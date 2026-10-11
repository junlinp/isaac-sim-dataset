#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
DATA_PYTHON=${DATA_PYTHON:-python3}
USD_PYTHON=${USD_PYTHON:-$DATA_PYTHON}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
"$USD_PYTHON" "$ROOT/scripts/cache_office.py"
"$USD_PYTHON" "$ROOT/scripts/extract_office_mesh.py"
"$DATA_PYTHON" "$ROOT/scripts/plan_office.py"
"$DATA_PYTHON" "$ROOT/scripts/build_route.py"
"$DATA_PYTHON" "$ROOT/scripts/check_clearance.py"
echo OFFICE_PREPARATION_COMPLETE
