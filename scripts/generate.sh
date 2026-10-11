#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ISAAC_PYTHON=${ISAAC_PYTHON:-python3}
DATA_PYTHON=${DATA_PYTHON:-python3}
USD_PYTHON=${USD_PYTHON:-$DATA_PYTHON}
ROS_PYTHON=${ROS_PYTHON:-/usr/bin/python3}
OUTPUT= DURATION=120 TRAJECTORY_DURATION= GPU=0 NATIVE=0 PACKAGE=0
usage() {
  echo "Usage: bash scripts/generate.sh --output FRESH_DIRECTORY [--duration SECONDS] [--trajectory-duration SECONDS] [--gpu INDEX] [--native-playback] [--package]"
}
while [ "$#" -gt 0 ]; do
  case "$1" in
    --output) OUTPUT=${2:?missing output}; shift 2 ;;
    --duration) DURATION=${2:?missing duration}; shift 2 ;;
    --trajectory-duration) TRAJECTORY_DURATION=${2:?missing trajectory duration}; shift 2 ;;
    --gpu) GPU=${2:?missing GPU}; shift 2 ;;
    --native-playback) NATIVE=1; shift ;;
    --package) PACKAGE=1; NATIVE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done
[ -n "$OUTPUT" ] || { usage; exit 2; }
if [ -z "$TRAJECTORY_DURATION" ]; then
  if [ "$DURATION" = 5 ]; then TRAJECTORY_DURATION=120; else TRAJECTORY_DURATION=$DURATION; fi
fi
[ ! -e "$OUTPUT" ] || { echo "Output already exists: $OUTPUT" >&2; exit 2; }
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1} OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
"$ISAAC_PYTHON" "$ROOT/scripts/render_dataset.py" --output "$OUTPUT" --duration "$DURATION" --trajectory-duration "$TRAJECTORY_DURATION" --gpu "$GPU"
"$DATA_PYTHON" "$ROOT/scripts/export_dataset.py" --dataset "$OUTPUT"
"$DATA_PYTHON" "$ROOT/scripts/validate_dataset.py" --dataset "$OUTPUT"
if [ "$NATIVE" -eq 1 ]; then
  set +u
  source "${ROS_SETUP:-/opt/ros/humble/setup.bash}"
  set -u
  export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-82}
  # Distro cv_bridge needs distro NumPy. Use system Python only for this check.
  PYTHONPATH="/usr/lib/python3/dist-packages:${PYTHONPATH:-}" "$ROS_PYTHON" "$ROOT/scripts/check_ros2_playback.py" --dataset "$OUTPUT" --rate "${ROS_PLAYBACK_RATE:-2}"
fi
if [ "$PACKAGE" -eq 1 ]; then
  "$USD_PYTHON" "$ROOT/scripts/finalize_dataset.py" --dataset "$OUTPUT" --project "$ROOT"
fi
echo DATASET_GENERATION_COMPLETE
