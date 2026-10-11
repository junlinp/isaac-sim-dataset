# TinyNav integration

The reference run used TinyNav commit `687199e7febe39d0774a72787c50ac2b870f4044` with its existing runtime configuration.

These optional scripts depend on a working TinyNav checkout and its model/runtime dependencies. TinyNav code and weights are not vendored. The adapter uses native callback APIs from the tested checkout and may need updates for other TinyNav revisions.

Use a ROS 2 Humble environment where `tinynav` is importable, with `cv_bridge`, `rosbag2_py`, and TinyNav's custom `message_filters` overlay. The tested setup required the overlay providing `InputAligner`; the standard distro module alone was insufficient. Source your TinyNav setup and that overlay before running the adapter.

From this repository, in the TinyNav runtime:

```bash
python integrations/tinynav/run_mapping.py \
  --bag /mnt/data/isaac_datasets/office_full01/rosbag \
  --output /mnt/data/isaac_maps/office_map01 --seconds 120

# Run this in the SAME runtime that created the map databases.
python integrations/tinynav/export_depth_samples.py \
  --map /mnt/data/isaac_maps/office_map01
```

The adapter reads camera calibration, stereo images, and noisy IMU in timestamp order, calls native estimation/mapping callbacks, and saves the map. It ignores ideal IMU and simulation ground truth. It preserves the estimator's image throttle and parameters. Left grayscale images are converted to BGR for the mapper's RGB storage input, with identity RGB-to-left extrinsics. The adapter also stores the pose component of TinyNav's `(pose, velocity)` recorder output.

Finish/close the mapper before exporting verification depth samples. Python `shelve`/dbm formats vary by runtime: the tested TinyNav container created Berkeley DB files that the host data environment could not open. `export_depth_samples.py` reads twelve deterministic keyframes with the original backend and exports a portable NPZ for evaluation.

Then use the data environment from the main README:

```bash
"$DATA_PYTHON" integrations/tinynav/evaluate_mapping.py \
  --dataset /mnt/data/isaac_datasets/office_full01 \
  --map /mnt/data/isaac_maps/office_map01

"$DATA_PYTHON" integrations/tinynav/diagnose_tracking.py \
  --dataset /mnt/data/isaac_datasets/office_full01 \
  --map /mnt/data/isaac_maps/office_map01
```

Evaluation matches timestamps and fits one rigid SE(3) transform to visual odometry positions. Scale is fixed at 1. The same alignment is reused for orientations, map keyframes, and occupied voxels. Ground truth is used only after mapping. Surface distances and ray depths reference the actual NVIDIA Office triangle mesh included with the dataset.

Outputs include `verification_report.json`, `verification.png`, aligned TUM trajectories, `occupied_voxels_world.ply`, and `tracking_diagnostics.json`. The local diagnostic acceptance criteria (trajectory RMSE below 0.20 m, more than 70% of occupied voxels within 0.20 m of the reference mesh) are not an upstream benchmark standard. Occupancy surface consistency does not establish complete scene coverage.

The reference Office run failed these criteria; see [measured results](../../docs/office-results.md). Use fresh map output directories for comparisons, and record the TinyNav commit, runtime/model versions, and any parameter changes separately.
