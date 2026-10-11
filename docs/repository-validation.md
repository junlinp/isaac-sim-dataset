# Repository validation

The portable repository workflow was tested on 2026-10-11 on Linux, with Isaac Sim 5.1 and a TITAN RTX 24 GB.

Checks completed:

1. Three CPU trajectory tests passed, including independent pose derivative checks, stationary optical-frame gravity, and integration of the full 120 s ideal IMU stream.
2. `prepare_office.sh` reused the selective NVIDIA Office cache, rebuilt geometry/floorplan/route in a fresh work directory, and passed both-camera mesh clearance checks.
3. `generate.sh --duration 5 --gpu 1 --package` completed with a fresh output directory.
4. All 100 stereo pairs and 1,000 samples each of noisy IMU, ideal IMU, and ground truth passed read-back validation.
5. Native ROS 2 Humble playback received every expected message and timestamp.
6. USD scene dependency resolution, camera projection/extrinsics, final rendered pose, source snapshot, and packaging passed.

The short GPU test covers the packaged workflow and initial stationary capture, with six inspection renders along the full route. The separate [reference 120 s run](office-results.md) covers full moving-image capture and TinyNav mapping. GitHub Actions tests trajectory math on CPU; it does not perform GPU rendering or estimator benchmarking.

Large assets, generated recordings, test logs, and archives remain outside the Git repository. The tiny committed spline contains generated route parameters, not NVIDIA scene geometry.
