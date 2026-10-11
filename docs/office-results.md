# Reference Office run

Reference recording: 120 seconds in Isaac Sim 5.1, rendered on a TITAN RTX 24 GB. The scene contained 3,784 visible meshes and 486,607 triangles. The route length was 121.78 m, with five stationary seconds at each end and minimum left/right camera mesh clearance 0.586 m.

## Dataset validation: passed

- 2,400 stereo pairs, grayscale width 544 × height 640, 20 Hz.
- 24,000 samples each for noisy IMU, ideal IMU, and ground-truth odometry, 200 Hz.
- Every image and recorded message read back; CSV/TUM/NPZ agreement checked.
- Identity left-camera/IMU extrinsics, timestamps, frames, camera projection, baseline, and sampled stereo geometry checked.
- Native ROS 2 Humble playback delivered every expected message and timestamp.
- Maximum ideal discrete IMU integration errors: 7.79 mm position, 6.84 mm/s velocity, 0.00269 degrees rotation.
- Self-contained USD dependencies and packaged file/archive checksums verified.

Start and end ground-truth position: `(-17.037984, 29.967763, 1.350000)` m. They overlap because the route is closed. Quaternion signs may differ while representing the same orientation.

## TinyNav mapping: completed, accuracy criteria failed

| Metric | Result |
|---|---:|
| Visual trajectory position RMSE | 2.694 m |
| Pose-graph position RMSE | 2.811 m |
| Visual orientation RMSE | 5.86 degrees |
| Keyframes / loop constraints | 700 / 57 |
| Occupied voxels within 20 cm of reference mesh | 25.4% |
| Median occupied-voxel surface distance | 0.562 m |
| Median sampled stereo depth absolute error | 0.043 m |
| Median sampled stereo depth relative error | 1.59% |

Alignment is rigid SE(3), with no scale fitting. Ground truth and ideal IMU were excluded from estimator input. Finite poses, proper rotations, and the stereo baseline passed. The actual Office triangles provide the geometry/depth reference; the map is not treated as a complete mesh.

The largest relative tracking errors occur around 42–45, 59, and 66–67 seconds. Native TinyNav throttling yielded visual updates every 0.15 s (about 6.67 Hz), while the camera reached 3.636 m/s: up to 0.55 m between updates. Repetitive/glossy appearance and high inter-frame motion are possible contributors; this run does not isolate a cause. A slower route or a separate all-20-Hz visual-processing run would be useful controlled comparisons. No ground-truth feedback or estimator parameter tuning was applied to this result.

The generator's dataset validation and the estimator's mapping accuracy are separate outcomes. This reference run is evidence that the dataset pipeline works, not a claim that TinyNav maps this scene accurately.
