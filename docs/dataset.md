# Dataset contract

## ROS 2 bag topics

| Topic | Message | Rate |
|---|---|---:|
| `/camera/camera/infra1/image_rect_raw` | `sensor_msgs/msg/Image`, mono8 | 20 Hz |
| `/camera/camera/infra2/image_rect_raw` | `sensor_msgs/msg/Image`, mono8 | 20 Hz |
| `/camera/camera/infra1/camera_info` | `sensor_msgs/msg/CameraInfo` | 20 Hz |
| `/camera/camera/infra2/camera_info` | `sensor_msgs/msg/CameraInfo` | 20 Hz |
| `/camera/camera/imu` | `sensor_msgs/msg/Imu`, noisy | 200 Hz |
| `/simulation/imu_ideal` | `sensor_msgs/msg/Imu`, ideal | 200 Hz |
| `/simulation/ground_truth/odometry` | `nav_msgs/msg/Odometry` | 200 Hz |
| `/tf_static` | `tf2_msgs/msg/TFMessage` | Once |

Stereo pairs and both CameraInfo messages share each camera timestamp. Ideal IMU, noisy IMU, and ground truth share every IMU timestamp. Epoch is the synthetic integer nanosecond value `1800000000000000000`; camera and IMU steps are 50,000,000 ns and 5,000,000 ns. A full recording samples `[0, 120)`; the final IMU timestamp is at 119.995 s.

Images are width 544, height 640, step 544, encoding mono8. The bag uses SQLite3 with ROS 2 Humble CDR message definitions. The exporter uses `rosbags`; live ROS publication is not required to generate it. Reliable QoS is recorded, and `/tf_static` is transient-local.

## Frames

`T_A_B` maps coordinates in B into A. Left camera and IMU share `left_camera_optical_frame`, including origin and orientation. Right messages use `right_camera_optical_frame`.

```text
T_left_imu = identity(4)
T_left_right = [[1, 0, 0, 0.099401839],
                [0, 1, 0, 0          ],
                [0, 0, 1, 0          ],
                [0, 0, 0, 1          ]]
```

Camera/IMU axes are x right, y down, z forward. World is Z up, with gravity `[0, 0, -9.81]` m/s². Ground-truth odometry has `header.frame_id="world"` and `child_frame_id="left_camera_optical_frame"`. Pose is `T_world_left`; both components of its twist are expressed in the child frame. EuRoC ground-truth CSV linear velocities are expressed in world coordinates instead.

Only the left-to-right transform is broadcast on `/tf_static`. There is no ground-truth world-to-camera TF. IMU orientation is unavailable: `orientation_covariance[0] = -1`. Ground-truth odometry covariance is zero because it is exact generating data.

## Calibration

```text
K = [[305.453491, 0,          269.162323],
     [0,          305.453491, 313.403473],
     [0,          0,          1         ]]
D = [0, 0, 0, 0, 0]
R_rect = identity(3)
P_left[0,3] = 0
P_right[0,3] = -305.453491 * 0.099401839
```

TinyNav's baseline extraction `-P_right[0,3] / K[0,0]` therefore returns the positive baseline. Left image x minus right image x is positive for ordinary visible stereo correspondences.

## IMU model

For continuous camera pose `R(t), p(t)`:

```text
angular velocity = vee(Rᵀ dR/dt)
specific force   = Rᵀ (d²p/dt² − gravity_world)
```

The specific-force formula includes gravity. At rest its norm is 9.81 m/s²; its direction depends on optical-frame orientation, so it is not assumed to point along camera z.

Seed: `20261011`; initial accelerometer and gyro biases are zero. White-noise standard deviation is density / sqrt(dt), and bias random-walk increment standard deviation is random_walk × sqrt(dt), with dt=0.005 s.

| Parameter | Value |
|---|---:|
| Accelerometer noise density | 0.002 m/s²/√Hz |
| Gyroscope noise density | 0.0002 rad/s/√Hz |
| Accelerometer bias random walk | 0.003 m/s³/√Hz |
| Gyroscope bias random walk | 0.00002 rad/s²/√Hz |

The code analytically differentiates spline position and estimates dR/dt with a five-point central difference. Ground truth contains no measurement noise.

## Files

```text
mav0/cam0/{data.csv,sensor.yaml,data/*.png}
mav0/cam1/{data.csv,sensor.yaml,data/*.png}
mav0/imu0/{data.csv,data_ideal.csv,sensor.yaml}
mav0/state_groundtruth_estimate0/{data.csv,sensor.yaml}
rosbag/{metadata.yaml,*.db3}
ground_truth.tum                 # t x y z qx qy qz qw
sensor_truth.npz                # ideal/noisy states and biases
calibration.json
manifest.json
scene/{room.usda,assets/,office_mesh.npz,...}
stereo_preview.mp4
validation_report.json
```

Native playback adds `native_playback_report.json`; packaging adds `camera_projection_report.json`, `source/`, `SHA256SUMS`, and an adjacent archive/checksum/delivery JSON.
