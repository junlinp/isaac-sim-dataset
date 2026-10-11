"""Export optical-frame IMU, ground truth, calibration, and TinyNav ROS2 SQLite bag."""

import argparse, json, csv
from pathlib import Path
import numpy as np, cv2, yaml
from trajectory import *
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore
from rosbags.interfaces import (
    Qos,
    QosHistory,
    QosReliability,
    QosDurability,
    QosLiveliness,
    QosTime,
)

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True)
a = p.parse_args()
root = Path(a.dataset)
report = json.loads((root / "render_report.json").read_text())
duration = report["duration_s"]
N = int(round(duration * 200))
ncam = report["stereo_pairs"]
assert N == ncam * 10
states = evaluate(np.arange(N) / 200)
stamps = T0_NS + np.arange(N, dtype=np.int64) * IMU_DT_NS
rng = np.random.default_rng(20261011)
dt = 0.005
noise = dict(
    seed=20261011,
    accelerometer_noise_density=0.002,
    gyroscope_noise_density=0.0002,
    accelerometer_random_walk=0.003,
    gyroscope_random_walk=0.00002,
    initial_bias=[0.0, 0.0, 0.0],
    discretization="white std=density/sqrt(dt); bias increment std=random_walk*sqrt(dt)",
)
ba = np.zeros((N, 3))
bg = np.zeros((N, 3))
ba[1:] = np.cumsum(
    rng.normal(0, noise["accelerometer_random_walk"] * np.sqrt(dt), (N - 1, 3)), axis=0
)
bg[1:] = np.cumsum(
    rng.normal(0, noise["gyroscope_random_walk"] * np.sqrt(dt), (N - 1, 3)), axis=0
)
acc = (
    states["f_body"]
    + ba
    + rng.normal(0, noise["accelerometer_noise_density"] / np.sqrt(dt), (N, 3))
)
gyro = (
    states["w_body"]
    + bg
    + rng.normal(0, noise["gyroscope_noise_density"] / np.sqrt(dt), (N, 3))
)
cal = calibration()
cal["noise"] = noise
(root / "calibration.json").write_text(json.dumps(cal, indent=2))
np.savez_compressed(
    root / "sensor_truth.npz",
    timestamps_ns=stamps,
    **states,
    acc_noisy=acc,
    gyro_noisy=gyro,
    bias_acc=ba,
    bias_gyro=bg,
)


def write_csv(path, header, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(header + "\n")
        for i in range(N):
            f.write(
                str(int(stamps[i]))
                + ","
                + ",".join(format(float(x), ".17g") for x in columns[i])
                + "\n"
            )


write_csv(
    root / "mav0/imu0/data.csv",
    "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]",
    np.hstack([gyro, acc]),
)
write_csv(
    root / "mav0/imu0/data_ideal.csv",
    "#timestamp [ns],wx,wy,wz,ax,ay,az",
    np.hstack([states["w_body"], states["f_body"]]),
)
q = states["q_xyzw"]
q_wxyz = q[:, [3, 0, 1, 2]]
write_csv(
    root / "mav0/state_groundtruth_estimate0/data.csv",
    "#timestamp [ns],p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],q_RS_w [],q_RS_x [],q_RS_y [],q_RS_z [],v_RS_R_x [m s^-1],v_RS_R_y [m s^-1],v_RS_R_z [m s^-1],b_w_RS_S_x [rad s^-1],b_w_RS_S_y [rad s^-1],b_w_RS_S_z [rad s^-1],b_a_RS_S_x [m s^-2],b_a_RS_S_y [m s^-2],b_a_RS_S_z [m s^-2]",
    np.hstack([states["p"], q_wxyz, states["v_world"], bg, ba]),
)
with (root / "ground_truth.tum").open("w") as f:
    for i in range(N):
        sec, nsec = divmod(int(stamps[i]), 1_000_000_000)
        f.write(
            f"{sec}.{nsec:09d} "
            + " ".join(format(float(x), ".17g") for x in np.r_[states["p"][i], q[i]])
            + "\n"
        )


def matrix(T):
    return dict(rows=4, cols=4, data=np.asarray(T).reshape(-1).tolist())


for j in [0, 1]:
    T = np.eye(4)
    T[0, 3] = j * BASELINE
    cfg = dict(
        sensor_type="camera",
        comment="T_BS maps camera into body; body is left optical camera / IMU",
        T_BS=matrix(T),
        rate_hz=20,
        resolution=[WIDTH, HEIGHT],
        camera_model="pinhole",
        intrinsics=[FX, FY, CX, CY],
        distortion_model="radial-tangential",
        distortion_coefficients=[0.0] * 4,
    )
    (root / f"mav0/cam{j}/sensor.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
cfg = dict(
    sensor_type="imu",
    T_BS=matrix(np.eye(4)),
    rate_hz=200,
    **{k: v for k, v in noise.items() if "density" in k or "random_walk" in k},
)
(root / "mav0/imu0/sensor.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
(root / "mav0/state_groundtruth_estimate0/sensor.yaml").write_text(
    yaml.safe_dump(
        dict(sensor_type="ground_truth", T_BS=matrix(np.eye(4)), rate_hz=200),
        sort_keys=False,
    )
)
s = get_typestore(Stores.ROS2_HUMBLE)
types = s.types
Time = types["builtin_interfaces/msg/Time"]
Header = types["std_msgs/msg/Header"]
Vec = types["geometry_msgs/msg/Vector3"]
Point = types["geometry_msgs/msg/Point"]
Quat = types["geometry_msgs/msg/Quaternion"]
Imu = types["sensor_msgs/msg/Imu"]
Image = types["sensor_msgs/msg/Image"]
Info = types["sensor_msgs/msg/CameraInfo"]
Odom = types["nav_msgs/msg/Odometry"]
Pose = types["geometry_msgs/msg/Pose"]
PoseCov = types["geometry_msgs/msg/PoseWithCovariance"]
Twist = types["geometry_msgs/msg/Twist"]
TwistCov = types["geometry_msgs/msg/TwistWithCovariance"]
TF = types["tf2_msgs/msg/TFMessage"]
TS = types["geometry_msgs/msg/TransformStamped"]
Transform = types["geometry_msgs/msg/Transform"]


def header(stamp, frame):
    sec, ns = divmod(int(stamp), 1_000_000_000)
    return Header(Time(sec, ns), frame)


def vec(a):
    return Vec(*map(float, a))


def quat(a):
    return Quat(*map(float, a))


left = cal["left_frame"]
right = cal["right_frame"]
zero9 = np.zeros(9)
unknown = zero9.copy()
unknown[0] = -1
acc_cov = np.eye(3).reshape(-1) * noise["accelerometer_noise_density"] ** 2 / dt
gyro_cov = np.eye(3).reshape(-1) * noise["gyroscope_noise_density"] ** 2 / dt
roi = types["sensor_msgs/msg/RegionOfInterest"](0, 0, 0, 0, False)


def info(stamp, j):
    P = np.array([[FX, 0, CX, -FX * BASELINE * j], [0, FY, CY, 0], [0, 0, 1, 0]], float)
    return Info(
        header(stamp, [left, right][j]),
        HEIGHT,
        WIDTH,
        "plumb_bob",
        np.zeros(5),
        np.asarray(cal["K"]).reshape(-1).astype(float),
        np.eye(3).reshape(-1),
        P.reshape(-1),
        0,
        0,
        roi,
    )


def imu(stamp, w, f, noisy):
    return Imu(
        header(stamp, left),
        Quat(0.0, 0.0, 0.0, 1.0),
        unknown.copy(),
        vec(w),
        gyro_cov.copy() if noisy else zero9.copy(),
        vec(f),
        acc_cov.copy() if noisy else zero9.copy(),
    )


forever = QosTime(9223372036, 854775807)
qos = Qos(
    QosHistory.KEEP_LAST,
    1000,
    QosReliability.RELIABLE,
    QosDurability.VOLATILE,
    forever,
    forever,
    QosLiveliness.AUTOMATIC,
    forever,
    False,
)
image_qos = Qos(
    QosHistory.KEEP_LAST,
    100,
    QosReliability.RELIABLE,
    QosDurability.VOLATILE,
    forever,
    forever,
    QosLiveliness.AUTOMATIC,
    forever,
    False,
)
static_qos = Qos(
    QosHistory.KEEP_LAST,
    1,
    QosReliability.RELIABLE,
    QosDurability.TRANSIENT_LOCAL,
    forever,
    forever,
    QosLiveliness.AUTOMATIC,
    forever,
    False,
)
topics = {
    "left": ("/camera/camera/infra1/image_rect_raw", "sensor_msgs/msg/Image"),
    "right": ("/camera/camera/infra2/image_rect_raw", "sensor_msgs/msg/Image"),
    "info_left": ("/camera/camera/infra1/camera_info", "sensor_msgs/msg/CameraInfo"),
    "info_right": ("/camera/camera/infra2/camera_info", "sensor_msgs/msg/CameraInfo"),
    "imu": ("/camera/camera/imu", "sensor_msgs/msg/Imu"),
    "ideal": ("/simulation/imu_ideal", "sensor_msgs/msg/Imu"),
    "gt": ("/simulation/ground_truth/odometry", "nav_msgs/msg/Odometry"),
    "tf": ("/tf_static", "tf2_msgs/msg/TFMessage"),
}
bag = root / "rosbag"
with Writer(bag, version=8) as writer:
    connections = {
        key: writer.add_connection(
            topic,
            typ,
            typestore=s,
            offered_qos_profiles=[
                (
                    static_qos
                    if key == "tf"
                    else image_qos if key in ["left", "right"] else qos
                )
            ],
        )
        for key, (topic, typ) in topics.items()
    }

    def emit(key, stamp, msg):
        writer.write(
            connections[key], int(stamp), s.serialize_cdr(msg, msg.__msgtype__)
        )

    emit(
        "tf",
        stamps[0],
        TF(
            [
                TS(
                    header(stamps[0], left),
                    right,
                    Transform(Vec(BASELINE, 0.0, 0.0), Quat(0.0, 0.0, 0.0, 1.0)),
                )
            ]
        ),
    )
    for i, stamp in enumerate(stamps):
        emit("imu", stamp, imu(stamp, gyro[i], acc[i], True))
        emit(
            "ideal", stamp, imu(stamp, states["w_body"][i], states["f_body"][i], False)
        )
        odom = Odom(
            header(stamp, "world"),
            left,
            PoseCov(Pose(Point(*map(float, states["p"][i])), quat(q[i])), np.zeros(36)),
            TwistCov(
                Twist(vec(states["v_body"][i]), vec(states["w_body"][i])), np.zeros(36)
            ),
        )
        emit("gt", stamp, odom)
        if i % 10 == 0:
            for j, key in enumerate(["info_left", "info_right"]):
                emit(key, stamp, info(stamp, j))
            for j, key in enumerate(["left", "right"]):
                im = cv2.imread(
                    str(root / f"mav0/cam{j}/data/{stamp}.png"), cv2.IMREAD_GRAYSCALE
                )
                assert im is not None and im.shape == (HEIGHT, WIDTH)
                emit(
                    key,
                    stamp,
                    Image(
                        header(stamp, [left, right][j]),
                        HEIGHT,
                        WIDTH,
                        "mono8",
                        0,
                        WIDTH,
                        im.reshape(-1),
                    ),
                )
        if i % 4000 == 0:
            print("EXPORT_PROGRESS", i, N, flush=True)
# Rosbags writes metadata v8; retain it. Native ROS2 Humble playback is checked separately.
manifest = dict(
    status="generated_pending_validation",
    duration_s=duration,
    camera_pairs=ncam,
    imu_samples=N,
    ground_truth_samples=N,
    topics={
        k: dict(
            name=v[0],
            type=v[1],
            count=(
                1
                if k == "tf"
                else ncam if k in ["left", "right", "info_left", "info_right"] else N
            ),
        )
        for k, v in topics.items()
    },
    calibration=cal,
    trajectory="stationary 0..5s and 115..120s; one closed NVIDIA Office corridor route with quintic easing and pitch/roll; IMU co-located with left optical camera",
    timestamps=dict(start_ns=T0_NS, camera_dt_ns=CAMERA_DT_NS, imu_dt_ns=IMU_DT_NS),
    simulator="Isaac Sim 5.1.0",
    ground_truth_twist="linear and angular expressed in left camera optical frame",
    ground_truth_covariance="zero: exact generating trajectory",
    bag_format="rosbag2 SQLite3, CDR serialized ROS2 Humble message definitions",
    estimator_benchmarked=False,
)
(root / "manifest.json").write_text(json.dumps(manifest, indent=2))
print("EXPORT_COMPLETE", root, flush=True)
