"""Read every recorded message and check optical-frame sensor/ground-truth consistency."""

import argparse, json, csv, hashlib, subprocess
from pathlib import Path
from collections import Counter
import numpy as np, cv2
from scipy.spatial.transform import Rotation
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
from trajectory import *

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True)
a = p.parse_args()
root = Path(a.dataset)
m = json.loads((root / "manifest.json").read_text())
truth_file = np.load(root / "sensor_truth.npz")
truth = {k: truth_file[k] for k in truth_file.files}
truth_file.close()
N = m["imu_samples"]
C = m["camera_pairs"]
cal = m["calibration"]
stamps = truth["timestamps_ns"]
assert np.array_equal(np.asarray(cal["T_left_imu"]), np.eye(4))
assert N == 10 * C
assert np.array_equal(np.diff(stamps), np.full(N - 1, IMU_DT_NS))
assert stamps[0] == T0_NS
for j in [0, 1]:
    rows = list(csv.reader((root / f"mav0/cam{j}/data.csv").open()))[1:]
    assert len(rows) == C
    assert [int(row[0]) for row in rows] == stamps[::10].tolist()
    assert len(list((root / f"mav0/cam{j}/data").glob("*.png"))) == C
    for stamp, filename in rows:
        im = cv2.imread(
            str(root / f"mav0/cam{j}/data" / filename), cv2.IMREAD_UNCHANGED
        )
        assert im.shape == (HEIGHT, WIDTH) and im.dtype == np.uint8
s = get_typestore(Stores.ROS2_HUMBLE)
counts = Counter()
previous = {}
max_errors = Counter()
tf_checked = False
names = {v["name"]: k for k, v in m["topics"].items()}


def vector(v):
    return np.array([v.x, v.y, v.z])


def msg_stamp(h):
    return h.stamp.sec * 1_000_000_000 + h.stamp.nanosec


with Reader(root / "rosbag") as reader:
    assert {c.topic: c.msgtype for c in reader.connections} == {
        v["name"]: v["type"] for v in m["topics"].values()
    }
    assert reader.message_count == sum(v["count"] for v in m["topics"].values())
    for con, stamp, raw in reader.messages():
        key = names[con.topic]
        msg = s.deserialize_cdr(raw, con.msgtype)
        counts[key] += 1
        if key in previous:
            assert stamp > previous[key]
        previous[key] = stamp
        if key == "tf":
            assert len(msg.transforms) == 1
            tf = msg.transforms[0]
            assert (
                tf.header.frame_id == cal["left_frame"]
                and tf.child_frame_id == cal["right_frame"]
            )
            assert np.allclose(vector(tf.transform.translation), [BASELINE, 0, 0])
            assert tf.transform.rotation.w == 1
            tf_checked = True
            continue
        assert msg_stamp(msg.header) == stamp
        i = (stamp - T0_NS) // IMU_DT_NS
        assert stamps[i] == stamp
        if key in ["imu", "ideal"]:
            assert (
                msg.header.frame_id == cal["left_frame"]
                and msg.orientation_covariance[0] == -1
            )
            w = truth["gyro_noisy"][i] if key == "imu" else truth["w_body"][i]
            acc = truth["acc_noisy"][i] if key == "imu" else truth["f_body"][i]
            assert np.array_equal(vector(msg.angular_velocity), w) and np.array_equal(
                vector(msg.linear_acceleration), acc
            )
        elif key == "gt":
            assert (
                msg.header.frame_id == "world"
                and msg.child_frame_id == cal["left_frame"]
            )
            pose = msg.pose.pose
            twist = msg.twist.twist
            assert np.array_equal(vector(pose.position), truth["p"][i])
            assert np.array_equal(vector(twist.linear), truth["v_body"][i])
            assert np.array_equal(vector(twist.angular), truth["w_body"][i])
            q = np.array(
                [
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                ]
            )
            assert np.allclose(np.linalg.norm(q), 1, atol=1e-12)
            assert np.array_equal(q, truth["q_xyzw"][i])
        elif key.startswith("info_"):
            j = int(key == "info_right")
            assert msg.header.frame_id == [cal["left_frame"], cal["right_frame"]][j]
            assert (msg.width, msg.height) == (WIDTH, HEIGHT)
            assert np.array_equal(msg.k, np.array(cal["K"]).reshape(-1))
            assert np.array_equal(msg.r, np.eye(3).reshape(-1))
            assert np.count_nonzero(msg.d) == 0
            assert np.isclose(msg.p[3], -FX * BASELINE * j, atol=1e-10)
            if j:
                assert abs(-msg.p[3] / msg.k[0] - BASELINE) < 1e-12
        else:
            j = int(key == "right")
            assert msg.header.frame_id == [cal["left_frame"], cal["right_frame"]][j]
            assert (msg.height, msg.width, msg.step, msg.encoding) == (
                HEIGHT,
                WIDTH,
                WIDTH,
                "mono8",
            )
            im = cv2.imread(str(root / f"mav0/cam{j}/data/{stamp}.png"), 0)
            assert np.array_equal(msg.data.reshape(HEIGHT, WIDTH), im)
assert tf_checked and counts == Counter({k: v["count"] for k, v in m["topics"].items()})
# Check CSV values without converting integer nanosecond timestamps to floats.
for file, cols in [
    ("mav0/imu0/data.csv", np.hstack([truth["gyro_noisy"], truth["acc_noisy"]])),
    ("mav0/imu0/data_ideal.csv", np.hstack([truth["w_body"], truth["f_body"]])),
    (
        "mav0/state_groundtruth_estimate0/data.csv",
        np.hstack(
            [
                truth["p"],
                truth["q_xyzw"][:, [3, 0, 1, 2]],
                truth["v_world"],
                truth["bias_gyro"],
                truth["bias_acc"],
            ]
        ),
    ),
]:
    rows = list(csv.reader((root / file).open()))[1:]
    assert len(rows) == N
    for i, row in enumerate(rows):
        assert int(row[0]) == stamps[i] and np.allclose(
            np.array(row[1:], float), cols[i], rtol=0, atol=1e-14
        )
tum_rows = [
    line.split()
    for line in (root / "ground_truth.tum").read_text().splitlines()
    if line and not line.startswith("#")
]
assert len(tum_rows) == N
for i, row in enumerate(tum_rows):
    sec, ns = row[0].split(".")
    assert int(sec) * 1_000_000_000 + int(ns.ljust(9, "0")) == stamps[i]
    assert np.allclose(
        np.array(row[1:], float),
        np.r_[truth["p"][i], truth["q_xyzw"][i]],
        rtol=0,
        atol=1e-14,
    )
# Stationary optical frame: gravity is not assumed to lie on optical z.
stationary = (truth["t"] < 5) | (truth["t"] >= 115)
assert np.max(np.linalg.norm(truth["w_body"][stationary], axis=1)) < 1e-7
assert np.allclose(
    truth["f_body"][stationary],
    np.einsum("nji,j->ni", truth["R"][stationary], -GRAVITY),
    atol=1e-8,
)
assert np.max(np.linalg.norm(truth["v_world"][stationary], axis=1)) < 1e-10
# Integrate ideal samples using midpoint attitude/acceleration and compare generating poses.
R = truth["R"][0].copy()
pos = truth["p"][0].copy()
vel = truth["v_world"][0].copy()
max_p = max_v = max_r = 0.0
for i in range(N - 1):
    w = (truth["w_body"][i] + truth["w_body"][i + 1]) / 2
    f = (truth["f_body"][i] + truth["f_body"][i + 1]) / 2
    dt = 0.005
    mid = R @ Rotation.from_rotvec(w * dt / 2).as_matrix()
    acc = mid @ f + GRAVITY
    pos = pos + vel * dt + 0.5 * acc * dt**2
    vel = vel + acc * dt
    R = R @ Rotation.from_rotvec(w * dt).as_matrix()
    max_p = max(max_p, float(np.linalg.norm(pos - truth["p"][i + 1])))
    max_v = max(max_v, float(np.linalg.norm(vel - truth["v_world"][i + 1])))
    max_r = max(max_r, float(Rotation.from_matrix(R.T @ truth["R"][i + 1]).magnitude()))
assert max_p < 0.05 and max_v < 0.01 and max_r < 0.002, (max_p, max_v, max_r)
# Rectification/disparity sanity check on spatially distributed matched features.
orb = cv2.ORB_create(2500)
stereo_checks = []
for ci in sorted(set(np.linspace(0, C - 1, min(8, C)).astype(int))):
    stamp = stamps[ci * 10]
    ims = [cv2.imread(str(root / f"mav0/cam{j}/data/{stamp}.png"), 0) for j in [0, 1]]
    k0, d0 = orb.detectAndCompute(ims[0], None)
    k1, d1 = orb.detectAndCompute(ims[1], None)
    assert d0 is not None and d1 is not None
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d0, d1, k=2)
    good = [
        x
        for pair in matches
        if len(pair) == 2
        for x, y in [pair]
        if x.distance < 0.65 * y.distance
    ]
    p0 = np.array([k0[x.queryIdx].pt for x in good])
    p1 = np.array([k1[x.trainIdx].pt for x in good])
    assert len(good) > 20
    err = np.abs(p0[:, 1] - p1[:, 1])
    inlier = err < 1.5
    assert inlier.mean() > 0.75
    disp = p0[inlier, 0] - p1[inlier, 0]
    assert (disp > 0).mean() > 0.9
    stereo_checks.append(
        dict(
            camera_index=int(ci),
            matches=len(good),
            epipolar_inlier_fraction=float(inlier.mean()),
            median_vertical_error_px=float(np.median(err)),
            positive_disparity_fraction=float((disp > 0).mean()),
        )
    )
# Preview is derived from the exact recorded grayscale image pairs.
preview = root / "stereo_preview.mp4"
log = (root / "preview_ffmpeg.log").open("w")
proc = subprocess.Popen(
    [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{WIDTH*2}x{HEIGHT+40}",
        "-r",
        "20",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-threads",
        "2",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(preview),
    ],
    stdin=subprocess.PIPE,
    stderr=log,
)
for ci, stamp in enumerate(stamps[::10]):
    panel = np.zeros((HEIGHT + 40, WIDTH * 2, 3), np.uint8)
    for j in [0, 1]:
        panel[40:, j * WIDTH : (j + 1) * WIDTH] = cv2.cvtColor(
            cv2.imread(str(root / f"mav0/cam{j}/data/{stamp}.png"), 0),
            cv2.COLOR_GRAY2RGB,
        )
    cv2.putText(
        panel,
        f"Left / IMU     {ci/20:.2f}s",
        (12, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "Right     stereo 544 x 640",
        (WIDTH + 12, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    proc.stdin.write(panel.tobytes())
    if ci == min(C - 1, 600):
        cv2.imwrite(
            str(root / "stereo_preview.jpg"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
        )
proc.stdin.close()
assert proc.wait(timeout=180) == 0
log.close()
probe = json.loads(
    subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,nb_read_frames,duration",
            "-of",
            "json",
            str(preview),
        ]
    )
)
assert int(probe["streams"][0]["nb_read_frames"]) == C
validation = dict(
    status="passed",
    ideal_integration_tolerances=dict(
        position_m=0.05,
        velocity_m_s=0.01,
        rotation_rad=0.002,
        method="second-order midpoint integration of discrete 200 Hz samples",
    ),
    stereo_pairs=C,
    imu_samples=N,
    ground_truth_messages=counts["gt"],
    all_rosbag_messages_read=True,
    all_images_compared_to_bag=True,
    all_csv_values_compared=True,
    tum_trajectory_compared=True,
    identity_left_imu_extrinsics=True,
    ground_truth_frame="world -> left_camera_optical_frame",
    ground_truth_twist_frame="left_camera_optical_frame",
    ideal_integration_max_position_error_m=max_p,
    ideal_integration_max_velocity_error_m_s=max_v,
    ideal_integration_max_rotation_error_rad=max_r,
    stereo_feature_checks=stereo_checks,
    preview_probe=probe,
    estimator_benchmarked=False,
)
(root / "validation_report.json").write_text(json.dumps(validation, indent=2))
m["status"] = "generated_and_validated"
(root / "manifest.json").write_text(json.dumps(m, indent=2))
print("VALIDATION_PASSED", json.dumps(validation), flush=True)
