"""Native ROS2 Humble playback smoke test with TinyNav-compatible subscribers."""

import argparse, json, subprocess, time, os
from pathlib import Path
from collections import Counter
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, CameraInfo, Imu
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage
from cv_bridge import CvBridge
from rosbag2_interfaces.srv import Resume

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True)
p.add_argument("--rate", type=float, default=1.0)
a = p.parse_args()
root = Path(a.dataset)
manifest = json.loads((root / "manifest.json").read_text())
counts = Counter()
bridge = CvBridge()
errors = []
seen = {k: [] for k in manifest["topics"]}
rclpy.init()
node = rclpy.create_node("vio_dataset_playback_check")
subs = []


def callback(key, msg):
    try:
        if key in ["left", "right"]:
            image = bridge.imgmsg_to_cv2(msg, "mono8")
            assert image.shape == (640, 544)
        elif key == "info_right":
            assert abs(-msg.p[3] / msg.k[0] - 0.099401839) < 1e-12
        elif key in ["imu", "ideal"]:
            assert (
                msg.header.frame_id == "left_camera_optical_frame"
                and msg.orientation_covariance[0] == -1
            )
        elif key == "gt":
            assert (
                msg.header.frame_id == "world"
                and msg.child_frame_id == "left_camera_optical_frame"
            )
        elif key == "tf":
            assert (
                len(msg.transforms) == 1
                and msg.transforms[0].header.frame_id == "left_camera_optical_frame"
            )
        counts[key] += 1
        if key != "tf":
            seen[key].append(
                msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
            )
    except Exception as e:
        errors.append(repr(e))


msgtypes = {
    "sensor_msgs/msg/Image": Image,
    "sensor_msgs/msg/CameraInfo": CameraInfo,
    "sensor_msgs/msg/Imu": Imu,
    "nav_msgs/msg/Odometry": Odometry,
    "tf2_msgs/msg/TFMessage": TFMessage,
}
for key, item in manifest["topics"].items():
    qos = QoSProfile(
        depth=2000,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=(
            DurabilityPolicy.TRANSIENT_LOCAL
            if key == "tf"
            else DurabilityPolicy.VOLATILE
        ),
    )
    subs.append(
        node.create_subscription(
            msgtypes[item["type"]],
            item["name"],
            lambda msg, key=key: callback(key, msg),
            qos,
        )
    )
log = (root / "native_playback.log").open("w")
proc = subprocess.Popen(
    [
        "ros2",
        "bag",
        "play",
        str(root / "rosbag"),
        "--clock",
        "--rate",
        str(a.rate),
        "--wait-for-all-acked",
        "5000",
        "--disable-loan-message",
        "--start-paused",
        "--read-ahead-queue-size",
        "10000",
    ],
    stdout=log,
    stderr=subprocess.STDOUT,
)
client = node.create_client(Resume, "/rosbag2_player/resume")
assert client.wait_for_service(timeout_sec=15)
ready = time.monotonic() + 2
while time.monotonic() < ready:
    rclpy.spin_once(node, timeout_sec=0.02)
assert all(node.count_publishers(v["name"]) >= 1 for v in manifest["topics"].values())
future = client.call_async(Resume.Request())
rclpy.spin_until_future_complete(node, future, timeout_sec=10)
assert future.done() and future.result() is not None
start = time.monotonic()
exit_time = None
while time.monotonic() - start < manifest["duration_s"] + 40:
    rclpy.spin_once(node, timeout_sec=0.02)
    if proc.poll() is not None:
        if exit_time is None:
            exit_time = time.monotonic()
        if time.monotonic() - exit_time > 2:
            break
if proc.poll() is None:
    proc.terminate()
    proc.wait(timeout=10)
log.close()
node.destroy_node()
rclpy.shutdown()
expected = {k: v["count"] for k, v in manifest["topics"].items()}
received_sets = {k: set(v) for k, v in seen.items()}
expected_sets = {
    k: {
        manifest["timestamps"]["start_ns"]
        + i
        * (
            manifest["timestamps"]["camera_dt_ns"]
            if k in ["left", "right", "info_left", "info_right"]
            else manifest["timestamps"]["imu_dt_ns"]
        )
        for i in range(expected[k])
    }
    for k in seen
    if k != "tf"
}
result = dict(
    status=(
        "passed"
        if not errors
        and dict(counts) == expected
        and all(received_sets[k] == v for k, v in expected_sets.items())
        and proc.returncode == 0
        else "failed"
    ),
    counts=dict(counts),
    expected=expected,
    errors=errors,
    player_exit_code=proc.returncode,
    cv_bridge_mono8_checked=True,
    playback_started_after_subscriber_discovery=True,
    playback_rate=a.rate,
    received_timestamp_ranges={k: [min(v), max(v)] for k, v in seen.items() if v},
    missing_timestamps={
        k: sorted(v - received_sets[k])[:20]
        for k, v in expected_sets.items()
        if v != received_sets[k]
    },
    unexpected_timestamps={
        k: sorted(received_sets[k] - v)[:20]
        for k, v in expected_sets.items()
        if v != received_sets[k]
    },
)
(root / "native_playback_report.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
assert result["status"] == "passed"
