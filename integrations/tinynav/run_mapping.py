import argparse
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from tinynav.core.perception_node import PerceptionNode, StereoPairMsg
from tinynav.core.imu_propagator_node import ImuPropagatorNode
from tinynav.core.build_map_node import BuildMapNode
from tinynav.core.math_utils import msg2np


class Tap:
    def __init__(self, publisher, callback):
        self.publisher = publisher
        self.callback = callback

    def publish(self, msg):
        self.publisher.publish(msg)
        self.callback(msg)


def stamp(msg):
    return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seconds", type=float, default=120)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    rclpy.init()
    perception = PerceptionNode(verbose_timer=False)
    propagator = ImuPropagatorNode()
    mapper = BuildMapNode(str(out), verbose_timer=False)
    mapper.T_rgb_to_infra1 = np.eye(4)
    visual, propagated, pending = {}, {}, {}
    stats = []

    def record_visual(msg):
        visual[stamp(msg)] = msg2np(msg)[0]
        propagator.odom_callback(msg)

    def record_propagated(msg):
        pose = msg2np(msg)[0]
        propagated[stamp(msg)] = pose
        # Upstream OdomPoseRecorder stores the (pose, velocity) tuple as pose.
        mapper.continuous_odom_recorder.poses[stamp(msg)] = pose

    perception.odom_pub = Tap(perception.odom_pub, record_visual)
    propagator.odom_pub = Tap(propagator.odom_pub, record_propagated)
    for name in ["keyframe_pose_pub", "keyframe_image_pub", "keyframe_depth_pub"]:
        setattr(
            perception,
            name,
            Tap(
                getattr(perception, name),
                lambda msg, name=name: pending.__setitem__(name, msg),
            ),
        )
    perception.stats_pub = Tap(
        perception.stats_pub, lambda msg: stats.append(json.loads(msg.data))
    )
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=args.bag, storage_id="sqlite3"),
        ConverterOptions("cdr", "cdr"),
    )
    types = {
        t.name: get_message(t.type)
        for t in reader.get_all_topics_and_types()
        if t.name.startswith("/camera/camera/")
    }
    counts = {}
    pairs = {}
    first = None
    start = time.monotonic()
    while reader.has_next():
        topic, raw, timestamp = reader.read_next()
        if first is None:
            first = timestamp
        if timestamp - first >= int(args.seconds * 1e9):
            break
        if topic not in types:
            continue
        msg = deserialize_message(raw, types[topic])
        counts[topic] = counts.get(topic, 0) + 1
        if topic.endswith("infra2/camera_info") and perception.K is None:
            perception.info_callback(msg)
            mapper.info_callback(msg)
            mapper.rgb_camera_K = perception.K.copy()
        elif topic.endswith("/imu"):
            perception._aligned_imu_callback(msg)
            propagator.imu_callback(msg)
        elif topic.endswith("/image_rect_raw"):
            t = stamp(msg)
            pair = pairs.setdefault(t, {})
            pair["left" if "/infra1/" in topic else "right"] = msg
            if len(pair) == 2:
                pending.clear()
                perception._aligned_stereo_callback(
                    StereoPairMsg(pair["left"].header, pair["left"], pair["right"])
                )
                if len(pending) == 3:
                    left = pending["keyframe_image_pub"]
                    gray = mapper.bridge.imgmsg_to_cv2(left, "mono8")
                    rgb = mapper.bridge.cv2_to_imgmsg(
                        cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), "bgr8"
                    )
                    rgb.header = left.header
                    mapper.keyframe_callback(
                        left,
                        pending["keyframe_pose_pub"],
                        pending["keyframe_depth_pub"],
                        rgb,
                    )
                del pairs[t]
                if counts[topic] % 100 == 0:
                    print(
                        "PROGRESS",
                        (timestamp - first) / 1e9,
                        "visual",
                        len(visual),
                        "keyframes",
                        len(mapper.pose_graph_used_pose),
                        flush=True,
                    )
    np.save(out / "visual_odometry.npy", visual, allow_pickle=True)
    np.save(out / "propagated_odometry.npy", propagated, allow_pickle=True)
    (out / "estimator_stats.json").write_text(json.dumps(stats))
    (out / "run_report.json").write_text(
        json.dumps(
            dict(
                bag=args.bag,
                seconds=args.seconds,
                sensor_counts=counts,
                visual_poses=len(visual),
                propagated_poses=len(propagated),
                keyframes=len(mapper.pose_graph_used_pose),
                constraints=len(mapper.relative_pose_constraint),
                elapsed_seconds=time.monotonic() - start,
                mode="offline chronological native callbacks; unchanged estimator and map algorithms; native 0.1333s image throttle; grayscale left converted to BGR for storage; ground truth and ideal IMU excluded",
            ),
            indent=2,
        )
    )
    mapper.save_mapping()
    print("MAPPING_COMPLETE", flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
