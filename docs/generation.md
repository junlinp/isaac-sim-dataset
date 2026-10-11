# Generation code

## Scene preparation

Run `scripts/prepare_office.sh` once before capture:

1. `cache_office.py` downloads the selective Office dependency closure. Existing nonempty files are reused; retries and a final cache report expose failed downloads.
2. `extract_office_mesh.py` traverses visible meshes, including instance proxies, applies their world transforms, and triangulates polygon faces. Authored centimetre-scale props become metre-scale world geometry through their USD transforms.
3. `plan_office.py` rasterizes the floor and indoor geometry into a 10 cm grid. It finds the largest free component with at least 45 cm clearance.
4. `build_route.py` connects explicit Office waypoints with A*, smooths the loop, and fits a periodic degree-five spline. Curvature-weighted parameterization slows turns.
5. `check_clearance.py` queries the actual triangle mesh at all 24,000 left/right camera positions, rejecting routes with clearance at or below 30 cm.

Working files are written under `ISAAC_DATASET_WORKDIR` (default `work/`). A small reference route spline is committed in `config/` for CPU tests; generation normally uses the newly prepared work-directory spline.

## Continuous pose and IMU

`trajectory.py` defines the sensor contract and `evaluate(t)`. It returns position, rotation, quaternion, world/body velocity, world acceleration, body angular velocity, and body specific force. The quintic timing function holds still before 5 s and after 115 s. Height varies around 1.35 m with 6 cm amplitude; pitch and roll vary gently.

All image poses, IMU readings, and ground-truth values come from this same function. Ground truth uses the left camera/IMU pose, not a displaced robot body or simulator viewport camera.

## Isaac Sim rendering

`render_dataset.py` starts a headless `SimulationApp` on the selected GPU and creates a metre-scale Z-up USD stage. It references the locally cached Office scene, adds fixed illumination, and creates two USD cameras with intrinsics derived from physical aperture/focal length parameters. Projection checks verify the resulting pixel coordinates against K.

At camera sample i:

```python
t = i / 20
state = evaluate([t])
T_world_right = T_world_left @ T_left_right
# Optical x-right/y-down/z-forward to USD x-right/y-up/z-backward.
R_world_usd_camera = R_world_optical @ diag([1, -1, -1])
rep.orchestrator.step(rt_subframes=4)
rgb = annotator.get_data()
```

The implementation writes both transforms, lets the renderer update, reads both RGB annotators, converts to grayscale, and saves lossless PNGs. Both cameras receive the same sample-index timestamp. Renderer updates/subframes are for image convergence; they do not define dataset time. These images are global-shutter synthetic observations, with fixed exposure and no motion blur.

Scene assets are hardlinked into the output when possible, with a copy fallback across filesystems. The exported `scene/room.usda` uses a relative Office asset reference.

## Export and validation

`export_dataset.py` evaluates the continuous trajectory independently at 200 Hz, adds seeded IMU noise/bias, and writes EuRoC CSV/YAML, TUM, NPZ, and a ROS 2 bag. Camera images are read from the exact PNGs already rendered. The writer serializes Humble message definitions without requiring a ROS node or Isaac ROS bridge.

`validate_dataset.py` reads every bag message, compares all recorded data to source files, checks timestamps/calibration/frames, integrates ideal IMU samples, samples stereo feature matches, and produces a side-by-side video. Numerical integration tolerances are recorded in the report; sampled stereo checks do not prove every correspondence is correct.

`check_ros2_playback.py` uses native Humble subscribers and cv_bridge. It starts the player paused, waits for discovery, resumes, and verifies every expected count and timestamp at 2× playback.

`finalize_dataset.py` checks resolved USD dependencies, actual camera projection/extrinsics and final pose, then snapshots sources and creates file/archive SHA-256 checksums. Use an explicit `--archive NEW_PATH` if repackaging an existing dataset after updating source documentation.

## Customization

- Change camera/IMU constants and calibration together in `scripts/trajectory.py`; the current validator intentionally enforces the documented contract.
- Change Office waypoints in `scripts/build_route.py`, then rerun preparation and the clearance check.
- Change trajectory timing, height, or orientation in `scripts/trajectory.py`. Changing recording length alone does not rescale the 120 s route; the wrapper therefore supports only 5 s smoke and 120 s full recordings.
- Change noise in `scripts/export_dataset.py`; its calibration/manifest export records the values.
- For another USD environment, change the asset entry, floor/obstacle extraction, route, and lighting together. The existing floor mesh name filters and waypoint coordinates are specific to Office.

This release generates static-scene, prescribed-camera-motion data. It does not simulate rolling shutter, exposure integration, hardware clock drift, camera thermal noise, robot contact dynamics, or a physical IMU device model. Those would require additional models and validation.
