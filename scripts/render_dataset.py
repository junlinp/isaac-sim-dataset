"""Render calibrated stereo images in a deterministic self-contained mesh room."""

import argparse, json, os, sys, time, traceback, shutil
from pathlib import Path
import numpy as np
from trajectory import *
from settings import ASSET_CACHE, WORK_DIR
from io_utils import link_or_copy

p = argparse.ArgumentParser()
p.add_argument("--output", required=True)
p.add_argument("--duration", type=float, default=120)
p.add_argument("--trajectory-duration", type=float, default=120)
p.add_argument("--gpu", type=int, default=1)
a = p.parse_args()
configure_trajectory(a.trajectory_duration)
if a.duration <= 0 or a.duration > a.trajectory_duration:
    p.error("Recording duration must be positive and no longer than trajectory duration")
out = Path(a.output)
out.mkdir(parents=True, exist_ok=False)
for folder in ["mav0/cam0/data", "mav0/cam1/data", "scene/textures"]:
    (out / folder).mkdir(parents=True)
from isaacsim import SimulationApp

app = SimulationApp(
    {
        "headless": True,
        "width": WIDTH,
        "height": HEIGHT,
        "active_gpu": a.gpu,
        "physics_gpu": a.gpu,
        "multi_gpu": False,
    }
)
import omni.usd, omni.replicator.core as rep, carb, cv2
from pxr import Usd, UsdGeom, UsdShade, UsdLux, Sdf, Gf

try:
    settings = carb.settings.get_settings()
    for key, val in [
        ("/rtx/post/aa/op", 0),
        ("/rtx/post/histogram/enabled", False),
        ("/rtx/post/tonemap/op", 2),
        ("/rtx/post/motionblur/enabled", False),
        ("/rtx/post/dof/enabled", False),
    ]:
        settings.set(key, val)
    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    cache = ASSET_CACHE / "cloud"
    shutil.copytree(cache, out / "scene/assets", copy_function=link_or_copy)
    entry = out / "scene/assets/Assets/Isaac/5.1/Isaac/Environments/Office/office.usd"
    office = UsdGeom.Xform.Define(stage, "/World/Office")
    office.GetPrim().GetReferences().AddReference(str(entry))
    # The provider disables most fixture lights. Add fixed downward area lights
    # at those authored fixture locations; preserve all geometry and materials.
    fixture_positions = []
    for prim in stage.Traverse():
        if "BP_CeilingLight" in str(prim.GetPath()) and prim.HasAPI(UsdLux.LightAPI):
            position = (
                UsdGeom.Xformable(prim)
                .ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                .ExtractTranslation()
            )
            fixture_positions.append(tuple(position))
    for li, (x, y, z) in enumerate(fixture_positions):
        lamp = UsdLux.RectLight.Define(stage, f"/World/DatasetLighting/Panel_{li:03d}")
        lamp.CreateIntensityAttr(10000.0)
        lamp.CreateWidthAttr(2.0)
        lamp.CreateHeightAttr(0.8)
        lamp.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.9))
        UsdGeom.Xformable(lamp).AddTranslateOp().Set(Gf.Vec3d(x, y, z - 0.18))
    # Some corridors have emissive meshes without authored LightAPI fixtures.
    # Fixed ceiling fill lights follow the route, spaced at least 3 m apart.
    route_points = evaluate(np.linspace(5, trajectory_duration() - 5, 4401))["p"]
    fill_positions = []
    for point in route_points:
        if (
            not fill_positions
            or min(np.linalg.norm(point[:2] - q[:2]) for q in fill_positions) > 3.0
        ):
            fill_positions.append(point)
    for li, position in enumerate(fill_positions):
        lamp = UsdLux.RectLight.Define(
            stage, f"/World/DatasetLighting/CorridorFill_{li:03d}"
        )
        lamp.CreateIntensityAttr(10000.0)
        lamp.CreateWidthAttr(1.2)
        lamp.CreateHeightAttr(0.8)
        lamp.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.9))
        UsdGeom.Xformable(lamp).AddTranslateOp().Set(
            Gf.Vec3d(float(position[0]), float(position[1]), 2.65)
        )
    project = WORK_DIR
    for asset in ["office_mesh.npz", "floorplan.npz", "mesh_manifest.json"]:
        shutil.copy2(project / asset, out / "scene" / asset)
    shutil.copy2(project / "route_report.json", out / "scene/clearance_report.json")
    shutil.copy2(project / "route.png", out / "scene/layout.png")
    cameras = []
    annotators = []
    ops = []
    products = []
    for name in ["Left", "Right"]:
        camera = UsdGeom.Camera.Define(stage, "/World/" + name)
        ha = 20.955
        va = ha * HEIGHT * FX / (WIDTH * FY)
        camera.CreateHorizontalApertureAttr(ha)
        camera.CreateVerticalApertureAttr(va)
        camera.CreateFocalLengthAttr(FX * ha / WIDTH)
        camera.CreateHorizontalApertureOffsetAttr(-(CX - WIDTH / 2) * ha / WIDTH)
        camera.CreateVerticalApertureOffsetAttr((CY - HEIGHT / 2) * va / HEIGHT)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.05, 100))
        camera.CreateFStopAttr(0)
        projection = camera.GetCamera().frustum.ComputeProjectionMatrix()
        for local, expected in [
            ((0, 0, -1), (CX, CY)),
            ((0.1, 0, -1), (CX + 0.1 * FX, CY)),
            ((0, -0.1, -1), (CX, CY + 0.1 * FY)),
        ]:
            ndc = projection.Transform(Gf.Vec3d(*local))
            pixel = np.array([(ndc[0] + 1) * WIDTH / 2, (1 - ndc[1]) * HEIGHT / 2])
            assert np.max(np.abs(pixel - expected)) < 0.001, (pixel, expected)
        ops.append(UsdGeom.Xformable(camera).AddTransformOp())
        cameras.append(camera)
        product = rep.create.render_product(str(camera.GetPath()), (WIDTH, HEIGHT))
        products.append(product)
        ann = rep.AnnotatorRegistry.get_annotator("rgb")
        ann.attach(product)
        annotators.append(ann)
    cal = calibration()
    (out / "calibration.json").write_text(json.dumps(cal, indent=2))
    (out / "scene/scene_manifest.json").write_text(
        json.dumps(
            dict(
                provider="NVIDIA",
                scene_name="Isaac Sim 5.1 Office",
                source_url="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Environments/Office/office.usd",
                scene_type="NVIDIA furnished multi-room office",
                mesh_count=json.loads((project / "mesh_manifest.json").read_text())[
                    "mesh_count"
                ],
                triangle_count=json.loads((project / "mesh_manifest.json").read_text())[
                    "triangle_count"
                ],
                objects=[],
                lighting="original emissive fixtures plus fixed 10000-intensity 2x0.8m downward area lights at authored ceiling fixture positions plus fixed 1.2x0.8m corridor fill panels spaced 3m along route at z=2.65m; fixed exposure; no randomization",
                external_assets=False,
                assets_cached_locally=True,
                route=json.loads((project / "route_report.json").read_text()),
            ),
            indent=2,
        )
    )
    inspection = []
    for ii, t in enumerate(np.array([0, 25, 45, 65, 85, 110]) / 120 * trajectory_duration()):
        state = evaluate([t])
        left = np.eye(4)
        left[:3, :3] = state["R"][0]
        left[:3, 3] = state["p"][0]
        right = left.copy()
        right[:3, 3] += left[:3, 0] * BASELINE
        for op, T in zip(ops, [left, right]):
            usd = T.copy()
            usd[:3, :3] = usd[:3, :3] @ np.diag([1, -1, -1])
            op.Set(Gf.Matrix4d(usd.T.tolist()))
        for _ in range(30 if ii == 0 else 2):
            app.update()
        rep.orchestrator.step(rt_subframes=4)
        rgb = np.asarray(annotators[0].get_data())[..., :3].copy()
        inspection.append(rgb)
        cv2.imwrite(
            str(out / f"inspection_{int(t):03d}.jpg"),
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        )
    panel = np.concatenate(
        [
            np.concatenate(inspection[:3], axis=1),
            np.concatenate(inspection[3:], axis=1),
        ],
        axis=0,
    )
    cv2.imwrite(
        str(out / "inspection_views.jpg"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
    )
    n = int(round(a.duration * 20))
    start = time.monotonic()
    rows = []
    for i in range(n):
        t = i / 20
        state = evaluate([t])
        left = np.eye(4)
        left[:3, :3] = state["R"][0]
        left[:3, 3] = state["p"][0]
        right = left.copy()
        right[:3, 3] += left[:3, 0] * BASELINE
        for op, T in zip(ops, [left, right]):
            usd = T.copy()
            usd[:3, :3] = usd[:3, :3] @ np.diag([1, -1, -1])
            op.Set(Gf.Matrix4d(usd.T.tolist()))
        for _ in range(30 if i == 0 else 2):
            app.update()
        rep.orchestrator.step(rt_subframes=4)
        stamp = T0_NS + i * CAMERA_DT_NS
        filename = f"{stamp}.png"
        for j, ann in enumerate(annotators):
            rgb = np.asarray(ann.get_data())[..., :3]
            assert rgb.shape == (HEIGHT, WIDTH, 3), rgb.shape
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            if i == 0:
                cv2.imwrite(str(out / f"first_camera{j}.png"), gray)
                print(
                    "FIRST_IMAGE",
                    j,
                    float(gray.min()),
                    float(gray.max()),
                    float(gray.mean()),
                    float(gray.std()),
                    flush=True,
                )
            assert (
                gray.std() > 15 and 20 < gray.mean() < 235
            ), "Empty/flat or badly exposed rendering"
            assert cv2.imwrite(
                str(out / f"mav0/cam{j}/data" / filename),
                gray,
                [cv2.IMWRITE_PNG_COMPRESSION, 3],
            )
        rows.append(f"{stamp},{filename}\n")
        if i % 100 == 0 or i == n - 1:
            print(
                "RENDER_PROGRESS",
                json.dumps(
                    dict(
                        frame=i + 1,
                        total=n,
                        elapsed_s=round(time.monotonic() - start, 1),
                    )
                ),
                flush=True,
            )
    for j in [0, 1]:
        (out / f"mav0/cam{j}/data.csv").write_text(
            "#timestamp [ns],filename\n" + "".join(rows)
        )
    office.GetPrim().GetReferences().ClearReferences()
    office.GetPrim().GetReferences().AddReference(
        "assets/Assets/Isaac/5.1/Isaac/Environments/Office/office.usd"
    )
    stage.GetRootLayer().Export(str(out / "scene/room.usda"))
    (out / "render_report.json").write_text(
        json.dumps(
            dict(
                status="rendered",
                duration_s=a.duration,
                trajectory_duration_s=trajectory_duration(),
                stereo_pairs=n,
                width=WIDTH,
                height=HEIGHT,
                wall_time_s=time.monotonic() - start,
                gpu=a.gpu,
                global_shutter=True,
                motion_blur=False,
                usd_projection_matches_K=True,
                multi_gpu=False,
                timestamps="integer fixed-step, independent of render updates",
            ),
            indent=2,
        )
    )
    print("RENDER_COMPLETE", out, flush=True)
except BaseException:
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)
finally:
    app.close()
