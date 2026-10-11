import json
from pathlib import Path
import numpy as np
from pxr import Usd, UsdGeom

from settings import WORK_DIR, OFFICE_ENTRY

PROJECT = WORK_DIR
PROJECT.mkdir(parents=True, exist_ok=True)
ENTRY = str(OFFICE_ENTRY)
stage = Usd.Stage.Open(ENTRY)
all_triangles, records = [], []
cache = UsdGeom.XformCache(Usd.TimeCode.Default())
for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
    if not prim.IsA(UsdGeom.Mesh):
        continue
    if UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible:
        continue
    mesh = UsdGeom.Mesh(prim)
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
    if not len(points):
        continue
    matrix = np.asarray(cache.GetLocalToWorldTransform(prim))
    points = points @ matrix[:3, :3] + matrix[3, :3]
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=int)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)
    offsets = np.r_[0, counts.cumsum()[:-1]]
    faces = []
    for n in np.unique(counts):
        starts = offsets[counts == n]
        for j in range(1, n - 1):
            faces.append(
                np.stack(
                    [indices[starts], indices[starts + j], indices[starts + j + 1]],
                    axis=1,
                )
            )
    triangles = points[np.concatenate(faces)].astype(np.float32)
    # Exterior skyline and the oversized support ground do not define indoor free space.
    indoor = not any(
        x in str(prim.GetPath()) for x in ["/GroundPlane/", "/SM_Buildings/"]
    )
    start = sum(r["triangles"] for r in records)
    records.append(
        dict(
            path=str(prim.GetPath()),
            minimum=points.min(0).tolist(),
            maximum=points.max(0).tolist(),
            triangles=len(triangles),
            start=start,
            indoor=indoor,
        )
    )
    all_triangles.append(triangles)
triangles = np.concatenate(all_triangles)
np.savez_compressed(PROJECT / "office_mesh.npz", triangles=triangles)
(PROJECT / "mesh_manifest.json").write_text(
    json.dumps(
        dict(
            entry=ENTRY,
            meters_per_unit=UsdGeom.GetStageMetersPerUnit(stage),
            up_axis=str(UsdGeom.GetStageUpAxis(stage)),
            mesh_count=len(records),
            triangle_count=len(triangles),
            meshes=records,
        ),
        indent=2,
    )
)
print("EXTRACTED", len(records), "meshes", len(triangles), "triangles", flush=True)
