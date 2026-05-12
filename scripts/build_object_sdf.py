from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh


def _build_sdf(
  mesh_file: Path,
  output_file: Path,
  *,
  resolution: int,
  padding: float,
  chunk_size: int,
) -> None:
  mesh = trimesh.load(mesh_file, force="mesh")
  if mesh.is_empty:
    raise ValueError(f"Empty mesh: {mesh_file}")
  mesh.remove_unreferenced_vertices()
  mesh.fix_normals()

  bounds = np.asarray(mesh.bounds, dtype=np.float32)
  bbox_min = bounds[0] - padding
  bbox_max = bounds[1] + padding
  axes = [
    np.linspace(bbox_min[axis], bbox_max[axis], resolution, dtype=np.float32)
    for axis in range(3)
  ]
  grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
  points = grid.reshape(-1, 3)

  sdf = np.empty(points.shape[0], dtype=np.float32)
  for start in range(0, points.shape[0], chunk_size):
    end = min(start + chunk_size, points.shape[0])
    query_points = points[start:end]
    signed_distance = trimesh.proximity.signed_distance(mesh, query_points)
    unsigned_distance = np.abs(signed_distance).astype(np.float32)
    inside = mesh.contains(query_points)
    sdf[start:end] = np.where(inside, -unsigned_distance, unsigned_distance)

  sdf = sdf.reshape((resolution, resolution, resolution))
  voxel_size = (bbox_max - bbox_min) / float(resolution - 1)
  output_file.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
    output_file,
    sdf=sdf.astype(np.float32),
    bbox_min=bbox_min.astype(np.float32),
    bbox_max=bbox_max.astype(np.float32),
    voxel_size=voxel_size.astype(np.float32),
    mesh_file=str(mesh_file),
    resolution=np.array([resolution, resolution, resolution], dtype=np.int32),
  )
  print(f"saved {output_file} shape={sdf.shape} voxel={voxel_size}")


def main() -> None:
  parser = argparse.ArgumentParser(description="Build local-object SDF grids from STL.")
  parser.add_argument("--object", choices=("left", "right", "both"), default="both")
  parser.add_argument("--resolution", type=int, default=48)
  parser.add_argument("--padding", type=float, default=0.03)
  parser.add_argument("--chunk-size", type=int, default=20000)
  args = parser.parse_args()

  repo_root = Path(__file__).resolve().parents[1]
  specs = {
    "left": (
      repo_root / "src/assets/robots/left_obj/left_scan.stl",
      repo_root / "src/assets/robots/left_obj/left_scan_sdf.npz",
    ),
    "right": (
      repo_root / "src/assets/robots/right_obj/right_scan.stl",
      repo_root / "src/assets/robots/right_obj/right_scan_sdf.npz",
    ),
  }
  names = ("left", "right") if args.object == "both" else (args.object,)
  for name in names:
    mesh_file, output_file = specs[name]
    _build_sdf(
      mesh_file,
      output_file,
      resolution=args.resolution,
      padding=args.padding,
      chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
  main()
