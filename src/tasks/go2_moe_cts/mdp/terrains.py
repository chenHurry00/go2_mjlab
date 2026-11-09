"""Reference terrains (go2_rl_gym ``utils/terrain.py``) as MuJoCo height fields.

Each tile reproduces ``Terrain.make_terrain`` with the Isaac Gym ``terrain_utils``
algorithms on an 80x80 grid (0.1 m, 0.005 m per unit). A tile is 8.5 m wide: the
8 m terrain plus the reference 0.5 m spacing, split as 3 + 3 zero samples so that
neighbouring tiles leave exactly 5 zero samples in between, as in the reference map.

``HeightMap`` rebuilds the global sample grid from the compiled model so height
scans can use the reference lookup (min of 3 neighbouring samples) instead of rays.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import mujoco
import numpy as np
import torch
from mjlab.terrains.heightfield_terrains import color_by_height
from mjlab.terrains.terrain_generator import (
  SubTerrainCfg,
  TerrainGeneratorCfg,
  TerrainGeometry,
  TerrainOutput,
)
from scipy.interpolate import RegularGridInterpolator

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

H_SCALE = 0.1
V_SCALE = 0.005
TILE_PX = 80
PAD_PX = 3
TILE_SIZE = (TILE_PX + 2 * PAD_PX - 1) * H_SCALE  # 8.5 m, 86 samples per side.

Kind = Literal[
  "wave", "slope_neg", "slope_pos", "rough_slope", "stairs_up", "stairs_down", "obstacles", "flat"
]

# Reference terrain id per kind (``terrain.terrain_id``), used by dynamic sigma and
# the per-terrain command ranges.
TERRAIN_ID: dict[str, int] = {
  "wave": 0, "slope_neg": 1, "slope_pos": 1, "rough_slope": 2,
  "stairs_up": 3, "stairs_down": 4, "obstacles": 5, "flat": 8,
}


# ---------------------------------------------------------------------------
# Isaac Gym terrain_utils, operating on an int16 array indexed [x, y].
# ---------------------------------------------------------------------------


def random_uniform(raw, rng, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2):
  lo, hi, st = int(min_height / V_SCALE), int(max_height / V_SCALE), int(step / V_SCALE)
  w, l = raw.shape
  coarse = rng.choice(np.arange(lo, hi + st, st), (int(w * H_SCALE / downsampled_scale), int(l * H_SCALE / downsampled_scale)))
  x = np.linspace(0, w * H_SCALE, coarse.shape[0])
  y = np.linspace(0, l * H_SCALE, coarse.shape[1])
  xx, yy = np.meshgrid(np.linspace(0, w * H_SCALE, w), np.linspace(0, l * H_SCALE, l), indexing="ij")
  # interp2d(kind="linear") on a rectilinear grid is bilinear interpolation.
  raw += np.rint(RegularGridInterpolator((x, y), coarse)((xx, yy))).astype(np.int16)


def pyramid_sloped(raw, slope, platform_size=1.0):
  w, l = raw.shape
  cx, cy = int(w / 2), int(l / 2)
  xx = ((cx - np.abs(cx - np.arange(w))) / cx).reshape(w, 1)
  yy = ((cy - np.abs(cy - np.arange(l))) / cy).reshape(1, l)
  max_height = int(slope * (H_SCALE / V_SCALE) * (w / 2))
  raw += (max_height * xx * yy).astype(raw.dtype)
  p = int(platform_size / H_SCALE / 2)
  x1, y1 = w // 2 - p, l // 2 - p
  edge = raw[x1, y1]
  np.clip(raw, min(edge, 0), max(edge, 0), out=raw)


def pyramid_stairs(raw, step_width, step_height, platform_size=1.0):
  sw, sh, p = int(step_width / H_SCALE), int(step_height / V_SCALE), int(platform_size / H_SCALE)
  height, x0, x1, y0, y1 = 0, 0, raw.shape[0], 0, raw.shape[1]
  while (x1 - x0) > p and (y1 - y0) > p:
    x0, x1, y0, y1 = x0 + sw, x1 - sw, y0 + sw, y1 - sw
    height += sh
    raw[x0:x1, y0:y1] = height


def discrete_obstacles(raw, rng, max_height, min_size, max_size, num_rects, platform_size=1.0):
  mh, lo, hi, p = int(max_height / V_SCALE), int(min_size / H_SCALE), int(max_size / H_SCALE), int(platform_size / H_SCALE)
  w, l = raw.shape
  heights = [-mh, -mh // 2, mh // 2, mh]
  sizes = range(lo, hi, 4)
  for _ in range(num_rects):
    width, length = rng.choice(sizes), rng.choice(sizes)
    i0, j0 = rng.choice(range(0, w - width, 4)), rng.choice(range(0, l - length, 4))
    raw[i0:i0 + width, j0:j0 + length] = rng.choice(heights)
  raw[(w - p) // 2:(w + p) // 2, (l - p) // 2:(l + p) // 2] = 0


def wave(raw, num_waves, amplitude):
  a = int(0.5 * amplitude / V_SCALE)
  w, l = raw.shape
  div = l / (num_waves * np.pi * 2)
  xx = np.arange(w).reshape(w, 1)
  yy = np.arange(l).reshape(1, l)
  raw += (a * np.cos(yy / div) + a * np.sin(xx / div)).astype(raw.dtype)


def make_tile(kind: str, difficulty: float, rng: np.random.Generator) -> np.ndarray:
  """``Terrain.make_terrain`` (IS_HARD branch) for one kind."""
  raw = np.zeros((TILE_PX, TILE_PX), dtype=np.int16)
  slope = 0.1 + difficulty * 0.52
  step_height = 0.05 + 0.23 * difficulty
  if kind == "wave":
    wave(raw, num_waves=5, amplitude=0.1 + 0.2 * difficulty)
    random_uniform(raw, rng)
  elif kind in ("slope_neg", "slope_pos"):
    pyramid_sloped(raw, -slope if kind == "slope_neg" else slope, platform_size=3.0)
  elif kind == "rough_slope":
    pyramid_sloped(raw, slope, platform_size=3.0)
    random_uniform(raw, rng)
  elif kind in ("stairs_up", "stairs_down"):
    pyramid_stairs(raw, 0.31, -step_height if kind == "stairs_up" else step_height, platform_size=3.0)
  elif kind == "obstacles":
    discrete_obstacles(raw, rng, 0.05 + difficulty * 0.25, 1.0, 2.0, 20, platform_size=3.0)
  elif kind != "flat":
    raise ValueError(f"Unknown terrain kind: {kind}")
  return raw


# ---------------------------------------------------------------------------
# Vertical walls (reference ``convert_heightfield_to_trimesh``)
# ---------------------------------------------------------------------------

# Go2 config ``slope_threshold = 1.5``: a step between neighbouring samples higher than
# this is made vertical in the reference mesh (the lower vertices move onto the higher
# ones), smaller steps stay ramps. MuJoCo height fields cannot represent vertical walls.
SLOPE_THRESHOLD = 1.5
WALL_RAW = SLOPE_THRESHOLD * H_SCALE / V_SCALE  # 30 units = 0.15 m.
BOX_THICKNESS = 0.5  # Larger than the highest step (0.257 m) so side walls are solid.


def has_walls(full: np.ndarray) -> bool:
  return max(np.abs(np.diff(full, axis=0)).max(), np.abs(np.diff(full, axis=1)).max()) > WALL_RAW


def wall_boxes(full: np.ndarray) -> list[tuple[int, int, int, int, int]]:
  """Flat-topped boxes ``(i0, i1, j0, j1, height)`` in cell units for the reference mesh.

  With vertical walls the lower sample owns the whole cell between two samples, so a cell's
  height is the minimum of its four corners. Cells of equal height are merged row by row.
  """
  cells = np.minimum(np.minimum(full[:-1, :-1], full[1:, :-1]), np.minimum(full[:-1, 1:], full[1:, 1:]))
  boxes = []
  for h in np.unique(cells):
    open_runs: dict[tuple[int, int], int] = {}  # (j0, j1) -> first row of the run
    for i in range(cells.shape[0] + 1):
      runs = set()
      if i < cells.shape[0]:
        m = np.concatenate([[0], (cells[i] == h).astype(np.int8), [0]])
        d = np.diff(m)
        runs = set(zip(np.where(d == 1)[0].tolist(), np.where(d == -1)[0].tolist(), strict=True))
      for run in [r for r in open_runs if r not in runs]:
        boxes.append((open_runs.pop(run), i, run[0], run[1], int(h)))
      for run in runs:
        open_runs.setdefault(run, i)
  return boxes


# ---------------------------------------------------------------------------
# mjlab sub-terrain
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class RefTerrainCfg(SubTerrainCfg):
  kind: Kind
  num_rows: int = 10
  """Rows of the generator; mjlab draws difficulty uniformly inside a row band,
  the reference uses ``row / num_rows``, so the band index is recovered here."""
  box_walls: bool = False
  """Collide with flat-topped boxes (vertical walls, as the reference mesh) instead of the
  height field on tiles whose steps exceed ``SLOPE_THRESHOLD``. The height field stays in
  the model, without collision, for the height scan."""

  def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
    row = min(int(difficulty * self.num_rows), self.num_rows - 1)
    tile = make_tile(self.kind, row / self.num_rows, rng)
    full = np.zeros((TILE_PX + 2 * PAD_PX,) * 2, dtype=np.int32)
    full[PAD_PX:PAD_PX + TILE_PX, PAD_PX:PAD_PX + TILE_PX] = tile
    lo, hi = int(full.min()), int(full.max())
    span = max(hi - lo, 1)
    norm = ((full - lo) / span).T  # MuJoCo rows run along y.
    uid = uuid.uuid4().hex
    n = full.shape[0]
    field = spec.add_hfield(
      name=f"hfield_{uid}",
      size=[TILE_SIZE / 2, TILE_SIZE / 2, span * V_SCALE, 1.0],
      nrow=n, ncol=n, userdata=norm.flatten().astype(np.float32).tolist(),
    )
    geom = spec.body("terrain").add_geom(
      type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname=field.name,
      pos=[TILE_SIZE / 2, TILE_SIZE / 2, lo * V_SCALE],
      material=color_by_height(spec, full, uid, norm),
    )
    geometries = [TerrainGeometry(geom=geom, hfield=field)]
    if self.box_walls and has_walls(full):
      geom.contype = geom.conaffinity = 0
      geom.group = 3
      body = spec.body("terrain")
      for i0, i1, j0, j1, h in wall_boxes(full):
        box = body.add_geom(
          type=mujoco.mjtGeom.mjGEOM_BOX,
          size=[(i1 - i0) * H_SCALE / 2, (j1 - j0) * H_SCALE / 2, BOX_THICKNESS / 2],
          pos=[(i0 + i1) * H_SCALE / 2, (j0 + j1) * H_SCALE / 2, h * V_SCALE - BOX_THICKNESS / 2],
        )
        geometries.append(TerrainGeometry(geom=box, color=(0.55, 0.55, 0.6, 1.0)))
    # Reference spawn: terrain pixel 40 (4 m into the tile), z = max over the centre 2 x 2 m.
    c = int((TILE_PX * H_SCALE / 2 - 1) / H_SCALE), int((TILE_PX * H_SCALE / 2 + 1) / H_SCALE)
    xy = (PAD_PX + TILE_PX // 2) * H_SCALE
    origin = np.array([xy, xy, tile[c[0]:c[1], c[0]:c[1]].max() * V_SCALE])
    return TerrainOutput(origin=origin, geometries=geometries)


# Column layout: proportions [0.05, 0.2, 0.05, 0.25, 0.1, 0.2, 0, 0, 0.15] with the
# slope split in half by sign, so cumulative sums match the reference choices.
PROPORTIONS: dict[str, float] = {
  "wave": 0.05, "slope_neg": 0.10, "slope_pos": 0.10, "rough_slope": 0.05,
  "stairs_up": 0.25, "stairs_down": 0.10, "obstacles": 0.20, "flat": 0.15,
}


def terrain_generator_cfg(num_rows: int = 10, num_cols: int = 20, box_kinds: tuple[str, ...] = ()) -> TerrainGeneratorCfg:
  return TerrainGeneratorCfg(
    size=(TILE_SIZE, TILE_SIZE),
    border_width=25.0,
    num_rows=num_rows,
    num_cols=num_cols,
    curriculum=True,
    sub_terrains={k: RefTerrainCfg(kind=k, proportion=p, num_rows=num_rows, box_walls=k in box_kinds) for k, p in PROPORTIONS.items()},  # type: ignore[arg-type]
  )


def column_terrain_ids(num_cols: int) -> list[int]:
  """Reference terrain id of every column (mjlab and reference use the same rule)."""
  cum = np.cumsum(list(PROPORTIONS.values()))
  kinds = list(PROPORTIONS)
  return [TERRAIN_ID[kinds[int(np.where(j / num_cols + 0.001 < cum)[0][0])]] for j in range(num_cols)]


# ---------------------------------------------------------------------------
# Height samples
# ---------------------------------------------------------------------------


class HeightMap:
  """Global height samples rebuilt from all height-field geoms of the model."""

  def __init__(self, model: mujoco.MjModel, device: str):
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    geoms = np.where(model.geom_type == mujoco.mjtGeom.mjGEOM_HFIELD)[0]
    if len(geoms) == 0:  # Flat plane: every query returns 0.
      self.x0 = self.y0 = 0.0
      self.samples = torch.zeros(2, 2, device=device)
      return
    tiles = []
    for g in geoms:
      h = model.geom_dataid[g]
      nrow, ncol = model.hfield_nrow[h], model.hfield_ncol[h]
      sx, sy, sz = model.hfield_size[h][:3]
      adr = model.hfield_adr[h]
      z = data.geom_xpos[g][2] + sz * model.hfield_data[adr:adr + nrow * ncol].reshape(nrow, ncol)
      tiles.append((data.geom_xpos[g][0] - sx, data.geom_xpos[g][1] - sy, z.T))  # -> [x, y]
    self.x0 = min(t[0] for t in tiles)
    self.y0 = min(t[1] for t in tiles)
    nx = max(round((t[0] - self.x0) / H_SCALE) + t[2].shape[0] for t in tiles)
    ny = max(round((t[1] - self.y0) / H_SCALE) + t[2].shape[1] for t in tiles)
    grid = np.zeros((nx, ny), dtype=np.float32)
    for x, y, z in tiles:
      i, j = round((x - self.x0) / H_SCALE), round((y - self.y0) / H_SCALE)
      grid[i:i + z.shape[0], j:j + z.shape[1]] = z
    self.samples = torch.from_numpy(grid).to(device)

  def query(self, xy: torch.Tensor) -> torch.Tensor:
    """Reference ``_get_heights``: min of the sample and its +x, +y neighbours."""
    px = ((xy[..., 0] - self.x0) / H_SCALE).floor().long().clamp(0, self.samples.shape[0] - 2)
    py = ((xy[..., 1] - self.y0) / H_SCALE).floor().long().clamp(0, self.samples.shape[1] - 2)
    s = self.samples
    return torch.minimum(torch.minimum(s[px, py], s[px + 1, py]), s[px, py + 1])


def height_map(env: ManagerBasedRlEnv) -> HeightMap:
  hm = getattr(env, "_moe_height_map", None)
  if hm is None:
    hm = HeightMap(env.sim.mj_model, env.device)
    env._moe_height_map = hm  # type: ignore[attr-defined]
  return hm


# Reference ``measured_points_x/y``: meshgrid(x, y) flattened with x outer (187 points).
SCAN_X = [i / 10 for i in range(-8, 9)]
SCAN_Y = [i / 10 for i in range(-5, 6)]


def scan_offsets(device: str) -> torch.Tensor:
  gx, gy = torch.meshgrid(torch.tensor(SCAN_X), torch.tensor(SCAN_Y), indexing="ij")
  return torch.stack([gx.flatten(), gy.flatten()], dim=-1).to(device)


def base_scan_mask(device: str) -> torch.Tensor:
  """Points under the trunk used for the base height estimate (|x| <= 0.2, |y| <= 0.15)."""
  p = scan_offsets(device)
  return (p[:, 0].abs() <= 0.2 + 1e-6) & (p[:, 1].abs() <= 0.15 + 1e-6)


def measured_heights(env: ManagerBasedRlEnv, entity_name: str = "robot") -> torch.Tensor:
  """Terrain heights at the yaw-rotated scan points (reference ``_get_heights``)."""
  cache = getattr(env, "_moe_scan_offsets", None)
  if cache is None:
    cache = scan_offsets(env.device)
    env._moe_scan_offsets = cache  # type: ignore[attr-defined]
  data = env.scene[entity_name].data
  q = data.root_link_quat_w  # (w, x, y, z)
  yaw = 2.0 * torch.atan2(q[:, 3], q[:, 0])  # ``quat_apply_yaw``: drop x, y and renormalise.
  c, s = torch.cos(yaw).unsqueeze(1), torch.sin(yaw).unsqueeze(1)
  x = c * cache[:, 0] - s * cache[:, 1] + data.root_link_pos_w[:, :1]
  y = s * cache[:, 0] + c * cache[:, 1] + data.root_link_pos_w[:, 1:2]
  return height_map(env).query(torch.stack([x, y], dim=-1))
