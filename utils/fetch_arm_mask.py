"""Build a 2D image-space mask of the Fetch arm from per-frame joint state.

Used by:
  * ``scripts/visualize_arm_mask.py`` to overlay the projected arm on RGB
    for visual verification.
  * ``scripts/visualize_voxel_obs.py`` (with ``--mask-arm``) to zero out
    arm pixels in depth *before* ``VoxelObservability.integrate_depth``
    so the robot itself doesn't pollute the occupancy grid.

The mask is the union of per-link convex hulls of projected mesh
vertices, then morphologically dilated to cover slack at link boundaries.
Dilation is exposed as a knob so callers can tune coverage vs. tightness.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from utils.fetch_arm_fk import FetchArmFK


# Default URDF + mesh root (sibling project robi_butler; URDF filename
# contains a space — quote when scripting).
DEFAULT_URDF = (
    "/Volumes/External/Workspace/nus_deliver/robi_butler/"
    "resources/fetch_ext/fetch ext.urdf"
)
DEFAULT_MESH_ROOT = (
    "/Volumes/External/Workspace/nus_deliver/robi_butler/"
    "resources/fetch_ext"
)

# Arm + gripper links, base → end-effector. The torso_lift_link is
# included because it's part of the visible robot mass (column rising
# from the base) that the head camera occasionally sees.
DEFAULT_ARM_LINKS: List[str] = [
    "torso_lift_link",
    "shoulder_pan_link",
    "shoulder_lift_link",
    "upperarm_roll_link",
    "elbow_flex_link",
    "forearm_roll_link",
    "wrist_flex_link",
    "wrist_roll_link",
    "gripper_link",
    "l_gripper_finger_link",
    "r_gripper_finger_link",
]


def _load_meshes(fk: FetchArmFK,
                  links: List[str],
                  max_vertices_per_link: int = 2000,
                  ) -> Dict[str, np.ndarray]:
    """Load each link's collision STL, sub-sample to ``max_vertices_per_link``,
    and return per-link vertices already transformed by the link's
    collision-origin offset (so they're in link frame).

    Skips links whose mesh file is missing — a warning is printed but the
    builder continues with the available links.
    """
    import trimesh
    out: Dict[str, np.ndarray] = {}
    rng = np.random.default_rng(0)
    for link in links:
        mp = fk.collision_mesh_path(link)
        if mp is None or not os.path.exists(mp):
            print(f"[arm_mask] skip {link}: no collision mesh ({mp})")
            continue
        try:
            mesh = trimesh.load(mp, force="mesh")
        except Exception as e:
            print(f"[arm_mask] skip {link}: load failed ({e})")
            continue
        V = np.asarray(mesh.vertices, dtype=np.float64)
        if V.shape[0] == 0:
            continue
        if V.shape[0] > max_vertices_per_link:
            idx = rng.permutation(V.shape[0])[:max_vertices_per_link]
            V = V[idx]
        # Apply the link's collision-origin transform (mesh frame → link frame).
        T_origin = fk.collision_origin(link)
        Vh = np.hstack([V, np.ones((V.shape[0], 1))])
        out[link] = (T_origin @ Vh.T).T[:, :3]
    return out


def _project_link_to_mask(verts_link: np.ndarray,
                            T_bl: np.ndarray,
                            T_cb: np.ndarray,
                            K: np.ndarray,
                            image_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """Project link mesh vertices to image and return a convex-hull mask.

    Returns ``None`` if all vertices project behind the camera or land
    well outside the image (in either case the link contributes nothing
    to the mask).
    """
    H, W = image_shape
    T_cl = T_cb @ T_bl
    Vh = np.hstack([verts_link, np.ones((verts_link.shape[0], 1))])
    V_cam = (T_cl @ Vh.T).T[:, :3]
    z = V_cam[:, 2]
    front = z > 1e-3
    if not np.any(front):
        return None
    Xc = V_cam[front]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = fx * (Xc[:, 0] / Xc[:, 2]) + cx
    v = fy * (Xc[:, 1] / Xc[:, 2]) + cy
    pts = np.stack([u, v], axis=1).astype(np.float32)
    margin = 50
    inside = ((pts[:, 0] >= -margin) & (pts[:, 0] < W + margin)
               & (pts[:, 1] >= -margin) & (pts[:, 1] < H + margin))
    if not np.any(inside):
        return None
    pts = pts[inside]
    if pts.shape[0] < 3:
        return None
    hull = cv2.convexHull(pts.astype(np.int32))
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 1)
    return mask


class ArmMaskBuilder:
    """Per-frame projector: joint state + T_bc → ``(H, W) uint8`` arm mask.

    Construction is heavy (parses URDF, loads STLs); call once. ``build``
    is fast (per-frame FK + projection + cv2.dilate).

    Args:
        urdf_path: path to a Fetch URDF file (visual or collision). The
            mesh references inside are resolved relative to ``mesh_root``.
        mesh_root: directory under which ``package://`` references resolve.
        K: (3, 3) camera intrinsics.
        image_shape: ``(H, W)``.
        links: which links to include in the mask. Defaults to the full
            arm + torso + gripper chain; pass a subset to exclude e.g. the
            torso.
        dilate_px: morphological dilation (elliptical kernel) applied to
            the union mask. 0 disables. Default 8 is a few cm at 480×640
            on a Fetch head camera at ~1 m.
        max_vertices_per_link: sub-sample count per mesh for projection
            (more = tighter convex hull but slower).
    """

    def __init__(self, *,
                 urdf_path: str = DEFAULT_URDF,
                 mesh_root: str = DEFAULT_MESH_ROOT,
                 K: np.ndarray,
                 image_shape: Tuple[int, int],
                 links: Optional[List[str]] = None,
                 dilate_px: int = 8,
                 max_vertices_per_link: int = 2000):
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(urdf_path)
        self.K = np.asarray(K, dtype=np.float64)
        self.H, self.W = int(image_shape[0]), int(image_shape[1])
        self.links: List[str] = list(links) if links else list(DEFAULT_ARM_LINKS)
        self.dilate_px = max(0, int(dilate_px))
        self.fk = FetchArmFK(urdf_path, mesh_root=mesh_root)
        self.meshes = _load_meshes(self.fk, self.links,
                                    max_vertices_per_link=max_vertices_per_link)
        self._kernel = (
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (self.dilate_px, self.dilate_px))
            if self.dilate_px > 0 else None
        )

    def build(self,
               joint_angles: Dict[str, float],
               T_bc: np.ndarray) -> np.ndarray:
        """Return ``(H, W) uint8 {0, 1}`` arm mask for one frame."""
        T_bc = np.asarray(T_bc, dtype=np.float64)
        T_cb = np.linalg.inv(T_bc)
        link_T = self.fk.fk(joint_angles, targets=self.links)
        mask = np.zeros((self.H, self.W), dtype=np.uint8)
        for link in self.links:
            verts = self.meshes.get(link)
            if verts is None:
                continue
            link_mask = _project_link_to_mask(
                verts, link_T[link], T_cb, self.K, (self.H, self.W))
            if link_mask is not None:
                mask |= link_mask
        if self._kernel is not None:
            mask = cv2.dilate(mask, self._kernel, iterations=1)
        return mask
