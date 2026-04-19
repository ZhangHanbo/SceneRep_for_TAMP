"""
Visibility predicate p_v^(i) for the Bernoulli-EKF tracker
(bernoulli_ekf.tex §9 eq. eq:pv).

A bounded, deterministic approximation of the true observer-dependent
visibility. For each track i we compute

    p_v^(i) = 1[pi_K(T_cw^-1 T^(i)_{k|k-1}) in Omega_img]
              * prod_{j != i} (1 - rho_{ij,k})

where pi_K is the pinhole projection of the object origin into the image
plane (with a positive-depth gate), Omega_img = [0, W] x [0, H] is the
image domain, and rho_{ij,k} is the fraction of track i's projected
footprint that is both covered by track j and in front of it along the
viewing ray. Distant or non-overlapping j contribute rho = 0.

Input shape. Each track is a dict with at least::

    {
        "oid": int,
        "T": (4,4) np.ndarray,                # world-frame mean
        "bbox_image": (x_min, y_min, x_max, y_max) or None,
        "mean_depth_camera": float or None,   # along +Z in the camera
    }

Tracks without a projected bbox or depth skip the occlusion product
(their p_v depends only on the centroid projection gate) -- useful for
freshly-born tracks with no shape summary yet.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def project_centroid(T_wo: np.ndarray,
                     T_cw: np.ndarray,
                     K: np.ndarray,
                     image_shape: tuple) -> Optional[tuple]:
    """Project the object origin into the image plane.

    Returns (u, v, depth) if the projection is in front of the camera and
    inside [0, W] x [0, H], else None.
    """
    T_co = np.linalg.inv(T_cw) @ T_wo
    p = T_co[:3, 3]
    depth = float(p[2])
    if depth <= 0.0:
        return None
    uv_h = K @ p
    if uv_h[2] <= 0.0:
        return None
    u = float(uv_h[0] / uv_h[2])
    v = float(uv_h[1] / uv_h[2])
    H, W = int(image_shape[0]), int(image_shape[1])
    if u < 0.0 or u >= W or v < 0.0 or v >= H:
        return None
    return (u, v, depth)


def _bbox_area(bbox: tuple) -> float:
    xmin, ymin, xmax, ymax = bbox
    w = max(0.0, float(xmax) - float(xmin))
    h = max(0.0, float(ymax) - float(ymin))
    return w * h


def _bbox_intersection_area(a: tuple, b: tuple) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    x0 = max(ax0, bx0); y0 = max(ay0, by0)
    x1 = min(ax1, bx1); y1 = min(ay1, by1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def occlusion_coefficient(bbox_i: tuple, depth_i: float,
                          bbox_j: tuple, depth_j: float) -> float:
    """Eq. eq:rho.

        rho_{ij} = 1[d_j < d_i] * |A_j & A_i| / |A_i|   in [0, 1]
    """
    if bbox_i is None or bbox_j is None:
        return 0.0
    if depth_i is None or depth_j is None:
        return 0.0
    if float(depth_j) >= float(depth_i):
        return 0.0
    area_i = _bbox_area(bbox_i)
    if area_i <= 0.0:
        return 0.0
    inter = _bbox_intersection_area(bbox_i, bbox_j)
    return max(0.0, min(1.0, inter / area_i))


def visibility_p_v(tracks: List[Dict[str, Any]],
                   K: np.ndarray,
                   T_cw: np.ndarray,
                   image_shape: tuple) -> Dict[int, float]:
    """Compute p_v^(i) for every track supplied.

    Args:
        tracks: list of per-track dicts (see module docstring).
        K: (3, 3) camera intrinsics.
        T_cw: (4, 4) camera-to-world pose (the inverse maps world to camera).
        image_shape: (H, W) or (H, W, C).

    Returns:
        dict oid -> p_v in [0, 1]. A track whose centroid does not project
        into the image receives 0; a track that projects in-frame but has
        no bbox/depth information receives the pure frustum-gate value
        (prod over occluders is 1).
    """
    out: Dict[int, float] = {}

    projections: Dict[int, Optional[tuple]] = {}
    for t in tracks:
        oid = int(t["oid"])
        T_wo = np.asarray(t["T"], dtype=np.float64)
        projections[oid] = project_centroid(T_wo, T_cw, K, image_shape)

    # Precompute bboxes and depths for occlusion.
    bbox_map: Dict[int, Optional[tuple]] = {}
    depth_map: Dict[int, Optional[float]] = {}
    for t in tracks:
        oid = int(t["oid"])
        bbox_map[oid] = t.get("bbox_image")
        depth_map[oid] = t.get("mean_depth_camera")

    for t in tracks:
        oid = int(t["oid"])
        p = projections.get(oid)
        if p is None:
            out[oid] = 0.0
            continue
        bbox_i = bbox_map.get(oid)
        depth_i = depth_map.get(oid)
        prod = 1.0
        if bbox_i is not None and depth_i is not None:
            for j, bbox_j in bbox_map.items():
                if j == oid:
                    continue
                depth_j = depth_map.get(j)
                if bbox_j is None or depth_j is None:
                    continue
                rho = occlusion_coefficient(bbox_i, depth_i, bbox_j, depth_j)
                prod *= (1.0 - rho)
        out[oid] = max(0.0, min(1.0, prod))
    return out
