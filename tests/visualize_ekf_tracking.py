#!/usr/bin/env python3
"""
Per-frame 5-panel visualization of the Bernoulli-EKF scene-graph tracker.

Panel layout (2 rows x 3 cols):
  [1] Perception overlay   [2] Top-down: state entering frame   [3] EKF step intermediates
  [4] Top-down: post-predict   [5] Top-down: post-update       [6] r / cov evolution

Drives a thin instrumented replica of pose_update.orchestrator._fast_tier_bernoulli
(so we can snapshot state between predict / associate / update / birth / prune
and dump per-track intermediates — d2, log_lik, Huber weight, existence r delta, etc.).

Data: expects apple_in_the_tray dataset layout:
  datasets/apple_in_the_tray/
    rgb/rgb_NNNNNN.png
    depth/depth_NNNNNN.npy
    pose_txt/amcl_pose.txt          (world <- base)
  tests/visualization_pipeline/apple_in_the_tray/perception/detection_h/
    detection_NNNNNN_final.json     (SAM2-tracked detections with masks + IDs)

Output: tests/visualization_pipeline/apple_in_the_tray/ekf_debug/
    frame_NNNNNN.png

Run:
    conda run -n ocmp_test python tests/visualize_ekf_tracking.py \
        --max-frame 700 --step 1
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

SCENEREP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCENEREP_ROOT)

from pose_update.association import hungarian_associate, oracle_associate
from pose_update.bernoulli import (
    r_predict, r_assoc_update_loglik, r_miss_update, r_birth,
)
from pose_update.ekf_se3 import (
    huber_weight, process_noise_for_phase, saturate_covariance,
)
from pose_update.icp_pose import PoseEstimator
from pose_update.orchestrator import BernoulliConfig
from pose_update.rbpf_state import RBPFState
from pose_update.slam_interface import PoseEstimate
from pose_update.visibility import visibility_p_v


# ─── data paths ──────────────────────────────────────────────────────────
DATASET_DIR = os.path.join(SCENEREP_ROOT, "datasets")
VIZ_BASE = os.path.join(SCENEREP_ROOT, "tests", "visualization_pipeline")

# ─── Fetch head camera intrinsics (from configs/*.yaml) ──────────────────
K_DEFAULT = np.array([
    [554.3827, 0.0, 320.5],
    [0.0, 554.3827, 240.5],
    [0.0, 0.0,     1.0],
], dtype=np.float64)

# palette matches visualize_sam2_observations for inter-viz consistency
_PALETTE_RGB = [
    (  0, 200,  80), (220,  60,  40), ( 40, 140, 220), (245, 200,  20),
    (160,  80, 200), (240, 130,  30), ( 20, 180, 160), (230, 120, 110),
    (100, 160, 230), (250, 220,  60), ( 80, 200, 120), (200, 100, 160),
    (140, 200,  50), (100, 100, 240), (230, 160,  80), ( 40, 220, 200),
    (220,  80, 200), (120, 120, 120), (200, 220, 120), ( 60, 100, 180),
]


def _palette_color(oid: int) -> Tuple[int, int, int]:
    return _PALETTE_RGB[int(oid) % len(_PALETTE_RGB)]


def _palette_color_f(oid: int) -> Tuple[float, float, float]:
    r, g, b = _palette_color(oid)
    return (r / 255.0, g / 255.0, b / 255.0)


# ─────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────

def _load_amcl_poses(path: str) -> List[np.ndarray]:
    """Parse `amcl_pose.txt` lines: `idx x y z qx qy qz qw`."""
    out: List[np.ndarray] = []
    with open(path, "r") as f:
        for line in f:
            arr = line.strip().split()
            if len(arr) != 8:
                continue
            _, tx, ty, tz, qx, qy, qz, qw = map(float, arr)
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            T[:3, 3] = [tx, ty, tz]
            out.append(T)
    return out


def _load_detection_json(path: str) -> List[Dict[str, Any]]:
    """Decode one detection_h JSON into a list of dicts (mask decoded)."""
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        data = json.load(f)
    out: List[Dict[str, Any]] = []
    for det in data.get("detections", []):
        mask_b64 = det.get("mask", "")
        if not mask_b64:
            continue
        try:
            mask_bytes = base64.b64decode(mask_b64)
            mask = np.array(Image.open(BytesIO(mask_bytes)).convert("L"))
            mask = (mask > 128).astype(np.uint8)
        except Exception:
            continue
        out.append({
            "id": int(det.get("object_id")),
            "label": det.get("label", "unknown"),
            "mask": mask,
            "score": float(det.get("score", 0.0)),
            "mean_score": float(det.get("mean_score", 0.0)),
            "n_obs": int(det.get("n_obs", 0)),
            "box": det.get("box"),
        })
    return out


def _load_rgb(path: str) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        return None
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _load_depth(path: str) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        return None
    d = np.load(path)
    return d.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────
# Instrumented tracker
# ─────────────────────────────────────────────────────────────────────────

class InstrumentedTracker:
    """Thin replica of `orchestrator._fast_tier_bernoulli` with snapshots.

    Owns an RBPFState (n=1 single-particle, effectively a plain EKF on
    SE(3)), a PoseEstimator (ICP chain mode) that converts (mask, depth)
    into `(T_co, R_icp)` per detection, and Bernoulli bookkeeping in
    `self.existence / self.object_labels / self.sam2_tau`.

    `step()` returns a debug dict containing:
      enter_tracks      — tracks at start of frame
      post_predict_tracks — after predict step
      assoc             — {match, unmatched_tracks, unmatched_dets, cost_matrix}
      matched           — per-pair {d2, w, log_lik, r_prev, r_new}
      missed            — per-track {p_v, p_d_tilde, r_prev, r_new}
      births            — per-birth {det_idx, new_oid, r_new, label, score}
      pruned            — per-prune {oid, r}
      post_update_tracks — final state
      slam_pose         — T_wb this frame
    """

    def __init__(self,
                 K: np.ndarray,
                 bernoulli_cfg: BernoulliConfig,
                 pose_method: str = "icp_chain"):
        self.K = np.asarray(K, dtype=np.float64)
        self.cfg = bernoulli_cfg
        self.pose_est = PoseEstimator(K=self.K, method=pose_method)
        # Single-particle RBPF collapses to a plain per-object EKF.
        self.state = RBPFState(n_particles=1)
        self.object_labels: Dict[int, str] = {}
        self.frames_since_obs: Dict[int, int] = {}
        self.existence: Dict[int, float] = {}
        self.sam2_tau: Dict[int, int] = {}
        self._frame_count = 0

    # ────────── state capture helper ──────────
    def _capture_tracks(self) -> Dict[int, Dict[str, Any]]:
        out: Dict[int, Dict[str, Any]] = {}
        for oid in self.object_labels:
            pe = self.state.collapsed_object(oid)
            if pe is None:
                continue
            out[int(oid)] = {
                "T": pe.T.copy(),
                "cov": pe.cov.copy(),
                "label": self.object_labels[oid],
                "r": float(self.existence.get(oid, 0.0)),
                "frames_since_obs": int(self.frames_since_obs.get(oid, 0)),
                "sam2_tau": int(self.sam2_tau.get(oid, -1)),
            }
        return out

    # ────────── birth / prune ──────────
    def _birth(self, det: Dict[str, Any]) -> Optional[int]:
        raw_id = det.get("id")
        if raw_id is None:
            d_id = max(self.object_labels.keys(), default=0) + 1
        else:
            d_id = int(raw_id)
            if d_id in self.object_labels:
                d_id = max(self.object_labels.keys(), default=0) + 1
        T_co = det.get("T_co")
        if T_co is None:
            return None
        T_co = np.asarray(T_co, dtype=np.float64)
        R_icp = np.asarray(det.get("R_icp", np.eye(6) * 1e-3),
                           dtype=np.float64)
        init_cov = (0.5 * (R_icp + R_icp.T)
                    if self.cfg.init_cov_from_R
                    else np.diag([0.05] * 6))
        self.state.ensure_object(d_id, T_co, init_cov)
        self.object_labels[d_id] = det.get("label", "unknown")
        self.frames_since_obs[d_id] = 0
        score = float(det.get("score", 1.0))
        self.existence[d_id] = r_birth(score,
                                        lambda_b=self.cfg.lambda_b,
                                        lambda_c=self.cfg.lambda_c)
        tau_raw = det.get("sam2_id", det.get("id"))
        if tau_raw is not None:
            try:
                self.sam2_tau[d_id] = int(tau_raw)
            except (TypeError, ValueError):
                self.sam2_tau[d_id] = -1
        else:
            self.sam2_tau[d_id] = -1
        return d_id

    def _prune(self, oid: int) -> None:
        self.state.delete_object(oid)
        self.object_labels.pop(oid, None)
        self.frames_since_obs.pop(oid, None)
        self.existence.pop(oid, None)
        self.sam2_tau.pop(oid, None)

    def _compute_visibility(self,
                            dets: List[Dict[str, Any]],
                            image_shape: tuple) -> Dict[int, float]:
        slam = self.state.collapsed_base()
        T_wb = slam.T
        tracks: List[Dict[str, Any]] = []
        for oid in self.object_labels:
            pe = self.state.collapsed_object(oid)
            if pe is None:
                continue
            bbox_im = None
            mean_depth = None
            for det in dets:
                if det.get("label") == self.object_labels.get(oid):
                    if det.get("box") is not None:
                        try:
                            bbox_im = tuple(float(v) for v in det["box"])
                        except Exception:
                            bbox_im = None
                    if det.get("T_co") is not None:
                        mean_depth = float(det["T_co"][2, 3])
                    break
            tracks.append({
                "oid": int(oid),
                "T": pe.T,
                "bbox_image": bbox_im,
                "mean_depth_camera": mean_depth,
            })
        return visibility_p_v(tracks, self.cfg.K if self.cfg.K is not None
                              else self.K,
                              T_wb,
                              self.cfg.image_shape or image_shape)

    # ────────── one step ──────────
    def step(self,
             rgb: np.ndarray,
             depth: np.ndarray,
             T_wb: np.ndarray,
             detections: List[Dict[str, Any]],
             phase: str = "idle") -> Tuple[Dict[str, Any],
                                            List[Dict[str, Any]]]:
        dbg: Dict[str, Any] = {"frame": self._frame_count,
                                "slam_pose": T_wb.copy()}

        # 0. SLAM ingest (very small covariance: no SLAM uncertainty stress).
        slam_pe = PoseEstimate(T=T_wb.astype(np.float64),
                               cov=np.diag([1e-6] * 6))
        self.state.ingest_slam(slam_pe)

        # 1. Per-detection ICP → (T_co, R_icp).
        dets_with_pose: List[Dict[str, Any]] = []
        for det in detections:
            T_co, R_icp, fitness, rmse = self.pose_est.estimate(
                oid=det["id"], mask=det["mask"], depth=depth,
            )
            d2 = dict(det)
            if T_co is None:
                d2["T_co"] = None
                d2["R_icp"] = None
                d2["fitness"] = fitness
                d2["rmse"] = rmse
                d2["_icp_ok"] = False
            else:
                d2["T_co"] = T_co
                d2["R_icp"] = R_icp
                d2["fitness"] = fitness
                d2["rmse"] = rmse
                d2["_icp_ok"] = True
            dets_with_pose.append(d2)

        # Enter snapshot (before predict).
        dbg["enter_tracks"] = self._capture_tracks()

        # 2. Predict state + existence.
        def Q_fn(oid: int, _particle) -> np.ndarray:
            return process_noise_for_phase(
                phase=phase,
                is_target=False,
                frames_since_observation=self.frames_since_obs.get(oid, 0),
                frame="world",
            )
        self.state.predict_objects(Q_fn, P_max=self.cfg.P_max)
        for oid in self.frames_since_obs:
            self.frames_since_obs[oid] += 1
        for oid in list(self.existence.keys()):
            self.existence[oid] = r_predict(self.existence[oid],
                                             self.cfg.p_s)

        dbg["post_predict_tracks"] = self._capture_tracks()

        # 3. Associate (Hungarian or oracle).
        # Feed ONLY detections with a valid ICP pose into association;
        # others go to birth as None-T_co (but that would fail birth check,
        # so they're dropped silently).
        dets_for_assoc = [d for d in dets_with_pose if d.get("_icp_ok")]
        det_idx_in_assoc = [i for i, d in enumerate(dets_with_pose)
                            if d.get("_icp_ok")]
        track_oids = list(self.object_labels.keys())
        if self.cfg.association_mode == "oracle":
            assoc = oracle_associate(track_oids, dets_for_assoc)
        else:
            assoc = hungarian_associate(
                track_oids=track_oids,
                detections=dets_for_assoc,
                innovation_fn=self.state.innovation_stats,
                track_labels=self.object_labels,
                track_tau=self.sam2_tau,
                alpha=self.cfg.alpha,
                G_out=self.cfg.G_out,
                enforce_label_match=self.cfg.enforce_label_match,
            )
        # Map assoc local indices back to dets_with_pose indices
        local_to_global = {li: gi for li, gi in enumerate(det_idx_in_assoc)}
        match_global = {oid: local_to_global[l]
                        for oid, l in assoc.match.items()}
        dbg["assoc"] = {
            "track_oids": [int(o) for o in track_oids],
            "match": {int(o): int(l) for o, l in match_global.items()},
            "unmatched_tracks": [int(o) for o in assoc.unmatched_tracks],
            "unmatched_dets_local": [int(l) for l in assoc.unmatched_detections],
            "cost_matrix": assoc.cost_matrix.tolist()
                if assoc.cost_matrix.size else [],
            "n_dets_for_assoc": len(dets_for_assoc),
            "n_dets_total": len(dets_with_pose),
        }

        # 4. Visibility.
        image_shape = rgb.shape[:2]
        if self.cfg.enable_visibility:
            p_v_map = self._compute_visibility(dets_with_pose, image_shape)
        else:
            p_v_map = {int(oid): 1.0 for oid in track_oids}

        # 5. Matched updates.
        consumed_global: set = set()
        dbg["matched"] = []
        for oid, l_local in list(assoc.match.items()):
            l_global = local_to_global[l_local]
            det = dets_with_pose[l_global]
            T_co = np.asarray(det["T_co"], dtype=np.float64)
            R_icp = np.asarray(det["R_icp"], dtype=np.float64)
            stats = self.state.innovation_stats(oid, T_co, R_icp)
            if stats is None:
                continue
            _nu, _S, d2, log_lik = stats
            w = (huber_weight(d2, self.cfg.G_in, self.cfg.G_out)
                 if self.cfg.enable_huber else 1.0)
            r_prev = self.existence.get(oid, 1.0)

            if w <= 0.0:
                # Outer-gate reject: track goes to miss branch; det to birth.
                assoc.unmatched_tracks.append(oid)
                del assoc.match[oid]
                del match_global[oid]
                dbg["matched"].append({
                    "oid": int(oid), "det_idx": int(l_global),
                    "d2": float(d2), "w": 0.0,
                    "reject_outer_gate": True,
                    "log_lik": float(log_lik),
                    "r_prev": float(r_prev), "r_new": float(r_prev),
                    "fitness": float(det.get("fitness", 0.0)),
                    "rmse": float(det.get("rmse", 0.0)),
                })
                continue

            self.state.update_observation(
                oid=oid, T_co_meas=T_co, R_icp=R_icp,
                iekf_iters=2, huber_w=w, P_max=self.cfg.P_max,
            )
            self.frames_since_obs[oid] = 0
            consumed_global.add(l_global)
            r_new = r_assoc_update_loglik(
                r_prev, log_L=log_lik,
                p_d=self.cfg.p_d, lambda_c=self.cfg.lambda_c)
            self.existence[oid] = r_new
            dbg["matched"].append({
                "oid": int(oid), "det_idx": int(l_global),
                "d2": float(d2), "w": float(w),
                "reject_outer_gate": False,
                "log_lik": float(log_lik),
                "r_prev": float(r_prev), "r_new": float(r_new),
                "fitness": float(det.get("fitness", 0.0)),
                "rmse": float(det.get("rmse", 0.0)),
            })
            tau_raw = det.get("sam2_id", det.get("id"))
            if tau_raw is not None:
                try:
                    self.sam2_tau[oid] = int(tau_raw)
                except (TypeError, ValueError):
                    pass

        # 6. Missed updates.
        dbg["missed"] = []
        for oid in assoc.unmatched_tracks:
            if oid not in self.existence:
                continue
            p_v = float(p_v_map.get(int(oid), 1.0))
            pdt = self.cfg.p_d * p_v
            r_prev = self.existence[oid]
            r_new = r_miss_update(r_prev, pdt)
            self.existence[oid] = r_new
            dbg["missed"].append({
                "oid": int(oid), "p_v": p_v, "p_d_tilde": pdt,
                "r_prev": float(r_prev), "r_new": float(r_new),
            })

        # 7. Birth from unassigned detections.
        dbg["births"] = []
        for g_idx, det in enumerate(dets_with_pose):
            if g_idx in consumed_global:
                continue
            if not det.get("_icp_ok"):
                continue
            # An assoc-unmatched detection becomes a birth candidate.
            new_oid = self._birth(det)
            if new_oid is not None:
                dbg["births"].append({
                    "det_idx": int(g_idx), "new_oid": int(new_oid),
                    "label": str(det.get("label", "unknown")),
                    "score": float(det.get("score", 0.0)),
                    "r_new": float(self.existence.get(new_oid, 0.0)),
                })

        # 8. Prune.
        dbg["pruned"] = []
        if self.cfg.r_min > 0.0:
            to_prune = [o for o, r in self.existence.items()
                        if r < self.cfg.r_min]
            for oid in to_prune:
                dbg["pruned"].append({"oid": int(oid),
                                       "r": float(self.existence[oid])})
                self._prune(oid)

        dbg["post_update_tracks"] = self._capture_tracks()
        self._frame_count += 1
        return dbg, dets_with_pose


# ─────────────────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────────────────

def _overlay_detections(rgb: np.ndarray,
                         detections: List[Dict[str, Any]],
                         alpha: float = 0.45) -> np.ndarray:
    out = rgb.copy()
    h, w = out.shape[:2]
    for det in detections:
        oid = det.get("id")
        if oid is None:
            continue
        color = _palette_color(oid)
        color_bgr = (int(color[2]), int(color[1]), int(color[0]))
        mask = det.get("mask")
        if mask is not None:
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            mb = mask.astype(bool)
            if mb.any():
                colored = np.zeros_like(out)
                colored[mb] = color
                out = np.where(mb[..., None],
                                (alpha * colored + (1 - alpha) * out).astype(np.uint8),
                                out)
        bb = det.get("box")
        if bb is not None and len(bb) == 4:
            x0, y0, x1, y1 = map(int, bb)
            cv2.rectangle(out, (x0, y0), (x1, y1), color_bgr[::-1], 2)
            tag = f"id:{oid} {det.get('label','?')} s={det.get('score',0):.2f}"
            ty = max(y0 - 4, 12)
            cv2.rectangle(out, (x0, ty - 10), (x0 + 10 + 8 * len(tag), ty + 3),
                          (255, 255, 255), -1)
            cv2.putText(out, tag, (x0 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, color_bgr[::-1], 1, cv2.LINE_AA)
    return out


def _plot_topdown(ax,
                  tracks: Dict[int, Dict[str, Any]],
                  dets_with_pose: List[Dict[str, Any]],
                  T_wb: np.ndarray,
                  xlim: Tuple[float, float],
                  ylim: Tuple[float, float],
                  title: str,
                  show_obs: bool = False) -> None:
    """Top-down scatter in the world frame.

    Each tracked object: filled circle at (x, y) + dashed uncertainty ellipse
    from the xy block of cov. Radius / ellipse scaled for visibility.

    If `show_obs` True, also draws detection centroids (from T_wb · T_co)
    as unfilled black squares, so you can see measurement vs. state.
    """
    # Camera frustum proxy: robot base location as a small triangle.
    bx, by = float(T_wb[0, 3]), float(T_wb[1, 3])
    # Heading = x-axis of the base in world.
    hx, hy = float(T_wb[0, 0]), float(T_wb[1, 0])
    theta = np.arctan2(hy, hx)

    ax.plot([bx], [by], marker="^", markersize=10,
            color="black", zorder=3)
    # Draw heading arrow.
    ax.annotate("", xy=(bx + 0.15 * np.cos(theta), by + 0.15 * np.sin(theta)),
                xytext=(bx, by),
                arrowprops=dict(arrowstyle="->", color="black", lw=1))

    # Track circles + uncertainty.
    for oid, tr in tracks.items():
        T = tr["T"]
        cov = tr["cov"]
        r_ex = tr["r"]
        x, y = float(T[0, 3]), float(T[1, 3])
        col = _palette_color_f(oid)
        # Scale circle by existence (more opaque when confident).
        alpha = 0.25 + 0.65 * max(0.0, min(1.0, r_ex))
        ax.scatter([x], [y], s=120, c=[col], alpha=alpha,
                   edgecolors="black", linewidths=0.8, zorder=4)
        # Uncertainty ellipse (3sigma in xy).
        cov_xy = cov[:2, :2]
        try:
            w_eig, v_eig = np.linalg.eigh(cov_xy)
            w_eig = np.clip(w_eig, 1e-10, None)
            width, height = 2 * 3 * np.sqrt(w_eig)
            angle = np.degrees(np.arctan2(v_eig[1, 1], v_eig[0, 1]))
            ell = mpatches.Ellipse(
                (x, y), width=float(width), height=float(height),
                angle=float(angle),
                fill=False, edgecolor=col, lw=1.0, linestyle="--",
                alpha=0.7, zorder=2,
            )
            ax.add_patch(ell)
        except Exception:
            pass
        ax.text(x + 0.015, y + 0.015,
                f"id:{oid}\nr={r_ex:.2f}",
                fontsize=6.5, color="black",
                bbox=dict(facecolor="white", alpha=0.7, pad=0.8,
                          edgecolor="none"), zorder=5)

    if show_obs:
        for det in dets_with_pose:
            if not det.get("_icp_ok"):
                continue
            T_co = det["T_co"]
            T_wo = T_wb @ T_co
            dx, dy = float(T_wo[0, 3]), float(T_wo[1, 3])
            oid = det.get("id")
            col = _palette_color_f(oid) if oid is not None else (0, 0, 0)
            ax.scatter([dx], [dy], s=60, facecolors="none",
                       edgecolors=col, marker="s", linewidths=1.4, zorder=6)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x [m]", fontsize=8)
    ax.set_ylabel("y [m]", fontsize=8)
    ax.tick_params(labelsize=7)


def _format_intermediates_text(dbg: Dict[str, Any]) -> str:
    """Compose the multi-line text block for panel 3."""
    lines: List[str] = []

    enter = dbg["enter_tracks"]
    post_p = dbg["post_predict_tracks"]
    post_u = dbg["post_update_tracks"]
    lines.append(f"tracks before -> after predict -> after update: "
                 f"{len(enter)} / {len(post_p)} / {len(post_u)}")

    # Predict deltas per track: tr(P) grows, T unchanged.
    lines.append("")
    lines.append("PREDICT  (tr P before -> after; T unchanged):")
    pred_rows = []
    for oid in sorted(enter.keys()):
        if oid not in post_p:
            continue
        trP0 = float(np.trace(enter[oid]["cov"]))
        trP1 = float(np.trace(post_p[oid]["cov"]))
        pred_rows.append(f"  id:{oid:<3d} tr(P): {trP0:.2e} -> {trP1:.2e}")
    lines.extend(pred_rows[:12] if pred_rows
                 else ["  (no tracks)"])

    # Association results.
    assoc = dbg.get("assoc", {})
    lines.append("")
    lines.append(f"ASSOC  n_tracks={len(assoc.get('track_oids', []))}  "
                 f"n_dets_icp={assoc.get('n_dets_for_assoc', 0)}  "
                 f"(total {assoc.get('n_dets_total', 0)})")
    m = assoc.get("match", {})
    lines.append(f"  matched: {len(m)}  "
                 f"unmatched_tr: {len(assoc.get('unmatched_tracks', []))}  "
                 f"unmatched_det: {len(assoc.get('unmatched_dets_local', []))}")

    # Matched pairs table.
    lines.append("")
    lines.append("MATCHED (d^2 gate = 25):")
    matched = dbg.get("matched", [])
    if matched:
        lines.append("  id   det  d^2   w    logL     r_prev -> r_new  fit/rmse")
        for m_row in matched[:12]:
            flag = "[REJ]" if m_row.get("reject_outer_gate") else "     "
            lines.append(
                f"  {m_row['oid']:<3d} {m_row['det_idx']:<3d}  "
                f"{m_row['d2']:5.2f} {m_row['w']:4.2f} "
                f"{m_row['log_lik']:7.1f}   "
                f"{m_row['r_prev']:.3f} -> {m_row['r_new']:.3f}  "
                f"{m_row['fitness']:.2f}/{m_row['rmse']*1e3:4.1f}mm {flag}"
            )
    else:
        lines.append("  (none)")

    # Missed branch.
    lines.append("")
    lines.append("MISSED (eq:r_miss):")
    missed = dbg.get("missed", [])
    if missed:
        lines.append("  id   p_v   p~_d   r_prev -> r_new")
        for m_row in missed[:10]:
            lines.append(
                f"  {m_row['oid']:<3d}  {m_row['p_v']:.2f}  "
                f"{m_row['p_d_tilde']:.2f}   "
                f"{m_row['r_prev']:.3f} -> {m_row['r_new']:.3f}"
            )
    else:
        lines.append("  (none)")

    # Births + prunes.
    births = dbg.get("births", [])
    prunes = dbg.get("pruned", [])
    lines.append("")
    lines.append(f"BIRTHS: {len(births)}    PRUNES: {len(prunes)}")
    for b in births[:8]:
        lines.append(f"  +id:{b['new_oid']} {b['label']} s={b['score']:.2f} "
                     f"r_new={b['r_new']:.3f}")
    for p in prunes[:8]:
        lines.append(f"  -id:{p['oid']} r={p['r']:.2e}")

    return "\n".join(lines)


def _plot_intermediates(ax, dbg: Dict[str, Any]) -> None:
    ax.axis("off")
    txt = _format_intermediates_text(dbg)
    ax.text(0.0, 1.0, txt, transform=ax.transAxes,
            fontsize=7.5, family="monospace",
            verticalalignment="top", horizontalalignment="left")
    ax.set_title("[3] EKF step intermediates", fontsize=10)


def _plot_r_evolution(ax,
                       r_history: Dict[int, List[Tuple[int, float]]],
                       xlim: Tuple[int, int]) -> None:
    ax.set_title("[6] existence r(t) per track", fontsize=10)
    any_plotted = False
    for oid, hist in r_history.items():
        if not hist:
            continue
        xs = [h[0] for h in hist]
        ys = [h[1] for h in hist]
        ax.plot(xs, ys, marker=".", markersize=3, linewidth=1.0,
                color=_palette_color_f(oid), label=f"id:{oid}")
        any_plotted = True
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(xlim)
    ax.set_xlabel("frame", fontsize=8)
    ax.set_ylabel("r", fontsize=8)
    ax.axhline(0.5, linestyle="--", color="gray", alpha=0.5)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)
    if any_plotted:
        # Unique handles (matplotlib legend picks duplicates up otherwise).
        handles, labels = ax.get_legend_handles_labels()
        seen_l = []
        seen_h = []
        for h, l in zip(handles, labels):
            if l not in seen_l:
                seen_l.append(l)
                seen_h.append(h)
        if seen_l:
            ax.legend(seen_h[:12], seen_l[:12], fontsize=6, ncol=2,
                      loc="lower left", framealpha=0.85)


def _compute_topdown_extent(tracks_snapshots: List[Dict[int, Dict[str, Any]]],
                             dets_with_pose: List[Dict[str, Any]],
                             T_wb: np.ndarray,
                             pad: float = 0.4) -> Tuple[Tuple[float, float],
                                                         Tuple[float, float]]:
    xs: List[float] = [float(T_wb[0, 3])]
    ys: List[float] = [float(T_wb[1, 3])]
    for snap in tracks_snapshots:
        for oid, tr in snap.items():
            xs.append(float(tr["T"][0, 3]))
            ys.append(float(tr["T"][1, 3]))
    for det in dets_with_pose:
        if det.get("_icp_ok"):
            T_wo = T_wb @ det["T_co"]
            xs.append(float(T_wo[0, 3]))
            ys.append(float(T_wo[1, 3]))
    if not xs:
        return (-1.0, 1.0), (-1.0, 1.0)
    x0, x1 = min(xs) - pad, max(xs) + pad
    y0, y1 = min(ys) - pad, max(ys) + pad
    # Enforce a minimum span so an almost-stationary camera doesn't look
    # infinitely zoomed.
    min_span = 0.6
    if x1 - x0 < min_span:
        c = 0.5 * (x0 + x1)
        x0, x1 = c - min_span / 2, c + min_span / 2
    if y1 - y0 < min_span:
        c = 0.5 * (y0 + y1)
        y0, y1 = c - min_span / 2, c + min_span / 2
    return (x0, x1), (y0, y1)


def render_frame(rgb: np.ndarray,
                 detections: List[Dict[str, Any]],
                 dbg: Dict[str, Any],
                 dets_with_pose: List[Dict[str, Any]],
                 r_history: Dict[int, List[Tuple[int, float]]],
                 frame_idx: int,
                 max_frame: int,
                 out_path: str,
                 traj: str) -> None:
    T_wb = dbg["slam_pose"]
    xlim, ylim = _compute_topdown_extent(
        [dbg["enter_tracks"], dbg["post_predict_tracks"],
         dbg["post_update_tracks"]],
        dets_with_pose, T_wb)

    fig = plt.figure(figsize=(19, 10.5), dpi=105)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.3, 1.0, 1.6],
                           hspace=0.28, wspace=0.22)

    # Panel 1: perception overlay.
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(_overlay_detections(rgb, detections))
    ax1.set_title(f"[1] Perception masks + ids  "
                  f"({len(detections)} dets)",
                  fontsize=10)
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Panel 2: top-down "entering frame".
    ax2 = fig.add_subplot(gs[0, 1])
    _plot_topdown(ax2, dbg["enter_tracks"], dets_with_pose, T_wb,
                  xlim, ylim, "[2] State entering frame",
                  show_obs=False)

    # Panel 3: EKF intermediates.
    ax3 = fig.add_subplot(gs[0, 2])
    _plot_intermediates(ax3, dbg)

    # Panel 4: top-down after predict.
    ax4 = fig.add_subplot(gs[1, 0])
    _plot_topdown(ax4, dbg["post_predict_tracks"], dets_with_pose, T_wb,
                  xlim, ylim,
                  "[4] After EKF predict (mean fixed, cov inflated)",
                  show_obs=False)

    # Panel 5: top-down after update.
    ax5 = fig.add_subplot(gs[1, 1])
    _plot_topdown(ax5, dbg["post_update_tracks"], dets_with_pose, T_wb,
                  xlim, ylim,
                  "[5] After observation update",
                  show_obs=True)

    # Panel 6: r(t) evolution.
    ax6 = fig.add_subplot(gs[1, 2])
    _plot_r_evolution(ax6, r_history, xlim=(0, max_frame))

    fig.suptitle(
        f"EKF tracking   traj={traj}   frame={frame_idx:04d}   "
        f"base=({T_wb[0,3]:.2f},{T_wb[1,3]:.2f})   "
        f"dets={len(detections)}",
        fontsize=12,
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
# Main driver
# ─────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory", default="apple_in_the_tray")
    ap.add_argument("--max-frame", type=int, default=700,
                    help="exclusive upper bound on frame index")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--pose-method", default="icp_chain",
                    choices=("centroid", "icp_chain", "icp_anchor",
                             "icp_chain_strict", "icp_anchor_strict"))
    ap.add_argument("--out-subdir", default="ekf_debug")
    ap.add_argument("--no-visibility", action="store_true")
    args = ap.parse_args()

    traj = args.trajectory
    ds_root = os.path.join(DATASET_DIR, traj)
    viz_root = os.path.join(VIZ_BASE, traj)
    rgb_dir = os.path.join(ds_root, "rgb")
    depth_dir = os.path.join(ds_root, "depth")
    det_dir = os.path.join(viz_root, "perception", "detection_h")
    pose_path = os.path.join(ds_root, "pose_txt", "amcl_pose.txt")
    out_dir = os.path.join(viz_root, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    # ── load poses ──
    slam_poses = _load_amcl_poses(pose_path)
    if not slam_poses:
        raise SystemExit(f"no AMCL poses loaded from {pose_path}")

    # ── tracker setup ──
    cfg = BernoulliConfig(
        association_mode="hungarian",
        p_s=1.0,
        p_d=0.9,
        alpha=4.4,
        lambda_c=1.0,
        lambda_b=1.0,
        r_conf=0.5,
        r_min=1e-3,
        G_in=12.59,
        G_out=25.0,
        P_max=np.diag([0.25**2] * 3 + [(np.pi / 4) ** 2] * 3),
        enable_visibility=(not args.no_visibility),
        enable_huber=True,
        init_cov_from_R=True,
        enforce_label_match=True,
        K=K_DEFAULT,
        image_shape=(480, 640),
    )
    tracker = InstrumentedTracker(K_DEFAULT, cfg, pose_method=args.pose_method)

    # ── per-track r(t) history ──
    r_history: Dict[int, List[Tuple[int, float]]] = {}

    max_frame = min(args.max_frame, len(slam_poses))
    frames_processed = 0
    frames_written = 0
    for idx in range(args.start, max_frame):
        if (idx - args.start) % args.step != 0:
            continue

        rgb = _load_rgb(os.path.join(rgb_dir, f"rgb_{idx:06d}.png"))
        depth = _load_depth(os.path.join(depth_dir, f"depth_{idx:06d}.npy"))
        if rgb is None or depth is None:
            continue

        detections = _load_detection_json(
            os.path.join(det_dir, f"detection_{idx:06d}_final.json"))

        T_wb = slam_poses[idx]
        dbg, dets_with_pose = tracker.step(
            rgb=rgb, depth=depth, T_wb=T_wb,
            detections=detections, phase="idle",
        )
        frames_processed += 1

        for oid, tr in dbg["post_update_tracks"].items():
            r_history.setdefault(oid, []).append((idx, float(tr["r"])))

        out_path = os.path.join(out_dir, f"frame_{idx:06d}.png")
        try:
            render_frame(
                rgb=rgb, detections=detections, dbg=dbg,
                dets_with_pose=dets_with_pose,
                r_history=r_history,
                frame_idx=idx,
                max_frame=max_frame,
                out_path=out_path,
                traj=traj,
            )
            frames_written += 1
        except Exception as e:
            print(f"[WARN] render failed at frame {idx}: {e}")

        if frames_processed % 20 == 0:
            print(f"[{traj}] frame {idx}: processed {frames_processed}, "
                  f"written {frames_written}, tracks={len(tracker.object_labels)}")

    print(f"[done] wrote {frames_written} frames under {out_dir}")


if __name__ == "__main__":
    main()
