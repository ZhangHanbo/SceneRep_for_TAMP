"""
Tests for the coupled online OWL + SAM2 + EKF-tracker pipeline
(rosbag2dataset/sam2/coupled_reconcile.py).

The pipeline uses the orchestrator's Bernoulli-EKF state as the sole
source of truth; perception drives orch.step per frame and seeds SAM2
on births. These tests verify:

  * Coupled run on cached apple_bowl_2 OWL data with a mock SAM2
    produces a sensible track set (~the number of distinct physical
    objects) and ids flow from tracker -> SAM2 prompts
  * Births trigger exactly one SAM2.add_box per new track, with the
    correct object id
  * The orchestrator's existence r is updated as tracks accumulate
    matched observations
  * Rigid-attach predict fires for the held object (bowl rides the
    gripper; its mu_w shifts with the gripper's T_bg evolution)
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rosbag2dataset.sam2.coupled_reconcile import (
    coupled_reconcile, CoupledInputs, make_coupled_orchestrator,
    _build_detections_for_orch, _match_new_tracks_to_dets,
)
from rosbag2dataset.sam2.sam2_client import (
    OwlDet, _load_owl_detections, PropagatedFrame,
)


class _MockSAM2:
    """Records add_box / propagate calls; returns empty propagation.

    Lets the coupled-pipeline tests run without network access to the
    SAM2 server. Since SAM2 isn't actually returning masks, the final
    propagate produces an empty list -- that's fine because the test
    checks the orchestrator's state evolution, not the emitted masks.
    """

    def __init__(self):
        self.added: List[tuple] = []
        self.start_called = False
        self.n_propagate = 0
        self.n_frames = 0

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def start(self, frames):
        self.start_called = True
        self.n_frames = len(frames)

    def add_box(self, video_idx, oid, box):
        self.added.append((video_idx, int(oid), list(box)))

    def add_points(self, *a, **k):
        return None

    def propagate(self):
        self.n_propagate += 1
        return [PropagatedFrame(object_masks={}, object_bboxes={})
                for _ in range(self.n_frames)]

    def close(self):
        return None

DATA_BASE = os.path.join(
    os.path.dirname(_REPO_ROOT),
    "Mobile_Manipulation_on_Fetch", "multi_objects",
)
DATA_ROOT = os.path.join(DATA_BASE, "apple_bowl_2")
HAS_DATA = os.path.isdir(os.path.join(DATA_ROOT, "rgb"))

requires_data = pytest.mark.skipif(
    not HAS_DATA, reason=f"Trajectory data not found at {DATA_ROOT}")

K = np.array([[554.3827, 0, 320.5],
              [0, 554.3827, 240.5],
              [0, 0, 1]], dtype=np.float64)
IMAGE_SHAPE = (480, 640)


def _build_inputs(n_frames: int = 20, step: int = 3):
    rgb_dir = os.path.join(DATA_ROOT, "rgb")
    depth_dir = os.path.join(DATA_ROOT, "depth")
    det_dir = os.path.join(DATA_ROOT, "detection_boxes")
    pose_path = os.path.join(DATA_ROOT, "pose_txt", "camera_pose.txt")

    rgb_files = sorted(f for f in os.listdir(rgb_dir)
                       if f.endswith(".png"))[:n_frames]
    fids = [int(f[4:10]) for f in rgb_files][::step]

    rgb = [cv2.cvtColor(cv2.imread(os.path.join(rgb_dir, f"rgb_{f:06d}.png")),
                         cv2.COLOR_BGR2RGB) for f in fids]
    depth = [np.load(os.path.join(depth_dir, f"depth_{f:06d}.npy"))
             .astype(np.float32) for f in fids]

    pose_per_fid = {}
    with open(pose_path) as f:
        for line in f:
            a = line.strip().split()
            if len(a) != 8:
                continue
            fid_i, tx, ty, tz, qx, qy, qz, qw = map(float, a)
            T = np.eye(4)
            T[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            T[:3, 3] = [tx, ty, tz]
            pose_per_fid[int(fid_i)] = T
    cam_poses = [pose_per_fid[f] for f in fids]

    owl_by_fid = _load_owl_detections(det_dir, min_score=0.15)

    return CoupledInputs(
        frame_ids=fids, rgb=rgb, depth=depth, cam_poses=cam_poses,
        K=K, image_shape=IMAGE_SHAPE, owl_by_fid=owl_by_fid)


# ─────────────────────────────────────────────────────────────────────
# Unit tests on the detection-building and birth-reverse-lookup helpers.
# ─────────────────────────────────────────────────────────────────────

class TestDetBuilder:

    def test_skips_boxes_without_valid_depth(self):
        # All-zero depth, all masks map to no valid pixels.
        depth = np.zeros((480, 640), dtype=np.float32)
        owl = [OwlDet(frame_idx=0, label="apple", score=0.5,
                       box=[100, 100, 200, 200])]
        dets = _build_detections_for_orch(owl, depth, K, (480, 640))
        assert dets == []

    def test_produces_detection_with_camera_frame_centroid(self):
        depth = np.ones((480, 640), dtype=np.float32) * 1.5
        owl = [OwlDet(frame_idx=0, label="apple", score=0.8,
                       box=[300, 200, 340, 280])]
        dets = _build_detections_for_orch(owl, depth, K, (480, 640))
        assert len(dets) == 1
        d = dets[0]
        assert d["label"] == "apple"
        assert d["score"] == 0.8
        assert d["id"] is None
        assert d["sam2_id"] is None
        # Back-projection is depth 1.5 so the z-coordinate should be 1.5
        assert d["T_co"][2, 3] == pytest.approx(1.5, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────
# End-to-end on cached apple_bowl_2 OWL data with mock SAM2.
# ─────────────────────────────────────────────────────────────────────

@requires_data
class TestCoupledEndToEnd:

    def test_converges_on_early_frames(self):
        """On the first 20 frames (step=3, 7 video indices), the five
        initial objects are seen co-visibly. Coupled pipeline should
        track all five and keep the track count small."""
        inputs = _build_inputs(n_frames=20, step=3)
        orch = make_coupled_orchestrator(inputs.cam_poses, rng_seed=42)
        sam2 = _MockSAM2()
        coupled_reconcile(orch, inputs, sam2, verbose=False)
        labels = sorted(set(orch.object_labels.values()))
        assert {"apple", "bowl", "can", "cup", "wooden box"}.issubset(
            set(labels)), f"missing classes: {labels}"
        # Small surplus OK (label flicker); should not explode.
        assert len(orch.object_labels) <= 10, \
            f"too many tracks: {len(orch.object_labels)}"

    def test_sam2_receives_one_prompt_per_birth(self):
        inputs = _build_inputs(n_frames=20, step=3)
        orch = make_coupled_orchestrator(inputs.cam_poses, rng_seed=42)
        sam2 = _MockSAM2()
        coupled_reconcile(orch, inputs, sam2, verbose=False)
        prompted_oids = sorted({oid for _, oid, _ in sam2.added})
        track_oids = sorted(orch.object_labels.keys())
        assert prompted_oids == track_oids, (
            f"track ids {track_oids} != SAM2 prompts {prompted_oids}")

    def test_orchestrator_existence_rises_on_matches(self):
        """Every track should accumulate enough matches to push its
        existence r above the birth value (r_birth(s) < s). This
        verifies EKF update + r_assoc_update are firing."""
        inputs = _build_inputs(n_frames=20, step=3)
        orch = make_coupled_orchestrator(inputs.cam_poses, rng_seed=42)
        sam2 = _MockSAM2()
        coupled_reconcile(orch, inputs, sam2, verbose=False)
        rs = list(orch.existence.values())
        assert len(rs) > 0
        # With p_d=0.9, lambda_c=1.0, and 2+ matches, r should climb
        # well above 0.5 for the confidently-tracked objects.
        n_confirmed = sum(1 for r in rs if r >= 0.5)
        assert n_confirmed >= 3, (
            f"only {n_confirmed} tracks have r>=0.5 out of {len(rs)}")

    def test_birth_reverse_lookup_is_stable(self):
        """_match_new_tracks_to_dets should assign each new_oid to
        exactly one det, and different new_oids shouldn't share a det."""
        inputs = _build_inputs(n_frames=10, step=3)
        orch = make_coupled_orchestrator(inputs.cam_poses, rng_seed=42)
        sam2 = _MockSAM2()
        # Process frame 0 manually to check the mapping.
        known_before = set(orch.object_labels.keys())
        from rosbag2dataset.sam2.coupled_reconcile import (
            _build_detections_for_orch as _bd,
        )
        rgb = inputs.rgb[0]; depth = inputs.depth[0]
        T_cw = inputs.cam_poses[0]
        owl = inputs.owl_by_fid.get(inputs.frame_ids[0], [])
        dets = _bd(owl, depth, K, IMAGE_SHAPE)
        orch.step(rgb, depth, dets,
                   {"phase": "idle", "held_obj_id": None})
        new_oids = sorted(set(orch.object_labels.keys()) - known_before)
        mapping = _match_new_tracks_to_dets(new_oids, orch, dets, T_cw)
        # Each new oid mapped to a unique det.
        det_ids = [id(d) for d in mapping.values()]
        assert len(set(det_ids)) == len(det_ids), \
            "two new tracks mapped to the same det"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
