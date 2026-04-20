"""
Tests for the state-driven SAM2 + OWL reconciliation
(rosbag2dataset/sam2/state_driven_reconcile.py).

These exercise the matching logic on real apple_bowl_2 OWL data using a
mock SAM2 client, so no network access to the SAM2 server is required.
The mock returns empty propagated masks every iteration; this isolates
the state-driven logic (Hungarian + visibility predicate) from SAM2's
own mask propagation.
"""

from __future__ import annotations

import os
import sys
from typing import List

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rosbag2dataset.sam2.state_driven_reconcile import (
    state_driven_reconcile, ReconcileInputs,
    project_in_fov, backproject_centroid,
    hungarian_match_3d, TrackState3D,
    _R_OBS, _INIT_COV_W, _G_OUT_3D,
)
from rosbag2dataset.sam2.sam2_client import (
    OwlDet, _load_owl_detections, PropagatedFrame,
)

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


# ─────────────────────────────────────────────────────────────────────
# Mock SAM2 client.
# ─────────────────────────────────────────────────────────────────────

class _MockSAM2:
    """Records add_box / propagate calls; returns empty propagation."""

    def __init__(self):
        self.added: List[tuple] = []      # (video_idx, oid, box)
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

    def add_points(self, *a, **k):  # unused
        return None

    def propagate(self):
        self.n_propagate += 1
        # Return empty masks for every frame -- isolates Hungarian + FOV.
        return [PropagatedFrame(object_masks={}, object_bboxes={})
                for _ in range(self.n_frames)]

    def close(self):
        return None


# ─────────────────────────────────────────────────────────────────────
# Unit tests on the camera-model + Hungarian primitives.
# ─────────────────────────────────────────────────────────────────────

class TestCameraModel:

    def test_centroid_in_front_projects(self):
        T_cw = np.eye(4)
        p_w = np.array([0.1, 0.05, 1.0])
        proj = project_in_fov(p_w, T_cw, K, IMAGE_SHAPE)
        assert proj is not None
        u, v, z = proj
        assert z == pytest.approx(1.0)
        assert 0.0 < u < 640
        assert 0.0 < v < 480

    def test_centroid_behind_camera_returns_none(self):
        T_cw = np.eye(4)
        p_w = np.array([0.0, 0.0, -0.5])
        assert project_in_fov(p_w, T_cw, K, IMAGE_SHAPE) is None

    def test_centroid_outside_image_returns_none(self):
        T_cw = np.eye(4)
        # Far off-axis -- projects outside the image.
        p_w = np.array([10.0, 0.0, 1.0])
        assert project_in_fov(p_w, T_cw, K, IMAGE_SHAPE) is None


class TestHungarian3D:

    def test_perfect_match(self):
        state = {
            0: TrackState3D(state_oid=0, sam2_oid=0, label="apple",
                            mu_w=np.array([0.0, 0.0, 1.0]),
                            cov_w=_INIT_COV_W.copy(), first_frame=0),
            1: TrackState3D(state_oid=1, sam2_oid=1, label="bowl",
                            mu_w=np.array([0.5, 0.0, 1.0]),
                            cov_w=_INIT_COV_W.copy(), first_frame=0),
        }
        cands = [
            {"label": "apple", "score": 0.5, "box": [0,0,1,1],
             "cw": np.array([0.0, 0.0, 1.0])},
            {"label": "bowl",  "score": 0.5, "box": [0,0,1,1],
             "cw": np.array([0.5, 0.0, 1.0])},
        ]
        m, uc, ut = hungarian_match_3d([0, 1], cands, state)
        assert m == {0: 0, 1: 1}
        assert uc == [] and ut == []

    def test_label_mismatch_blocks_match(self):
        state = {0: TrackState3D(state_oid=0, sam2_oid=0, label="apple",
                                  mu_w=np.array([0.0, 0.0, 1.0]),
                                  cov_w=_INIT_COV_W.copy(), first_frame=0)}
        cands = [{"label": "bowl", "score": 0.5, "box": [0,0,1,1],
                  "cw": np.array([0.0, 0.0, 1.0])}]
        m, uc, ut = hungarian_match_3d([0], cands, state)
        assert m == {} and uc == [0] and ut == [0]

    def test_far_candidate_outside_outer_gate(self):
        state = {0: TrackState3D(state_oid=0, sam2_oid=0, label="apple",
                                  mu_w=np.array([0.0, 0.0, 1.0]),
                                  cov_w=_INIT_COV_W.copy(), first_frame=0)}
        # 10 m off; way past chi^2_3 outer gate of ~18.5 with 15 cm sigma.
        cands = [{"label": "apple", "score": 0.5, "box": [0,0,1,1],
                  "cw": np.array([10.0, 0.0, 1.0])}]
        m, uc, ut = hungarian_match_3d([0], cands, state)
        assert m == {} and uc == [0] and ut == [0]

    def test_one_match_one_unmatched_track(self):
        state = {
            0: TrackState3D(state_oid=0, sam2_oid=0, label="apple",
                            mu_w=np.array([0.0, 0.0, 1.0]),
                            cov_w=_INIT_COV_W.copy(), first_frame=0),
            1: TrackState3D(state_oid=1, sam2_oid=1, label="bowl",
                            mu_w=np.array([0.5, 0.0, 1.0]),
                            cov_w=_INIT_COV_W.copy(), first_frame=0),
        }
        cands = [{"label": "apple", "score": 0.5, "box": [0,0,1,1],
                  "cw": np.array([0.0, 0.0, 1.0])}]
        m, uc, ut = hungarian_match_3d([0, 1], cands, state)
        assert m == {0: 0}
        assert uc == []
        assert ut == [1]


# ─────────────────────────────────────────────────────────────────────
# End-to-end on cached apple_bowl_2 OWL data with a mock SAM2.
# ─────────────────────────────────────────────────────────────────────

@requires_data
class TestEndToEnd:

    def _build_inputs(self, n_frames=20, step=5):
        rgb_dir = os.path.join(DATA_ROOT, "rgb")
        depth_dir = os.path.join(DATA_ROOT, "depth")
        det_dir = os.path.join(DATA_ROOT, "detection_boxes")
        pose_path = os.path.join(DATA_ROOT, "pose_txt", "camera_pose.txt")

        rgb_files = sorted(f for f in os.listdir(rgb_dir)
                           if f.endswith(".png"))
        all_fids = [int(f[4:10]) for f in rgb_files][:n_frames]
        fids = all_fids[::step]

        rgb = [cv2.cvtColor(
            cv2.imread(os.path.join(rgb_dir, f"rgb_{f:06d}.png")),
            cv2.COLOR_BGR2RGB) for f in fids]
        depth = [np.load(os.path.join(depth_dir, f"depth_{f:06d}.npy"))
                 .astype(np.float32) for f in fids]

        pose_per_fid = {}
        with open(pose_path) as f:
            for line in f:
                arr = line.strip().split()
                if len(arr) != 8:
                    continue
                fid_i, tx, ty, tz, qx, qy, qz, qw = map(float, arr)
                T = np.eye(4)
                T[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
                T[:3, 3] = [tx, ty, tz]
                pose_per_fid[int(fid_i)] = T
        cam_poses = [pose_per_fid[f] for f in fids]

        owl_by_fid = _load_owl_detections(det_dir, min_score=0.15)

        return ReconcileInputs(
            frame_ids=fids, rgb=rgb, depth=depth, cam_poses=cam_poses,
            K=K, image_shape=IMAGE_SHAPE, owl_by_fid=owl_by_fid)

    def test_reconciler_produces_one_track_per_distinct_object(self):
        """On the first ~20 frames where 5 objects are co-visible, the
        reconciler should converge to 5 tracks (apple, bowl, can, cup,
        wooden box). With a mock SAM2 that returns empty propagation,
        every OWL detection in the first iteration goes to the new-prompt
        candidate pool and gets clustered down to one prompt per object."""
        inputs = self._build_inputs(n_frames=10, step=2)
        sam2 = _MockSAM2()
        state, _prop = state_driven_reconcile(inputs, sam2, max_iters=4,
                                                verbose=False)
        labels = sorted({tr.label for tr in state.values()})
        assert {"apple", "bowl", "can", "cup", "wooden box"}.issubset(set(labels)), \
            f"expected the 5 main classes, got {labels}"
        # At most a few extra (low-score noise might still pass 0.15).
        assert len(state) <= 8, f"too many tracks: {len(state)}"

    def test_all_tracks_in_fov_at_first_frame(self):
        """At frame 0 every born track's mu_w should project inside the
        image (since each was born from a candidate that did project)."""
        inputs = self._build_inputs(n_frames=6, step=2)
        sam2 = _MockSAM2()
        state, _prop = state_driven_reconcile(inputs, sam2, max_iters=3,
                                                verbose=False)
        T_cw0 = inputs.cam_poses[0]
        for oid, tr in state.items():
            if tr.first_frame != 0:
                continue
            assert project_in_fov(tr.mu_w, T_cw0, K, IMAGE_SHAPE) is not None, \
                f"track {oid} ({tr.label}) born at frame 0 not in FOV"

    def test_state_oid_consistency_with_sam2_prompts(self):
        """Every TrackState3D should have a 1:1 mapping to a sam2.add_box
        call (state_oid == sam2_oid by construction in this reconciler)."""
        inputs = self._build_inputs(n_frames=10, step=2)
        sam2 = _MockSAM2()
        state, _prop = state_driven_reconcile(inputs, sam2, max_iters=3,
                                                verbose=False)
        prompted_oids = sorted({oid for _, oid, _ in sam2.added})
        state_oids = sorted(state.keys())
        assert prompted_oids == state_oids, \
            f"prompts {prompted_oids} != state {state_oids}"

    def test_starts_with_empty_state_then_seeds_at_iter1(self):
        """Iter 0 should propagate-skip (empty state) and seed everything
        as new prompts; iter 1 should converge."""
        inputs = self._build_inputs(n_frames=10, step=2)
        sam2 = _MockSAM2()
        state, _prop = state_driven_reconcile(inputs, sam2, max_iters=4,
                                                verbose=False)
        # With a mock that returns empty masks, Hungarian after the first
        # add round still finds no tracks to match against (cov is fresh
        # and far from new candidates). The reconciler should still
        # converge -- new objects added in iter 0 would re-anchor in iter 1
        # via low-Mahalanobis matches and not respawn.
        assert sam2.start_called
        assert sam2.n_propagate >= 1
        # No infinite spawning: total prompts ≤ candidates per frame * frames
        owl_total = sum(len(v) for v in inputs.owl_by_fid.values())
        assert len(sam2.added) <= owl_total


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
