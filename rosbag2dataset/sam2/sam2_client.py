#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SAM2 video-predictor client — replaces the per-frame SAM + Hungarian tracker.

Pipeline position:

    owl_client.py                   →  detection_boxes/*.json   (boxes)
    THIS FILE (sam2_client.py)      →  detection_h/*_final.json (masks + stable object_ids)

Algorithm
─────────
1. Upload every RGB frame of the dataset to the SAM2 server in one call
   (/sam2_start_session); this materialises the video in GPU memory and
   returns a session_id.
2. Seed the tracker: every OWL detection in frame 0 becomes an initial
   prompt via /sam2_add_box with a fresh ``object_id``.
3. /sam2_propagate — get SAM2's masks for every frame, given the prompts.
4. Walk forward through the video. At each frame, compare OWL's boxes
   against SAM2's current propagated masks. A box whose IoU against every
   same-class mask is below ``new_obj_iou`` is treated as a newly-appeared
   object: it's added as a new prompt (fresh object_id, current frame)
   and we'll re-propagate on the next round.
5. Repeat (3)–(4) until no new prompts fire, or ``max_iters`` is hit.
6. Write ``detection_h/detection_NNNNNN_final.json`` in the existing
   schema so the visualiser and downstream consumers need no changes.
7. /sam2_close_session — free the GPU memory.

Why OWL + SAM2 together?
    OWL tells us *which classes* exist at every frame (and detects new
    instances when they enter the scene). SAM2 supplies *temporally
    consistent masks with stable identity*. This pairing fixes the
    fragmentation we saw with per-frame SAM + 2-D Hungarian.
"""

from __future__ import annotations

import argparse
import base64
import glob
import io
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
from PIL import Image

# Allow running as `python rosbag2dataset/sam2/sam2_client.py`
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rosbag2dataset.server_configs import (  # noqa: E402
    SAM2_SERVER_URL,
    SAM2_START_PATH, SAM2_ADD_BOX_PATH,
    SAM2_PROPAGATE_PATH, SAM2_CLOSE_PATH,
)


# ---------------------------------------------------------------------------
# HTTP helpers (with retry — SAM2 calls can take >30s)
# ---------------------------------------------------------------------------

def _post(url: str, payload: dict, timeout: float) -> dict:
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _encode_png(img_rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(img_rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _decode_mask(b64: Optional[str]) -> Optional[np.ndarray]:
    if not b64:
        return None
    raw = base64.b64decode(b64, validate=True)
    arr = np.array(Image.open(io.BytesIO(raw)))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return (arr > 127).astype(bool)


def _mask_to_png_b64(mask: np.ndarray) -> str:
    arr = (mask.astype(np.uint8) * 255)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _bbox_from_mask(mask: np.ndarray) -> List[int]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def _bbox_mask_iou(box_xyxy: List[int],
                   mask: np.ndarray) -> float:
    """IoU of a bbox region against a binary mask."""
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    h, w = mask.shape
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    box_area = (x2 - x1) * (y2 - y1)
    mask_area = int(mask.sum())
    if box_area == 0 or mask_area == 0:
        return 0.0
    inter = int(mask[y1:y2, x1:x2].sum())
    union = box_area + mask_area - inter
    return inter / union if union > 0 else 0.0


def _bbox_iou(a: List[int], b: List[int]) -> float:
    ax1, ay1, ax2, ay2 = [int(v) for v in a]
    bx1, by1, bx2, by2 = [int(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# New-prompt clustering
# ---------------------------------------------------------------------------

def _cluster_new_prompts(new_prompts, iou_thresh: float = 0.3):
    """Collapse a list of ``(video_idx, OwlDet)`` into one entry per
    plausible distinct object.

    Two entries belong to the same cluster iff:
      * same label, and
      * bbox IoU with the cluster's rolling "last seen" box ≥ iou_thresh.

    The list is processed in (label, video_idx) order so that a detection
    smoothly moving across consecutive frames ends up in a single cluster.
    Representative entry = highest-score frame in the cluster — gives the
    cleanest prompt to seed SAM2 with.
    """
    if not new_prompts:
        return []
    from collections import defaultdict
    by_label = defaultdict(list)
    for video_idx, owl in new_prompts:
        by_label[owl.label].append((video_idx, owl))

    out = []
    for label, items in by_label.items():
        items.sort(key=lambda p: p[0])
        clusters = []           # list of list of (video_idx, owl)
        for video_idx, owl in items:
            placed = False
            for cl in clusters:
                _, last_owl = cl[-1]
                if _bbox_iou(last_owl.box, owl.box) >= iou_thresh:
                    cl.append((video_idx, owl))
                    placed = True
                    break
            if not placed:
                clusters.append([(video_idx, owl)])
        for cl in clusters:
            video_idx, rep = max(cl, key=lambda p: p[1].score)
            out.append((video_idx, rep))
    return out


# ---------------------------------------------------------------------------
# Dataset IO
# ---------------------------------------------------------------------------

@dataclass
class OwlDet:
    frame_idx: int
    label: str
    score: float
    box: List[int]            # [x1, y1, x2, y2] pixel coords


def _load_owl_detections(det_dir: str,
                         min_score: float = 0.0) -> Dict[int, List[OwlDet]]:
    """Return {frame_idx: [OwlDet, ...]} from the OWL JSONs."""
    files = sorted(f for f in os.listdir(det_dir)
                   if f.startswith("detection_") and f.endswith(".json")
                   and not f.endswith("_final.json"))
    out: Dict[int, List[OwlDet]] = {}
    for fn in files:
        stem = fn[len("detection_"):-len(".json")]
        try:
            fid = int(stem)
        except ValueError:
            continue
        with open(os.path.join(det_dir, fn), "r") as f:
            data = json.load(f)
        dets: List[OwlDet] = []
        for d in data.get("detections", []):
            if "detection" in d and d["detection"]:
                top = max(d["detection"], key=lambda x: x.get("score", 0.0))
                label = str(top.get("label", "unknown"))
                score = float(top.get("score", 0.0))
            else:
                label = str(d.get("label", "unknown"))
                score = float(d.get("score", 0.0))
            box = d.get("box")
            if not box or len(box) != 4:
                continue
            if score < min_score:
                continue
            dets.append(OwlDet(frame_idx=fid, label=label, score=score,
                               box=[int(v) for v in box]))
        out[fid] = dets
    return out


def _load_frames(rgb_dir: str) -> Tuple[List[int], List[np.ndarray]]:
    files = sorted(f for f in os.listdir(rgb_dir) if f.endswith(".png"))
    frame_ids: List[int] = []
    frames: List[np.ndarray] = []
    for fn in files:
        try:
            fid = int(os.path.splitext(fn)[0].split("_")[-1])
        except ValueError:
            continue
        img = Image.open(os.path.join(rgb_dir, fn)).convert("RGB")
        frame_ids.append(fid)
        frames.append(np.asarray(img, dtype=np.uint8))
    return frame_ids, frames


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

@dataclass
class TrackState:
    """Client-side metadata parallel to SAM2's internal per-object state."""
    label: str
    first_frame: int
    score: float
    scores_by_frame: Dict[int, float] = field(default_factory=dict)


@dataclass
class PropagatedFrame:
    """Masks that SAM2 returns for one frame."""
    object_masks: Dict[int, np.ndarray]   # object_id -> (H, W) bool
    object_bboxes: Dict[int, List[int]]


class SAM2Client:
    def __init__(self, server_url: str = SAM2_SERVER_URL,
                 timeout_start: float = 300.0,
                 timeout_prompt: float = 60.0,
                 timeout_propagate: float = 1800.0):
        self.server_url = server_url.rstrip("/")
        self.t_start = timeout_start
        self.t_prompt = timeout_prompt
        self.t_prop = timeout_propagate
        self.session_id: Optional[str] = None

    # -- session lifecycle --------------------------------------------------

    def start(self, frames: List[np.ndarray]) -> dict:
        frames_b64 = [_encode_png(f) for f in frames]
        t0 = time.time()
        r = _post(self.server_url + SAM2_START_PATH,
                  {"frames": frames_b64}, timeout=self.t_start)
        self.session_id = r["session_id"]
        print(f"[sam2] session {self.session_id}  "
              f"{r['n_frames']} frames  {r['width']}x{r['height']}  "
              f"upload={time.time()-t0:.1f}s")
        return r

    def close(self) -> None:
        if self.session_id is None:
            return
        try:
            _post(self.server_url + SAM2_CLOSE_PATH,
                  {"session_id": self.session_id}, timeout=30.0)
        except Exception as e:
            print(f"[sam2] close failed: {e}")
        self.session_id = None

    def __enter__(self) -> "SAM2Client":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- prompts / propagate ------------------------------------------------

    def add_box(self, frame_idx: int, object_id: int,
                box: List[int]) -> np.ndarray:
        assert self.session_id is not None, "session not started"
        r = _post(self.server_url + SAM2_ADD_BOX_PATH, {
            "session_id": self.session_id,
            "frame_idx":  int(frame_idx),
            "object_id":  int(object_id),
            "box":        [float(v) for v in box],
        }, timeout=self.t_prompt)
        return _decode_mask(r.get("mask"))

    def propagate(self) -> List[PropagatedFrame]:
        assert self.session_id is not None, "session not started"
        r = _post(self.server_url + SAM2_PROPAGATE_PATH,
                  {"session_id": self.session_id}, timeout=self.t_prop)
        out: Dict[int, PropagatedFrame] = {}
        for frame in r.get("results", []):
            masks: Dict[int, np.ndarray] = {}
            bboxes: Dict[int, List[int]] = {}
            for o in frame.get("objects", []):
                oid = int(o["object_id"])
                m = _decode_mask(o.get("mask"))
                if m is None:
                    continue
                masks[oid] = m
                bboxes[oid] = o.get("bbox") or _bbox_from_mask(m)
            out[int(frame["frame_idx"])] = PropagatedFrame(
                object_masks=masks, object_bboxes=bboxes)
        # Return as list sorted by frame_idx
        return [out[f] for f in sorted(out.keys())]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def track_dataset(dataset_root: str,
                  server_url: str = SAM2_SERVER_URL,
                  det_dir_name: str = "detection_boxes",
                  out_dir_name: str = "detection_h",
                  new_obj_iou: float = 0.3,
                  max_iters: int = 4,
                  min_score: float = 0.0) -> None:

    rgb_dir = os.path.join(dataset_root, "rgb")
    det_dir = os.path.join(dataset_root, det_dir_name)
    out_dir = os.path.join(dataset_root, out_dir_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[sam2-client] dataset={dataset_root}  server={server_url}")

    frame_ids, frames = _load_frames(rgb_dir)
    if not frames:
        raise FileNotFoundError(f"no RGB in {rgb_dir}")
    owl_dets = _load_owl_detections(det_dir, min_score=min_score)
    print(f"[sam2-client] frames={len(frames)}  "
          f"owl-dets-per-frame (mean)={np.mean([len(v) for v in owl_dets.values()] or [0]):.1f}")

    # Map dataset frame_id → video index (SAM2 uses contiguous 0..N-1).
    fid_to_idx = {fid: i for i, fid in enumerate(frame_ids)}
    idx_to_fid = {i: fid for fid, i in fid_to_idx.items()}

    tracks: Dict[int, TrackState] = {}
    next_id = 0

    def _seed_prompt(sam2, video_idx: int, owl: OwlDet) -> int:
        nonlocal next_id
        obj_id = next_id; next_id += 1
        tracks[obj_id] = TrackState(
            label=owl.label, first_frame=video_idx, score=owl.score)
        tracks[obj_id].scores_by_frame[video_idx] = owl.score
        sam2.add_box(video_idx, obj_id, owl.box)
        return obj_id

    with SAM2Client(server_url=server_url) as sam2:
        sam2.start(frames)

        # No explicit "seed from frame 0" step. The reconciliation loop
        # below handles initialization itself: in iter 0 with no prompts
        # yet, ``prop_by_idx`` is empty, every OWL detection across the
        # whole video is "unmatched", and the cluster + add-prompt pass
        # picks one high-score representative per (label, IoU-cluster)
        # to seed SAM2. This covers the case where frame 0 is empty and
        # there's no special meaning to "first frame".
        prop_by_idx: List[PropagatedFrame] = []

        for it in range(max_iters):
            # Skip propagate when state is empty (no prompts yet) — SAM2
            # legitimately returns no masks, and the server's empty-state
            # path also returns []. Treat as the same "no propagated
            # masks anywhere" state for the reconciliation pass.
            if tracks:
                t0 = time.time()
                prop_by_idx = sam2.propagate()
                print(f"[sam2-client] iter={it}  propagate took "
                      f"{time.time()-t0:.1f}s  "
                      f"frames_with_masks={len(prop_by_idx)}")
            else:
                prop_by_idx = []
                print(f"[sam2-client] iter={it}  no prompts yet — skipping "
                      f"propagate; treating all OWL detections as unmatched")

            # Walk every frame (not just frames SAM2 returned) so the loop
            # works equally well with an empty propagated state.
            new_prompts: List[Tuple[int, OwlDet]] = []
            for i, fid in enumerate(frame_ids):
                prop = (prop_by_idx[i] if i < len(prop_by_idx)
                        else PropagatedFrame(object_masks={},
                                              object_bboxes={}))
                for owl in owl_dets.get(fid, []):
                    best_iou = 0.0
                    best_oid = None
                    for oid, mask in prop.object_masks.items():
                        if tracks[oid].label != owl.label:
                            continue
                        iou = _bbox_mask_iou(owl.box, mask)
                        if iou > best_iou:
                            best_iou = iou
                            best_oid = oid
                    if best_iou < new_obj_iou:
                        new_prompts.append((i, owl))
                    elif best_oid is not None:
                        # Record observed score on the matched track.
                        tracks[best_oid].scores_by_frame[i] = owl.score

            if not new_prompts:
                print(f"[sam2-client] converged after {it+1} iteration(s)")
                break

            # Dedup: an object visible across consecutive frames shouldn't
            # produce N new prompts just because SAM2 hasn't yet propagated
            # the first one. Cluster unmatched boxes by (label, rolling
            # bbox-IoU) — one prompt per cluster, at its highest-score
            # frame. This collapses a smoothly-moving detection from a
            # dozen entries into one.
            new_prompts = _cluster_new_prompts(new_prompts,
                                               iou_thresh=0.3)

            # Cap how many new prompts we add per iteration so propagate
            # doesn't explode; prefer high-score ones first.
            new_prompts.sort(key=lambda p: -p[1].score)
            MAX_NEW_PER_ITER = 20
            added = 0
            for video_idx, owl in new_prompts[:MAX_NEW_PER_ITER]:
                _seed_prompt(sam2, video_idx, owl)
                added += 1
            print(f"[sam2-client] iter={it}  added {added} new tracks  "
                  f"(pending {len(new_prompts)-added})")

        # Final propagate for consistency after any last additions.
        # Only meaningful if we ever seeded anything.
        if tracks and (prop_by_idx is None or not prop_by_idx):
            prop_by_idx = sam2.propagate()

    # -----------------------------------------------------------------
    #  Write detection_h/*_final.json
    # -----------------------------------------------------------------
    print(f"[sam2-client] writing {len(frame_ids)} JSONs → {out_dir}")
    for i, prop in enumerate(prop_by_idx):
        fid = idx_to_fid.get(i, i)
        dets_out = []
        for oid, mask in prop.object_masks.items():
            bbox = prop.object_bboxes.get(oid) or _bbox_from_mask(mask)
            # Use the OWL-observed score at this frame when available,
            # fall back to the seeding score.
            sc = tracks[oid].scores_by_frame.get(i, tracks[oid].score)
            dets_out.append({
                "object_id": int(oid),
                "label":     tracks[oid].label,
                "score":     float(sc),
                "box":       list(map(int, bbox)),
                "mask":      _mask_to_png_b64(mask),
            })
        out_path = os.path.join(out_dir, f"detection_{fid:06d}_final.json")
        with open(out_path, "w") as f:
            json.dump({"detections": dets_out}, f, indent=2)

    # Fill in any frames SAM2 didn't produce results for (shouldn't happen
    # after a full propagate, but be defensive).
    written_idxs = {i for i in range(len(prop_by_idx))}
    for i, fid in enumerate(frame_ids):
        if i in written_idxs:
            continue
        out_path = os.path.join(out_dir, f"detection_{fid:06d}_final.json")
        with open(out_path, "w") as f:
            json.dump({"detections": []}, f, indent=2)

    print(f"[sam2-client] done.  final track count = {len(tracks)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", help="trajectory folder "
                                     "(e.g. datasets/apple_drop)")
    ap.add_argument("--server", default=SAM2_SERVER_URL,
                    help="SAM2 server URL "
                         "(override with $SAM2_SERVER_URL)")
    ap.add_argument("--det-dir", default="detection_boxes",
                    help="OWL JSONs subdir (default: detection_boxes)")
    ap.add_argument("--out-dir", default="detection_h",
                    help="output subdir (default: detection_h)")
    ap.add_argument("--new-obj-iou", type=float, default=0.3,
                    help="IoU (bbox vs propagated mask) below which an OWL "
                         "detection spawns a NEW track (default: 0.3, "
                         "legacy 2-D reconciliation path only)")
    ap.add_argument("--max-iters", type=int, default=4,
                    help="max propagate+add-prompt cycles (default: 4)")
    ap.add_argument("--min-score", type=float, default=0.0,
                    help="drop OWL dets below this score "
                         "before seeding (default: 0)")
    ap.add_argument("--state-driven", action="store_true",
                    help="use the state-driven reconciliation "
                         "(3-D Hungarian + FOV gate from bernoulli_ekf.tex). "
                         "Requires --pose-txt and --K-fx/--K-cx/etc. for "
                         "back-projection; defaults match the Fetch dataset")
    ap.add_argument("--pose-txt", default=None,
                    help="path to camera_pose.txt (state-driven only)")
    ap.add_argument("--K-fx", type=float, default=554.3827)
    ap.add_argument("--K-fy", type=float, default=554.3827)
    ap.add_argument("--K-cx", type=float, default=320.5)
    ap.add_argument("--K-cy", type=float, default=240.5)
    args = ap.parse_args()

    if args.state_driven:
        _track_dataset_state_driven(
            dataset_root=args.dataset,
            server_url=args.server,
            det_dir_name=args.det_dir,
            out_dir_name=args.out_dir,
            max_iters=args.max_iters,
            min_score=args.min_score,
            pose_txt=args.pose_txt,
            K=np.array([[args.K_fx, 0.0, args.K_cx],
                        [0.0, args.K_fy, args.K_cy],
                        [0.0, 0.0, 1.0]], dtype=np.float64),
        )
    else:
        track_dataset(
            dataset_root=args.dataset,
            server_url=args.server,
            det_dir_name=args.det_dir,
            out_dir_name=args.out_dir,
            new_obj_iou=args.new_obj_iou,
            max_iters=args.max_iters,
            min_score=args.min_score,
        )
    return 0


def _track_dataset_state_driven(dataset_root: str,
                                 server_url: str,
                                 det_dir_name: str,
                                 out_dir_name: str,
                                 max_iters: int,
                                 min_score: float,
                                 pose_txt: Optional[str],
                                 K: np.ndarray) -> None:
    """Run the state-driven reconciler end-to-end, writing detection_h/
    from the final SAM2 propagation.

    Requires camera intrinsics (K) and per-frame camera poses (from
    pose_txt). For the Fetch-style `Mobile_Manipulation_on_Fetch` layout
    the poses live under `<dataset>/pose_txt/camera_pose.txt` and can be
    omitted from the CLI (auto-located).
    """
    from rosbag2dataset.sam2.state_driven_reconcile import (
        state_driven_reconcile, ReconcileInputs,
    )

    rgb_dir = os.path.join(dataset_root, "rgb")
    det_dir = os.path.join(dataset_root, det_dir_name)
    depth_dir = os.path.join(dataset_root, "depth")
    out_dir = os.path.join(dataset_root, out_dir_name)
    os.makedirs(out_dir, exist_ok=True)

    pose_path = pose_txt or os.path.join(
        dataset_root, "pose_txt", "camera_pose.txt")
    if not os.path.exists(pose_path):
        raise FileNotFoundError(f"state-driven needs pose file: {pose_path}")

    print(f"[state-driven] dataset={dataset_root}  server={server_url}")
    frame_ids, rgb_frames = _load_frames(rgb_dir)
    if not rgb_frames:
        raise FileNotFoundError(f"no RGB in {rgb_dir}")
    depth_frames: List[np.ndarray] = []
    for fid in frame_ids:
        depth_path = os.path.join(depth_dir, f"depth_{fid:06d}.npy")
        depth_frames.append(np.load(depth_path).astype(np.float32)
                             if os.path.exists(depth_path)
                             else np.zeros(rgb_frames[0].shape[:2],
                                           dtype=np.float32))

    # Camera poses (T_cw, 4x4 per frame) -- one line per frame:
    # frame_id tx ty tz qx qy qz qw.
    from scipy.spatial.transform import Rotation
    pose_per_fid: Dict[int, np.ndarray] = {}
    with open(pose_path, "r") as f:
        for line in f:
            arr = line.strip().split()
            if len(arr) != 8:
                continue
            fid_i, tx, ty, tz, qx, qy, qz, qw = [float(x) for x in arr]
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            T[:3, 3] = [tx, ty, tz]
            pose_per_fid[int(fid_i)] = T
    cam_poses = [pose_per_fid.get(fid, np.eye(4)) for fid in frame_ids]

    owl_by_fid = _load_owl_detections(det_dir, min_score=min_score)
    H, W = rgb_frames[0].shape[:2]

    inputs = ReconcileInputs(
        frame_ids=frame_ids, rgb=rgb_frames, depth=depth_frames,
        cam_poses=cam_poses, K=K, image_shape=(H, W),
        owl_by_fid=owl_by_fid,
    )

    with SAM2Client(server_url=server_url) as sam2:
        state, prop_by_idx = state_driven_reconcile(
            inputs, sam2, max_iters=max_iters)

    # Write detection_h/*_final.json (same schema the legacy path uses).
    _write_detection_h(dataset_root, out_dir_name, frame_ids,
                        prop_by_idx, state)


def _write_detection_h(dataset_root: str, out_dir_name: str,
                        frame_ids: List[int],
                        prop_by_idx: List["PropagatedFrame"],
                        state: Dict[int, "TrackState3D"]) -> None:
    """Serialise per-frame {object_id, label, box, mask, score} to
    detection_h/*_final.json, matching the legacy output schema."""
    out_dir = os.path.join(dataset_root, out_dir_name)
    for i, fid in enumerate(frame_ids):
        prop = (prop_by_idx[i] if i < len(prop_by_idx)
                else PropagatedFrame(object_masks={}, object_bboxes={}))
        entries = []
        for oid, mask in prop.object_masks.items():
            tr = state.get(oid)
            if tr is None:
                continue
            box = prop.object_bboxes.get(oid)
            if box is None:
                box = _bbox_from_mask(mask)
            entries.append({
                "object_id": int(oid),
                "label": tr.label,
                "score": float(tr.scores_by_frame.get(i, 0.0)),
                "box": [int(v) for v in box],
                "mask": _mask_to_png_b64(mask),
            })
        out_path = os.path.join(out_dir, f"detection_{fid:06d}_final.json")
        with open(out_path, "w") as f:
            json.dump({"detections": entries}, f, indent=2)


if __name__ == "__main__":
    sys.exit(main())
