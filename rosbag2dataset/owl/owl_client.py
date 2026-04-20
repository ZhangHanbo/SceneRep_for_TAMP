#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Drop-in replacement for ``owl_object_scores.py`` that calls the
alpha_robot OWL-ViT server instead of running inference locally.

Server contract (from ``alpha_robot/service/owl_vit/server.py``):

    POST <OWL_SERVER_URL>/owl_detect
    body = {
        "text_queries":        [str, ...],
        "image":               base64 PNG,
        "bbox_conf_threshold": float,
        "with_nms":            bool,
        "nms_threshold":       float,
        "nms_cat_threshold":   float,
    }
    response = {
        "scores":    [float, ...],
        "bboxes":    [[x1, y1, x2, y2], ...]    # normalized 0-1
        "box_names": [str,   ...],
    }

Output compatibility
────────────────────
For each ``rgb/rgb_NNNNNN.png`` we write:

    detection_boxes/detection_NNNNNN.json  — the same schema the old script
                                              emitted, so ``sam_client.py``
                                              and ``track_object_ids.py``
                                              need no changes.
    detection_boxes/detection_NNNNNN.png   — overlay preview

The JSON is:

    {"detections": [
        {"label": <str>, "score": <float>, "box": [x1, y1, x2, y2]},
        ...
    ]}

Pixel coordinates are in the original RGB image's frame (not normalized),
matching the old script.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
from typing import List, Tuple

import cv2
import numpy as np
import requests
from PIL import Image

# Allow running as `python rosbag2dataset/owl/owl_client.py ...`
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rosbag2dataset.server_configs import (  # noqa: E402
    OWL_SERVER_URL, OWL_DETECT_PATH, DEFAULT_OBJECTS,
    OWL_BBOX_CONF, OWL_NMS_IOU,
)


# ---------------------------------------------------------------------------
# HTTP call
# ---------------------------------------------------------------------------

def _encode_png_b64(bgr_or_rgb: np.ndarray) -> str:
    """Encode an HxWx3 uint8 image (RGB) as base64 PNG."""
    buf = io.BytesIO()
    Image.fromarray(bgr_or_rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def call_owl(rgb: np.ndarray,
             text_queries: List[str],
             bbox_conf_threshold: float = OWL_BBOX_CONF,
             server_url: str = OWL_SERVER_URL,
             with_nms: bool = True,
             nms_threshold: float = OWL_NMS_IOU,
             nms_cat_threshold: float = OWL_NMS_IOU,
             timeout: float = 60.0) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """POST one RGB frame to the OWL server.

    Returns:
        box_names (list[str]): label per detection.
        bboxes_norm (np.ndarray): (N, 4) normalized 0-1 [x1, y1, x2, y2].
        scores (np.ndarray): (N,) confidence.
    """
    url = server_url.rstrip("/") + OWL_DETECT_PATH
    payload = {
        "text_queries":        list(text_queries),
        "image":               _encode_png_b64(rgb),
        "bbox_conf_threshold": float(bbox_conf_threshold),
        "with_nms":            bool(with_nms),
        "nms_threshold":       float(nms_threshold),
        "nms_cat_threshold":   float(nms_cat_threshold),
    }
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    box_names = list(data.get("box_names", []))
    scores = np.asarray(data.get("scores", []),  dtype=np.float32)
    bboxes = np.asarray(data.get("bboxes", []),  dtype=np.float32).reshape(-1, 4)
    return box_names, bboxes, scores


# ---------------------------------------------------------------------------
# Per-frame processing -- direct server call, no client-side post-processing
# ---------------------------------------------------------------------------

def process_frame(rgb_path: str,
                  frame_idx: int,
                  output_folder: str,
                  text_queries: List[str],
                  server_url: str,
                  thresh: float = OWL_BBOX_CONF,
                  nms_iou: float = OWL_NMS_IOU,
                  with_nms: bool = True) -> int:
    img_bgr = cv2.imread(rgb_path)
    if img_bgr is None:
        print(f"[warn] cannot read {rgb_path}")
        return 0
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    box_names, bboxes_norm, scores = call_owl(
        img, text_queries,
        bbox_conf_threshold=thresh,
        nms_threshold=nms_iou,
        nms_cat_threshold=nms_iou,
        server_url=server_url, with_nms=with_nms,
    )

    # Pixel-space boxes, in the same [x1, y1, x2, y2] format the old script used.
    pix_boxes: List[List[int]] = []
    for bx in bboxes_norm:
        x1 = int(round(bx[0] * w))
        y1 = int(round(bx[1] * h))
        x2 = int(round(bx[2] * w))
        y2 = int(round(bx[3] * h))
        pix_boxes.append([max(0, x1), max(0, y1),
                          min(w - 1, x2), min(h - 1, y2)])
    scores_list = [float(s) for s in scores]

    # Write overlay + JSON (same filenames the old script produced).
    overlay = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).copy()
    for name, bx, sc in zip(box_names, pix_boxes, scores_list):
        x1, y1, x2, y2 = bx
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label_txt = f"{name}: {sc:.2f}"
        (tw, th), _ = cv2.getTextSize(label_txt,
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(overlay, (x1, max(0, y1 - th - 6)),
                      (x1 + tw + 4, y1), (255, 255, 255), -1)
        cv2.putText(overlay, label_txt, (x1 + 2, max(th, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imwrite(os.path.join(output_folder, f"detection_{frame_idx:06d}.png"),
                overlay)

    results = {"detections": [
        {"label": name, "score": sc, "box": bx}
        for name, bx, sc in zip(box_names, pix_boxes, scores_list)
    ]}
    with open(os.path.join(output_folder, f"detection_{frame_idx:06d}.json"),
              "w") as f:
        json.dump(results, f, indent=2)

    return len(box_names)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset_name", help="dataset sub-dir under DATASET_PATH "
                                         "(or a full path)")
    ap.add_argument("--dataset-root", default=None,
                    help="dataset root; defaults to $DATASET_PATH or '.'")
    ap.add_argument("--out-dir", default="detection_boxes",
                    help="output subdir (default: detection_boxes)")
    ap.add_argument("--objects", nargs="*", default=DEFAULT_OBJECTS,
                    help=f"text queries (default: {DEFAULT_OBJECTS})")
    ap.add_argument("--server", default=OWL_SERVER_URL,
                    help="OWL server URL (override with $OWL_SERVER_URL)")
    ap.add_argument("--thresh", type=float, default=OWL_BBOX_CONF,
                    help=f"bbox confidence threshold (uniform over frames "
                         f"and classes, default {OWL_BBOX_CONF})")
    ap.add_argument("--nms-iou", type=float, default=OWL_NMS_IOU,
                    help=f"server-side NMS IoU (default {OWL_NMS_IOU})")
    ap.add_argument("--no-nms", action="store_true",
                    help="disable server-side NMS entirely")
    args = ap.parse_args()

    if os.path.isabs(args.dataset_name) and os.path.isdir(args.dataset_name):
        dataset_dir = args.dataset_name
    else:
        root = args.dataset_root or os.environ.get("DATASET_PATH", ".")
        dataset_dir = os.path.join(root, args.dataset_name)
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"no such dataset: {dataset_dir}")

    rgb_dir = os.path.join(dataset_dir, "rgb")
    out_dir = os.path.join(dataset_dir, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(rgb_dir)
                   if f.endswith((".png", ".jpg")))
    if not files:
        print(f"no RGB frames in {rgb_dir}")
        return 1

    print(f"[owl] server={args.server}  objects={args.objects}  "
          f"{len(files)} frames")

    total_detections = 0
    t0 = time.time()
    for i, fname in enumerate(files):
        try:
            fid = int(os.path.splitext(fname)[0].split("_")[-1])
        except ValueError:
            print(f"[skip] unexpected filename {fname}")
            continue
        try:
            n = process_frame(
                rgb_path=os.path.join(rgb_dir, fname),
                frame_idx=fid,
                output_folder=out_dir,
                text_queries=args.objects,
                server_url=args.server,
                thresh=args.thresh,
                nms_iou=args.nms_iou,
                with_nms=not args.no_nms,
            )
            total_detections += n
        except requests.RequestException as e:
            print(f"[http error] frame {fid}: {e}")
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(files)}] "
                  f"{(time.time() - t0) / (i + 1):.2f}s/frame  "
                  f"total_dets={total_detections}")

    print(f"[done] {dataset_dir} → {out_dir}  "
          f"{total_detections} detections across {len(files)} frames "
          f"in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
