"""Lightweight URDF parser + forward-kinematics chain reader for the Fetch arm.

Used by ``scripts/visualize_arm_mask.py`` to project per-link mesh silhouettes
onto RGB images for visual verification of the robot-arm mask before any
depth-side change.

Intentionally minimal: only what's needed to build per-link
``T_base_link`` from joint angles and to look up each link's collision
mesh path. No GL context, no urdfpy / pybullet dependency — only stdlib
``xml.etree.ElementTree`` plus scipy.

URDF conventions used here:
    * Per-joint ``<origin xyz rpy>``: SE(3) from parent link frame to the
      *static* part of the joint frame (before motion is applied).
    * ``rpy`` is intrinsic Z-Y-X / extrinsic X-Y-Z Tait-Bryan; we use
      ``scipy.spatial.transform.Rotation.from_euler('xyz', [r, p, y])``
      which matches.
    * ``<axis xyz>`` is in the child link frame. For revolute /
      continuous joints, motion is ``Rodrigues(angle * axis)``; for
      prismatic, it's ``translate(angle * axis)``; for fixed, identity.

Per-link transform:
    ``T_parent_child = T_origin @ T_motion(angle, axis)``
    ``T_base_child   = T_base_parent @ T_parent_child``
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R_


@dataclass
class JointInfo:
    name: str
    type: str            # 'fixed' | 'revolute' | 'continuous' | 'prismatic'
    parent: str
    child: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis: np.ndarray


@dataclass
class LinkMesh:
    name: str
    collision_mesh_path: Optional[str]
    collision_origin_xyz: np.ndarray
    collision_origin_rpy: np.ndarray


def _parse_origin(elt: Optional[ET.Element]) -> Tuple[np.ndarray, np.ndarray]:
    if elt is None:
        return np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
    xyz = elt.get("xyz", "0 0 0").split()
    rpy = elt.get("rpy", "0 0 0").split()
    return (np.asarray([float(x) for x in xyz], dtype=np.float64),
            np.asarray([float(x) for x in rpy], dtype=np.float64))


def _parse_axis(elt: Optional[ET.Element]) -> np.ndarray:
    if elt is None:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    xyz = elt.get("xyz", "1 0 0").split()
    return np.asarray([float(x) for x in xyz], dtype=np.float64)


def _resolve_mesh_path(filename: str, mesh_root: str) -> str:
    """Translate ``package://meshes/foo.stl`` (or similar) to a disk path
    rooted at ``mesh_root``.

    Strips the leading ``package://`` and any first segment that names a
    package; the remainder is treated as a relative path under
    ``mesh_root``. For Fetch URDFs the references look like
    ``package://meshes/<file>`` — we strip ``package://`` and join.
    """
    if filename.startswith("package://"):
        rel = filename[len("package://"):]
        return os.path.join(mesh_root, rel)
    if os.path.isabs(filename):
        return filename
    return os.path.join(mesh_root, filename)


def _SE3(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_.from_euler("xyz", rpy).as_matrix()
    T[:3, 3] = xyz
    return T


def _axis_motion(joint_type: str, axis: np.ndarray, angle: float) -> np.ndarray:
    """SE(3) for the motion component of a single joint."""
    T = np.eye(4, dtype=np.float64)
    if joint_type == "fixed":
        return T
    if joint_type in ("revolute", "continuous"):
        T[:3, :3] = R_.from_rotvec(angle * axis).as_matrix()
        return T
    if joint_type == "prismatic":
        T[:3, 3] = angle * axis
        return T
    return T  # unknown type → no motion


class FetchArmFK:
    """Parse a Fetch URDF and compute per-link transforms in base_link frame."""

    def __init__(self, urdf_path: str, mesh_root: Optional[str] = None):
        self.urdf_path = urdf_path
        # Default mesh root: directory containing the URDF.
        self.mesh_root = mesh_root or os.path.dirname(urdf_path)
        self.joints: Dict[str, JointInfo] = {}
        self.parent_joint_of: Dict[str, str] = {}
        self.links: Dict[str, LinkMesh] = {}
        self._parse(urdf_path)

    def _parse(self, urdf_path: str) -> None:
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        for j in root.findall("joint"):
            name = j.get("name")
            jtype = j.get("type")
            parent_e = j.find("parent")
            child_e = j.find("child")
            if parent_e is None or child_e is None:
                continue
            parent = parent_e.get("link")
            child = child_e.get("link")
            ox, op = _parse_origin(j.find("origin"))
            ax = _parse_axis(j.find("axis"))
            self.joints[name] = JointInfo(
                name=name, type=jtype, parent=parent, child=child,
                origin_xyz=ox, origin_rpy=op, axis=ax,
            )
            # Multiple joints can share a child only in trees with cycles
            # (none in URDF). First wins.
            self.parent_joint_of.setdefault(child, name)
        for L in root.findall("link"):
            lname = L.get("name")
            mesh_path: Optional[str] = None
            mxyz = np.zeros(3, dtype=np.float64)
            mrpy = np.zeros(3, dtype=np.float64)
            col = L.find("collision")
            if col is not None:
                geom = col.find("geometry")
                if geom is not None:
                    mesh = geom.find("mesh")
                    if mesh is not None:
                        fname = mesh.get("filename", "")
                        mesh_path = _resolve_mesh_path(fname, self.mesh_root)
                mxyz, mrpy = _parse_origin(col.find("origin"))
            self.links[lname] = LinkMesh(
                name=lname,
                collision_mesh_path=mesh_path,
                collision_origin_xyz=mxyz,
                collision_origin_rpy=mrpy,
            )

    def _link_chain(self, target: str, base_link: str) -> List[str]:
        """Return ``[base_link, ..., target]`` in parent-to-child order.

        Raises ``ValueError`` if ``target`` is not in the descendant
        subtree of ``base_link``.
        """
        chain = [target]
        cur = target
        while cur in self.parent_joint_of and cur != base_link:
            j = self.joints[self.parent_joint_of[cur]]
            chain.append(j.parent)
            cur = j.parent
            if len(chain) > 200:
                raise ValueError(f"chain depth > 200 walking from {target}")
        if chain[-1] != base_link:
            raise ValueError(f"link {target!r} not reachable from {base_link!r}")
        return list(reversed(chain))

    def fk(self, joint_angles: Dict[str, float],
            targets: List[str], *,
            base_link: str = "base_link") -> Dict[str, np.ndarray]:
        """Return ``{link: T_base_link}`` for every link in ``targets``.

        Re-uses already-computed transforms across overlapping chain
        prefixes so deep arms are evaluated once.
        """
        cache: Dict[str, np.ndarray] = {base_link: np.eye(4, dtype=np.float64)}
        out: Dict[str, np.ndarray] = {}
        for target in targets:
            if target == base_link:
                out[target] = cache[base_link].copy()
                continue
            chain = self._link_chain(target, base_link=base_link)
            T = cache[chain[0]]
            for child_link in chain[1:]:
                if child_link in cache:
                    T = cache[child_link]
                    continue
                joint = self.joints[self.parent_joint_of[child_link]]
                angle = float(joint_angles.get(joint.name, 0.0))
                T_origin = _SE3(joint.origin_xyz, joint.origin_rpy)
                T_motion = _axis_motion(joint.type, joint.axis, angle)
                T = T @ T_origin @ T_motion
                cache[child_link] = T
            out[target] = T
        return out

    def collision_origin(self, link: str) -> np.ndarray:
        m = self.links.get(link)
        if m is None:
            return np.eye(4, dtype=np.float64)
        return _SE3(m.collision_origin_xyz, m.collision_origin_rpy)

    def collision_mesh_path(self, link: str) -> Optional[str]:
        m = self.links.get(link)
        return None if m is None else m.collision_mesh_path
