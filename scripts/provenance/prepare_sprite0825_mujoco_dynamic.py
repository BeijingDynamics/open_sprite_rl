#!/usr/bin/env python3
"""Prepare the articulated Sprite0825 MuJoCo model for external-PD sim2sim."""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


ARMATURE_GROUPS = (
    ((r".*_hip_.*", r".*_knee_joint"), 0.032),
    ((r".*_ankle_pitch_joint", r".*_ankle_roll_joint"), 0.0036),
    ((r"waist_roll_joint",), 0.01),
    ((r"waist_yaw_joint",), 0.032),
    ((r".*_shoulder_.*", r".*_elbow_joint", r".*_wrist_yaw_joint"), 0.0018),
    ((r".*_wrist_pitch_joint", r".*_wrist_roll_joint", r"head_.*"), 0.005),
)


def indent(element: ET.Element, level: int = 0) -> None:
    pad = "\n" + "  " * level
    if len(element):
        if not element.text or not element.text.strip():
            element.text = pad + "  "
        for child in element:
            indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = pad
    if level and (not element.tail or not element.tail.strip()):
        element.tail = pad


def find_body(root: ET.Element, name: str) -> ET.Element:
    for body in root.iter("body"):
        if body.get("name") == name:
            return body
    raise RuntimeError(f"Missing body: {name}")


def add_foot_visuals(root: ET.Element) -> None:
    asset = root.find("asset")
    if asset is None:
        raise RuntimeError("Missing MJCF asset section")
    for side in ("left", "right"):
        mesh_name = f"{side}_ankle_roll_shell"
        asset.append(
            ET.Element(
                "mesh",
                {
                    "name": mesh_name,
                    "content_type": "model/stl",
                    "file": f"meshes/{side}_ankle_roll_link.STL",
                },
            )
        )
        body = find_body(root, f"{side}_ankle_roll_link")
        boxes = [geom for geom in body.findall("geom") if geom.get("type") == "box"]
        if len(boxes) != 1:
            raise RuntimeError(f"Expected one {side} sole proxy, found {len(boxes)}")
        boxes[0].set("name", f"{side}_sole")
        boxes[0].set("rgba", "0 0 0 0")
        body.append(
            ET.Element(
                "geom",
                {
                    "name": f"{side}_foot_shell_visual",
                    "type": "mesh",
                    "mesh": mesh_name,
                    "contype": "0",
                    "conaffinity": "0",
                    "group": "2",
                    "rgba": "0.776471 0.756863 0.737255 1",
                },
            )
        )


def set_external_pd_joint_dynamics(root: ET.Element) -> None:
    matched = set()
    for joint in root.iter("joint"):
        name = joint.get("name", "")
        if name == "pelvis_joint":
            continue
        armature = None
        for patterns, value in ARMATURE_GROUPS:
            if any(re.fullmatch(pattern, name) for pattern in patterns):
                armature = value
                break
        if armature is None:
            raise RuntimeError(f"No armature group for joint: {name}")
        joint.set("armature", str(armature))
        # The sim2sim runner applies the Isaac Kd term in qfrc_applied.
        # Passive joint damping here would count the same damping twice.
        joint.set("damping", "0")
        matched.add(name)
    if len(matched) != 31:
        raise RuntimeError(f"Expected 31 actuated joints, matched {len(matched)}")


def set_root_height(root: ET.Element, height: float) -> None:
    pelvis = find_body(root, "pelvis_link")
    position = [float(value) for value in pelvis.get("pos", "0 0 0").split()]
    if len(position) != 3:
        raise RuntimeError(f"Invalid pelvis_link position: {pelvis.get('pos')}")
    position[2] = height
    pelvis.set("pos", " ".join(f"{value:g}" for value in position))


def set_sim2sim_contact_contract(root: ET.Element, friction: tuple[float, float, float]) -> None:
    default = root.find("default")
    if default is None:
        default = ET.Element("default")
        compiler = root.find("compiler")
        root.insert(list(root).index(compiler) + 1 if compiler is not None else 0, default)
    geom = default.find("geom")
    if geom is None:
        geom = ET.SubElement(default, "geom")
    # Isaac runs Sprite with self-collision disabled. Robot geoms still collide
    # with the explicitly affine floor in the generated scene.
    geom.set("contype", "1")
    geom.set("conaffinity", "0")
    geom.set("friction", " ".join(f"{value:g}" for value in friction))


def write_scene(path: Path, robot_path: Path, friction: tuple[float, float, float]) -> None:
    include_path = robot_path.relative_to(path.parent).as_posix()
    friction_text = " ".join(f"{value:g}" for value in friction)
    text = f"""<mujoco model="sprite0825 v4 external PD sim2sim">
  <include file="{include_path}"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
  <visual>
    <headlight diffuse="0.7 0.7 0.7" ambient="0.35 0.35 0.35" specular="0.1 0.1 0.1"/>
    <global azimuth="135" elevation="-12"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.18 0.22 0.28" rgb2="0.02 0.03 0.05" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.25 0.28 0.30" rgb2="0.12 0.14 0.16" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="8 8" reflectance="0.1"/>
  </asset>
  <worldbody>
    <light pos="0 -2 2" dir="0 1 -1" directional="true"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="groundplane" contype="1" conaffinity="1" friction="{friction_text}"/>
  </worldbody>
</mujoco>
"""
    path.write_text(text, encoding="utf-8")


def names(model: mujoco.MjModel, object_type: int, count: int) -> list[str]:
    return [mujoco.mj_id2name(model, object_type, index) or "" for index in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--deploy-contract", type=Path, required=True)
    parser.add_argument("--skip-foot-visuals", action="store_true")
    args = parser.parse_args()

    contract = json.loads(args.deploy_contract.read_text(encoding="utf-8"))
    root_height = float(contract["deployment_initial_root_height_m"])
    contact = contract["mujoco_contact"]
    friction = (
        float(contact["sliding_friction"]),
        float(contact["torsional_friction"]),
        float(contact["rolling_friction"]),
    )
    if bool(contact["robot_self_collision_enabled"]):
        raise RuntimeError("This generator currently requires robot self-collision disabled")

    root = ET.parse(args.input).getroot()
    set_root_height(root, root_height)
    if not args.skip_foot_visuals:
        add_foot_visuals(root)
    set_external_pd_joint_dynamics(root)
    set_sim2sim_contact_contract(root, friction)
    indent(root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(args.output, encoding="utf-8", xml_declaration=False)
    write_scene(args.scene, args.output, friction)

    model = mujoco.MjModel.from_xml_path(str(args.output))
    scene = mujoco.MjModel.from_xml_path(str(args.scene))
    joint_names = names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
    if joint_names[0] != "pelvis_joint" or model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
        raise RuntimeError("Floating pelvis_joint is missing")
    if model.njnt != 32 or model.nq != 38 or model.nv != 37:
        raise RuntimeError(f"Unexpected model dimensions: njnt={model.njnt} nq={model.nq} nv={model.nv}")
    if not np.allclose(model.dof_damping[6:], 0.0):
        raise RuntimeError("Actuated joint damping must be zero for external-PD parity")
    robot_geoms = model.geom_bodyid > 0
    if np.any(model.geom_conaffinity[robot_geoms] != 0):
        raise RuntimeError("Robot self-collision mask does not match Isaac enabled_self_collisions=False")
    floor_id = mujoco.mj_name2id(scene, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id < 0 or scene.geom_contype[floor_id] != 1 or scene.geom_conaffinity[floor_id] != 1:
        raise RuntimeError("Scene floor contact mask is missing")
    if not np.allclose(model.geom_friction[robot_geoms], friction):
        raise RuntimeError(f"Robot friction must match contract {friction}")
    if not np.allclose(scene.geom_friction[floor_id], friction):
        raise RuntimeError(f"Floor friction must match contract {friction}")
    if not np.isclose(model.qpos0[2], root_height):
        raise RuntimeError(f"Root height {model.qpos0[2]} does not match contract {root_height}")

    print(
        f"robot bodies={model.nbody} joints={model.njnt} geoms={model.ngeom} "
        f"meshes={model.nmesh} nq={model.nq} nv={model.nv} mass={model.body_mass.sum():.9f}"
    )
    print(
        f"scene bodies={scene.nbody} joints={scene.njnt} geoms={scene.ngeom} "
        f"meshes={scene.nmesh}"
    )
    print(f"passive_damping_max={float(np.abs(model.dof_damping[6:]).max()):.9g}")
    print(
        f"deployment_contract=root_height={root_height:g} "
        f"self_collision_off friction={friction}"
    )
    print(f"WROTE {args.output}")
    print(f"WROTE {args.scene}")


if __name__ == "__main__":
    main()
