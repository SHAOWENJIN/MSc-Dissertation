#!/usr/bin/env python3
"""Regression check for contact-gated static and moving gripper latches."""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from controller.grasp_controller import StaticLadderGraspController
from run_assisted_cyclic_crawl_v3 import AssistedCyclicCrawler


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def minimum_visible_limb_rung_clearance(model, data) -> tuple[float, str]:
    """Measure the rendered thigh/calf meshes, not only collision proxies."""
    rung_ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "").startswith("rung_")
    ]
    minimum = float("inf")
    minimum_pair = ""
    for leg in ("FL", "FR", "RL", "RR"):
        body_ids = {
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_{part}")
            for part in ("thigh", "calf")
        }
        for limb_geom in range(model.ngeom):
            if int(model.geom_bodyid[limb_geom]) not in body_ids:
                continue
            # Group 2 contains the actual rendered mesh envelope.
            if int(model.geom_group[limb_geom]) != 2:
                continue
            limb_name = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, limb_geom)
                or f"geom_{limb_geom}"
            )
            for rung_id in rung_ids:
                distance = float(
                    mujoco.mj_geomDistance(
                        model, data, limb_geom, rung_id, 1.0, None
                    )
                )
                if distance < minimum:
                    minimum = distance
                    rung_name = mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_GEOM, rung_id
                    )
                    minimum_pair = f"{limb_name}/{rung_name}"
    return minimum, minimum_pair


def has_visible_grasp_enclosure(model, data, controller, leg: str) -> bool:
    """Require the palm above and both distal hooks below one wrapped rung."""
    contacts = controller._detect_contacts(model, data)
    rung_names = {
        str(item["rung_geom_name"])
        for item in contacts.get("contact_details", [])
        if str(item.get("leg")) == leg
    }
    for rung_name in rung_names:
        if not controller.has_geometric_wrap(contacts, leg, rung_name):
            continue
        rung_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, rung_name
        )
        rung = np.asarray(data.geom_xpos[rung_id], dtype=float)
        palm = np.asarray(data.xpos[controller._palm_body_ids[leg]], dtype=float)
        distal = []
        for side in ("L", "R"):
            body_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                f"{leg.lower()}_finger_{side}3",
            )
            distal.append(np.asarray(data.xpos[body_id], dtype=float))
        if palm[2] > rung[2] + 0.020 and all(
            point[2] < rung[2] - 0.002 for point in distal
        ):
            return True
    return False


def main() -> None:
    scene = Path(__file__).resolve().parent / "scene_hill_frame.xml"
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    controller = StaticLadderGraspController(model)
    controller.reset(model, data)

    # The latch must fail closed before any physical contact exists.
    try:
        controller.set_grasp_weld(model, data, "RR", True)
    except RuntimeError:
        pass
    else:
        raise RuntimeError("Ghost-contact regression: empty RR grasp was latched.")

    one_sided = {
        "contact_details": [
            {"leg": "RR", "side": "L", "rung_geom_name": "rung_6", "relative_x": -0.01},
            {"leg": "RR", "side": "R", "rung_geom_name": "rung_6", "relative_x": -0.005},
        ]
    }
    require(
        not controller.has_geometric_wrap(one_sided, "RR", "rung_6"),
        "Ghost-contact regression: same-face contacts passed as a wrap.",
    )
    crossed_sides = {
        "contact_details": [
            {"leg": "RR", "side": "L", "rung_geom_name": "rung_6", "relative_x": -0.01},
            {"leg": "RR", "side": "R", "rung_geom_name": "rung_6", "relative_x": 0.006},
        ]
    }
    require(
        not controller.has_geometric_wrap(crossed_sides, "RR", "rung_6"),
        "Ghost-contact regression: crossed hook chains passed as a wrap.",
    )
    proper_wrap = {
        "contact_details": [
            {"leg": "RR", "side": "L", "rung_geom_name": "rung_6", "relative_x": 0.01},
            {"leg": "RR", "side": "R", "rung_geom_name": "rung_6", "relative_x": -0.006},
        ]
    }
    require(
        controller.has_geometric_wrap(proper_wrap, "RR", "rung_6"),
        "A correctly ordered opposite-face wrap was rejected.",
    )

    status = controller.status
    while data.time < 8.0 and not status.verified:
        status = controller.update(model, data)
        mujoco.mj_step(model, data)
        controller.post_step(model, data)
    require(status.verified, "Static four-gripper grasp was not verified.")

    crawler = AssistedCyclicCrawler(
        model,
        data,
        controller,
        sequence=("RR", "RL", "FR", "FL"),
        cycles=1,
        approach_clearance=0.055,
        open_time=1.5,
        lift_time=2.0,
        transfer_time=3.0,
        lower_time=2.0,
        close_time=7.0,
        hold_time=0.5,
        body_shift_time=3.0,
    )
    crawler.begin()
    deadline = float(data.time) + 140.0
    next_clearance_sample = float(data.time)
    minimum_transient_clearance = float("inf")
    minimum_transient_detail = ""
    while data.time < deadline and not crawler.finished:
        crawler.apply()
        mujoco.mj_step(model, data)
        crawler.post_step()
        if float(data.time) >= next_clearance_sample:
            clearance, pair = minimum_visible_limb_rung_clearance(model, data)
            if clearance < minimum_transient_clearance:
                minimum_transient_clearance = clearance
                minimum_transient_detail = (
                    f"t={float(data.time):.3f} phase={crawler.phase} {pair}"
                )
            next_clearance_sample += 0.020

    initial_base_x = float(crawler.body_shift_start_base[0])
    require(crawler.finished, "Four-leg grasp gait and body shift did not complete.")
    require(crawler.completed_steps == 4, "Full gait completed the wrong number of steps.")
    final_base_x = float(data.qpos[controller._base_qpos_adr])
    require(
        final_base_x - initial_base_x > 0.15,
        f"Robot did not advance: base displacement={final_base_x - initial_base_x:.3f} m.",
    )
    visible_clearance, visible_pair = minimum_visible_limb_rung_clearance(
        model, data
    )
    require(
        visible_clearance >= 0.002,
        "Visible-limb ghost intersection remains: "
        f"{visible_pair} clearance={visible_clearance:.4f} m.",
    )
    require(
        minimum_transient_clearance >= 0.002,
        "Transient visible-limb ghost intersection: "
        f"{minimum_transient_detail} clearance={minimum_transient_clearance:.4f} m.",
    )
    for leg in ("FL", "FR", "RL", "RR"):
        require(
            has_visible_grasp_enclosure(model, data, controller, leg),
            f"{leg} has contacts but its distal hooks do not visibly enclose the rung.",
        )
    contacts = controller._detect_contacts(model, data)
    wraps = set(contacts.get("geometrically_wrapped_grippers", set()))
    require(
        wraps == {"FL", "FR", "RL", "RR"},
        f"Final physical wrap set is incomplete: {sorted(wraps)}",
    )
    active_latches = {
        leg
        for leg, weld_id in controller._grasp_weld_ids.items()
        if weld_id >= 0 and bool(data.eq_active[weld_id])
    }
    require(
        active_latches == {"FL", "FR", "RL", "RR"},
        f"Unexpected final latch set: {sorted(active_latches)}",
    )
    output_dir = PACKAGE_ROOT / "results" / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "verified_final_grasp.png"
    renderer = None
    rendered = False
    try:
        # Headless macOS sessions may not have a CoreGraphics connection.  A
        # missing diagnostic image must not invalidate the completed physics
        # and clearance assertions above.
        renderer = mujoco.Renderer(model, height=480, width=640)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = [0.1, 0.0, -0.10]
        camera.distance = 1.45
        camera.azimuth = -125.0
        camera.elevation = -18.0
        renderer.update_scene(data, camera=camera)
        Image.fromarray(renderer.render()).save(image_path)
        rendered = True
    except Exception as exc:
        print(f"Final visual skipped: {type(exc).__name__}: {exc}")
    finally:
        if renderer is not None:
            renderer.close()
    print(
        f"Minimum visible limb clearance: {minimum_transient_clearance:.4f} m "
        f"at {minimum_transient_detail}"
    )
    if rendered:
        print(f"Final visual: {image_path}")
    print(
        "PASS: empty, same-face, and crossed grasps were rejected; the full "
        "four-leg gait advanced with four correctly ordered physical wraps."
    )


if __name__ == "__main__":
    main()
