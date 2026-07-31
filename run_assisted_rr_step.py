#!/usr/bin/env python3
"""
Assisted one-leg ladder step for the modified Unitree Go2.

Purpose
-------
This is deliberately simple and highly assisted.  It is intended to obtain
the first visible and repeatable RR contact transition:

    rung_5 -> rung_6

The verified StaticLadderGraspController is used unchanged to establish the
initial four-gripper grasp.  After that:

1. the floating base is temporarily held at its measured pose;
2. FL, FR and RL joint positions are held;
3. RR finger collisions are temporarily disabled;
4. RR fingers are opened kinematically;
5. RR leg joints are moved through three IK waypoints:
       lift -> translate one rung -> lower;
6. RR collisions are restored;
7. RR fingers are closed to the measured source-grasp pose.

This is an assisted/kinematic development baseline.  Do not use its motion
for free-floating dynamic metrics.  Its purpose is to prove the rung mapping,
IK waypoints and regrasp geometry before assistance is removed.
"""

from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path
from typing import Iterable

import mujoco
import numpy as np

from controller.grasp_controller import StaticLadderGraspController


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SCENE = PROJECT_ROOT / "scene.xml"


def smoothstep(value: float) -> float:
    x = float(np.clip(value, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def interpolate(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    return np.asarray(start, dtype=float) + smoothstep(alpha) * (
        np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    )


def geom_name(model: mujoco.MjModel, geom_id: int) -> str:
    return (
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id))
        or f"geom_{int(geom_id)}"
    )


def build_rung_table(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller: StaticLadderGraspController,
) -> list[tuple[float, int, str, np.ndarray]]:
    rows: list[tuple[float, int, str, np.ndarray]] = []
    for geom_id in controller._rung_geom_ids:
        position = np.asarray(data.geom_xpos[int(geom_id)], dtype=float).copy()
        rows.append((float(position[0]), int(geom_id), geom_name(model, int(geom_id)), position))
    rows.sort(key=lambda item: item[0])
    return rows


def source_rung_for_leg(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller: StaticLadderGraspController,
    leg: str,
) -> str:
    contacts = controller._detect_contacts(model, data)
    details = contacts.get("contact_details", [])
    names = [
        str(item["rung_geom_name"])
        for item in details
        if str(item.get("leg")) == leg
    ]
    if not names:
        raise RuntimeError(f"{leg} has no measured rung contact after static grasp.")
    return collections.Counter(names).most_common(1)[0][0]


def count_leg_rung_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller: StaticLadderGraspController,
    leg: str,
    rung_name: str,
) -> int:
    contacts = controller._detect_contacts(model, data)
    details = contacts.get("contact_details", [])
    return sum(
        1
        for item in details
        if str(item.get("leg")) == leg
        and str(item.get("rung_geom_name")) == rung_name
    )


def rr_finger_geom_ids(
    model: mujoco.MjModel,
    controller: StaticLadderGraspController,
) -> np.ndarray:
    ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if controller._finger_body_to_leg.get(int(model.geom_bodyid[geom_id])) == "RR"
    ]
    if len(ids) != 6:
        raise RuntimeError(f"Expected six RR finger collision geoms, found {len(ids)}.")
    return np.asarray(ids, dtype=int)


def solve_leg_ik(
    model: mujoco.MjModel,
    reference_qpos: np.ndarray,
    controller: StaticLadderGraspController,
    *,
    leg: str,
    target_palm: np.ndarray,
    initial_leg_qpos: np.ndarray,
    iterations: int = 400,
    tolerance: float = 0.0025,
    damping: float = 0.003,
    step_limit: float = 0.035,
) -> tuple[np.ndarray, float]:
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = np.asarray(reference_qpos, dtype=float)
    scratch.qvel[:] = 0.0
    scratch.ctrl[:] = 0.0
    scratch.qpos[controller._leg_qpos_adrs[leg]] = np.asarray(initial_leg_qpos, dtype=float)
    scratch.qpos[controller._finger_qpos_adrs[leg]] = controller.open_finger_pose
    mujoco.mj_forward(model, scratch)

    body_id = controller._palm_body_ids[leg]
    dof_addresses = controller._leg_dof_adrs[leg]
    qpos_addresses = controller._leg_qpos_adrs[leg]
    regulariser = damping * damping * np.eye(3)

    for _ in range(iterations):
        actual = np.asarray(scratch.xpos[body_id], dtype=float)
        error = np.asarray(target_palm, dtype=float) - actual
        if float(np.linalg.norm(error)) <= tolerance:
            break

        jacp = np.zeros((3, model.nv), dtype=float)
        jacr = np.zeros((3, model.nv), dtype=float)
        mujoco.mj_jacBody(model, scratch, jacp, jacr, body_id)
        jacobian = jacp[:, dof_addresses]

        system = jacobian @ jacobian.T + regulariser
        try:
            delta = jacobian.T @ np.linalg.solve(system, error)
        except np.linalg.LinAlgError:
            delta = np.linalg.pinv(jacobian) @ error

        scratch.qpos[qpos_addresses] += np.clip(delta, -step_limit, step_limit)
        controller._clip_joint_positions(scratch, controller._leg_joint_ids[leg])
        mujoco.mj_forward(model, scratch)

    residual = float(
        np.linalg.norm(
            np.asarray(target_palm, dtype=float)
            - np.asarray(scratch.xpos[body_id], dtype=float)
        )
    )
    return scratch.qpos[qpos_addresses].copy(), residual


class AssistedRRStep:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        controller: StaticLadderGraspController,
        *,
        lift_height: float,
        open_time: float,
        lift_time: float,
        transfer_time: float,
        lower_time: float,
        close_time: float,
        final_hold_time: float,
    ) -> None:
        self.model = model
        self.data = data
        self.controller = controller

        self.lift_height = float(lift_height)
        self.open_time = float(open_time)
        self.lift_time = float(lift_time)
        self.transfer_time = float(transfer_time)
        self.lower_time = float(lower_time)
        self.close_time = float(close_time)
        self.final_hold_time = float(final_hold_time)

        self.started = False
        self.finished = False
        self.start_time = 0.0
        self.phase = "WAITING_FOR_STATIC_GRASP"

        self.base_qpos = np.zeros(7)
        self.support_leg_qpos: dict[str, np.ndarray] = {}
        self.support_finger_qpos: dict[str, np.ndarray] = {}
        self.rr_start_leg_qpos = np.zeros(3)
        self.rr_source_finger_qpos = np.zeros(6)

        self.q_lift = np.zeros(3)
        self.q_transfer = np.zeros(3)
        self.q_goal = np.zeros(3)

        self.source_rung_name = ""
        self.target_rung_name = ""
        self.start_palm = np.zeros(3)
        self.lift_palm = np.zeros(3)
        self.transfer_palm = np.zeros(3)
        self.goal_palm = np.zeros(3)

        self.rr_geom_ids = rr_finger_geom_ids(model, controller)
        self.saved_contype = model.geom_contype[self.rr_geom_ids].copy()
        self.saved_conaffinity = model.geom_conaffinity[self.rr_geom_ids].copy()
        self.collisions_disabled = False

    @property
    def total_time(self) -> float:
        return (
            self.open_time
            + self.lift_time
            + self.transfer_time
            + self.lower_time
            + self.close_time
            + self.final_hold_time
        )

    def begin(self) -> None:
        data = self.data
        controller = self.controller

        self.base_qpos = data.qpos[
            controller._base_qpos_adr : controller._base_qpos_adr + 7
        ].copy()

        for leg in ("FL", "FR", "RL"):
            self.support_leg_qpos[leg] = data.qpos[
                controller._leg_qpos_adrs[leg]
            ].copy()
            self.support_finger_qpos[leg] = data.qpos[
                controller._finger_qpos_adrs[leg]
            ].copy()

        self.rr_start_leg_qpos = data.qpos[
            controller._leg_qpos_adrs["RR"]
        ].copy()
        self.rr_source_finger_qpos = data.qpos[
            controller._finger_qpos_adrs["RR"]
        ].copy()

        self.source_rung_name = source_rung_for_leg(
            self.model, data, controller, "RR"
        )
        table = build_rung_table(self.model, data, controller)
        source_index = next(
            index for index, row in enumerate(table) if row[2] == self.source_rung_name
        )
        if source_index + 1 >= len(table):
            raise RuntimeError(f"No forward target rung exists after {self.source_rung_name}.")
        _, _, self.target_rung_name, target_position = table[source_index + 1]
        _, _, _, source_position = table[source_index]

        rung_translation = target_position - source_position
        self.start_palm = np.asarray(
            data.xpos[controller._palm_body_ids["RR"]], dtype=float
        ).copy()
        self.lift_palm = self.start_palm + np.array([0.0, 0.0, self.lift_height])
        self.transfer_palm = self.lift_palm + rung_translation
        self.goal_palm = self.start_palm + rung_translation

        reference_qpos = data.qpos.copy()
        self.q_lift, residual_lift = solve_leg_ik(
            self.model,
            reference_qpos,
            controller,
            leg="RR",
            target_palm=self.lift_palm,
            initial_leg_qpos=self.rr_start_leg_qpos,
        )
        q_reference = reference_qpos.copy()
        q_reference[controller._leg_qpos_adrs["RR"]] = self.q_lift

        self.q_transfer, residual_transfer = solve_leg_ik(
            self.model,
            q_reference,
            controller,
            leg="RR",
            target_palm=self.transfer_palm,
            initial_leg_qpos=self.q_lift,
        )
        q_reference[controller._leg_qpos_adrs["RR"]] = self.q_transfer

        self.q_goal, residual_goal = solve_leg_ik(
            self.model,
            q_reference,
            controller,
            leg="RR",
            target_palm=self.goal_palm,
            initial_leg_qpos=self.q_transfer,
        )

        print("\nASSISTED RR STEP SETUP")
        print("-" * 72)
        print(f"Source rung:       {self.source_rung_name}")
        print(f"Target rung:       {self.target_rung_name}")
        print(f"Rung translation:  {rung_translation}")
        print(f"Start palm:        {self.start_palm}")
        print(f"Lift palm:         {self.lift_palm}")
        print(f"Transfer palm:     {self.transfer_palm}")
        print(f"Goal palm:         {self.goal_palm}")
        print(
            "IK residuals:      "
            f"lift={residual_lift:.6f} m, "
            f"transfer={residual_transfer:.6f} m, "
            f"goal={residual_goal:.6f} m"
        )
        print("-" * 72)

        worst = max(residual_lift, residual_transfer, residual_goal)
        if worst > 0.010:
            raise RuntimeError(
                f"Assisted RR waypoints are not sufficiently reachable; "
                f"worst IK residual={worst:.6f} m."
            )

        self.start_time = float(data.time)
        self.started = True
        self.phase = "OPEN_RR"
        self._disable_rr_collisions()
        print(
            "\nASSISTED STEP STARTED: base/supports are held, "
            "RR collisions are temporarily disabled."
        )

    def _disable_rr_collisions(self) -> None:
        if self.collisions_disabled:
            return
        self.model.geom_contype[self.rr_geom_ids] = 0
        self.model.geom_conaffinity[self.rr_geom_ids] = 0
        self.collisions_disabled = True

    def _restore_rr_collisions(self) -> None:
        if not self.collisions_disabled:
            return
        self.model.geom_contype[self.rr_geom_ids] = self.saved_contype
        self.model.geom_conaffinity[self.rr_geom_ids] = self.saved_conaffinity
        self.collisions_disabled = False

    def _hold_base_and_supports(self) -> None:
        data = self.data
        c = self.controller

        data.qpos[c._base_qpos_adr : c._base_qpos_adr + 7] = self.base_qpos
        data.qvel[c._base_dof_adr : c._base_dof_adr + 6] = 0.0

        for leg in ("FL", "FR", "RL"):
            data.qpos[c._leg_qpos_adrs[leg]] = self.support_leg_qpos[leg]
            data.qvel[c._leg_dof_adrs[leg]] = 0.0
            data.qpos[c._finger_qpos_adrs[leg]] = self.support_finger_qpos[leg]
            data.qvel[c._finger_dof_adrs[leg]] = 0.0

    def apply(self) -> None:
        if not self.started or self.finished:
            return

        data = self.data
        c = self.controller
        elapsed = float(data.time) - self.start_time

        data.ctrl[:] = 0.0
        self._hold_base_and_supports()

        t0 = 0.0
        t1 = t0 + self.open_time
        t2 = t1 + self.lift_time
        t3 = t2 + self.transfer_time
        t4 = t3 + self.lower_time
        t5 = t4 + self.close_time
        t6 = t5 + self.final_hold_time

        if elapsed < t1:
            self.phase = "OPEN_RR"
            alpha = (elapsed - t0) / self.open_time
            rr_fingers = interpolate(
                self.rr_source_finger_qpos, c.open_finger_pose, alpha
            )
            rr_leg = self.rr_start_leg_qpos

        elif elapsed < t2:
            self.phase = "LIFT_RR"
            alpha = (elapsed - t1) / self.lift_time
            rr_fingers = c.open_finger_pose
            rr_leg = interpolate(self.rr_start_leg_qpos, self.q_lift, alpha)

        elif elapsed < t3:
            self.phase = "TRANSFER_RR"
            alpha = (elapsed - t2) / self.transfer_time
            rr_fingers = c.open_finger_pose
            rr_leg = interpolate(self.q_lift, self.q_transfer, alpha)

        elif elapsed < t4:
            self.phase = "LOWER_RR"
            alpha = (elapsed - t3) / self.lower_time
            rr_fingers = c.open_finger_pose
            rr_leg = interpolate(self.q_transfer, self.q_goal, alpha)

        elif elapsed < t5:
            self.phase = "CLOSE_RR"
            self._restore_rr_collisions()
            alpha = (elapsed - t4) / self.close_time
            rr_fingers = interpolate(
                c.open_finger_pose, self.rr_source_finger_qpos, alpha
            )
            rr_leg = self.q_goal

        elif elapsed < t6:
            self.phase = "HOLD_TARGET"
            self._restore_rr_collisions()
            rr_fingers = self.rr_source_finger_qpos
            rr_leg = self.q_goal

        else:
            self.phase = "COMPLETE"
            self._restore_rr_collisions()
            rr_fingers = self.rr_source_finger_qpos
            rr_leg = self.q_goal
            self.finished = True

        data.qpos[c._leg_qpos_adrs["RR"]] = rr_leg
        data.qvel[c._leg_dof_adrs["RR"]] = 0.0
        data.qpos[c._finger_qpos_adrs["RR"]] = rr_fingers
        data.qvel[c._finger_dof_adrs["RR"]] = 0.0
        mujoco.mj_forward(self.model, data)

    def post_step(self) -> None:
        if not self.started:
            return
        # Re-apply the assisted state after dynamics so the scripted pose is
        # not displaced by contact impulses.
        self.apply()

    def report(self) -> None:
        source_contacts = count_leg_rung_contacts(
            self.model,
            self.data,
            self.controller,
            "RR",
            self.source_rung_name,
        )
        target_contacts = count_leg_rung_contacts(
            self.model,
            self.data,
            self.controller,
            "RR",
            self.target_rung_name,
        )
        palm = np.asarray(
            self.data.xpos[self.controller._palm_body_ids["RR"]], dtype=float
        )
        print(
            f"t={float(self.data.time):7.3f} | phase={self.phase:12s} | "
            f"RR palm=({palm[0]:+.4f}, {palm[1]:+.4f}, {palm[2]:+.4f}) | "
            f"old={source_contacts} | target={target_contacts}"
        )

    def final_result(self) -> None:
        self._restore_rr_collisions()
        source_contacts = count_leg_rung_contacts(
            self.model, self.data, self.controller, "RR", self.source_rung_name
        )
        target_contacts = count_leg_rung_contacts(
            self.model, self.data, self.controller, "RR", self.target_rung_name
        )
        print("\n" + "=" * 72)
        print("ASSISTED RR STEP RESULT")
        print("=" * 72)
        print(f"Final phase:          {self.phase}")
        print(f"Source rung:          {self.source_rung_name}")
        print(f"Target rung:          {self.target_rung_name}")
        print(f"Old-rung contacts:    {source_contacts}")
        print(f"Target-rung contacts: {target_contacts}")
        if target_contacts > 0:
            print("Result:               ASSISTED REGRASP DETECTED")
        else:
            print("Result:               MOTION COMPLETED, BUT TARGET CONTACT NOT DETECTED")
        print("=" * 72)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one simple assisted RR ladder step."
    )
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--lift-height", type=float, default=0.080)
    parser.add_argument("--open-time", type=float, default=1.0)
    parser.add_argument("--lift-time", type=float, default=1.5)
    parser.add_argument("--transfer-time", type=float, default=2.5)
    parser.add_argument("--lower-time", type=float, default=1.5)
    parser.add_argument("--close-time", type=float, default=1.5)
    parser.add_argument("--final-hold-time", type=float, default=1.5)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--real-time", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--hold-viewer-on-exit", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    if not args.scene.exists():
        raise FileNotFoundError(f"Scene XML not found: {args.scene}")

    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    model.opt.gravity[:] = [0.0, 0.0, 0.0]
    data = mujoco.MjData(model)

    static = StaticLadderGraspController(model)
    static.reset(model, data)

    assisted = AssistedRRStep(
        model,
        data,
        static,
        lift_height=args.lift_height,
        open_time=args.open_time,
        lift_time=args.lift_time,
        transfer_time=args.transfer_time,
        lower_time=args.lower_time,
        close_time=args.close_time,
        final_hold_time=args.final_hold_time,
    )

    next_report = 0.0

    def loop(viewer: object | None) -> None:
        nonlocal next_report
        while float(data.time) < args.duration:
            wall_start = time.perf_counter()

            if not assisted.started:
                status = static.update(model, data)
                mujoco.mj_step(model, data)
                static.post_step(model, data)

                if status.state == "FAILED":
                    raise RuntimeError("Static grasp acquisition failed.")
                if status.state == "HOLDING":
                    assisted.begin()
            else:
                assisted.apply()
                mujoco.mj_step(model, data)
                assisted.post_step()

            if viewer is not None:
                viewer.sync()

            if float(data.time) >= next_report:
                if assisted.started:
                    assisted.report()
                else:
                    status = static.status
                    print(
                        f"t={float(data.time):7.3f} | "
                        f"phase=STATIC_{status.state:10s} | "
                        f"contacts={status.physical_contacts} | "
                        f"grippers={status.distinct_grippers}"
                    )
                next_report = float(data.time) + 0.5

            if assisted.finished:
                break

            if args.real_time:
                elapsed = time.perf_counter() - wall_start
                remaining = float(model.opt.timestep) - elapsed
                if remaining > 0.0:
                    time.sleep(remaining)

    if args.headless:
        loop(None)
    else:
        from mujoco import viewer as mj_viewer

        with mj_viewer.launch_passive(model, data) as viewer:
            loop(viewer)
            if args.hold_viewer_on_exit:
                print("Close the MuJoCo window to exit.")
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(0.02)

    assisted.final_result()


def main() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
