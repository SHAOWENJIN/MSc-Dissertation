#!/usr/bin/env python3
"""
Assisted rear-approach cyclic ladder crawler for the modified Unitree Go2.

This runner extends the successful assisted single-RR step to all four legs.

One cycle:
    RR -> RL -> FR -> FL -> assisted body shift

For each leg:
1. keep the floating base fixed;
2. hold the other three legs and grippers at their measured poses;
3. keep collision enabled throughout the leg motion;
4. open the moving gripper;
5. withdraw radially away from the source rung;
6. transfer through a collision-free outside corridor;
7. approach the target rung from the outside and close.

After all four legs move one rung forward, the base is shifted forward by one
rung spacing while all four palm world positions are held by assisted IK.

This is an assisted/kinematic gait-development baseline. Do not use it as
free-floating dynamic evidence or for final dynamic metrics.
"""

from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

from controller.grasp_controller import StaticLadderGraspController

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SCENE = PROJECT_ROOT / "scene.xml"
LEGS = ("FL", "FR", "RL", "RR")


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


def build_rung_table(model, data, controller):
    rows = []
    for geom_id in controller._rung_geom_ids:
        pos = np.asarray(data.geom_xpos[int(geom_id)], dtype=float).copy()
        rows.append((float(pos[0]), int(geom_id), geom_name(model, int(geom_id)), pos))
    rows.sort(key=lambda item: item[0])
    return rows


def source_rung_for_leg(model, data, controller, leg: str) -> str:
    contacts = controller._detect_contacts(model, data)
    names = [
        str(item["rung_geom_name"])
        for item in contacts.get("contact_details", [])
        if str(item.get("leg")) == leg
    ]
    if not names:
        # Assisted fallback: use nearest rung to palm x.
        palm_x = float(data.xpos[controller._palm_body_ids[leg]][0])
        table = build_rung_table(model, data, controller)
        return min(table, key=lambda row: abs(row[0] - palm_x))[2]
    return collections.Counter(names).most_common(1)[0][0]


def count_leg_rung_contacts(model, data, controller, leg: str, rung_name: str) -> int:
    contacts = controller._detect_contacts(model, data)
    return sum(
        1
        for item in contacts.get("contact_details", [])
        if str(item.get("leg")) == leg
        and str(item.get("rung_geom_name")) == rung_name
    )


def finger_geom_ids_for_leg(model, controller, leg: str) -> np.ndarray:
    ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if controller._finger_body_to_leg.get(int(model.geom_bodyid[geom_id])) == leg
    ]
    if len(ids) != 6:
        raise RuntimeError(f"Expected six {leg} finger collision geoms, found {len(ids)}.")
    return np.asarray(ids, dtype=int)


def solve_leg_ik(
    model,
    reference_qpos,
    controller,
    *,
    leg: str,
    target_palm: np.ndarray,
    initial_leg_qpos: np.ndarray,
    iterations: int = 500,
    tolerance: float = 1e-5,
):
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = reference_qpos
    qaddrs = controller._leg_qpos_adrs[leg]
    scratch.qpos[qaddrs] = initial_leg_qpos
    mujoco.mj_forward(model, scratch)

    body_id = controller._palm_body_ids[leg]
    dof_ids = np.asarray(controller._leg_dof_adrs[leg], dtype=int)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    for _ in range(iterations):
        mujoco.mj_forward(model, scratch)
        current = np.asarray(scratch.xpos[body_id], dtype=float)
        error = np.asarray(target_palm, dtype=float) - current
        if float(np.linalg.norm(error)) <= tolerance:
            break

        jacp.fill(0.0)
        jacr.fill(0.0)
        mujoco.mj_jacBody(model, scratch, jacp, jacr, body_id)
        J = jacp[:, dof_ids]
        damping = 1e-4
        dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(3), error)
        dq = np.clip(dq, -0.05, 0.05)
        scratch.qpos[qaddrs] += dq

        # Respect joint limits when present.
        for local_i, qadr in enumerate(qaddrs):
            joint_id = int(model.dof_jntid[dof_ids[local_i]])
            if model.jnt_limited[joint_id]:
                lo, hi = model.jnt_range[joint_id]
                scratch.qpos[qadr] = np.clip(scratch.qpos[qadr], lo, hi)

    mujoco.mj_forward(model, scratch)
    residual = float(
        np.linalg.norm(np.asarray(target_palm) - np.asarray(scratch.xpos[body_id]))
    )
    return scratch.qpos[qaddrs].copy(), residual


class AssistedCyclicCrawler:
    def __init__(
        self,
        model,
        data,
        controller,
        *,
        sequence: tuple[str, ...],
        cycles: int,
        approach_clearance: float,
        open_time: float,
        lift_time: float,
        transfer_time: float,
        lower_time: float,
        close_time: float,
        hold_time: float,
        body_shift_time: float,
    ):
        self.model = model
        self.data = data
        self.controller = controller
        self.sequence = sequence
        self.requested_cycles = cycles
        self.approach_clearance = approach_clearance
        self.open_time = open_time
        self.lift_time = lift_time
        self.transfer_time = transfer_time
        self.lower_time = lower_time
        self.close_time = close_time
        self.hold_time = hold_time
        self.body_shift_time = body_shift_time

        self.started = False
        self.finished = False
        self.phase = "WAITING_FOR_STATIC_GRASP"
        self.phase_start = 0.0
        self.cycle_index = 0
        self.leg_index = 0
        self.completed_steps = 0

        self.base_hold = np.zeros(7)
        self.held_leg_q = {}
        self.held_finger_q = {}
        self.geom_ids = {leg: finger_geom_ids_for_leg(model, controller, leg) for leg in LEGS}
        self.saved_contype = {leg: model.geom_contype[self.geom_ids[leg]].copy() for leg in LEGS}
        self.saved_conaffinity = {
            leg: model.geom_conaffinity[self.geom_ids[leg]].copy() for leg in LEGS
        }
        self.disabled = set()

        self.moving_leg = ""
        self.source_rung = ""
        self.target_rung = ""
        self.source_finger_q = np.zeros(6)
        self.q_start = np.zeros(3)
        self.q_withdraw = np.zeros(3)
        self.q_pregrasp = np.zeros(3)
        self.q_goal = np.zeros(3)
        self.start_palm = np.zeros(3)
        self.goal_palm = np.zeros(3)

        self.body_shift_start_base = np.zeros(7)
        self.body_shift_goal_base = np.zeros(7)
        self.body_shift_start_q = {}
        self.body_shift_goal_q = {}

    def _disable_leg_collision(self, leg: str):
        if leg in self.disabled:
            return
        ids = self.geom_ids[leg]
        self.model.geom_contype[ids] = 0
        self.model.geom_conaffinity[ids] = 0
        self.disabled.add(leg)

    def _restore_leg_collision(self, leg: str):
        if leg not in self.disabled:
            return
        ids = self.geom_ids[leg]
        self.model.geom_contype[ids] = self.saved_contype[leg]
        self.model.geom_conaffinity[ids] = self.saved_conaffinity[leg]
        self.disabled.remove(leg)

    def _restore_all_collisions(self):
        for leg in tuple(self.disabled):
            self._restore_leg_collision(leg)

    def _capture_current_hold(self):
        c, d = self.controller, self.data
        self.base_hold = d.qpos[c._base_qpos_adr : c._base_qpos_adr + 7].copy()
        for leg in LEGS:
            self.held_leg_q[leg] = d.qpos[c._leg_qpos_adrs[leg]].copy()
            self.held_finger_q[leg] = d.qpos[c._finger_qpos_adrs[leg]].copy()

    def begin(self):
        self._capture_current_hold()
        self.started = True
        self._begin_leg_step()

    def _next_rung(self, source_name: str):
        table = build_rung_table(self.model, self.data, self.controller)
        idx = next(i for i, row in enumerate(table) if row[2] == source_name)
        if idx + 1 >= len(table):
            raise RuntimeError(f"No forward target rung exists after {source_name}.")
        return table[idx], table[idx + 1]

    def _begin_leg_step(self):
        leg = self.sequence[self.leg_index]
        self.moving_leg = leg
        c, d = self.controller, self.data

        self.source_rung = source_rung_for_leg(self.model, d, c, leg)
        source_row, target_row = self._next_rung(self.source_rung)
        self.target_rung = target_row[2]
        source_center = source_row[3]
        target_center = target_row[3]
        translation = target_center - source_center

        self.q_start = d.qpos[c._leg_qpos_adrs[leg]].copy()
        self.source_finger_q = d.qpos[c._finger_qpos_adrs[leg]].copy()
        self.start_palm = np.asarray(d.xpos[c._palm_body_ids[leg]], dtype=float).copy()
        self.goal_palm = self.start_palm + translation

        # Collision-aware path in the rung cross-section.
        #
        # Departure:
        #   keep the successful radial withdrawal from the source rung.
        #
        # Arrival:
        #   do NOT place the pre-grasp waypoint above the target rung.
        #   Place it behind the target, opposite the direction of travel,
        #   with only a small upward bias.  The final approach therefore
        #   moves mostly forward (+/-x depending on ladder direction) and
        #   slightly downward, rather than dropping vertically through it.
        radial = self.start_palm - source_center
        radial[1] = 0.0
        radial_norm = float(np.linalg.norm(radial))
        if radial_norm < 1e-8:
            radial = np.array([-1.0, 0.0, 0.0], dtype=float)
            radial_norm = float(np.linalg.norm(radial))
        radial /= radial_norm

        travel = translation.copy()
        travel[1] = 0.0
        travel_norm = float(np.linalg.norm(travel))
        if travel_norm < 1e-8:
            raise RuntimeError("Source and target rung positions are identical.")
        travel /= travel_norm

        withdraw_palm = self.start_palm + self.approach_clearance * radial

        # Behind the target = opposite the travel direction.
        # A small z bias gives a shallow diagonal approach, not a vertical drop.
        approach_height = min(0.015, 0.30 * self.approach_clearance)
        pregrasp_palm = (
            self.goal_palm
            - self.approach_clearance * travel
            + np.array([0.0, 0.0, approach_height], dtype=float)
        )

        reference = d.qpos.copy()
        self.q_withdraw, r1 = solve_leg_ik(
            self.model, reference, c, leg=leg,
            target_palm=withdraw_palm, initial_leg_qpos=self.q_start
        )
        reference[c._leg_qpos_adrs[leg]] = self.q_withdraw
        self.q_pregrasp, r2 = solve_leg_ik(
            self.model, reference, c, leg=leg,
            target_palm=pregrasp_palm, initial_leg_qpos=self.q_withdraw
        )
        reference[c._leg_qpos_adrs[leg]] = self.q_pregrasp
        self.q_goal, r3 = solve_leg_ik(
            self.model, reference, c, leg=leg,
            target_palm=self.goal_palm, initial_leg_qpos=self.q_pregrasp
        )

        worst = max(r1, r2, r3)
        if worst > 0.02:
            raise RuntimeError(
                f"{leg} collision-aware waypoints not reachable enough; "
                f"worst residual={worst:.4f} m"
            )

        # Collision stays enabled.  The path itself avoids the rung.
        self._restore_leg_collision(leg)
        self.phase = f"OPEN_{leg}"
        self.phase_start = float(d.time)
        print(
            f"\nSTEP {self.completed_steps + 1}: {leg} "
            f"{self.source_rung} -> {self.target_rung} | "
            f"withdrawal={self.approach_clearance:.3f} m | "
            f"rear diagonal approach={self.approach_clearance:.3f} m | "
            f"IK residuals=({r1:.4f}, {r2:.4f}, {r3:.4f}) m"
        )

    def _hold_state(self):
        c, d = self.controller, self.data
        d.qpos[c._base_qpos_adr : c._base_qpos_adr + 7] = self.base_hold
        d.qvel[c._base_dof_adr : c._base_dof_adr + 6] = 0.0
        for leg in LEGS:
            d.qpos[c._leg_qpos_adrs[leg]] = self.held_leg_q[leg]
            d.qvel[c._leg_dof_adrs[leg]] = 0.0
            d.qpos[c._finger_qpos_adrs[leg]] = self.held_finger_q[leg]
            d.qvel[c._finger_dof_adrs[leg]] = 0.0

    def _finish_leg_step(self):
        leg = self.moving_leg
        c, d = self.controller, self.data
        self._restore_leg_collision(leg)
        self.held_leg_q[leg] = self.q_goal.copy()
        self.held_finger_q[leg] = self.source_finger_q.copy()
        self.completed_steps += 1
        target_contacts = count_leg_rung_contacts(
            self.model, d, c, leg, self.target_rung
        )
        print(
            f"{leg} step complete: target={self.target_rung}, "
            f"detected target contacts={target_contacts}"
        )

        self.leg_index += 1
        if self.leg_index < len(self.sequence):
            self._begin_leg_step()
        else:
            self._begin_body_shift()

    def _begin_body_shift(self):
        c, d = self.controller, self.data
        table = build_rung_table(self.model, d, c)
        spacing = float(np.median(np.diff([row[0] for row in table])))
        self.body_shift_start_base = self.base_hold.copy()
        self.body_shift_goal_base = self.base_hold.copy()
        self.body_shift_goal_base[0] += spacing
        self.body_shift_start_q = {leg: self.held_leg_q[leg].copy() for leg in LEGS}

        palm_targets = {
            leg: np.asarray(d.xpos[c._palm_body_ids[leg]], dtype=float).copy()
            for leg in LEGS
        }
        reference = d.qpos.copy()
        reference[c._base_qpos_adr : c._base_qpos_adr + 7] = self.body_shift_goal_base

        residuals = {}
        for leg in LEGS:
            q, residual = solve_leg_ik(
                self.model,
                reference,
                c,
                leg=leg,
                target_palm=palm_targets[leg],
                initial_leg_qpos=self.held_leg_q[leg],
            )
            self.body_shift_goal_q[leg] = q
            reference[c._leg_qpos_adrs[leg]] = q
            residuals[leg] = residual

        self.phase = "BODY_SHIFT"
        self.phase_start = float(d.time)
        print(
            f"\nBODY SHIFT: base +x by {spacing:.3f} m while assisted IK "
            f"holds palm world positions. residuals={residuals}"
        )

    def _finish_body_shift(self):
        self.base_hold = self.body_shift_goal_base.copy()
        for leg in LEGS:
            self.held_leg_q[leg] = self.body_shift_goal_q[leg].copy()
            self._restore_leg_collision(leg)

        self.cycle_index += 1
        print(f"CYCLE {self.cycle_index} COMPLETE")
        if self.cycle_index >= self.requested_cycles:
            self.phase = "COMPLETE"
            self.finished = True
            return

        self.leg_index = 0
        self._begin_leg_step()

    def apply(self):
        if not self.started or self.finished:
            return
        c, d = self.controller, self.data
        d.ctrl[:] = 0.0
        elapsed = float(d.time) - self.phase_start

        if self.phase == "BODY_SHIFT":
            alpha = elapsed / self.body_shift_time
            d.qpos[c._base_qpos_adr : c._base_qpos_adr + 7] = interpolate(
                self.body_shift_start_base, self.body_shift_goal_base, alpha
            )
            d.qvel[c._base_dof_adr : c._base_dof_adr + 6] = 0.0
            for leg in LEGS:
                d.qpos[c._leg_qpos_adrs[leg]] = interpolate(
                    self.body_shift_start_q[leg], self.body_shift_goal_q[leg], alpha
                )
                d.qvel[c._leg_dof_adrs[leg]] = 0.0
                d.qpos[c._finger_qpos_adrs[leg]] = self.held_finger_q[leg]
                d.qvel[c._finger_dof_adrs[leg]] = 0.0
            mujoco.mj_forward(self.model, d)
            if elapsed >= self.body_shift_time:
                self._finish_body_shift()
            return

        self._hold_state()
        leg = self.moving_leg
        t1 = self.open_time
        t2 = t1 + self.lift_time
        t3 = t2 + self.transfer_time
        t4 = t3 + self.lower_time
        t5 = t4 + self.close_time
        t6 = t5 + self.hold_time

        if elapsed < t1:
            self.phase = f"OPEN_{leg}"
            qleg = self.q_start
            qfinger = interpolate(
                self.source_finger_q, c.open_finger_pose, elapsed / self.open_time
            )
        elif elapsed < t2:
            self.phase = f"WITHDRAW_{leg}"
            qleg = interpolate(
                self.q_start, self.q_withdraw, (elapsed - t1) / self.lift_time
            )
            qfinger = c.open_finger_pose
        elif elapsed < t3:
            self.phase = f"TRANSFER_{leg}"
            qleg = interpolate(
                self.q_withdraw, self.q_pregrasp, (elapsed - t2) / self.transfer_time
            )
            qfinger = c.open_finger_pose
        elif elapsed < t4:
            self.phase = f"APPROACH_{leg}"
            qleg = interpolate(
                self.q_pregrasp, self.q_goal, (elapsed - t3) / self.lower_time
            )
            qfinger = c.open_finger_pose
        elif elapsed < t5:
            self.phase = f"CLOSE_{leg}"
            qleg = self.q_goal
            qfinger = interpolate(
                c.open_finger_pose,
                self.source_finger_q,
                (elapsed - t4) / self.close_time,
            )
        elif elapsed < t6:
            self.phase = f"HOLD_{leg}"
            qleg = self.q_goal
            qfinger = self.source_finger_q
        else:
            self._finish_leg_step()
            return

        d.qpos[c._leg_qpos_adrs[leg]] = qleg
        d.qvel[c._leg_dof_adrs[leg]] = 0.0
        d.qpos[c._finger_qpos_adrs[leg]] = qfinger
        d.qvel[c._finger_dof_adrs[leg]] = 0.0
        mujoco.mj_forward(self.model, d)

    def post_step(self):
        if self.started and not self.finished:
            self.apply()

    def report(self):
        c, d = self.controller, self.data
        base = d.qpos[c._base_qpos_adr : c._base_qpos_adr + 3]
        if self.moving_leg:
            palm = d.xpos[c._palm_body_ids[self.moving_leg]]
            print(
                f"t={d.time:7.3f} | cycle={self.cycle_index + 1}/"
                f"{self.requested_cycles} | phase={self.phase:14s} | "
                f"leg={self.moving_leg} | palm=({palm[0]:+.3f},"
                f"{palm[1]:+.3f},{palm[2]:+.3f}) | "
                f"base=({base[0]:+.3f},{base[1]:+.3f},{base[2]:+.3f})"
            )

    def final_result(self):
        self._restore_all_collisions()
        print("\n" + "=" * 72)
        print("ASSISTED CYCLIC CRAWL RESULT")
        print("=" * 72)
        print(f"Final phase:       {self.phase}")
        print(f"Cycles completed:  {self.cycle_index}")
        print(f"Leg steps:         {self.completed_steps}")
        contacts = self.controller._detect_contacts(self.model, self.data)
        print(f"Contacting legs:   {sorted(contacts.get('grippers', []))}")
        print(f"Physical contacts: {contacts.get('physical_contact_count', 0)}")
        print("=" * 72)


def parse_args():
    parser = argparse.ArgumentParser(description="Rear-approach assisted cyclic ladder crawler")
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--sequence", default="RR,RL,FR,FL")
    parser.add_argument("--approach-clearance", type=float, default=0.055)
    parser.add_argument("--open-time", type=float, default=1.5)
    parser.add_argument("--lift-time", type=float, default=2.0)
    parser.add_argument("--transfer-time", type=float, default=3.0)
    parser.add_argument("--lower-time", type=float, default=2.0)
    parser.add_argument("--close-time", type=float, default=1.5)
    parser.add_argument("--hold-time", type=float, default=1.0)
    parser.add_argument("--body-shift-time", type=float, default=3.0)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--status-rate", type=float, default=2.0)
    parser.add_argument("--real-time", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--hold-viewer-on-exit", action="store_true")
    return parser.parse_args()


def run(args):
    scene = args.scene.expanduser().resolve()
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    model.opt.gravity[:] = [0.0, 0.0, 0.0]

    static = StaticLadderGraspController(model)
    static.reset(model, data)

    sequence = tuple(part.strip().upper() for part in args.sequence.split(","))
    if sorted(sequence) != sorted(LEGS):
        raise ValueError("--sequence must contain FL, FR, RL and RR exactly once.")

    crawl = AssistedCyclicCrawler(
        model, data, static,
        sequence=sequence,
        cycles=max(1, args.cycles),
        approach_clearance=args.approach_clearance,
        open_time=args.open_time,
        lift_time=args.lift_time,
        transfer_time=args.transfer_time,
        lower_time=args.lower_time,
        close_time=args.close_time,
        hold_time=args.hold_time,
        body_shift_time=args.body_shift_time,
    )

    start_wall = time.perf_counter()
    next_report = 0.0

    def step_once():
        nonlocal next_report
        if not crawl.started:
            static.update(model, data)
            mujoco.mj_step(model, data)
            static.post_step(model, data)
            status = static.status
            if status.state == "HOLDING":
                print(
                    f"\nStatic grasp confirmed at t={data.time:.3f} s: "
                    f"contacts={status.physical_contacts}, "
                    f"grippers={status.distinct_grippers}. "
                    "Starting rear-approach assisted crawl."
                )
                crawl.begin()
        else:
            crawl.apply()
            mujoco.mj_step(model, data)
            crawl.post_step()

        if data.time >= next_report:
            if crawl.started:
                crawl.report()
            next_report = data.time + 1.0 / max(args.status_rate, 0.1)

        if args.real_time:
            target = start_wall + data.time
            delay = target - time.perf_counter()
            if delay > 0:
                time.sleep(delay)

    if args.headless:
        while data.time < args.duration and not crawl.finished:
            step_once()
    else:
        from mujoco import viewer as mj_viewer
        with mj_viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and data.time < args.duration and not crawl.finished:
                step_once()
                viewer.sync()
            if args.hold_viewer_on_exit:
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(0.02)

    crawl.final_result()


def main():
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
