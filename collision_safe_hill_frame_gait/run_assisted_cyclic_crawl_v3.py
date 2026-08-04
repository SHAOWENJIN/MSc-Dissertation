#!/usr/bin/env python3
"""
Assisted rear-approach cyclic ladder crawler for the modified Unitree Go2.

This runner extends the successful assisted single-RR step to all four legs.

One cycle:
    RR -> RL -> FR -> FL -> assisted body shift

For each leg:
1. keep the free-floating robot attached through the other verified grasps;
2. hold the other three legs and grippers at their measured poses;
3. keep collision enabled throughout the leg motion;
4. open the moving gripper;
5. withdraw radially away from the source rung;
6. transfer through a collision-free outside corridor;
7. approach the target rung from the outside and close.

After all four legs move one rung forward, the base is shifted forward by one
rung spacing while all four palm world positions are held by assisted IK.

The robot is a free dynamical body and the truss/ladder is fixed to the world
space-structure frame. Runtime leg and finger motion is actuator-torque driven;
qpos is only changed in scratch data used for IK and collision queries. This
version uses an idealized mechanical latch after physical contact is verified;
the moving latch is released before the gripper moves. This
remains a trajectory-assisted development controller, not an autonomous-
climbing validation result.
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
DEFAULT_SCENE = PROJECT_ROOT / "scene_hill_frame.xml"
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


def moving_collision_geom_ids_for_leg(model, leg: str) -> np.ndarray:
    root_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_hip"
    )
    if root_id < 0:
        raise RuntimeError(f"Body not found: {leg}_hip")

    def descends_from_root(body_id: int) -> bool:
        current = int(body_id)
        while current > 0:
            if current == root_id:
                return True
            current = int(model.body_parentid[current])
        return False

    ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if descends_from_root(int(model.geom_bodyid[geom_id]))
        and (
            int(model.geom_contype[geom_id]) != 0
            or int(model.geom_conaffinity[geom_id]) != 0
        )
    ]
    if not ids:
        raise RuntimeError(f"No active collision geometry found for {leg}.")
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
        self.held_wrist_q = {}
        self.geom_ids = {leg: finger_geom_ids_for_leg(model, controller, leg) for leg in LEGS}
        self.collision_geom_ids = {
            leg: moving_collision_geom_ids_for_leg(model, leg) for leg in LEGS
        }
        self.saved_contype = {leg: model.geom_contype[self.geom_ids[leg]].copy() for leg in LEGS}
        self.saved_conaffinity = {
            leg: model.geom_conaffinity[self.geom_ids[leg]].copy() for leg in LEGS
        }
        self.disabled = set()
        self.clearance_data = mujoco.MjData(model)
        self.minimum_path_clearance = 0.012
        # Front calves sweep closer to their source rungs than the rear links,
        # so they need a larger robust corridor around the nominal path.
        self.minimum_rear_link_clearance = 0.002
        self.minimum_front_link_clearance = 0.002
        # Use a wider aperture only for the moving gripper.  Static acquisition
        # starts closer to the rung and uses the controller's ordinary open
        # pose; transfer needs the extra retraction so neither hook blocks palm
        # centring at the next rung.
        self.transfer_open_pose = controller.open_finger_pose.copy()
        self.transfer_open_pose[:] = [-0.60, 0.0, 0.0, -0.60, 0.0, 0.0]

        self.moving_leg = ""
        self.source_rung = ""
        self.target_rung = ""
        self.source_finger_q = np.zeros(6)
        self.source_wrist_q = 0.0
        self.q_start = np.zeros(3)
        self.q_withdraw = np.zeros(3)
        self.q_pregrasp = np.zeros(3)
        self.q_goal = np.zeros(3)
        self.q_close_center = np.zeros(3)
        self.start_palm = np.zeros(3)
        self.goal_palm = np.zeros(3)
        self.travel_direction = np.zeros(3)
        self.motion_path: list[np.ndarray] = []
        self.motion_path_clearance = float("nan")
        self.motion_progress = 0.0
        self.motion_actual_progress = 0.0
        self.motion_last_time = 0.0
        self.wrist_progress_paused = False
        self.close_start_time: float | None = None
        self.regrasp_wrap_start: float | None = None
        self.regrasp_finger_hold: np.ndarray | None = None
        self.regrasp_leg_hold: np.ndarray | None = None
        self.regrasp_wrist_hold: float | None = None
        self.last_two_sided_diagnostic_time = -float("inf")
        # Configuration-space safety is only meaningful if the real joints stay
        # close to the collision-checked path. Integral action removes the
        # steady-state error created by the other three loaded grippers.
        self.maximum_path_tracking_error = 0.500
        self.maximum_wrist_tracking_error = 0.200
        self.motion_lookahead = 0.015
        # This is only the open-pose handoff into contact-guided closure.  A
        # small abduction residual moves the gripper along the rung axis and
        # does not compromise the checked x-z approach corridor.  Closure is
        # still accepted only after real two-sided contact and enclosure.
        self.maximum_goal_tracking_error = 0.050
        self.leg_integral_gain = 40.0
        self.leg_integral_limit = 0.50
        self.leg_integral_error = {
            leg: np.zeros(3, dtype=float) for leg in LEGS
        }
        self.leg_integral_time = {
            leg: float(data.time) for leg in LEGS
        }
        self.wrist_integral_error = {leg: 0.0 for leg in LEGS}
        self.wrist_integral_time = {leg: float(data.time) for leg in LEGS}

        self.body_shift_start_base = np.zeros(7)
        self.body_shift_goal_base = np.zeros(7)
        self.body_shift_start_q = {}
        self.body_shift_goal_q = {}
        self.body_shift_start_wrist = {}
        self.body_shift_goal_wrist = {}
        self.body_shift_finger_q = {}
        self.body_shift_wrap_start: float | None = None

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
            self.held_wrist_q[leg] = float(
                d.qpos[c._wrist_qpos_adrs[leg]]
            )

    def _wrist_for_leg_q(self, leg: str, qleg: np.ndarray) -> float:
        """Preserve gripper pitch while the thigh/calf configuration changes."""
        c = self.controller
        target = self.source_wrist_q + float(
            np.sum(self.q_start[1:]) - np.sum(np.asarray(qleg)[1:])
        )
        joint_id = c._wrist_joint_ids[leg]
        lower, upper = self.model.jnt_range[joint_id]
        return float(np.clip(target, lower, upper))

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

    def _minimum_finger_rung_distance(
        self,
        leg: str,
        qleg: np.ndarray,
        finger_pose: np.ndarray,
        *,
        rung_ids: tuple[int, ...] | None = None,
        finger_only: bool = False,
        nonfinger_only: bool = False,
    ) -> float:
        """Return signed finger-to-rung clearance for a candidate configuration."""
        c, scratch = self.controller, self.clearance_data
        scratch.qpos[:] = self.data.qpos
        scratch.qpos[c._leg_qpos_adrs[leg]] = qleg
        scratch.qpos[c._wrist_qpos_adrs[leg]] = self._wrist_for_leg_q(
            leg, qleg
        )
        scratch.qpos[c._finger_qpos_adrs[leg]] = finger_pose
        mujoco.mj_forward(self.model, scratch)

        obstacles = rung_ids or tuple(int(x) for x in c._rung_geom_ids)
        minimum = float("inf")
        if finger_only and nonfinger_only:
            raise ValueError("finger_only and nonfinger_only are mutually exclusive")
        if finger_only:
            candidate_geom_ids = self.geom_ids[leg]
        elif nonfinger_only:
            finger_ids = set(int(item) for item in self.geom_ids[leg])
            candidate_geom_ids = np.asarray(
                [
                    item
                    for item in self.collision_geom_ids[leg]
                    if int(item) not in finger_ids
                ],
                dtype=int,
            )
        else:
            candidate_geom_ids = self.collision_geom_ids[leg]
        for finger_id in candidate_geom_ids:
            for rung_id in obstacles:
                minimum = min(
                    minimum,
                    float(
                        mujoco.mj_geomDistance(
                            self.model,
                            scratch,
                            int(finger_id),
                            int(rung_id),
                            1.0,
                            None,
                        )
                    ),
                )
        return minimum

    def _required_link_clearance(self, leg: str) -> float:
        return (
            self.minimum_front_link_clearance
            if leg.startswith("F")
            else self.minimum_rear_link_clearance
        )

    def _predicted_target_contact_sides(
        self,
        leg: str,
        qleg: np.ndarray,
        finger_pose: np.ndarray,
        target_rung_id: int,
    ) -> set[str]:
        c, scratch = self.controller, self.clearance_data
        scratch.qpos[:] = self.data.qpos
        scratch.qpos[c._leg_qpos_adrs[leg]] = qleg
        scratch.qpos[c._wrist_qpos_adrs[leg]] = self._wrist_for_leg_q(
            leg, qleg
        )
        scratch.qpos[c._finger_qpos_adrs[leg]] = finger_pose
        mujoco.mj_forward(self.model, scratch)

        sides: set[str] = set()
        for finger_id in self.geom_ids[leg]:
            distance = float(
                mujoco.mj_geomDistance(
                    self.model,
                    scratch,
                    int(finger_id),
                    int(target_rung_id),
                    1.0,
                    None,
                )
            )
            if distance > 0.001:
                continue
            body_id = int(self.model.geom_bodyid[finger_id])
            body_name = c._finger_body_to_name.get(body_id, "")
            if "_finger_L" in body_name:
                sides.add("L")
            elif "_finger_R" in body_name:
                sides.add("R")
        return sides

    def _choose_collision_free_goal(
        self,
        leg: str,
        nominal_goal: np.ndarray,
        travel: np.ndarray,
        target_rung_id: int,
        reference_qpos: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Find an open pose that can close onto both sides of the target rung."""
        c = self.controller
        candidates = []
        # Search on both sides of the source-translated palm pose.  A strictly
        # positive standoff search can leave the palm behind the rung: the
        # leading hook touches first and the opposite hook can never close.
        # Small negative values move the palm forward and let the controller
        # centre the rung between the two hooks before closure.
        standoffs = np.linspace(-0.040, 0.050, 37)
        height_offsets = tuple(np.linspace(-0.030, 0.030, 13))

        for height in height_offsets:
            for standoff in standoffs:
                palm = (
                    nominal_goal
                    - float(standoff) * travel
                    + np.array([0.0, 0.0, height], dtype=float)
                )
                qleg, residual = solve_leg_ik(
                    self.model,
                    reference_qpos,
                    c,
                    leg=leg,
                    target_palm=palm,
                    initial_leg_qpos=self.q_start,
                )
                if residual > 0.005:
                    continue

                open_clearance = self._minimum_finger_rung_distance(
                    leg, qleg, self.transfer_open_pose, finger_only=True
                )
                if open_clearance < self.minimum_path_clearance:
                    continue
                link_clearance = self._minimum_finger_rung_distance(
                    leg, qleg, self.transfer_open_pose, nonfinger_only=True
                )
                if link_clearance < self._required_link_clearance(leg):
                    continue

                sides = self._predicted_target_contact_sides(
                    leg,
                    qleg,
                    self.source_finger_q,
                    target_rung_id,
                )
                if sides != {"L", "R"}:
                    continue

                closed_distance = self._minimum_finger_rung_distance(
                    leg,
                    qleg,
                    self.source_finger_q,
                    rung_ids=(target_rung_id,),
                    finger_only=True,
                )
                # This is only a reachability test for a torque-controlled
                # close. The runtime never teleports to this overlapping pose;
                # MuJoCo contact stops the fingers before penetration.
                if closed_distance < -0.040:
                    continue

                # The two hook roots are symmetric about the palm.  Prefer the
                # actual rung-centred pose; a forward bias places both visible
                # hooks beyond the rod and creates a false-looking closure.
                candidates.append(
                    (
                        -abs(float(standoff)),
                        -abs(float(height)),
                        -abs(closed_distance + 0.0005),
                        open_clearance,
                        palm,
                        qleg,
                        closed_distance,
                    )
                )

        if not candidates:
            raise RuntimeError(
                f"No collision-free two-sided target pose found for {leg} "
                f"at {self.target_rung}."
            )

        _, _, _, open_clearance, palm, qleg, closed_distance = max(
            candidates, key=lambda item: item[:4]
        )
        return qleg.copy(), palm.copy(), open_clearance, closed_distance

    def _segment_is_clear(
        self,
        leg: str,
        start: np.ndarray,
        end: np.ndarray,
    ) -> bool:
        distance = float(np.linalg.norm(np.asarray(end) - np.asarray(start)))
        sample_count = max(2, int(np.ceil(distance / 0.025)) + 1)
        for alpha in np.linspace(0.0, 1.0, sample_count):
            qleg = interpolate(start, end, alpha)
            finger_clearance = self._minimum_finger_rung_distance(
                leg,
                qleg,
                self.transfer_open_pose,
                finger_only=True,
            )
            link_clearance = self._minimum_finger_rung_distance(
                leg,
                qleg,
                self.transfer_open_pose,
                nonfinger_only=True,
            )
            if (
                finger_clearance < self.minimum_path_clearance
                or link_clearance < self._required_link_clearance(leg)
            ):
                return False
        return True

    def _plan_collision_free_path(
        self,
        leg: str,
        start: np.ndarray,
        goal: np.ndarray,
    ) -> list[np.ndarray]:
        """Plan a deterministic RRT path through collision-free joint space."""
        if self._segment_is_clear(leg, start, goal):
            return [start.copy(), goal.copy()]

        # Prefer a readable Cartesian lift-transfer-lower gait over a random
        # joint-space detour.  With the hook fully open, lifting the palm first
        # clears the source rung, translating moves to the next rung, and the
        # final descent approaches it from above.
        c = self.controller
        reference = self.data.qpos.copy()
        scratch = self.clearance_data
        scratch.qpos[:] = reference
        scratch.qpos[c._leg_qpos_adrs[leg]] = goal
        scratch.qpos[c._wrist_qpos_adrs[leg]] = self._wrist_for_leg_q(leg, goal)
        scratch.qpos[c._finger_qpos_adrs[leg]] = self.transfer_open_pose
        mujoco.mj_forward(self.model, scratch)
        start_palm = np.asarray(
            self.data.xpos[c._palm_body_ids[leg]], dtype=float
        ).copy()
        goal_palm = np.asarray(
            scratch.xpos[c._palm_body_ids[leg]], dtype=float
        ).copy()
        for lift in np.linspace(0.04, 0.20, 17):
            lifted_start, start_residual = solve_leg_ik(
                self.model,
                reference,
                c,
                leg=leg,
                target_palm=start_palm + np.array([0.0, 0.0, lift]),
                initial_leg_qpos=start,
            )
            lifted_goal, goal_residual = solve_leg_ik(
                self.model,
                reference,
                c,
                leg=leg,
                target_palm=goal_palm + np.array([0.0, 0.0, lift]),
                initial_leg_qpos=goal,
            )
            candidate_path = [start, lifted_start, lifted_goal, goal]
            if (
                start_residual <= 0.005
                and goal_residual <= 0.005
                and all(
                    self._segment_is_clear(leg, a, b)
                    for a, b in zip(candidate_path, candidate_path[1:])
                )
            ):
                return [node.copy() for node in candidate_path]

        limits = []
        for dof_id in c._leg_dof_adrs[leg]:
            joint_id = int(self.model.dof_jntid[dof_id])
            limits.append(np.asarray(self.model.jnt_range[joint_id], dtype=float))
        limits = np.asarray(limits)
        # Rungs are arranged along world x.  A full-range sample of the
        # abduction joint can produce a technically collision-free but visibly
        # twisted sideways leg sweep.  Keep transfer in the sagittal plane,
        # with only a small lateral allowance for real clearance.
        lateral_centre = 0.5 * float(start[0] + goal[0])
        limits[0, 0] = max(limits[0, 0], lateral_centre - 0.18)
        limits[0, 1] = min(limits[0, 1], lateral_centre + 0.18)

        seed = 1009 * (self.completed_steps + 1) + sum(ord(char) for char in leg)
        rng = np.random.default_rng(seed)
        nodes = [start.copy()]
        parents = [-1]
        goal_index = None

        for _ in range(10000):
            sample = (
                goal
                if rng.random() < 0.25
                else rng.uniform(limits[:, 0], limits[:, 1])
            )
            nearest = min(
                range(len(nodes)),
                key=lambda index: float(np.linalg.norm(nodes[index] - sample)),
            )
            delta = sample - nodes[nearest]
            distance = float(np.linalg.norm(delta))
            candidate = (
                sample
                if distance <= 0.12
                else nodes[nearest] + 0.12 * delta / distance
            )
            if not self._segment_is_clear(leg, nodes[nearest], candidate):
                continue

            nodes.append(candidate.copy())
            parents.append(nearest)
            candidate_index = len(nodes) - 1
            if self._segment_is_clear(leg, candidate, goal):
                nodes.append(goal.copy())
                parents.append(candidate_index)
                goal_index = len(nodes) - 1
                break

        if goal_index is None:
            raise RuntimeError(
                f"Could not find a collision-free open-gripper path for {leg}."
            )

        path = []
        index = goal_index
        while index >= 0:
            path.append(nodes[index])
            index = parents[index]
        path.reverse()

        # Deterministically remove unnecessary RRT vertices while rechecking
        # every shortcut against the same signed-clearance constraint.
        shortened = [path[0]]
        index = 0
        while index < len(path) - 1:
            next_index = len(path) - 1
            while (
                next_index > index + 1
                and not self._segment_is_clear(leg, path[index], path[next_index])
            ):
                next_index -= 1
            shortened.append(path[next_index])
            index = next_index
        return [q.copy() for q in shortened]

    def _sample_motion_path(self, alpha: float) -> np.ndarray:
        if len(self.motion_path) == 1:
            return self.motion_path[0].copy()
        segment_lengths = np.asarray(
            [
                np.linalg.norm(end - start)
                for start, end in zip(self.motion_path, self.motion_path[1:])
            ],
            dtype=float,
        )
        total = float(np.sum(segment_lengths))
        if total <= 1e-12:
            return self.motion_path[-1].copy()

        target = smoothstep(alpha) * total
        traversed = 0.0
        for index, length in enumerate(segment_lengths):
            if target <= traversed + length or index == len(segment_lengths) - 1:
                local = (target - traversed) / max(float(length), 1e-12)
                return (
                    self.motion_path[index]
                    + np.clip(local, 0.0, 1.0)
                    * (self.motion_path[index + 1] - self.motion_path[index])
                )
            traversed += float(length)
        return self.motion_path[-1].copy()

    def _project_onto_motion_path(self, qleg: np.ndarray) -> tuple[float, float]:
        """Return monotonic nearest path progress and joint-space distance.

        The leg is torque controlled and the robot base is free floating, so a
        purely time-driven reference can run far ahead of the real mechanism.
        Projecting the measured joints onto the already collision-checked path
        lets the controller use a bounded look-ahead without inventing a
        kinematic teleport through a rung.
        """
        lengths = np.asarray(
            [
                np.linalg.norm(end - start)
                for start, end in zip(self.motion_path, self.motion_path[1:])
            ],
            dtype=float,
        )
        total = float(np.sum(lengths))
        if total <= 1e-12:
            return 1.0, float(np.linalg.norm(qleg - self.motion_path[-1]))

        best_distance = float("inf")
        best_arc = 0.0
        traversed = 0.0
        for start, end, length in zip(
            self.motion_path, self.motion_path[1:], lengths
        ):
            delta = end - start
            local = float(
                np.clip(
                    np.dot(qleg - start, delta) / max(float(length * length), 1e-12),
                    0.0,
                    1.0,
                )
            )
            nearest = start + local * delta
            distance = float(np.linalg.norm(qleg - nearest))
            if distance < best_distance:
                best_distance = distance
                best_arc = traversed + local * float(length)
            traversed += float(length)

        # _sample_motion_path maps alpha through smoothstep before measuring
        # arc length. Invert that monotonic map with a cheap fixed bisection.
        arc_fraction = float(np.clip(best_arc / total, 0.0, 1.0))
        low, high = 0.0, 1.0
        for _ in range(20):
            middle = 0.5 * (low + high)
            if smoothstep(middle) < arc_fraction:
                low = middle
            else:
                high = middle
        nearest_progress = 0.5 * (low + high)
        projected = max(self.motion_actual_progress, nearest_progress)
        return projected, best_distance

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

        # Release only the moving gripper's idealized latch. The other
        # verified grasps keep the free robot attached to the fixed truss.
        c.set_grasp_weld(self.model, d, leg, False)
        self.regrasp_wrap_start = None
        self.regrasp_finger_hold = None
        self.regrasp_leg_hold = None
        self.regrasp_wrist_hold = None

        self.q_start = d.qpos[c._leg_qpos_adrs[leg]].copy()
        self.source_finger_q = d.qpos[c._finger_qpos_adrs[leg]].copy()
        self.source_wrist_q = float(d.qpos[c._wrist_qpos_adrs[leg]])
        self.wrist_integral_error[leg] = 0.0
        self.wrist_integral_time[leg] = float(d.time)
        self.start_palm = np.asarray(d.xpos[c._palm_body_ids[leg]], dtype=float).copy()
        nominal_goal = self.start_palm + translation

        travel = translation.copy()
        travel[1] = 0.0
        travel_norm = float(np.linalg.norm(travel))
        if travel_norm < 1e-8:
            raise RuntimeError("Source and target rung positions are identical.")
        travel /= travel_norm
        self.travel_direction = travel.copy()

        target_rung_id = int(target_row[1])
        reference = d.qpos.copy()
        (
            self.q_goal,
            self.goal_palm,
            open_clearance,
            predicted_closed_distance,
        ) = self._choose_collision_free_goal(
            leg,
            nominal_goal,
            travel,
            target_rung_id,
            reference,
        )
        # Close at the exact source-translated palm pose, which centres the
        # symmetric hooks over the target rung.  Physical contact remains
        # enabled throughout, and acquisition freezes immediately at the first
        # genuine two-sided wrap.
        close_center_palm = nominal_goal
        self.q_close_center, close_center_residual = solve_leg_ik(
            self.model,
            reference,
            c,
            leg=leg,
            target_palm=close_center_palm,
            initial_leg_qpos=self.q_goal,
        )
        if close_center_residual > 0.005:
            raise RuntimeError(
                f"Could not solve the contact-guided centring pose for {leg}; "
                f"IK residual={close_center_residual:.4f} m."
            )
        self.motion_path = self._plan_collision_free_path(
            leg, self.q_start, self.q_goal
        )
        self.q_withdraw = self._sample_motion_path(1.0 / 3.0)
        self.q_pregrasp = self._sample_motion_path(2.0 / 3.0)
        self.motion_path_clearance = min(
            self._minimum_finger_rung_distance(
                leg,
                interpolate(start, end, alpha),
                self.transfer_open_pose,
            )
            for start, end in zip(self.motion_path, self.motion_path[1:])
            for alpha in np.linspace(
                0.0,
                1.0,
                max(
                    2,
                    int(np.ceil(np.linalg.norm(end - start) / 0.025)) + 1,
                ),
            )
        )

        # Collision stays enabled.  The path itself avoids the rung.
        self._restore_leg_collision(leg)
        self.phase = f"OPEN_{leg}"
        self.phase_start = float(d.time)
        self.motion_progress = 0.0
        self.motion_actual_progress = 0.0
        self.motion_last_time = float(d.time) + self.open_time
        self.wrist_progress_paused = False
        self.close_start_time = None
        print(
            f"\nSTEP {self.completed_steps + 1}: {leg} "
            f"{self.source_rung} -> {self.target_rung} | "
            f"path_nodes={len(self.motion_path)} | "
            f"path_q={[np.round(node, 3).tolist() for node in self.motion_path]} | "
            f"minimum_open_clearance={self.motion_path_clearance:.4f} m | "
            f"goal_open_clearance={open_clearance:.4f} m | "
            f"predicted_closed_distance={predicted_closed_distance:.4f} m | "
            f"close_center={np.round(close_center_palm, 4).tolist()}"
        )

    def _hold_state(self, *, dynamic_finger_leg: str | None = None):
        c = self.controller
        # The robot base remains a true free body after grasp verification;
        # the verified contact-triggered latches attach it to the fixed truss.
        for leg in LEGS:
            if leg != dynamic_finger_leg:
                self._apply_leg_pd_target(
                    leg, self.held_leg_q[leg], integrate=False
                )
                self._apply_wrist_pd_target(leg, self.held_wrist_q[leg])
            if leg != dynamic_finger_leg:
                self._apply_moving_finger_pd(leg, self.held_finger_q[leg])

    def _apply_leg_pd_target(
        self, leg: str, target: np.ndarray, *, integrate: bool = True
    ):
        c, d = self.controller, self.data
        q = d.qpos[c._leg_qpos_adrs[leg]]
        qd = d.qvel[c._leg_dof_adrs[leg]]
        error = np.asarray(target) - q
        now = float(d.time)
        dt = max(0.0, now - self.leg_integral_time[leg])
        self.leg_integral_time[leg] = now
        if integrate and dt > 0.0:
            self.leg_integral_error[leg] = np.clip(
                self.leg_integral_error[leg] + dt * error,
                -self.leg_integral_limit,
                self.leg_integral_limit,
            )
        torque = (
            2.0 * c.leg_kp * error
            + self.leg_integral_gain * self.leg_integral_error[leg]
            - 2.0 * c.leg_kd * qd
        )
        c._write_clipped_control(d, c._leg_actuator_ids[leg], torque)

    def _apply_moving_finger_pd(self, leg: str, target: np.ndarray):
        c, d = self.controller, self.data
        q = d.qpos[c._finger_qpos_adrs[leg]]
        qd = d.qvel[c._finger_dof_adrs[leg]]
        torque = c.finger_kp * (np.asarray(target) - q) - c.finger_kd * qd
        c._write_clipped_control(d, c._finger_actuator_ids[leg], torque)

    def _apply_wrist_pd_target(
        self,
        leg: str,
        target: float,
        *,
        target_velocity: float = 0.0,
        integrate: bool = True,
    ):
        c, d = self.controller, self.data
        q = float(d.qpos[c._wrist_qpos_adrs[leg]])
        qd = float(d.qvel[c._wrist_dof_adrs[leg]])
        error = float(target) - q
        now = float(d.time)
        dt = max(0.0, now - self.wrist_integral_time[leg])
        self.wrist_integral_time[leg] = now
        if not integrate:
            self.wrist_integral_error[leg] = 0.0
        elif dt > 0.0:
            self.wrist_integral_error[leg] = float(
                np.clip(
                    self.wrist_integral_error[leg] + dt * error,
                    -0.20,
                    0.20,
                )
            )
        # The wrist counter-rotates the thigh and calf to preserve palm pitch.
        # Damping absolute wrist velocity fights that necessary motion and
        # creates a large transient orientation error.  Track the required
        # velocity as well as the position so the real gripper remains level
        # throughout a transfer.
        torque = (
            350.0 * error
            + 400.0 * self.wrist_integral_error[leg]
            + 8.0 * (float(target_velocity) - qd)
        )
        actuator_id = c._wrist_actuator_ids[leg]
        c._write_clipped_control(
            d,
            np.asarray([actuator_id], dtype=int),
            np.asarray([torque], dtype=float),
        )

    def _finish_leg_step(self):
        leg = self.moving_leg
        c, d = self.controller, self.data
        self._restore_leg_collision(leg)
        self.held_finger_q[leg] = d.qpos[c._finger_qpos_adrs[leg]].copy()
        target_contacts = count_leg_rung_contacts(
            self.model, d, c, leg, self.target_rung
        )
        contacts = c._detect_contacts(self.model, d)
        target_sides = {
            str(item["side"])
            for item in contacts.get("contact_details", [])
            if str(item.get("leg")) == leg
            and str(item.get("rung_geom_name")) == self.target_rung
        }
        geometric_wrap = c.has_geometric_wrap(contacts, leg, self.target_rung)
        if target_contacts <= 0 or target_sides != {"L", "R"} or not geometric_wrap:
            actual_finger_q = d.qpos[c._finger_qpos_adrs[leg]].copy()
            actual_distance = self._minimum_finger_rung_distance(
                leg,
                self.q_goal,
                actual_finger_q,
                rung_ids=(
                    mujoco.mj_name2id(
                        self.model,
                        mujoco.mjtObj.mjOBJ_GEOM,
                        self.target_rung,
                    ),
                ),
                finger_only=True,
            )
            raise RuntimeError(
                f"{leg} did not establish physical rung contact on "
                f"{self.target_rung}; contacts={target_contacts}, "
                f"sides={sorted(target_sides)}, "
                f"geometric_wrap={geometric_wrap}, "
                f"actual_target_distance={actual_distance:.4f} m, "
                f"finger_q={actual_finger_q.tolist()}."
            )
        c.set_grasp_weld(self.model, d, leg, True)
        self.held_leg_q[leg] = d.qpos[c._leg_qpos_adrs[leg]].copy()
        self.held_wrist_q[leg] = float(d.qpos[c._wrist_qpos_adrs[leg]])
        self.completed_steps += 1
        print(
            f"{leg} step complete: target={self.target_rung}, "
            f"detected target contacts={target_contacts}, "
            f"sides={sorted(target_sides)}, geometric_wrap={geometric_wrap}"
        )

        self.leg_index += 1
        if self.leg_index < len(self.sequence):
            self._begin_leg_step()
        elif len(self.sequence) < len(LEGS):
            self.cycle_index += 1
            print(f"GRASP GAIT CYCLE {self.cycle_index} COMPLETE")
            if self.cycle_index >= self.requested_cycles:
                self.phase = "COMPLETE"
                self.finished = True
            else:
                self.leg_index = 0
                self._begin_leg_step()
        else:
            self._begin_body_shift()

    def _begin_body_shift(self):
        c, d = self.controller, self.data
        table = build_rung_table(self.model, d, c)
        spacing = float(np.median(np.diff([row[0] for row in table])))
        self.body_shift_start_base = d.qpos[
            c._base_qpos_adr : c._base_qpos_adr + 7
        ].copy()
        self.body_shift_goal_base = self.body_shift_start_base.copy()
        self.body_shift_goal_base[0] += spacing
        self.body_shift_start_q = {leg: self.held_leg_q[leg].copy() for leg in LEGS}
        self.body_shift_start_wrist = {
            leg: float(self.held_wrist_q[leg]) for leg in LEGS
        }
        self.body_shift_finger_q = {}
        for leg in LEGS:
            preload = self.held_finger_q[leg].copy()
            preload[[0, 3]] = np.minimum(
                preload[[0, 3]] + 0.04,
                c.closed_finger_pose[[0, 3]],
            )
            self.body_shift_finger_q[leg] = preload
        self.body_shift_wrap_start = None

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
            wrist = self.body_shift_start_wrist[leg] + float(
                np.sum(self.body_shift_start_q[leg][1:]) - np.sum(q[1:])
            )
            wrist_joint_id = c._wrist_joint_ids[leg]
            self.body_shift_goal_wrist[leg] = float(
                np.clip(wrist, *self.model.jnt_range[wrist_joint_id])
            )
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
            self.held_wrist_q[leg] = self.body_shift_goal_wrist[leg]
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
            for leg in LEGS:
                self._apply_leg_pd_target(
                    leg,
                    interpolate(
                        self.body_shift_start_q[leg],
                        self.body_shift_goal_q[leg],
                        alpha,
                    ),
                    integrate=True,
                )
                self._apply_wrist_pd_target(
                    leg,
                    float(
                        interpolate(
                            np.asarray([self.body_shift_start_wrist[leg]]),
                            np.asarray([self.body_shift_goal_wrist[leg]]),
                            alpha,
                        )[0]
                    ),
                )
                self._apply_moving_finger_pd(leg, self.body_shift_finger_q[leg])
            if elapsed >= self.body_shift_time:
                contacts = c._detect_contacts(self.model, d)
                wraps = set(contacts.get("geometrically_wrapped_grippers", set()))
                base_error = abs(
                    float(d.qpos[c._base_qpos_adr])
                    - float(self.body_shift_goal_base[0])
                )
                if wraps == set(LEGS) and base_error <= 0.020:
                    if self.body_shift_wrap_start is None:
                        self.body_shift_wrap_start = float(d.time)
                    elif float(d.time) - self.body_shift_wrap_start >= 0.30:
                        for held_leg in LEGS:
                            self.held_finger_q[held_leg] = d.qpos[
                                c._finger_qpos_adrs[held_leg]
                            ].copy()
                        self._finish_body_shift()
                else:
                    self.body_shift_wrap_start = None
                if elapsed > self.body_shift_time + 30.0:
                    raise RuntimeError(
                        "Body shift failed to converge with four real wraps; "
                        f"base_error={base_error:.3f} m, wraps={sorted(wraps)}."
                    )
            return

        leg = self.moving_leg
        self._hold_state(dynamic_finger_leg=leg)
        motion_duration = self.lift_time + self.transfer_time + self.lower_time
        pause_for_wrist = False

        if elapsed < self.open_time:
            self.phase = f"OPEN_{leg}"
            qleg = self.q_start
            finger_target = self.transfer_open_pose
        else:
            actual_q = d.qpos[c._leg_qpos_adrs[leg]].copy()
            if self.close_start_time is None:
                now = float(d.time)
                dt = max(0.0, now - self.motion_last_time)
                self.motion_last_time = now
                projected_progress, path_tracking_error = (
                    self._project_onto_motion_path(actual_q)
                )
                self.motion_actual_progress = projected_progress
                actual_wrist = float(d.qpos[c._wrist_qpos_adrs[leg]])
                wrist_tracking_error = abs(
                    actual_wrist
                    - self._wrist_for_leg_q(leg, actual_q)
                )
                if wrist_tracking_error > 0.120:
                    self.wrist_progress_paused = True
                elif wrist_tracking_error < 0.060:
                    self.wrist_progress_paused = False
                if (
                    path_tracking_error > self.maximum_path_tracking_error
                    or wrist_tracking_error > self.maximum_wrist_tracking_error
                ):
                    wrist_reference = self._wrist_for_leg_q(leg, actual_q)
                    wrist_control = float(d.ctrl[c._wrist_actuator_ids[leg]])
                    raise RuntimeError(
                        f"{leg} left the collision-checked tracking tube; "
                        f"path_error={path_tracking_error:.3f} rad, "
                        f"wrist_error={wrist_tracking_error:.3f} rad "
                        f"(actual={actual_wrist:.3f}, "
                        f"reference={wrist_reference:.3f}, "
                        f"control={wrist_control:.1f})."
                    )
                pause_for_wrist = self.wrist_progress_paused
                time_limited_progress = self.motion_progress
                if not pause_for_wrist:
                    time_limited_progress += dt / max(motion_duration, 1e-12)
                self.motion_progress = min(
                    1.0,
                    time_limited_progress,
                )

                qleg = (
                    actual_q.copy()
                    if pause_for_wrist
                    else self._sample_motion_path(self.motion_progress)
                )
                finger_target = self.transfer_open_pose
                if self.motion_progress < 1.0 / 3.0:
                    self.phase = f"WITHDRAW_{leg}"
                elif self.motion_progress < 2.0 / 3.0:
                    self.phase = f"TRANSFER_{leg}"
                else:
                    self.phase = f"APPROACH_{leg}"

                if (
                    self.motion_progress >= 1.0
                    and np.linalg.norm(self.q_goal - actual_q)
                    <= self.maximum_goal_tracking_error
                ):
                    self.close_start_time = now
                elif self.motion_progress >= 0.90:
                    # Treat the first target-rung touch as a tactile pregrasp
                    # event.  Do not keep driving the open leading hook into
                    # the rung while waiting for an unreachable nominal joint
                    # tolerance; freeze the actual arm pose, curl the hooks,
                    # and advance the palm under contact guidance.
                    contacts = c._detect_contacts(self.model, d)
                    target_touch = any(
                        str(item.get("leg")) == leg
                        and str(item.get("rung_geom_name")) == self.target_rung
                        for item in contacts.get("contact_details", [])
                    )
                    if target_touch:
                        self.q_goal = actual_q.copy()
                        actual_palm = np.asarray(
                            d.xpos[c._palm_body_ids[leg]], dtype=float
                        ).copy()
                        centre_palm = (
                            actual_palm + 0.060 * self.travel_direction
                        )
                        self.q_close_center, residual = solve_leg_ik(
                            self.model,
                            d.qpos.copy(),
                            c,
                            leg=leg,
                            target_palm=centre_palm,
                            initial_leg_qpos=actual_q,
                        )
                        if residual > 0.005:
                            raise RuntimeError(
                                f"{leg} tactile pregrasp centring IK failed; "
                                f"residual={residual:.4f} m."
                            )
                        self.close_start_time = now
                elif elapsed > self.open_time + 8.0 * motion_duration + 30.0:
                    raise RuntimeError(
                        f"{leg} failed to reach its collision-checked regrasp "
                        "goal before timeout; refusing to close or latch."
                    )
            else:
                close_elapsed = float(d.time) - self.close_start_time
                close_progress = float(
                    np.clip(close_elapsed / max(self.close_time, 1e-12), 0.0, 1.0)
                )
                # Curl first, then advance the palm.  This lets the leading
                # hook guide around the rung instead of blocking the open palm
                # short of centre.
                centre_alpha = smoothstep(
                    np.clip((close_progress - 0.20) / 0.65, 0.0, 1.0)
                )
                qleg = interpolate(
                    self.q_goal, self.q_close_center, centre_alpha
                )
                contacts = c._detect_contacts(self.model, d)
                geometric_wrap = c.has_geometric_wrap(
                    contacts, leg, self.target_rung
                )
                target_sides = {
                    str(item.get("side"))
                    for item in contacts.get("contact_details", [])
                    if str(item.get("leg")) == leg
                    and str(item.get("rung_geom_name")) == self.target_rung
                }
                if (
                    target_sides == {"L", "R"}
                    and not geometric_wrap
                    and float(d.time) - self.last_two_sided_diagnostic_time >= 0.5
                ):
                    relative_x = [
                        (
                            str(item.get("side")),
                            round(float(item.get("relative_x", 0.0)), 5),
                            str(item.get("segment")),
                        )
                        for item in contacts.get("contact_details", [])
                        if str(item.get("leg")) == leg
                        and str(item.get("rung_geom_name")) == self.target_rung
                    ]
                    print(
                        f"{leg} two-sided touch is not yet a wrap: "
                        f"relative_x={relative_x}"
                    )
                    self.last_two_sided_diagnostic_time = float(d.time)
                if geometric_wrap:
                    if self.regrasp_wrap_start is None:
                        self.regrasp_wrap_start = float(d.time)
                        hold = d.qpos[c._finger_qpos_adrs[leg]].copy()
                        hold[[0, 3]] = np.minimum(
                            # Only a light preload is needed after contact.
                            # The previous 0.06 rad increment could rotate a
                            # newly wrapped hook back off the opposite face.
                            hold[[0, 3]] + 0.015,
                            c.closed_finger_pose[[0, 3]],
                        )
                        self.regrasp_finger_hold = hold
                        self.regrasp_leg_hold = d.qpos[
                            c._leg_qpos_adrs[leg]
                        ].copy()
                        self.regrasp_wrist_hold = float(
                            d.qpos[c._wrist_qpos_adrs[leg]]
                        )
                    self.phase = f"HOLD_{leg}"
                    finger_target = self.regrasp_finger_hold
                    qleg = self.regrasp_leg_hold
                    if (
                        float(d.time) - self.regrasp_wrap_start
                        >= self.hold_time
                    ):
                        self._finish_leg_step()
                        return
                else:
                    self.regrasp_wrap_start = None
                    self.regrasp_finger_hold = None
                    self.regrasp_leg_hold = None
                    self.regrasp_wrist_hold = None
                    if close_elapsed >= self.close_time:
                        raise RuntimeError(
                            f"{leg} close timed out without a two-sided "
                            f"geometric wrap on {self.target_rung}; refusing "
                            "to activate the grasp latch."
                        )
                    self.phase = f"CLOSE_{leg}"
                    finger_target = c.finger_closure_pose(
                        close_elapsed / self.close_time,
                        c.closed_finger_pose,
                        self.transfer_open_pose,
                    )
                    # Do not curl the first contacting hook deeper into the
                    # rung.  Hold that side at its real joint position while
                    # the opposite hook and palm continue to close/centre.
                    # This turns one-sided contact into a guide instead of a
                    # blocker and still leaves collision response enabled.
                    actual_finger_q = d.qpos[
                        c._finger_qpos_adrs[leg]
                    ].copy()
                    if target_sides == {"L"}:
                        finger_target[:3] = actual_finger_q[:3]
                    elif target_sides == {"R"}:
                        finger_target[3:] = actual_finger_q[3:]

        if pause_for_wrist:
            self.leg_integral_error[leg][:] = 0.0
        self._apply_leg_pd_target(leg, qleg, integrate=not pause_for_wrist)
        wrist_target = (
            self.regrasp_wrist_hold
            if self.regrasp_wrist_hold is not None
            # Compensate the wrist from the measured leg configuration, not
            # the look-ahead reference.  This preserves the real hook frame
            # even when torque-controlled joints lag slightly on the path.
            else self._wrist_for_leg_q(
                leg, d.qpos[c._leg_qpos_adrs[leg]]
            )
        )
        wrist_target_velocity = 0.0
        if (
            self.regrasp_wrist_hold is None
            and elapsed >= self.open_time
            and self.close_start_time is None
        ):
            leg_velocity = d.qvel[c._leg_dof_adrs[leg]]
            wrist_target_velocity = -float(np.sum(leg_velocity[1:]))
            wrist_joint_id = c._wrist_joint_ids[leg]
            lower, upper = self.model.jnt_range[wrist_joint_id]
            if wrist_target <= lower + 1e-9 or wrist_target >= upper - 1e-9:
                wrist_target_velocity = 0.0
        self._apply_wrist_pd_target(
            leg,
            wrist_target,
            target_velocity=wrist_target_velocity,
            integrate=elapsed >= self.open_time,
        )
        self._apply_moving_finger_pd(leg, finger_target)

    def post_step(self):
        if self.started and not self.finished:
            self.apply()

    def report(self):
        c, d = self.controller, self.data
        base = d.qpos[c._base_qpos_adr : c._base_qpos_adr + 3]
        if self.moving_leg:
            palm = d.xpos[c._palm_body_ids[self.moving_leg]]
            source_contacts = count_leg_rung_contacts(
                self.model, d, c, self.moving_leg, self.source_rung
            )
            target_contacts = count_leg_rung_contacts(
                self.model, d, c, self.moving_leg, self.target_rung
            )
            q_error = float(
                np.linalg.norm(
                    d.qpos[c._leg_qpos_adrs[self.moving_leg]]
                    - (
                        self.q_goal
                        if self.close_start_time is not None
                        else self._sample_motion_path(self.motion_progress)
                    )
                )
            )
            actual_leg_q = d.qpos[c._leg_qpos_adrs[self.moving_leg]]
            reference_leg_q = (
                self.q_goal
                if self.close_start_time is not None
                else self._sample_motion_path(self.motion_progress)
            )
            leg_ctrl = d.ctrl[c._leg_actuator_ids[self.moving_leg]]
            wrist_q = float(d.qpos[c._wrist_qpos_adrs[self.moving_leg]])
            wrist_ref = self._wrist_for_leg_q(self.moving_leg, actual_leg_q)
            wrist_ctrl = float(d.ctrl[c._wrist_actuator_ids[self.moving_leg]])
            leg_root = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                f"{self.moving_leg}_hip",
            )

            def belongs_to_moving_leg(body_id: int) -> bool:
                current = int(body_id)
                while current > 0:
                    if current == leg_root:
                        return True
                    current = int(self.model.body_parentid[current])
                return False

            blocking_pairs = set()
            rung_ids = set(int(item) for item in c._rung_geom_ids)
            for contact in d.contact:
                geom1, geom2 = int(contact.geom1), int(contact.geom2)
                if geom1 in rung_ids and belongs_to_moving_leg(
                    self.model.geom_bodyid[geom2]
                ):
                    moving_geom, rung_geom = geom2, geom1
                elif geom2 in rung_ids and belongs_to_moving_leg(
                    self.model.geom_bodyid[geom1]
                ):
                    moving_geom, rung_geom = geom1, geom2
                else:
                    continue
                moving_name = geom_name(self.model, moving_geom)
                moving_body = mujoco.mj_id2name(
                    self.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    int(self.model.geom_bodyid[moving_geom]),
                )
                blocking_pairs.add(
                    f"{moving_name}@{moving_body}/{geom_name(self.model, rung_geom)}"
                )
            print(
                f"t={d.time:7.3f} | cycle={self.cycle_index + 1}/"
                f"{self.requested_cycles} | phase={self.phase:14s} | "
                f"leg={self.moving_leg} | palm=({palm[0]:+.3f},"
                f"{palm[1]:+.3f},{palm[2]:+.3f}) | "
                f"base=({base[0]:+.3f},{base[1]:+.3f},{base[2]:+.3f}) | "
                f"qerr={q_error:.3f} | src={source_contacts} | "
                f"tgt={target_contacts} | q={np.round(actual_leg_q, 3).tolist()} | "
                f"qref={np.round(reference_leg_q, 3).tolist()} | "
                f"tau={np.round(leg_ctrl, 1).tolist()} | "
                f"wrist={wrist_q:+.3f}/{wrist_ref:+.3f} "
                f"tauw={wrist_ctrl:+.1f} | "
                f"blocks={sorted(blocking_pairs)}"
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
        print(
            "Geometric wraps:   "
            f"{sorted(contacts.get('geometrically_wrapped_grippers', []))}"
        )
        print(f"Physical contacts: {contacts.get('physical_count', 0)}")
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
    parser.add_argument("--close-time", type=float, default=7.0)
    parser.add_argument("--hold-time", type=float, default=0.5)
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
    if not sequence or len(set(sequence)) != len(sequence) or any(
        leg not in LEGS for leg in sequence
    ):
        raise ValueError("--sequence must contain unique names from FL, FR, RL, RR.")

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
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.fixedcamid = -1
            viewer.cam.lookat[:] = np.array([0.0, 0.0, -0.15])
            viewer.cam.distance = 3.4
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -20.0
            print(
                "Viewer camera: free (drag to orbit, Shift-drag to pan, "
                "scroll to zoom)."
            )
            while viewer.is_running() and data.time < args.duration and not crawl.finished:
                step_once()
                viewer.sync()
            if args.hold_viewer_on_exit:
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(0.02)

    if not crawl.finished:
        raise RuntimeError(
            f"Gait did not complete within {args.duration:.1f} simulated "
            "seconds; refusing to report an incomplete run as successful."
        )
    crawl.final_result()


def main():
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
