"""
Contact-aware static ladder grasp controller for the modified Unitree Go2.

Purpose
-------
This controller does one job only: establish and hold a real multi-contact
grasp on the ladder before any climbing gait is attempted.

It does not use the old gait implementation.  It:
1. solves a numerical IK problem for the four gripper palms,
2. opens and settles the fingers,
3. closes the fingers gradually with joint-space PD torque control,
4. keeps the floating base kinematically pinned during acquisition,
5. releases the base only after verified fingertip-rung contacts persist,
6. continues holding all leg and finger joints after release.

The temporary base pin is an acquisition fixture, not part of the measured
free-flight stability interval.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Iterable, Mapping

import mujoco
import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


class GraspState(Enum):
    OPEN_SETTLE = auto()
    CLOSING = auto()
    VERIFYING = auto()
    HOLDING = auto()
    FAILED = auto()


@dataclass(frozen=True)
class GraspStatus:
    state: str
    base_pinned: bool
    physical_contacts: int
    distinct_fingertips: int
    distinct_grippers: int
    verified: bool
    message: str


class StaticLadderGraspController:
    """Acquire and hold a four-gripper ladder grasp."""

    LEG_JOINTS: Mapping[str, tuple[str, str, str]] = {
        "FL": ("FL_hip_joint", "FL_thigh_joint", "FL_calf_joint"),
        "FR": ("FR_hip_joint", "FR_thigh_joint", "FR_calf_joint"),
        "RL": ("RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"),
        "RR": ("RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"),
    }

    LEG_ACTUATORS: Mapping[str, tuple[str, str, str]] = {
        "FL": ("FL_hip", "FL_thigh", "FL_calf"),
        "FR": ("FR_hip", "FR_thigh", "FR_calf"),
        "RL": ("RL_hip", "RL_thigh", "RL_calf"),
        "RR": ("RR_hip", "RR_thigh", "RR_calf"),
    }

    WRIST_JOINTS: Mapping[str, str] = {
        leg: f"{leg}_gripper_wrist_j" for leg in ("FL", "FR", "RL", "RR")
    }

    WRIST_ACTUATORS: Mapping[str, str] = {
        leg: f"{leg}_gripper_wrist" for leg in ("FL", "FR", "RL", "RR")
    }

    PALM_BODIES: Mapping[str, str] = {
        "FL": "FL_gripper_palm",
        "FR": "FR_gripper_palm",
        "RL": "RL_gripper_palm",
        "RR": "RR_gripper_palm",
    }

    FINGER_CONTACT_BODIES: Mapping[str, tuple[str, str]] = {
         "FL": (
            "fl_finger_L1",
            "fl_finger_L2",
            "fl_finger_L3",
            "fl_finger_R1",
            "fl_finger_R2",
            "fl_finger_R3",
        ),
        "FR": (
            "fr_finger_L1",
            "fr_finger_L2",
            "fr_finger_L3",
            "fr_finger_R1",
            "fr_finger_R2",
            "fr_finger_R3",
        ),
        "RL": (
            "rl_finger_L1",
            "rl_finger_L2",
            "rl_finger_L3",
            "rl_finger_R1",
            "rl_finger_R2",
            "rl_finger_R3",
        ),
        "RR": (
            "rr_finger_L1",
            "rr_finger_L2",
            "rr_finger_L3",
            "rr_finger_R1",
            "rr_finger_R2",
            "rr_finger_R3",
        ),
    }

    FINGER_JOINTS: Mapping[str, tuple[str, ...]] = {
        leg: tuple(
            f"{leg.lower()}_g_{side}{segment}_j"
            for side in ("L", "R")
            for segment in (1, 2, 3)
        )
        for leg in ("FL", "FR", "RL", "RR")
    }

    FINGER_ACTUATORS: Mapping[str, tuple[str, ...]] = {
        leg: tuple(
            f"{leg.lower()}_gripper_{side}{segment}"
            for side in ("L", "R")
            for segment in (1, 2, 3)
        )
        for leg in ("FL", "FR", "RL", "RR")
    }

    # The front pair is aligned with rung_5 and the rear pair with rung_3.
    # Each left/right gripper uses a separate y location on the same rung.
    DEFAULT_TARGETS = {
        # Centre each palm over its rung.  The previous x offsets were a
        # workaround for over-curled, crossing fingers and inherently made
        # only one side of a proper U-shaped hook reachable.
        "FL": (+0.175, +0.131, -0.186),
        "FR": (+0.175, -0.131, -0.186),
        "RL": (-0.275, +0.131, -0.186),
        "RR": (-0.275, -0.131, -0.186),
    }

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        base_position: Iterable[float] = (-0.05, 0.0, 0.08),
        base_quaternion: Iterable[float] = (1.0, 0.0, 0.0, 0.0),
        palm_targets: Mapping[str, Iterable[float]] | None = None,
        open_angle: float = 0.05,
        closed_angle: float = 0.95,
        open_settle_time: float = 0.50,
        closing_time: float = 2.00,
        verification_hold_time: float = 0.30,
        acquisition_timeout: float = 15.00,
        minimum_distinct_fingertips: int = 4,
        minimum_distinct_grippers: int = 3,
        leg_kp: float = 40.0,
        leg_kd: float = 4.0,
        finger_kp: float = 4.0,
        finger_kd: float = 0.20,
        ik_tolerance: float = 0.002,
        ik_iterations: int = 500,
        ik_damping: float = 2.0e-3,
        ik_step_limit: float = 0.04,
    ) -> None:
        self.model = model

        self.base_position = np.asarray(tuple(base_position), dtype=float)
        self.base_quaternion = np.asarray(tuple(base_quaternion), dtype=float)
        if self.base_position.shape != (3,):
            raise ValueError("base_position must have three values.")
        if self.base_quaternion.shape != (4,):
            raise ValueError("base_quaternion must have four values.")

        raw_targets = palm_targets or self.DEFAULT_TARGETS
        self.palm_targets = {
            leg: np.asarray(tuple(raw_targets[leg]), dtype=float)
            for leg in self.LEG_JOINTS
        }

        self.open_angle = float(open_angle)
        #self.closed_angle = float(closed_angle)
        self.open_finger_pose = np.array(
            [-0.25, 0.05, 0.05, -0.25, 0.05, 0.05],
            dtype=float,
        )

        self.closed_finger_pose = np.array(
            [
                # Mirror-symmetric U-shaped hooks.  The previous high angles
                # made the distal links cross the rung centre and turn back
                # upward: two-sided contact existed, but the fingers did not
                # visibly enclose the rod.  These cumulative angles keep both
                # distal chains below the rung and stop their tips near x=0.
                0.30, 0.50, -0.80,
                0.30, 0.50, -0.80,
            ],
            dtype=float,
        )
        self.open_settle_time = float(open_settle_time)
        self.closing_time = float(closing_time)
        self.verification_hold_time = float(verification_hold_time)
        self.acquisition_timeout = float(acquisition_timeout)
        self.minimum_distinct_fingertips = int(minimum_distinct_fingertips)
        self.minimum_distinct_grippers = int(minimum_distinct_grippers)

        self.leg_kp = float(leg_kp)
        self.leg_kd = float(leg_kd)
        self.finger_kp = float(finger_kp)
        self.finger_kd = float(finger_kd)

        self.ik_tolerance = float(ik_tolerance)
        self.ik_iterations = int(ik_iterations)
        self.ik_damping = float(ik_damping)
        self.ik_step_limit = float(ik_step_limit)

        self._base_joint_id = self._find_free_joint()
        self._base_qpos_adr = int(model.jnt_qposadr[self._base_joint_id])
        self._base_dof_adr = int(model.jnt_dofadr[self._base_joint_id])
        self._structure_joint_id = int(
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, "structure_freejoint"
            )
        )
        if self._structure_joint_id >= 0:
            self._structure_qpos_adr = int(
                model.jnt_qposadr[self._structure_joint_id]
            )
            self._structure_dof_adr = int(
                model.jnt_dofadr[self._structure_joint_id]
            )
        else:
            self._structure_qpos_adr = None
            self._structure_dof_adr = None

        self._leg_joint_ids = {
            leg: np.array([self._joint_id(name) for name in names], dtype=int)
            for leg, names in self.LEG_JOINTS.items()
        }
        self._leg_qpos_adrs = {
            leg: np.array([model.jnt_qposadr[jid] for jid in ids], dtype=int)
            for leg, ids in self._leg_joint_ids.items()
        }
        self._leg_dof_adrs = {
            leg: np.array([model.jnt_dofadr[jid] for jid in ids], dtype=int)
            for leg, ids in self._leg_joint_ids.items()
        }
        self._leg_actuator_ids = {
            leg: np.array([self._actuator_id(name) for name in names], dtype=int)
            for leg, names in self.LEG_ACTUATORS.items()
        }

        self._wrist_joint_ids = {
            leg: self._joint_id(name) for leg, name in self.WRIST_JOINTS.items()
        }
        self._wrist_qpos_adrs = {
            leg: int(model.jnt_qposadr[jid])
            for leg, jid in self._wrist_joint_ids.items()
        }
        self._wrist_dof_adrs = {
            leg: int(model.jnt_dofadr[jid])
            for leg, jid in self._wrist_joint_ids.items()
        }
        self._wrist_actuator_ids = {
            leg: self._actuator_id(name)
            for leg, name in self.WRIST_ACTUATORS.items()
        }
        self._wrist_target_qpos = {leg: 0.0 for leg in self.LEG_JOINTS}

        self._finger_joint_ids = {
            leg: np.array([self._joint_id(name) for name in names], dtype=int)
            for leg, names in self.FINGER_JOINTS.items()
        }
        self._finger_qpos_adrs = {
            leg: np.array([model.jnt_qposadr[jid] for jid in ids], dtype=int)
            for leg, ids in self._finger_joint_ids.items()
        }
        self._finger_dof_adrs = {
            leg: np.array([model.jnt_dofadr[jid] for jid in ids], dtype=int)
            for leg, ids in self._finger_joint_ids.items()
        }
        self._finger_actuator_ids = {
            leg: np.array([self._actuator_id(name) for name in names], dtype=int)
            for leg, names in self.FINGER_ACTUATORS.items()
        }

        self._palm_body_ids = {
            leg: self._body_id(name) for leg, name in self.PALM_BODIES.items()
        }
        self._ladder_body_id = self._body_id("ladder_structure")
        self._grasp_weld_ids = {
            leg: int(
                mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_EQUALITY,
                    f"{leg}_grasp_weld",
                )
            )
            for leg in self.LEG_JOINTS
        }
        self._finger_body_to_leg: Dict[int, str] = {}
        self._finger_body_to_name: Dict[int, str] = {}

        for leg, names in self.FINGER_CONTACT_BODIES.items():
            for name in names:
                body_id = self._body_id(name)
                self._finger_body_to_leg[body_id] = leg
                self._finger_body_to_name[body_id] = name

        self._rung_geom_ids = {
            geom_id
            for geom_id in range(model.ngeom)
            if (mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
            ) or "").startswith("rung_")
        }
        if not self._rung_geom_ids:
            raise ValueError("No rung_* geoms were found in the loaded scene.")

        self._leg_target_qpos: Dict[str, FloatArray] = {}
        self._pinned_base_qpos = np.zeros(7, dtype=float)
        self._pinned_structure_qpos: FloatArray | None = None
        self._state = GraspState.OPEN_SETTLE
        self._start_time = 0.0
        self._verification_start: float | None = None
        self._verification_finger_poses: Dict[str, FloatArray] = {}
        self._candidate_wrap_start: Dict[str, float] = {}
        self._candidate_finger_poses: Dict[str, FloatArray] = {}
        self._last_status = GraspStatus(
            state=self._state.name,
            base_pinned=True,
            physical_contacts=0,
            distinct_fingertips=0,
            distinct_grippers=0,
            verified=False,
            message="Controller has not been reset.",
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if model is not self.model:
            raise ValueError("Controller was constructed for a different model.")

        mujoco.mj_resetData(model, data)
        for weld_id in self._grasp_weld_ids.values():
            if weld_id >= 0:
                data.eq_active[weld_id] = False

        base_adr = self._base_qpos_adr
        data.qpos[base_adr:base_adr + 3] = self.base_position
        data.qpos[base_adr + 3:base_adr + 7] = self.base_quaternion

        for leg, addresses in self._finger_qpos_adrs.items():
            data.qpos[addresses] = self.open_finger_pose
            data.qpos[self._wrist_qpos_adrs[leg]] = 0.0

        data.qvel[:] = 0.0
        data.ctrl[:] = 0.0
        mujoco.mj_forward(model, data)

        self._solve_all_leg_ik(model, data)
        self._leg_target_qpos = {
            leg: data.qpos[addresses].copy()
            for leg, addresses in self._leg_qpos_adrs.items()
        }
        # The palm is mounted after the thigh and calf pitch joints.  A wrist
        # value of zero therefore inherits the leg pitch and presents one hook
        # above the rung and the other below it.  Cancel that accumulated pitch
        # so every gripper starts level in the world grasping plane.
        for leg, addresses in self._leg_qpos_adrs.items():
            wrist_joint_id = self._wrist_joint_ids[leg]
            lower, upper = model.jnt_range[wrist_joint_id]
            wrist_target = float(
                np.clip(-np.sum(data.qpos[addresses][1:]), lower, upper)
            )
            data.qpos[self._wrist_qpos_adrs[leg]] = wrist_target
            self._wrist_target_qpos[leg] = wrist_target

        self._pinned_base_qpos = data.qpos[base_adr:base_adr + 7].copy()
        if self._structure_qpos_adr is not None:
            structure_adr = self._structure_qpos_adr
            self._pinned_structure_qpos = data.qpos[
                structure_adr:structure_adr + 7
            ].copy()
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        self._state = GraspState.OPEN_SETTLE
        self._start_time = float(data.time)
        self._verification_start = None
        self._verification_finger_poses = {}
        self._candidate_wrap_start = {}
        self._candidate_finger_poses = {}
        self._last_status = self._build_status(
            model, data, "IK completed; fingers open and base pinned."
        )

        self._print_initialisation_report(model, data)

        print("\nInitial fingertip positions")
        print("-" * 72)

        for body_id, body_name in self._finger_body_to_name.items():
            position = data.xpos[body_id]

            print(
                f"{body_name:16s}: "
                f"x={position[0]: .4f}, "
                f"y={position[1]: .4f}, "
                f"z={position[2]: .4f}"
            )

        print("-" * 72)

    def finger_closure_pose(
        self,
        progress: float,
        terminal_pose: FloatArray | None = None,
        start_pose: FloatArray | None = None,
    ) -> FloatArray:
        """Return a coordinated hook-closing pose for ``progress`` in [0, 1].

        The proximal joints lead the motion so the opposing fingers move
        around the rung before the distal links curl. Interpolating all six
        joints together made the distal tips collide beneath the rung.
        """
        progress = float(np.clip(progress, 0.0, 1.0))
        terminal = np.asarray(
            self.closed_finger_pose if terminal_pose is None else terminal_pose,
            dtype=float,
        )
        if terminal.shape != (6,):
            raise ValueError("terminal_pose must contain six finger joint values.")
        start = np.asarray(
            self.open_finger_pose if start_pose is None else start_pose,
            dtype=float,
        )
        if start.shape != (6,):
            raise ValueError("start_pose must contain six finger joint values.")
        target = start.copy()
        stages = (
            ((0, 3), 0.00, 0.45),
            ((1, 4), 0.30, 0.75),
            ((2, 5), 0.60, 1.00),
        )
        for indices, stage_start, stage_end in stages:
            alpha = float(
                np.clip(
                    (progress - stage_start) / (stage_end - stage_start),
                    0.0,
                    1.0,
                )
            )
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            for index in indices:
                target[index] = (
                    start[index]
                    + alpha
                    * (terminal[index] - start[index])
                )
        return target

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> GraspStatus:
        """Compute and write actuator commands before mj_step()."""
        elapsed = float(data.time) - self._start_time
        contacts = self._detect_contacts(model, data)

        if self._state is GraspState.OPEN_SETTLE:
            finger_target = self.open_finger_pose.copy()
            if elapsed >= self.open_settle_time:
                self._state = GraspState.CLOSING

        elif self._state is GraspState.CLOSING:
            alpha = np.clip(
                (elapsed - self.open_settle_time) / self.closing_time,
                0.0,
                1.0,
            )
            closing_target = self.finger_closure_pose(alpha)
            wrapped = contacts["geometrically_wrapped_grippers"]
            assert isinstance(wrapped, set)

            # Freeze each gripper independently at its first real wrap. The
            # four fingers do not acquire contact at exactly the same time.
            for leg in self.LEG_JOINTS:
                if leg in wrapped:
                    if leg not in self._candidate_wrap_start:
                        self._candidate_wrap_start[leg] = float(data.time)
                        pose = data.qpos[
                            self._finger_qpos_adrs[leg]
                        ].copy()
                        # Maintain a small proximal preload against the two
                        # rung sides while leaving the distal joints frozen;
                        # this avoids both contact loss and fingertip crossing.
                        pose[[0, 3]] = np.minimum(
                            pose[[0, 3]] + 0.06,
                            self.closed_finger_pose[[0, 3]],
                        )
                        self._candidate_finger_poses[leg] = pose
                else:
                    self._candidate_wrap_start.pop(leg, None)
                    self._candidate_finger_poses.pop(leg, None)

            finger_target = {
                leg: self._candidate_finger_poses.get(leg, closing_target)
                for leg in self.LEG_JOINTS
            }
            persistent_wrap = (
                len(self._candidate_wrap_start) == len(self.LEG_JOINTS)
                and all(
                    float(data.time) - start >= self.verification_hold_time
                    for start in self._candidate_wrap_start.values()
                )
            )
            if persistent_wrap:
                self._state = GraspState.HOLDING
                self._verification_finger_poses = {
                    leg: pose.copy()
                    for leg, pose in self._candidate_finger_poses.items()
                }
                for leg in self.LEG_JOINTS:
                    self.set_grasp_weld(model, data, leg, True)

        elif self._state is GraspState.HOLDING:
            finger_target = self._verification_finger_poses

        else:
            finger_target = self.closed_finger_pose.copy()

        self._apply_leg_pd(data)
        self._apply_wrist_pd(data)
        self._apply_finger_pd(data, finger_target)

        if (
            self._state is not GraspState.HOLDING
            and elapsed >= self.acquisition_timeout
        ):
            self._state = GraspState.FAILED

        if self._state is GraspState.HOLDING:
            message = "Verified grasp acquired; floating base released."
        elif self._state is GraspState.FAILED:
            message = (
                "Grasp was not verified before timeout; base remains pinned. "
                "Inspect palm-target errors and contact report."
            )
        else:
            message = "Acquiring grasp; floating base remains pinned."

        self._last_status = self._build_status(model, data, message)
        return self._last_status

    def post_step(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """
        Apply temporary robot/structure acquisition fixtures after mj_step().

        The base is released only after a persistent real contact set has
        been verified.  Joint dynamics are never overwritten.
        """
        if not self.base_pinned:
            return

        qadr = self._base_qpos_adr
        dadr = self._base_dof_adr
        data.qpos[qadr:qadr + 7] = self._pinned_base_qpos
        data.qvel[dadr:dadr + 6] = 0.0
        if (
            self._structure_qpos_adr is not None
            and self._structure_dof_adr is not None
            and self._pinned_structure_qpos is not None
        ):
            structure_qadr = self._structure_qpos_adr
            structure_dadr = self._structure_dof_adr
            data.qpos[
                structure_qadr:structure_qadr + 7
            ] = self._pinned_structure_qpos
            data.qvel[structure_dadr:structure_dadr + 6] = 0.0
        mujoco.mj_forward(model, data)

    @property
    def base_pinned(self) -> bool:
        return self._state is not GraspState.HOLDING

    @property
    def grasp_verified(self) -> bool:
        return self._state is GraspState.HOLDING

    @property
    def status(self) -> GraspStatus:
        return self._last_status

    @staticmethod
    def _quat_conjugate(quaternion: FloatArray) -> FloatArray:
        return np.array(
            [quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]],
            dtype=float,
        )

    @staticmethod
    def _quat_multiply(left: FloatArray, right: FloatArray) -> FloatArray:
        w1, x1, y1, z1 = left
        w2, x2, y2, z2 = right
        return np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=float,
        )

    def set_grasp_weld(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        leg: str,
        active: bool,
    ) -> None:
        """Lock or release a verified palm-to-truss grasp at its current pose."""
        weld_id = self._grasp_weld_ids.get(leg, -1)
        if weld_id < 0:
            return
        if active:
            contacts = self._detect_contacts(model, data)
            wrapped = contacts.get("geometrically_wrapped_grippers", set())
            if leg not in wrapped:
                raise RuntimeError(
                    f"Refusing to latch {leg}: no two-sided geometric "
                    "finger-rung wrap is currently present."
                )
            palm_id = self._palm_body_ids[leg]
            ladder_id = self._ladder_body_id
            palm_rotation = data.xmat[palm_id].reshape(3, 3)
            relative_position = palm_rotation.T @ (
                data.xpos[ladder_id] - data.xpos[palm_id]
            )
            relative_quaternion = self._quat_multiply(
                self._quat_conjugate(data.xquat[palm_id]),
                data.xquat[ladder_id],
            )
            model.eq_data[weld_id, 3:6] = relative_position
            model.eq_data[weld_id, 6:10] = relative_quaternion
        data.eq_active[weld_id] = bool(active)

    # ------------------------------------------------------------------
    # IK
    # ------------------------------------------------------------------

    def _solve_all_leg_ik(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> None:
        # Neutral seeds are inside all four leg joint limits.
        neutral = {
            "FL": np.array([0.0, 0.90, -1.80]),
            "FR": np.array([0.0, 0.90, -1.80]),
            "RL": np.array([0.0, 1.10, -1.80]),
            "RR": np.array([0.0, 1.10, -1.80]),
        }
        for leg, addresses in self._leg_qpos_adrs.items():
            data.qpos[addresses] = neutral[leg]
        mujoco.mj_forward(model, data)

        for _ in range(self.ik_iterations):
            maximum_error = 0.0

            for leg in self.LEG_JOINTS:
                body_id = self._palm_body_ids[leg]
                target = self.palm_targets[leg]

                actual = data.xpos[body_id]
                error = target - actual
                maximum_error = max(
                    maximum_error, float(np.linalg.norm(error))
                )

                jacp = np.zeros((3, model.nv), dtype=float)
                jacr = np.zeros((3, model.nv), dtype=float)

                mujoco.mj_jacBody(
                    model,
                    data,
                    jacp,
                    jacr,
                    body_id,
                )

                dof_adrs = self._leg_dof_adrs[leg]
                jacobian = jacp[:, dof_adrs]

                regulariser = (
                    self.ik_damping**2
                    * np.eye(3)
                )

                delta = jacobian.T @ np.linalg.solve(
                    jacobian @ jacobian.T + regulariser,
                    error,
                )

                delta = np.clip(
                    delta,
                    -self.ik_step_limit,
                    self.ik_step_limit,
                )

                qpos_adrs = self._leg_qpos_adrs[leg]
                data.qpos[qpos_adrs] += delta

                self._clip_joint_positions(
                    data,
                    self._leg_joint_ids[leg],
                )

                # Important: update kinematics after modifying each leg.
                mujoco.mj_forward(model, data)
            if maximum_error <= self.ik_tolerance:
                break

        error_vectors = {
            leg: (
                self.palm_targets[leg]
                - data.xpos[self._palm_body_ids[leg]]
            )
            for leg in self.LEG_JOINTS
        }

        errors = {
            leg: float(np.linalg.norm(vector))
            for leg, vector in error_vectors.items()
        }

        for leg in self.LEG_JOINTS:
            actual = data.xpos[self._palm_body_ids[leg]]
            target = self.palm_targets[leg]
            vector = error_vectors[leg]

            print(
                f"{leg} IK residual: "
                f"target={target}, "
                f"actual={actual}, "
                f"delta={vector}, "
                f"norm={errors[leg]:.6f} m"
            )
    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def _apply_leg_pd(self, data: mujoco.MjData) -> None:
        for leg in self.LEG_JOINTS:
            q = data.qpos[self._leg_qpos_adrs[leg]]
            qd = data.qvel[self._leg_dof_adrs[leg]]
            torque = self.leg_kp * (self._leg_target_qpos[leg] - q)
            torque -= self.leg_kd * qd
            self._write_clipped_control(
                data, self._leg_actuator_ids[leg], torque
            )

    def _apply_finger_pd(
        self,
        data: mujoco.MjData,
        target_pose: FloatArray | Mapping[str, FloatArray],
    ) -> None:
        for leg in self.LEG_JOINTS:
            leg_target = np.asarray(
                target_pose[leg]
                if isinstance(target_pose, Mapping)
                else target_pose,
                dtype=float,
            )
            if leg_target.shape != (6,):
                raise ValueError(
                    "Each finger target must contain "
                    "[L1, L2, L3, R1, R2, R3]."
                )
            q = data.qpos[
                self._finger_qpos_adrs[leg]
            ]

            qd = data.qvel[
                self._finger_dof_adrs[leg]
            ]

            torque = (
                self.finger_kp * (leg_target - q)
                - self.finger_kd * qd
            )

            self._write_clipped_control(
                data,
                self._finger_actuator_ids[leg],
                torque,
            )

    def _apply_wrist_pd(self, data: mujoco.MjData) -> None:
        for leg in self.LEG_JOINTS:
            q = float(data.qpos[self._wrist_qpos_adrs[leg]])
            qd = float(data.qvel[self._wrist_dof_adrs[leg]])
            torque = 35.0 * (self._wrist_target_qpos[leg] - q) - 3.0 * qd
            actuator_id = self._wrist_actuator_ids[leg]
            self._write_clipped_control(
                data,
                np.asarray([actuator_id], dtype=int),
                np.asarray([torque], dtype=float),
            )

    def _write_clipped_control(
        self,
        data: mujoco.MjData,
        actuator_ids: NDArray[np.int_],
        command: FloatArray,
    ) -> None:
        lower = self.model.actuator_ctrlrange[actuator_ids, 0]
        upper = self.model.actuator_ctrlrange[actuator_ids, 1]
        data.ctrl[actuator_ids] = np.clip(command, lower, upper)

    # ------------------------------------------------------------------
    # Contact verification
    # ------------------------------------------------------------------

    @staticmethod
    def has_geometric_wrap(
        contacts: Mapping[str, object],
        leg: str,
        rung_name: str | None = None,
    ) -> bool:
        """Return true only for two hook chains contacting opposite rung faces.

        The wrist controller keeps the two hook roots in their intended world
        ordering.  L must contact the +x face and R the -x face.  Accepting the
        reversed ordering permits the chains to cross underneath the rung,
        which is visibly not a proper grasp.
        """
        details = contacts.get("contact_details", [])
        assert isinstance(details, list)
        rung_names = {
            str(item["rung_geom_name"])
            for item in details
            if str(item["leg"]) == leg
            and (rung_name is None or str(item["rung_geom_name"]) == rung_name)
        }
        for candidate_rung in rung_names:
            left_x = [
                float(item["relative_x"])
                for item in details
                if str(item["leg"]) == leg
                and str(item["rung_geom_name"]) == candidate_rung
                and str(item["side"]) == "L"
            ]
            right_x = [
                float(item["relative_x"])
                for item in details
                if str(item["leg"]) == leg
                and str(item["rung_geom_name"]) == candidate_rung
                and str(item["side"]) == "R"
            ]
            if (
                left_x
                and right_x
                and max(left_x) > 0.001
                and min(right_x) < -0.001
            ):
                return True
        return False

    def _detect_contacts(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> dict[str, object]:
        """
        Detect contacts between ladder rungs and finger collision geoms.

        The returned dictionary contains:

        physical_count:
            Number of MuJoCo contact points between a rung and a finger geom.

        finger_segment_names:
            Distinct finger bodies involved in contact.

        contacting_fingers:
            Logical fingers involved in contact, represented as (leg, side).

        grippers:
            Grippers with at least one finger contact.

        sides_by_gripper:
            Contacting sides for each gripper.

        fully_wrapped_grippers:
            Grippers for which both L and R fingers contact a rung.

            Important:
            This is only a preliminary contact-based classification. It does
            not yet prove that the rung is geometrically enclosed.

        contact_details:
            Detailed information for every rung-finger contact.
        """

        physical_count = 0

        finger_segment_names: set[str] = set()
        contacting_fingers: set[tuple[str, str]] = set()
        grippers: set[str] = set()

        sides_by_gripper: dict[str, set[str]] = {
            leg: set()
            for leg in self.LEG_JOINTS
        }

        contact_details: list[dict[str, object]] = []

        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]

            geom1_id = int(contact.geom1)
            geom2_id = int(contact.geom2)

            geom1_is_rung = geom1_id in self._rung_geom_ids
            geom2_is_rung = geom2_id in self._rung_geom_ids

            # Keep only contacts containing exactly one ladder rung.
            if geom1_is_rung and not geom2_is_rung:
                rung_geom_id = geom1_id
                finger_geom_id = geom2_id
                rung_is_geom1 = True

            elif geom2_is_rung and not geom1_is_rung:
                rung_geom_id = geom2_id
                finger_geom_id = geom1_id
                rung_is_geom1 = False

            else:
                continue

            finger_body_id = int(
                model.geom_bodyid[finger_geom_id]
            )

            # Ignore ladder contacts involving feet, palms, legs or base.
            if finger_body_id not in self._finger_body_to_leg:
                continue

            leg = self._finger_body_to_leg[
                finger_body_id
            ]

            finger_body_name = self._finger_body_to_name[
                finger_body_id
            ]

            if "_finger_L" in finger_body_name:
                side = "L"

            elif "_finger_R" in finger_body_name:
                side = "R"

            else:
                continue

            rung_geom_name = mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                rung_geom_id,
            )

            finger_geom_name = mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                finger_geom_id,
            )

            if rung_geom_name is None:
                rung_geom_name = f"geom_{rung_geom_id}"

            if finger_geom_name is None:
                finger_geom_name = f"geom_{finger_geom_id}"

            contact_position = np.asarray(
                contact.pos,
                dtype=float,
            ).copy()

            raw_contact_normal = np.asarray(
                contact.frame[:3],
                dtype=float,
            ).copy()

            # MuJoCo stores the contact normal according to geom1/geom2 order.
            # Convert it to one consistent convention: rung -> finger.
            if rung_is_geom1:
                rung_to_finger_normal = raw_contact_normal.copy()
            else:
                rung_to_finger_normal = -raw_contact_normal

            rung_center = np.asarray(
                data.geom_xpos[rung_geom_id],
                dtype=float,
            ).copy()

            relative_position = (
                contact_position - rung_center
            )

            radial_xz = np.array(
                [
                    relative_position[0],
                    relative_position[2],
                ],
                dtype=float,
            )

            radial_distance_xz = float(
                np.linalg.norm(radial_xz)
            )

            # Polar angle around the rung in the x-z grasping plane.
            #
            #   0 deg   : +x side
            #   90 deg  : above the rung
            #   180 deg : -x side
            #  -90 deg  : below the rung
            contact_angle_deg = float(
                np.degrees(
                    np.arctan2(
                        relative_position[2],
                        relative_position[0],
                    )
                )
            )

            physical_count += 1

            finger_segment_names.add(
                finger_body_name
            )

            contacting_fingers.add(
                (leg, side)
            )

            grippers.add(
                leg
            )

            sides_by_gripper[leg].add(
                side
            )

            contact_details.append(
                {
                    "contact_index": contact_index,
                    "leg": leg,
                    "side": side,
                    "segment": finger_body_name,
                    "finger_body_id": finger_body_id,
                    "finger_geom_id": finger_geom_id,
                    "finger_geom_name": finger_geom_name,
                    "rung_geom_id": rung_geom_id,
                    "rung_geom_name": rung_geom_name,
                    "position": contact_position,
                    "rung_center": rung_center,
                    "relative_position": relative_position,
                    "relative_x": float(relative_position[0]),
                    "relative_y": float(relative_position[1]),
                    "relative_z": float(relative_position[2]),
                    "radial_distance_xz": radial_distance_xz,
                    "contact_angle_deg": contact_angle_deg,
                    "raw_normal": raw_contact_normal,
                    "rung_to_finger_normal": (
                        rung_to_finger_normal
                    ),
                    "distance": float(contact.dist),
                }
            )

        fully_wrapped_grippers = {
            leg
            for leg, sides in sides_by_gripper.items()
            if {"L", "R"}.issubset(sides)
        }

        provisional = {"contact_details": contact_details}
        geometrically_wrapped_grippers = {
            leg
            for leg in self.LEG_JOINTS
            if self.has_geometric_wrap(provisional, leg)
        }

        return {
            "physical_count": physical_count,
            "finger_segment_names": finger_segment_names,
            "contacting_fingers": contacting_fingers,
            "grippers": grippers,
            "sides_by_gripper": sides_by_gripper,
            "fully_wrapped_grippers": fully_wrapped_grippers,
            "geometrically_wrapped_grippers": geometrically_wrapped_grippers,
            "contact_details": contact_details,
        }

    def contact_report(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> str:
        """Return a detailed diagnostic report for rung-finger contacts."""

        contacts = self._detect_contacts(
            model,
            data,
        )

        finger_segments = sorted(
            contacts["finger_segment_names"]
        )

        contacting_fingers = sorted(
            contacts["contacting_fingers"]
        )

        grippers = sorted(
            contacts["grippers"]
        )

        fully_wrapped_grippers = sorted(
            contacts["fully_wrapped_grippers"]
        )
        geometrically_wrapped_grippers = sorted(
            contacts["geometrically_wrapped_grippers"]
        )

        sides_by_gripper = contacts[
            "sides_by_gripper"
        ]

        contact_details = contacts[
            "contact_details"
        ]

        lines: list[str] = []

        # This title makes it obvious that the new function is running.
        lines.append(
            "=== DETAILED CONTACT REPORT V2 ==="
        )

        lines.append(
            f"segments={finger_segments}"
        )

        lines.append(
            f"fingers={contacting_fingers}"
        )

        lines.append(
            f"grippers={grippers}"
        )

        side_text = ", ".join(
            f"{leg}={sorted(sides_by_gripper[leg])}"
            for leg in self.LEG_JOINTS
        )

        lines.append(
            f"sides_by_gripper: {side_text}"
        )

        lines.append(
            f"fully_wrapped={fully_wrapped_grippers}"
        )
        lines.append(
            f"geometrically_wrapped={geometrically_wrapped_grippers}"
        )

        lines.append(
            f"detail_count={len(contact_details)}"
        )

        if not contact_details:
            lines.append(
                "contact_details: none"
            )
            return "\n".join(lines)

        lines.append(
            "contact_details:"
        )

        sorted_details = sorted(
            contact_details,
            key=lambda item: (
                str(item["leg"]),
                str(item["side"]),
                str(item["segment"]),
                int(item["contact_index"]),
            ),
        )

        for item in sorted_details:
            position = np.asarray(
                item["position"],
                dtype=float,
            )

            rung_center = np.asarray(
                item["rung_center"],
                dtype=float,
            )

            normal = np.asarray(
                item["rung_to_finger_normal"],
                dtype=float,
            )

            lines.append(
                "  "
                f"{item['leg']}-{item['side']} | "
                f"segment={item['segment']} | "
                f"finger_geom={item['finger_geom_name']} | "
                f"rung={item['rung_geom_name']} | "
                f"pos=("
                f"{position[0]:+.5f}, "
                f"{position[1]:+.5f}, "
                f"{position[2]:+.5f}) | "
                f"rung_center=("
                f"{rung_center[0]:+.5f}, "
                f"{rung_center[1]:+.5f}, "
                f"{rung_center[2]:+.5f}) | "
                f"rel_x={float(item['relative_x']):+.5f} | "
                f"rel_y={float(item['relative_y']):+.5f} | "
                f"rel_z={float(item['relative_z']):+.5f} | "
                f"radial_xz="
                f"{float(item['radial_distance_xz']):.5f} | "
                f"angle="
                f"{float(item['contact_angle_deg']):+.2f} deg | "
                f"normal=("
                f"{normal[0]:+.4f}, "
                f"{normal[1]:+.4f}, "
                f"{normal[2]:+.4f}) | "
                f"dist={float(item['distance']):+.7f}"
            )

        return "\n".join(lines)

    def _contact_requirements_met(
        self,
        contacts: Mapping[str, object],
    ) -> bool:
        """
        Verify a stable four-gripper grasp suitable for metrics collection.
        """

        wrapped = contacts["geometrically_wrapped_grippers"]
        assert isinstance(wrapped, set)
        return wrapped == set(self.LEG_JOINTS)

    def _build_status(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        message: str,
    ) -> GraspStatus:
        contacts = self._detect_contacts(
            model,
            data,
        )

        finger_segments = contacts[
            "finger_segment_names"
        ]

        grippers = contacts[
            "grippers"
        ]

        assert isinstance(
            finger_segments,
            set,
        )

        assert isinstance(
            grippers,
            set,
        )

        return GraspStatus(
            state=self._state.name,
            base_pinned=self.base_pinned,
            physical_contacts=int(
                contacts["physical_count"]
            ),
            distinct_fingertips=len(
                finger_segments
            ),
            distinct_grippers=len(
                grippers
            ),
            verified=self.grasp_verified,
            message=message,
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _clip_joint_positions(
        self,
        data: mujoco.MjData,
        joint_ids: NDArray[np.int_],
    ) -> None:
        for joint_id in joint_ids:
            if not bool(self.model.jnt_limited[joint_id]):
                continue
            address = int(self.model.jnt_qposadr[joint_id])
            low, high = self.model.jnt_range[joint_id]
            data.qpos[address] = np.clip(data.qpos[address], low, high)

    def _find_free_joint(self) -> int:
        for joint_id in range(self.model.njnt):
            if self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                return joint_id
        raise ValueError("The model does not contain a floating-base free joint.")

    def _joint_id(self, name: str) -> int:
        object_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        if object_id < 0:
            raise ValueError(f"Joint not found: {name}")
        return int(object_id)

    def _actuator_id(self, name: str) -> int:
        object_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
        )
        if object_id < 0:
            raise ValueError(f"Actuator not found: {name}")
        return int(object_id)

    def _body_id(self, name: str) -> int:
        object_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, name
        )
        if object_id < 0:
            raise ValueError(f"Body not found: {name}")
        return int(object_id)

    def _print_initialisation_report(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> None:
        print("\nInitial palm IK report")
        print("-" * 72)
        for leg in self.LEG_JOINTS:
            actual = data.xpos[self._palm_body_ids[leg]]
            target = self.palm_targets[leg]
            error = np.linalg.norm(target - actual)
            q = data.qpos[self._leg_qpos_adrs[leg]]
            print(
                f"{leg}: target={target}, actual={actual}, "
                f"error={error:.6f} m, q={q}"
            )
        print("-" * 72)
