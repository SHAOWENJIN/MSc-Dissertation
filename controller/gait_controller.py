"""
Direct vertical-lift one-leg ladder-step controller V8.0 for the modified Unitree Go2.

This controller deliberately builds on the verified static grasp controller.
It uses a rear-first crawl sequence, freezes the three supporting leg targets
while one gripper swings, verifies old-rung release and target-rung regrasp,
and divides the body translation into two half-rung shifts.  V8.0 changed the release strategy after the V5 diagnostics showed that a
fixed 35 mm palm waypoint could be kinematically reachable while the real
R2/R3 finger chain remained almost touching the source rung.  V6 searches
multiple escape directions in the rung cross-section, requires a preflight
clearance reserve, verifies that the fingers are truly open, and then extends
the release command online until the measured geometry (not the kinematic
prediction) has enough clearance.  After release, transfer follows the same
escape offset translated to the next rung instead of applying an arbitrary
global +z lift.  V8.0 replaces the oblique peel path with a deterministic three-segment
world-space path requested for the first validated step: open while lifting
vertically away from the ladder (+z), translate exactly one XML rung spacing
(+x), then lower vertically to the measured palm-to-rung offset and close.
During the entire swing, the three support palms are actively held at their
measured world positions by support-leg IK; the floating base is never pinned.  The first body shift occurs after the two rear grippers
advance; the second occurs after the two front grippers advance.

Important scope
---------------
* The floating base is pinned only during the inherited static acquisition.
* The base remains free during every walking/contact-transition phase.
* This is a conservative crawl-gait baseline, not RAMP.
* It contains no dissertation metric calculations.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Mapping, Sequence

import mujoco
import numpy as np
from numpy.typing import NDArray

from controller.grasp_controller import StaticLadderGraspController


FloatArray = NDArray[np.float64]


class GaitPhase(Enum):
    ACQUIRING_STATIC_GRASP = auto()
    INITIAL_HOLD = auto()
    OPENING_SWING_GRIPPER = auto()
    RETRACTING_SWING_LEG = auto()
    POST_RELEASE_SETTLE = auto()
    LIFTING_TO_TRANSFER_CLEARANCE = auto()
    TRANSFERRING_SWING_LEG = auto()
    APPROACHING_TARGET_RUNG = auto()
    CLOSING_SWING_GRIPPER = auto()
    VERIFYING_REGRASP = auto()
    POST_REGRASP_SETTLE = auto()
    SHIFTING_BODY = auto()
    CYCLE_SETTLE = auto()
    COMPLETE = auto()
    FAILED = auto()


@dataclass(frozen=True)
class GaitStatus:
    phase: str
    swing_leg: str | None
    cycles_completed: int
    completed_regrasps: int
    contacting_grippers: tuple[str, ...]
    physical_contacts: int
    old_rung_contacts: int
    target_rung_contacts: int
    swing_tracking_error: float
    source_clearance: float
    selected_release_displacement: float
    base_position: tuple[float, float, float]
    base_linear_speed: float
    base_angular_speed: float
    complete: bool
    failed: bool
    message: str


class GaitController(StaticLadderGraspController):
    """One-gripper-at-a-time crawl gait on the fixed ladder."""

    DEFAULT_SWING_SEQUENCE: tuple[str, ...] = ("RR", "RL", "FR", "FL")

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData | None = None,
        *,
        max_cycles: int = 1,
        swing_sequence: Sequence[str] = DEFAULT_SWING_SEQUENCE,
        initial_hold_time: float = 1.0,
        opening_time: float = 1.00,
        opening_timeout: float = 3.00,
        finger_open_tolerance: float = 0.10,
        release_start_finger_tolerance: float = 0.35,
        peel_start_time: float = 0.55,
        peel_start_max_contacts: int = 1,
        peel_open_time: float = 1.20,
        transport_finger_tolerance: float = 0.40,
        swing_open_kp_scale: float = 1.75,
        retract_time: float = 2.50,
        release_extension_speed: float = 0.010,
        preflight_clearance_margin: float = 0.010,
        escape_direction_samples: int = 16,
        clearance_lift_time: float = 2.00,
        clearance_hold_time: float = 0.15,
        clearance_timeout: float = 4.00,
        transfer_time: float = 4.00,
        approach_time: float = 2.50,
        closing_time_gait: float = 1.50,
        release_hold_time: float = 0.15,
        release_timeout: float = 3.00,
        source_recontact_timeout: float = 0.10,
        regrasp_hold_time: float = 0.30,
        regrasp_timeout: float = 3.00,
        post_regrasp_settle_time: float = 0.70,
        body_shift_time: float = 4.00,
        cycle_settle_time: float = 1.00,
        release_clearance: float = 0.035,
        release_clearance_step: float = 0.005,
        release_clearance_max: float = 0.085,
        minimum_source_clearance: float = 0.010,
        clearance_comparison_tolerance: float = 0.0005,
        release_pose_tolerance: float = 0.025,
        post_release_settle_time: float = 0.75,
        post_release_settle_timeout: float = 3.00,
        settle_hold_time: float = 0.20,
        max_settle_linear_speed: float = 0.025,
        max_settle_angular_speed: float = 0.20,
        transfer_lift_height: float = 0.025,
        body_shift_scale: float = 1.0,
        minimum_regrasp_contacts: int = 1,
        support_loss_timeout: float = 0.25,
        ik_update_period: float = 0.010,
        ik_iterations_per_update: int = 20,
        ik_tolerance_gait: float = 0.003,
        ik_damping_gait: float = 3.0e-3,
        ik_step_limit_gait: float = 0.035,
        max_joint_target_rate: float = 0.60,
        max_swing_base_drift: float = 0.10,
        max_swing_tracking_error: float = 0.06,
        tracking_error_timeout: float = 0.40,
        preflight_ik_tolerance: float = 0.025,
        preflight_ik_iterations: int = 120,
        early_close_min_time: float = 0.40,
        stop_after_regrasps: int | None = None,
        stop_after_release_settle: bool = False,
        **static_grasp_kwargs: object,
    ) -> None:
        del data  # Constructor compatibility with run_climbing.py.
        super().__init__(model, **static_grasp_kwargs)

        if max_cycles < 1:
            raise ValueError("max_cycles must be at least 1.")
        if release_clearance <= 0.0:
            raise ValueError("release_clearance must be positive.")
        if opening_timeout <= opening_time:
            raise ValueError("opening_timeout must be greater than opening_time.")
        if finger_open_tolerance <= 0.0:
            raise ValueError("finger_open_tolerance must be positive.")
        if release_start_finger_tolerance < finger_open_tolerance:
            raise ValueError(
                "release_start_finger_tolerance must be >= finger_open_tolerance."
            )
        if peel_start_time <= 0.0:
            raise ValueError("peel_start_time must be positive.")
        if peel_start_max_contacts < 0:
            raise ValueError("peel_start_max_contacts must be non-negative.")
        if peel_open_time <= 0.0:
            raise ValueError("peel_open_time must be positive.")
        if transport_finger_tolerance <= 0.0:
            raise ValueError("transport_finger_tolerance must be positive.")
        if early_close_min_time < 0.0:
            raise ValueError("early_close_min_time must be non-negative.")
        if swing_open_kp_scale < 1.0:
            raise ValueError("swing_open_kp_scale must be at least 1.0.")
        if release_extension_speed <= 0.0:
            raise ValueError("release_extension_speed must be positive.")
        if preflight_clearance_margin < 0.0:
            raise ValueError("preflight_clearance_margin must be non-negative.")
        if escape_direction_samples < 8:
            raise ValueError("escape_direction_samples must be at least 8.")
        if release_clearance_step <= 0.0:
            raise ValueError("release_clearance_step must be positive.")
        if release_clearance_max < release_clearance:
            raise ValueError(
                "release_clearance_max must be >= release_clearance."
            )
        if minimum_source_clearance <= 0.0:
            raise ValueError("minimum_source_clearance must be positive.")
        if clearance_comparison_tolerance < 0.0:
            raise ValueError(
                "clearance_comparison_tolerance must be non-negative."
            )
        if release_pose_tolerance <= 0.0:
            raise ValueError("release_pose_tolerance must be positive.")
        if transfer_lift_height <= 0.0:
            raise ValueError("transfer_lift_height must be positive.")
        if clearance_timeout <= clearance_lift_time:
            raise ValueError(
                "clearance_timeout must be greater than clearance_lift_time."
            )
        if body_shift_scale <= 0.0:
            raise ValueError("body_shift_scale must be positive.")
        if minimum_regrasp_contacts < 1:
            raise ValueError("minimum_regrasp_contacts must be at least 1.")

        sequence = tuple(str(leg).upper() for leg in swing_sequence)
        if len(sequence) != 4 or set(sequence) != set(self.LEG_JOINTS):
            raise ValueError(
                "swing_sequence must contain FL, FR, RL and RR exactly once."
            )

        self.max_cycles = int(max_cycles)
        self.swing_sequence = sequence

        self.initial_hold_time = float(initial_hold_time)
        self.opening_time = float(opening_time)
        self.opening_timeout = float(opening_timeout)
        self.finger_open_tolerance = float(finger_open_tolerance)
        self.release_start_finger_tolerance = float(
            release_start_finger_tolerance
        )
        self.peel_start_time = float(peel_start_time)
        self.peel_start_max_contacts = int(peel_start_max_contacts)
        self.peel_open_time = float(peel_open_time)
        self.transport_finger_tolerance = float(transport_finger_tolerance)
        self.swing_open_kp_scale = float(swing_open_kp_scale)
        self.retract_time = float(retract_time)
        self.release_extension_speed = float(release_extension_speed)
        self.preflight_clearance_margin = float(preflight_clearance_margin)
        self.escape_direction_samples = int(escape_direction_samples)
        self.clearance_lift_time = float(clearance_lift_time)
        self.clearance_hold_time = float(clearance_hold_time)
        self.clearance_timeout = float(clearance_timeout)
        self.transfer_time = float(transfer_time)
        self.approach_time = float(approach_time)
        self.closing_time_gait = float(closing_time_gait)
        self.release_hold_time = float(release_hold_time)
        self.release_timeout = float(release_timeout)
        self.source_recontact_timeout = float(source_recontact_timeout)
        self.regrasp_hold_time = float(regrasp_hold_time)
        self.regrasp_timeout = float(regrasp_timeout)
        self.post_regrasp_settle_time = float(post_regrasp_settle_time)
        self.body_shift_time = float(body_shift_time)
        self.cycle_settle_time = float(cycle_settle_time)
        self.release_clearance = float(release_clearance)
        self.release_clearance_step = float(release_clearance_step)
        self.release_clearance_max = float(release_clearance_max)
        self.minimum_source_clearance = float(minimum_source_clearance)
        self.clearance_comparison_tolerance = float(
            clearance_comparison_tolerance
        )
        self.release_pose_tolerance = float(release_pose_tolerance)
        self.post_release_settle_time = float(post_release_settle_time)
        self.post_release_settle_timeout = float(post_release_settle_timeout)
        self.settle_hold_time = float(settle_hold_time)
        self.max_settle_linear_speed = float(max_settle_linear_speed)
        self.max_settle_angular_speed = float(max_settle_angular_speed)
        # V8 uses transfer_lift_height as the direct world +z lift height.
        self.transfer_lift_height = float(transfer_lift_height)
        self.body_shift_scale = float(body_shift_scale)
        self.minimum_regrasp_contacts = int(minimum_regrasp_contacts)
        self.support_loss_timeout = float(support_loss_timeout)

        self.ik_update_period = float(ik_update_period)
        self.ik_iterations_per_update = int(ik_iterations_per_update)
        self.ik_tolerance_gait = float(ik_tolerance_gait)
        self.ik_damping_gait = float(ik_damping_gait)
        self.ik_step_limit_gait = float(ik_step_limit_gait)
        self.max_joint_target_rate = float(max_joint_target_rate)
        self.max_swing_base_drift = float(max_swing_base_drift)
        self.max_swing_tracking_error = float(max_swing_tracking_error)
        self.tracking_error_timeout = float(tracking_error_timeout)
        self.preflight_ik_tolerance = float(preflight_ik_tolerance)
        self.preflight_ik_iterations = int(preflight_ik_iterations)
        self.early_close_min_time = float(early_close_min_time)
        if self.preflight_ik_iterations < 1:
            raise ValueError("preflight_ik_iterations must be at least 1.")
        self.stop_after_regrasps = (
            None if stop_after_regrasps is None else int(stop_after_regrasps)
        )
        self.stop_after_release_settle = bool(stop_after_release_settle)
        if self.stop_after_regrasps is not None and self.stop_after_regrasps < 1:
            raise ValueError("stop_after_regrasps must be at least 1 when set.")

        positive_durations = {
            "initial_hold_time": self.initial_hold_time,
            "opening_time": self.opening_time,
            "opening_timeout": self.opening_timeout,
            "finger_open_tolerance": self.finger_open_tolerance,
            "release_start_finger_tolerance": (
                self.release_start_finger_tolerance
            ),
            "peel_start_time": self.peel_start_time,
            "peel_open_time": self.peel_open_time,
            "transport_finger_tolerance": self.transport_finger_tolerance,
            "swing_open_kp_scale": self.swing_open_kp_scale,
            "retract_time": self.retract_time,
            "release_extension_speed": self.release_extension_speed,
            "preflight_clearance_margin": self.preflight_clearance_margin,
            "clearance_lift_time": self.clearance_lift_time,
            "clearance_hold_time": self.clearance_hold_time,
            "clearance_timeout": self.clearance_timeout,
            "transfer_time": self.transfer_time,
            "approach_time": self.approach_time,
            "closing_time_gait": self.closing_time_gait,
            "release_hold_time": self.release_hold_time,
            "release_timeout": self.release_timeout,
            "source_recontact_timeout": self.source_recontact_timeout,
            "regrasp_hold_time": self.regrasp_hold_time,
            "regrasp_timeout": self.regrasp_timeout,
            "post_regrasp_settle_time": self.post_regrasp_settle_time,
            "body_shift_time": self.body_shift_time,
            "cycle_settle_time": self.cycle_settle_time,
            "support_loss_timeout": self.support_loss_timeout,
            "ik_update_period": self.ik_update_period,
            "max_swing_base_drift": self.max_swing_base_drift,
            "max_swing_tracking_error": self.max_swing_tracking_error,
            "tracking_error_timeout": self.tracking_error_timeout,
            "preflight_ik_tolerance": self.preflight_ik_tolerance,
            "minimum_source_clearance": self.minimum_source_clearance,
            "release_pose_tolerance": self.release_pose_tolerance,
            "post_release_settle_time": self.post_release_settle_time,
            "post_release_settle_timeout": self.post_release_settle_timeout,
            "settle_hold_time": self.settle_hold_time,
            "max_settle_linear_speed": self.max_settle_linear_speed,
            "max_settle_angular_speed": self.max_settle_angular_speed,
            "release_clearance_step": self.release_clearance_step,
            "release_clearance_max": self.release_clearance_max,
            "transfer_lift_height": self.transfer_lift_height,
        }
        for name, value in positive_durations.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be positive.")

        self._ik_data = mujoco.MjData(model)

        # Collision geom ids for each six-segment gripper.  These are used
        # with mj_geomDistance so that release is based on a positive
        # surface-to-surface clearance, not merely the absence of a contact
        # point in one simulation step.
        self._finger_geom_ids_by_leg: dict[str, tuple[int, ...]] = {}
        for leg in self.LEG_JOINTS:
            geom_ids = [
                geom_id
                for geom_id in range(model.ngeom)
                if self._finger_body_to_leg.get(int(model.geom_bodyid[geom_id]))
                == leg
            ]
            if len(geom_ids) != 6:
                raise ValueError(
                    f"Expected six finger collision geoms for {leg}, "
                    f"found {len(geom_ids)}."
                )
            self._finger_geom_ids_by_leg[leg] = tuple(geom_ids)

        self._phase = GaitPhase.ACQUIRING_STATIC_GRASP
        self._phase_start_time = 0.0
        self._message = "Controller has not been reset."

        self._cycles_completed = 0
        self._completed_regrasps = 0
        self._sequence_index = 0
        self._swing_leg: str | None = None

        self._palm_anchors: dict[str, FloatArray] = {}
        self._rung_names: list[str] = []
        self._rung_geom_ids_sorted: list[int] = []
        self._rung_positions: list[FloatArray] = []
        self._leg_rung_index: dict[str, int] = {}

        self._swing_start = np.zeros(3, dtype=float)
        self._swing_clear_start = np.zeros(3, dtype=float)
        self._clearance_lift_start = np.zeros(3, dtype=float)
        self._source_transfer_clear = np.zeros(3, dtype=float)
        self._transfer_start = np.zeros(3, dtype=float)
        self._target_transfer_clear = np.zeros(3, dtype=float)
        self._swing_clear_goal = np.zeros(3, dtype=float)
        self._swing_goal = np.zeros(3, dtype=float)
        self._source_rung_name: str | None = None
        self._target_rung_name: str | None = None
        self._target_rung_index: int | None = None

        self._body_hold_qpos = np.zeros(7, dtype=float)
        self._swing_base_reference_qpos = np.zeros(7, dtype=float)
        self._body_shift_start_qpos = np.zeros(7, dtype=float)
        self._body_shift_goal_qpos = np.zeros(7, dtype=float)
        self._last_body_shift_goal_qpos = np.zeros(7, dtype=float)

        self._release_verified_since: float | None = None
        self._settle_verified_since: float | None = None
        self._clearance_verified_since: float | None = None
        self._source_recontact_since: float | None = None
        self._regrasp_verified_since: float | None = None
        self._support_loss_since: float | None = None
        self._tracking_error_since: float | None = None
        self._body_shift_stage = 0
        self._body_shifts_completed = 0
        self._last_ik_update_time = -np.inf
        self._selected_release_displacement = float("nan")
        self._release_command_displacement = float("nan")
        self._release_direction = np.array([0.0, 0.0, 1.0], dtype=float)
        self._rung_translation = np.zeros(3, dtype=float)
        self._closest_source_segment = ""
        self._finger_open_error = float("nan")
        self._peel_finger_start_pose = self.open_finger_pose.copy()
        self._last_desired_palms: dict[str, FloatArray] = {}
        self._last_ik_residuals: dict[str, float] = {
            leg: np.inf for leg in self.LEG_JOINTS
        }

        self._last_gait_status = GaitStatus(
            phase=self._phase.name,
            swing_leg=None,
            cycles_completed=0,
            completed_regrasps=0,
            contacting_grippers=(),
            physical_contacts=0,
            old_rung_contacts=0,
            target_rung_contacts=0,
            swing_tracking_error=float("nan"),
            source_clearance=float("nan"),
            selected_release_displacement=float("nan"),
            base_position=(0.0, 0.0, 0.0),
            base_linear_speed=0.0,
            base_angular_speed=0.0,
            complete=False,
            failed=False,
            message=self._message,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        super().reset(model, data)

        self._phase = GaitPhase.ACQUIRING_STATIC_GRASP
        self._phase_start_time = float(data.time)
        self._message = "Acquiring the verified four-gripper static grasp."

        self._cycles_completed = 0
        self._completed_regrasps = 0
        self._sequence_index = 0
        self._swing_leg = None
        self._palm_anchors = {}
        self._leg_rung_index = {}
        self._source_rung_name = None
        self._target_rung_name = None
        self._clearance_lift_start[:] = 0.0
        self._source_transfer_clear[:] = 0.0
        self._transfer_start[:] = 0.0
        self._target_transfer_clear[:] = 0.0
        self._target_rung_index = None
        self._release_verified_since = None
        self._settle_verified_since = None
        self._clearance_verified_since = None
        self._source_recontact_since = None
        self._regrasp_verified_since = None
        self._support_loss_since = None
        self._tracking_error_since = None
        self._body_shift_stage = 0
        self._body_shifts_completed = 0
        self._last_ik_update_time = -np.inf
        self._selected_release_displacement = float("nan")
        self._release_command_displacement = float("nan")
        self._release_direction = np.array([0.0, 0.0, 1.0], dtype=float)
        self._rung_translation = np.zeros(3, dtype=float)
        self._closest_source_segment = ""
        self._finger_open_error = float("nan")
        self._peel_finger_start_pose = self.open_finger_pose.copy()

        self._build_rung_table(model, data)
        self._last_gait_status = self._make_status(model, data)

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> GaitStatus:
        if model is not self.model:
            raise ValueError("Controller was constructed for a different model.")

        if self._phase is GaitPhase.ACQUIRING_STATIC_GRASP:
            static_status = super().update(model, data)

            if static_status.state == "FAILED":
                self._fail("Static grasp acquisition failed; gait was not started.")

            elif static_status.state == "HOLDING":
                self._initialise_gait_after_static_grasp(model, data)

            self._last_gait_status = self._make_status(model, data)
            return self._last_gait_status

        contacts = self._detect_contacts(model, data)
        self._check_support_contacts(data, contacts)

        desired_palms, virtual_base_qpos = self._desired_targets(data)
        self._last_desired_palms = {
            leg: np.asarray(target, dtype=float).copy()
            for leg, target in desired_palms.items()
        }
        self._check_swing_motion_safety(data)
        self._refresh_leg_targets_if_due(
            model,
            data,
            desired_palms=desired_palms,
            virtual_base_qpos=virtual_base_qpos,
        )
        self._apply_leg_pd(data)
        self._apply_per_leg_finger_pd(data, self._finger_targets(data))

        if self._phase not in (GaitPhase.FAILED, GaitPhase.COMPLETE):
            self._advance_phase(model, data, contacts)

        self._last_gait_status = self._make_status(model, data)
        return self._last_gait_status

    def post_step(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        # Inherited post_step pins the base only before static grasp verification.
        # Once the inherited state reaches HOLDING, base_pinned is False and this
        # method becomes a no-op.  The gait itself never kinematically pins base.
        super().post_step(model, data)

    @property
    def gait_status(self) -> GaitStatus:
        return self._last_gait_status

    @property
    def gait_complete(self) -> bool:
        return self._phase is GaitPhase.COMPLETE

    @property
    def gait_failed(self) -> bool:
        return self._phase is GaitPhase.FAILED

    @property
    def phase(self) -> str:
        return self._phase.name

    @property
    def last_ik_residuals(self) -> Mapping[str, float]:
        return dict(self._last_ik_residuals)

    # ------------------------------------------------------------------
    # Gait initialisation and rung bookkeeping
    # ------------------------------------------------------------------

    def _build_rung_table(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> None:
        rows: list[tuple[float, int, str, FloatArray]] = []
        for geom_id in self._rung_geom_ids:
            name = mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                int(geom_id),
            )
            if name is None:
                continue
            position = np.asarray(data.geom_xpos[int(geom_id)], dtype=float).copy()
            rows.append((float(position[0]), int(geom_id), name, position))

        rows.sort(key=lambda item: item[0])
        if len(rows) < 2:
            raise RuntimeError("At least two rung_* geoms are required for gait.")

        self._rung_geom_ids_sorted = [row[1] for row in rows]
        self._rung_names = [row[2] for row in rows]
        self._rung_positions = [row[3] for row in rows]

    def _initialise_gait_after_static_grasp(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> None:
        # Use the *measured* palm positions and the *measured* rung contacts.
        # The inherited static controller currently reports large residuals
        # between its nominal palm_targets and the realised grasp.  Reusing
        # those nominal targets here would command an unsafe discontinuous
        # jump as soon as the gait begins.
        self._palm_anchors = {
            leg: np.asarray(
                data.xpos[self._palm_body_ids[leg]],
                dtype=float,
            ).copy()
            for leg in self.LEG_JOINTS
        }

        contacts = self._detect_contacts(model, data)
        details = contacts.get("contact_details", [])
        if not isinstance(details, list):
            self._fail("Static contact details are unavailable after grasp verification.")
            return

        rung_index_by_name = {
            name: index for index, name in enumerate(self._rung_names)
        }
        leg_rung_index: dict[str, int] = {}

        for leg in self.LEG_JOINTS:
            counts: dict[str, int] = {}
            for item in details:
                if not isinstance(item, dict) or item.get("leg") != leg:
                    continue
                rung_name = item.get("rung_geom_name")
                if isinstance(rung_name, str) and rung_name in rung_index_by_name:
                    counts[rung_name] = counts.get(rung_name, 0) + 1

            if not counts:
                self._fail(
                    f"No verified rung contact was found for {leg} at gait start."
                )
                return

            # Prefer the rung with the most physical contacts.  Resolve ties
            # by choosing the rung whose x coordinate is closest to the
            # measured palm position.
            palm_x = float(self._palm_anchors[leg][0])
            selected_name = min(
                counts,
                key=lambda name: (
                    -counts[name],
                    abs(
                        float(self._rung_positions[rung_index_by_name[name]][0])
                        - palm_x
                    ),
                ),
            )
            leg_rung_index[leg] = rung_index_by_name[selected_name]

        self._leg_rung_index = leg_rung_index

        print("\nMeasured gait start mapping")
        print("-" * 72)
        for leg in self.LEG_JOINTS:
            rung_name = self._rung_names[self._leg_rung_index[leg]]
            palm = self._palm_anchors[leg]
            print(
                f"{leg}: rung={rung_name:8s} | "
                f"palm=({palm[0]:+.4f}, {palm[1]:+.4f}, {palm[2]:+.4f})"
            )
        print("-" * 72)

        self._body_hold_qpos = self._current_base_qpos(data)
        self._last_body_shift_goal_qpos = self._body_hold_qpos.copy()
        self._set_phase(
            GaitPhase.INITIAL_HOLD,
            data,
            "Static grasp verified; base released. Holding before first step.",
        )

    # ------------------------------------------------------------------
    # Phase logic
    # ------------------------------------------------------------------

    def _advance_phase(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        contacts: Mapping[str, object],
    ) -> None:
        elapsed = self._phase_elapsed(data)

        if self._phase is GaitPhase.INITIAL_HOLD:
            if elapsed >= self.initial_hold_time:
                self._sequence_index = 0
                self._prepare_next_swing(data)

        elif self._phase is GaitPhase.OPENING_SWING_GRIPPER:
            self._finger_open_error = self._swing_finger_open_error(data)
            old_contact_count = self._count_leg_rung_contacts(
                contacts, self._swing_leg, self._source_rung_name
            )

            # Do not wait for a loaded finger chain to reach the exact open
            # angle.  After a short opening preload, begin the requested +z
            # lift while continuing to drive all six swing fingers open.
            if elapsed >= self.peel_start_time:
                if self._swing_leg is None:
                    self._fail("Swing leg is missing while starting vertical lift.")
                    return
                self._peel_finger_start_pose = np.asarray(
                    data.qpos[self._finger_qpos_adrs[self._swing_leg]],
                    dtype=float,
                ).copy()
                self._release_verified_since = None
                self._set_phase(
                    GaitPhase.RETRACTING_SWING_LEG,
                    data,
                    f"{self._swing_leg}: opening preload complete; source "
                    f"contacts={old_contact_count}, finger error="
                    f"{self._finger_open_error:.3f} rad. Beginning direct "
                    f"world +z lift while continuing to open the gripper.",
                )
            elif elapsed >= self.opening_timeout:
                self._fail(
                    f"{self._swing_leg} did not reach the vertical-lift start "
                    "time before the opening timeout."
                )

        elif self._phase is GaitPhase.RETRACTING_SWING_LEG:
            if self._swing_leg is None:
                self._fail("Swing leg is missing during vertical release.")
                return

            old_contact_count = self._count_leg_rung_contacts(
                contacts, self._swing_leg, self._source_rung_name
            )
            actual_palm = np.asarray(
                data.xpos[self._palm_body_ids[self._swing_leg]], dtype=float
            )
            desired = np.asarray(
                self._last_desired_palms.get(
                    self._swing_leg, self._source_transfer_clear
                ),
                dtype=float,
            )
            tracking_error = float(np.linalg.norm(actual_palm - desired))
            lift_achieved = float(actual_palm[2] - self._swing_start[2])
            self._finger_open_error = self._swing_finger_open_error(data)
            source_clearance, closest_segment = (
                self._minimum_leg_rung_clearance_details(
                    model, data, self._swing_leg, self._source_rung_name
                )
            )
            self._closest_source_segment = closest_segment

            release_ready = bool(
                old_contact_count == 0
                and lift_achieved >= self.transfer_lift_height - 0.020
                and tracking_error <= self.release_pose_tolerance
            )
            if release_ready:
                if self._release_verified_since is None:
                    self._release_verified_since = float(data.time)
                released_duration = (
                    float(data.time) - self._release_verified_since
                )
            else:
                self._release_verified_since = None
                released_duration = 0.0

            if released_duration >= self.release_hold_time:
                # Use the commanded high waypoint, not a drifting measured
                # offset, so the next segment remains tied to the XML rungs.
                self._swing_clear_start = self._source_transfer_clear.copy()
                self._transfer_start = self._source_transfer_clear.copy()
                self._settle_verified_since = None
                self._set_phase(
                    GaitPhase.POST_RELEASE_SETTLE,
                    data,
                    f"{self._swing_leg}: vertical release verified; old "
                    f"contacts=0 for {released_duration:.3f} s, actual lift="
                    f"{lift_achieved:.4f} m, tracking error="
                    f"{tracking_error:.4f} m. Holding before horizontal "
                    f"translation to {self._target_rung_name}.",
                )
            elif elapsed >= self.release_timeout:
                self._fail(
                    f"{self._swing_leg} could not complete direct vertical "
                    f"release from {self._source_rung_name}; old contacts="
                    f"{old_contact_count}, actual lift={lift_achieved:.4f} m, "
                    f"commanded lift={self._release_command_displacement:.4f} "
                    f"m, palm error={tracking_error:.4f} m, closest segment="
                    f"{closest_segment or 'unknown'}, finger error="
                    f"{self._finger_open_error:.4f} rad."
                )

        elif self._phase is GaitPhase.POST_RELEASE_SETTLE:
            if self._swing_leg is None:
                self._fail("Swing leg is missing during post-release hold.")
                return
            old_contact_count = self._count_leg_rung_contacts(
                contacts, self._swing_leg, self._source_rung_name
            )
            linear_speed, angular_speed = self._base_speed_norms(data)

            # Development-mode relaxation: do not require a 10 mm signed
            # geom-distance margin or very low base speed before transfer.
            # We still require true source-contact loss and all three support
            # grippers; _check_support_contacts enforces the latter.
            if old_contact_count == 0:
                if self._settle_verified_since is None:
                    self._settle_verified_since = float(data.time)
                clear_duration = float(data.time) - self._settle_verified_since
            else:
                self._settle_verified_since = None
                clear_duration = 0.0

            if (
                elapsed >= self.post_release_settle_time
                and clear_duration >= self.settle_hold_time
            ):
                if self.stop_after_release_settle:
                    self._set_phase(
                        GaitPhase.COMPLETE,
                        data,
                        f"{self._swing_leg}: direct vertical release held "
                        "successfully; stopped before transfer as requested.",
                    )
                    return
                self._transfer_start = self._source_transfer_clear.copy()
                self._source_recontact_since = None
                self._set_phase(
                    GaitPhase.TRANSFERRING_SWING_LEG,
                    data,
                    f"{self._swing_leg}: vertical release held for "
                    f"{clear_duration:.3f} s. Translating +x by XML rung "
                    f"spacing {self._rung_translation[0]:.3f} m at fixed "
                    f"height; base speeds={linear_speed:.3f} m/s and "
                    f"{angular_speed:.3f} rad/s.",
                )
            elif elapsed >= self.post_release_settle_timeout:
                self._fail(
                    f"{self._swing_leg} re-contacted {self._source_rung_name} "
                    f"during the high hold; old contacts={old_contact_count}."
                )

        elif self._phase is GaitPhase.LIFTING_TO_TRANSFER_CLEARANCE:
            old_contact_count = self._count_leg_rung_contacts(
                contacts,
                self._swing_leg,
                self._source_rung_name,
            )
            if self._swing_leg is None:
                self._fail("Swing leg is missing during clearance lift.")
                return

            actual_palm = np.asarray(
                data.xpos[self._palm_body_ids[self._swing_leg]],
                dtype=float,
            )
            source_clearance = self._minimum_leg_rung_clearance(
                model,
                data,
                self._swing_leg,
                self._source_rung_name,
            )

            # Once genuine geometric release has been accepted, falling back
            # below the same clearance margin is a real source-rung recontact,
            # not a harmless one-step contact flicker.
            release_lost = bool(
                old_contact_count > 0
                or not self._clearance_is_sufficient(source_clearance)
            )
            if release_lost:
                if self._source_recontact_since is None:
                    self._source_recontact_since = float(data.time)
                elif (
                    float(data.time) - self._source_recontact_since
                    >= self.source_recontact_timeout
                ):
                    self._fail(
                        f"{self._swing_leg} lost geometric source clearance "
                        f"during clearance lift; contacts={old_contact_count}, "
                        f"source clearance={source_clearance:.4f} m."
                    )
                    return
            else:
                self._source_recontact_since = None

            height_ready = bool(
                actual_palm[2] >= self._source_transfer_clear[2] - 0.015
            )
            if (
                elapsed >= self.clearance_lift_time
                and old_contact_count == 0
                and self._clearance_is_sufficient(source_clearance)
                and height_ready
            ):
                if self._clearance_verified_since is None:
                    self._clearance_verified_since = float(data.time)
                clear_duration = (
                    float(data.time) - self._clearance_verified_since
                )
            else:
                self._clearance_verified_since = None
                clear_duration = 0.0

            if clear_duration >= self.clearance_hold_time:
                self._transfer_start = actual_palm.copy()
                self._source_recontact_since = None
                self._set_phase(
                    GaitPhase.TRANSFERRING_SWING_LEG,
                    data,
                    f"{self._swing_leg}: high transfer clearance verified "
                    f"for {clear_duration:.3f} s; translating to "
                    f"{self._target_rung_name}.",
                )
            elif elapsed >= self.clearance_timeout:
                self._fail(
                    f"{self._swing_leg} could not reach a contact-free high "
                    f"transfer corridor; old-rung contacts={old_contact_count}, "
                    f"source clearance={source_clearance:.4f} m, "
                    f"palm_z={actual_palm[2]:+.4f} m, required_z>="
                    f"{self._source_transfer_clear[2] - 0.015:+.4f} m."
                )

        elif self._phase is GaitPhase.TRANSFERRING_SWING_LEG:
            old_contact_count = self._count_leg_rung_contacts(
                contacts,
                self._swing_leg,
                self._source_rung_name,
            )
            if old_contact_count > 0:
                if self._source_recontact_since is None:
                    self._source_recontact_since = float(data.time)
                elif (
                    float(data.time) - self._source_recontact_since
                    >= self.source_recontact_timeout
                ):
                    self._fail(
                        f"{self._swing_leg} persistently re-contacted "
                        f"{self._source_rung_name} during transfer; "
                        f"old-rung contacts={old_contact_count}."
                    )
            else:
                self._source_recontact_since = None

            if (
                self._phase is GaitPhase.TRANSFERRING_SWING_LEG
                and elapsed >= self.transfer_time
            ):
                self._set_phase(
                    GaitPhase.APPROACHING_TARGET_RUNG,
                    data,
                    f"{self._swing_leg}: approaching {self._target_rung_name} "
                    "from the cleared transfer pose.",
                )

        elif self._phase is GaitPhase.APPROACHING_TARGET_RUNG:
            target_contact_count = self._count_leg_rung_contacts(
                contacts, self._swing_leg, self._target_rung_name
            )
            if (
                target_contact_count > 0
                and elapsed >= self.early_close_min_time
            ):
                self._set_phase(
                    GaitPhase.CLOSING_SWING_GRIPPER,
                    data,
                    f"{self._swing_leg}: target contact detected early on "
                    f"{self._target_rung_name}; closing without pushing farther "
                    "through the rung.",
                )
            elif elapsed >= self.approach_time:
                self._set_phase(
                    GaitPhase.CLOSING_SWING_GRIPPER,
                    data,
                    f"{self._swing_leg}: closing around {self._target_rung_name}.",
                )

        elif self._phase is GaitPhase.CLOSING_SWING_GRIPPER:
            target_contact_count = self._count_leg_rung_contacts(
                contacts, self._swing_leg, self._target_rung_name
            )
            if (
                target_contact_count > 0
                and elapsed >= self.early_close_min_time
            ) or elapsed >= self.closing_time_gait:
                self._regrasp_verified_since = None
                self._set_phase(
                    GaitPhase.VERIFYING_REGRASP,
                    data,
                    f"{self._swing_leg}: verifying contact on {self._target_rung_name}.",
                )

        elif self._phase is GaitPhase.VERIFYING_REGRASP:
            if self._regrasp_is_valid(contacts):
                if self._regrasp_verified_since is None:
                    self._regrasp_verified_since = float(data.time)
                elif (
                    float(data.time) - self._regrasp_verified_since
                    >= self.regrasp_hold_time
                ):
                    self._accept_regrasp(data)
            else:
                self._regrasp_verified_since = None

            if (
                self._phase is GaitPhase.VERIFYING_REGRASP
                and elapsed >= self.regrasp_timeout
            ):
                self._fail(
                    f"{self._swing_leg} did not establish a persistent contact "
                    f"on {self._target_rung_name} before timeout."
                )

        elif self._phase is GaitPhase.POST_REGRASP_SETTLE:
            if elapsed >= self.post_regrasp_settle_time:
                if (
                    self.stop_after_regrasps is not None
                    and self._completed_regrasps >= self.stop_after_regrasps
                ):
                    self._swing_leg = None
                    self._set_phase(
                        GaitPhase.COMPLETE,
                        data,
                        f"Stopped after {self._completed_regrasps} verified "
                        "regrasp(s), as requested.",
                    )
                    return

                self._sequence_index += 1
                if self._sequence_index in (2, 4):
                    self._start_body_shift(data)
                elif self._sequence_index < len(self.swing_sequence):
                    self._prepare_next_swing(data)
                else:
                    self._fail("Unexpected gait sequence index after regrasp.")

        elif self._phase is GaitPhase.SHIFTING_BODY:
            if elapsed >= self.body_shift_time:
                self._set_phase(
                    GaitPhase.CYCLE_SETTLE,
                    data,
                    "Body shift complete; settling with all four grippers attached.",
                )

        elif self._phase is GaitPhase.CYCLE_SETTLE:
            if elapsed >= self.cycle_settle_time:
                self._body_shifts_completed += 1
                if self._body_shift_stage == 1:
                    self._prepare_next_swing(data)
                elif self._body_shift_stage == 2:
                    self._cycles_completed += 1
                    if self._cycles_completed >= self.max_cycles:
                        self._swing_leg = None
                        self._set_phase(
                            GaitPhase.COMPLETE,
                            data,
                            f"Completed {self._cycles_completed} crawl cycle(s).",
                        )
                    else:
                        self._sequence_index = 0
                        self._prepare_next_swing(data)
                else:
                    self._fail("Invalid body-shift stage.")

    def _prepare_next_swing(self, data: mujoco.MjData) -> None:
        """Prepare one deterministic vertical-lift step.

        V8 intentionally avoids the V7 oblique escape search.  The ladder XML
        places every rung on the same z plane and spaces adjacent rungs by
        0.225 m along +x.  We therefore preserve the measured palm-to-rung
        offset and use three explicit waypoints:

            source palm -> source palm + [0, 0, lift]
            -> previous waypoint + rung translation
            -> measured target palm offset

        The three supporting palm anchors remain fixed in world coordinates.
        """
        while self._sequence_index < len(self.swing_sequence):
            leg = self.swing_sequence[self._sequence_index]
            current_index = self._leg_rung_index[leg]
            next_index = current_index + 1

            if next_index >= len(self._rung_names):
                self._sequence_index += 1
                continue

            self._swing_leg = leg
            self._source_rung_name = self._rung_names[current_index]
            self._target_rung_index = next_index
            self._target_rung_name = self._rung_names[next_index]

            # Re-anchor every palm to the measured free-base pose immediately
            # before the step.  These are the world-space support references.
            for measured_leg in self.LEG_JOINTS:
                self._palm_anchors[measured_leg] = np.asarray(
                    data.xpos[self._palm_body_ids[measured_leg]], dtype=float
                ).copy()

            self._swing_start = self._palm_anchors[leg].copy()
            self._rung_translation = (
                self._rung_positions[next_index]
                - self._rung_positions[current_index]
            ).copy()
            self._swing_goal = self._swing_start + self._rung_translation

            lift_vector = np.array(
                [0.0, 0.0, self.transfer_lift_height], dtype=float
            )
            self._release_direction = np.array([0.0, 0.0, 1.0], dtype=float)
            self._selected_release_displacement = self.transfer_lift_height
            self._release_command_displacement = 0.0
            self._swing_clear_start = self._swing_start + lift_vector
            self._source_transfer_clear = self._swing_clear_start.copy()
            self._target_transfer_clear = (
                self._source_transfer_clear + self._rung_translation
            )
            self._swing_clear_goal = self._target_transfer_clear.copy()
            self._clearance_lift_start = self._swing_start.copy()
            self._transfer_start = self._source_transfer_clear.copy()

            # Initialise all leg targets from the measured state.  Unlike V7,
            # support legs will be re-solved online to hold their palms fixed.
            for item in self.LEG_JOINTS:
                self._leg_target_qpos[item] = np.asarray(
                    data.qpos[self._leg_qpos_adrs[item]], dtype=float
                ).copy()

            self._body_hold_qpos = self._current_base_qpos(data)
            self._swing_base_reference_qpos = self._body_hold_qpos.copy()
            self._release_verified_since = None
            self._settle_verified_since = None
            self._clearance_verified_since = None
            self._source_recontact_since = None
            self._regrasp_verified_since = None
            self._support_loss_since = None
            self._tracking_error_since = None

            if not self._preflight_swing_targets(data, leg):
                return

            self._set_phase(
                GaitPhase.OPENING_SWING_GRIPPER,
                data,
                f"Starting direct vertical {leg} step from "
                f"{self._source_rung_name} to {self._target_rung_name}; "
                f"lift={self.transfer_lift_height:.3f} m, XML rung "
                f"translation=({self._rung_translation[0]:+.3f}, "
                f"{self._rung_translation[1]:+.3f}, "
                f"{self._rung_translation[2]:+.3f}) m.",
            )
            return

        self._fail("No forward rung is available for the requested gait cycle.")

    def _accept_regrasp(self, data: mujoco.MjData) -> None:
        if self._swing_leg is None or self._target_rung_index is None:
            self._fail("Internal regrasp state is incomplete.")
            return

        self._palm_anchors[self._swing_leg] = np.asarray(
            data.xpos[self._palm_body_ids[self._swing_leg]],
            dtype=float,
        ).copy()
        self._leg_rung_index[self._swing_leg] = self._target_rung_index
        self._leg_target_qpos[self._swing_leg] = np.asarray(
            data.qpos[self._leg_qpos_adrs[self._swing_leg]],
            dtype=float,
        ).copy()
        self._completed_regrasps += 1
        self._regrasp_verified_since = None

        self._set_phase(
            GaitPhase.POST_REGRASP_SETTLE,
            data,
            f"{self._swing_leg} regrasp verified on {self._target_rung_name}.",
        )

    def _start_body_shift(self, data: mujoco.MjData) -> None:
        rung_spacings = np.diff(
            np.array([position[0] for position in self._rung_positions], dtype=float)
        )
        nominal_spacing = float(np.median(rung_spacings))

        self._swing_leg = None
        self._body_shift_stage = 1 if self._sequence_index == 2 else 2
        half_shift = 0.5 * self.body_shift_scale * nominal_spacing

        # Anchor the body-shift IK to the currently measured grasp geometry.
        for leg in self.LEG_JOINTS:
            self._palm_anchors[leg] = np.asarray(
                data.xpos[self._palm_body_ids[leg]],
                dtype=float,
            ).copy()
            self._leg_target_qpos[leg] = np.asarray(
                data.qpos[self._leg_qpos_adrs[leg]],
                dtype=float,
            ).copy()

        self._body_shift_start_qpos = self._current_base_qpos(data)
        self._body_shift_goal_qpos = self._body_shift_start_qpos.copy()
        self._body_shift_goal_qpos[0] += half_shift
        self._last_body_shift_goal_qpos = self._body_shift_goal_qpos.copy()
        self._support_loss_since = None
        self._tracking_error_since = None

        self._set_phase(
            GaitPhase.SHIFTING_BODY,
            data,
            f"Starting body-shift stage {self._body_shift_stage}; "
            f"desired forward displacement={half_shift:.4f} m.",
        )

    def _set_phase(
        self,
        phase: GaitPhase,
        data: mujoco.MjData,
        message: str,
    ) -> None:
        self._phase = phase
        self._phase_start_time = float(data.time)
        self._message = message
        print(f"[t={float(data.time):7.3f}] {phase.name}: {message}")

    def _fail(self, message: str) -> None:
        self._phase = GaitPhase.FAILED
        self._message = message
        print(f"GAIT FAILED: {message}")

    # ------------------------------------------------------------------
    # Desired trajectories
    # ------------------------------------------------------------------

    def _desired_targets(
        self,
        data: mujoco.MjData,
    ) -> tuple[dict[str, FloatArray], FloatArray | None]:
        palms = {leg: target.copy() for leg, target in self._palm_anchors.items()}

        if not palms:
            palms = {
                leg: np.asarray(self.palm_targets[leg], dtype=float).copy()
                for leg in self.LEG_JOINTS
            }

        # During release, transfer and regrasp, solve IK from the *actual*
        # floating-base pose.  Using the old held base pose caused the joint
        # targets to ignore base drift and allowed a stuck swing gripper to
        # drag the entire robot instead of following its world-space path.
        virtual_base: FloatArray | None = self._current_base_qpos(data)
        elapsed = self._phase_elapsed(data)

        if self._phase is GaitPhase.POST_RELEASE_SETTLE and self._swing_leg:
            palms[self._swing_leg] = self._swing_clear_start.copy()

        elif self._phase is GaitPhase.RETRACTING_SWING_LEG and self._swing_leg:
            # Direct world +z lift.  If source contacts remain after the
            # nominal lift, continue upward slowly up to release_clearance_max.
            nominal = float(self.transfer_lift_height)
            if elapsed <= self.retract_time:
                commanded = nominal * self._smooth_progress(
                    elapsed, self.retract_time
                )
            else:
                commanded = min(
                    self.release_clearance_max,
                    nominal + self.release_extension_speed
                    * (elapsed - self.retract_time),
                )
            self._release_command_displacement = float(commanded)
            palms[self._swing_leg] = self._swing_start + np.array(
                [0.0, 0.0, self._release_command_displacement], dtype=float
            )

        elif (
            self._phase is GaitPhase.LIFTING_TO_TRANSFER_CLEARANCE
            and self._swing_leg
        ):
            alpha = self._smooth_progress(elapsed, self.clearance_lift_time)
            palms[self._swing_leg] = self._lerp(
                self._clearance_lift_start,
                self._source_transfer_clear,
                alpha,
            )

        elif self._phase is GaitPhase.TRANSFERRING_SWING_LEG and self._swing_leg:
            alpha = self._smooth_progress(elapsed, self.transfer_time)
            palms[self._swing_leg] = self._lerp(
                self._transfer_start,
                self._target_transfer_clear,
                alpha,
            )

        elif self._phase is GaitPhase.APPROACHING_TARGET_RUNG and self._swing_leg:
            alpha = self._smooth_progress(elapsed, self.approach_time)
            palms[self._swing_leg] = self._lerp(
                self._target_transfer_clear,
                self._swing_goal,
                alpha,
            )

        elif self._phase in (
            GaitPhase.CLOSING_SWING_GRIPPER,
            GaitPhase.VERIFYING_REGRASP,
        ) and self._swing_leg:
            palms[self._swing_leg] = self._swing_goal.copy()

        elif self._phase is GaitPhase.SHIFTING_BODY:
            alpha = self._smooth_progress(elapsed, self.body_shift_time)
            virtual_base = self._interpolate_base_qpos(
                self._body_shift_start_qpos,
                self._body_shift_goal_qpos,
                alpha,
            )

        elif self._phase in (GaitPhase.CYCLE_SETTLE, GaitPhase.COMPLETE):
            virtual_base = self._last_body_shift_goal_qpos.copy()

        elif self._phase is GaitPhase.FAILED:
            virtual_base = self._current_base_qpos(data)

        return palms, virtual_base

    def _finger_targets(self, data: mujoco.MjData) -> dict[str, FloatArray]:
        targets = {
            leg: self.closed_finger_pose.copy()
            for leg in self.LEG_JOINTS
        }
        if self._swing_leg is None:
            return targets

        elapsed = self._phase_elapsed(data)

        if self._phase is GaitPhase.OPENING_SWING_GRIPPER:
            alpha = self._smooth_progress(elapsed, self.opening_time)
            targets[self._swing_leg] = self._lerp(
                self.closed_finger_pose,
                self.open_finger_pose,
                alpha,
            )

        elif self._phase is GaitPhase.RETRACTING_SWING_LEG:
            alpha = self._smooth_progress(elapsed, self.peel_open_time)
            targets[self._swing_leg] = self._lerp(
                self._peel_finger_start_pose,
                self.open_finger_pose,
                alpha,
            )

        elif self._phase in (
            GaitPhase.POST_RELEASE_SETTLE,
            GaitPhase.LIFTING_TO_TRANSFER_CLEARANCE,
            GaitPhase.TRANSFERRING_SWING_LEG,
            GaitPhase.APPROACHING_TARGET_RUNG,
        ):
            targets[self._swing_leg] = self.open_finger_pose.copy()

        elif self._phase is GaitPhase.CLOSING_SWING_GRIPPER:
            alpha = self._smooth_progress(elapsed, self.closing_time_gait)
            targets[self._swing_leg] = self._lerp(
                self.open_finger_pose,
                self.closed_finger_pose,
                alpha,
            )

        return targets

    def _clearance_is_sufficient(self, value: float) -> bool:
        return bool(
            np.isfinite(value)
            and value + self.clearance_comparison_tolerance
            >= self.minimum_source_clearance
        )

    def _candidate_escape_directions(
        self,
        data: mujoco.MjData,
        leg: str,
        source_rung: FloatArray,
    ) -> tuple[FloatArray, ...]:
        """Return unique x-z escape directions, contact geometry first."""
        candidates: list[FloatArray] = []

        contacts = self._detect_contacts(self.model, data)
        details = contacts.get("contact_details", [])
        contact_vectors: list[FloatArray] = []
        if isinstance(details, list):
            for item in details:
                if (
                    isinstance(item, dict)
                    and item.get("leg") == leg
                    and item.get("rung_geom_name") == self._source_rung_name
                ):
                    position = item.get("position")
                    if isinstance(position, (tuple, list, np.ndarray)) and len(position) == 3:
                        vector = np.array(
                            [float(position[0]) - float(source_rung[0]), 0.0,
                             float(position[2]) - float(source_rung[2])],
                            dtype=float,
                        )
                        if np.linalg.norm(vector) > 1.0e-8:
                            contact_vectors.append(vector)
        if contact_vectors:
            mean_vector = np.mean(np.vstack(contact_vectors), axis=0)
            if np.linalg.norm(mean_vector) > 1.0e-8:
                candidates.append(mean_vector)
            candidates.extend(contact_vectors)

        palm_vector = np.array(
            [
                self._swing_start[0] - source_rung[0],
                0.0,
                self._swing_start[2] - source_rung[2],
            ],
            dtype=float,
        )
        candidates.extend((palm_vector, -palm_vector))
        for index in range(self.escape_direction_samples):
            angle = 2.0 * np.pi * index / self.escape_direction_samples
            candidates.append(
                np.array([np.cos(angle), 0.0, np.sin(angle)], dtype=float)
            )

        unique: list[FloatArray] = []
        for candidate in candidates:
            norm = float(np.linalg.norm(candidate))
            if norm < 1.0e-8:
                continue
            direction = np.asarray(candidate, dtype=float) / norm
            if not any(float(np.dot(direction, old)) > 0.995 for old in unique):
                unique.append(direction)
        return tuple(unique)

    def _select_release_waypoint(
        self,
        data: mujoco.MjData,
        leg: str,
        directions: Sequence[FloatArray],
        rung_translation: FloatArray,
    ) -> bool:
        """Search direction and distance with a conservative clearance reserve."""
        base_qpos = self._current_base_qpos(data)
        max_steps = int(
            np.floor(
                (self.release_clearance_max - self.release_clearance)
                / self.release_clearance_step
            )
        )
        displacements = [
            self.release_clearance + index * self.release_clearance_step
            for index in range(max_steps + 1)
        ]
        if (
            not displacements
            or displacements[-1] < self.release_clearance_max - 1.0e-12
        ):
            displacements.append(self.release_clearance_max)

        required_preflight = (
            self.minimum_source_clearance + self.preflight_clearance_margin
        )
        print(
            f"Escape-path search {leg}: {len(directions)} directions; "
            f"dynamic clearance target={self.minimum_source_clearance:.6f} m; "
            f"preflight reserve target={required_preflight:.6f} m"
        )

        overall_best: tuple[float, float, FloatArray, FloatArray] | None = None
        for displacement in displacements:
            print(
                f"  evaluating escape={displacement:.4f} m across "
                f"{len(directions)} directions...",
                flush=True,
            )
            best_at_distance: tuple[float, float, FloatArray, FloatArray] | None = None
            for direction in directions:
                candidate = self._swing_start + displacement * direction
                desired = {
                    item: np.asarray(
                        data.xpos[self._palm_body_ids[item]], dtype=float
                    ).copy()
                    for item in self.LEG_JOINTS
                }
                desired[leg] = candidate.copy()
                solved, residuals = self._solve_leg_targets(
                    self.model,
                    data,
                    desired_palms=desired,
                    virtual_base_qpos=base_qpos,
                    legs_to_solve=(leg,),
                    initial_qpos=np.asarray(data.qpos, dtype=float).copy(),
                    iteration_limit=self.preflight_ik_iterations,
                )
                self._ik_data.qpos[self._leg_qpos_adrs[leg]] = solved[leg]
                self._ik_data.qpos[self._finger_qpos_adrs[leg]] = (
                    self.open_finger_pose
                )
                mujoco.mj_forward(self.model, self._ik_data)
                clearance = self._minimum_leg_rung_clearance(
                    self.model, self._ik_data, leg, self._source_rung_name
                )
                residual = float(residuals[leg])
                if residual > self.preflight_ik_tolerance:
                    continue
                record = (clearance, residual, direction.copy(), candidate.copy())
                if (
                    best_at_distance is None
                    or clearance > best_at_distance[0]
                ):
                    best_at_distance = record
                if overall_best is None or clearance > overall_best[0]:
                    overall_best = record

            if best_at_distance is None:
                print(
                    f"  escape={displacement:.4f} m | no IK-feasible direction"
                )
                continue
            clearance, residual, direction, candidate = best_at_distance
            angle_deg = float(np.degrees(np.arctan2(direction[2], direction[0])))
            print(
                f"  escape={displacement:.4f} m | best angle={angle_deg:+.1f} deg "
                f"| residual={residual:.6f} m | predicted clearance="
                f"{clearance:+.6f} m"
            )
            if clearance >= required_preflight:
                self._release_direction = direction.copy()
                self._selected_release_displacement = float(displacement)
                self._release_command_displacement = 0.0
                self._swing_clear_start = candidate.copy()
                self._swing_clear_goal = candidate + rung_translation
                print(
                    f"Selected {leg} escape={displacement:.4f} m at "
                    f"{angle_deg:+.1f} deg; predicted clearance reserve="
                    f"{clearance:.6f} m."
                )
                return True

        if overall_best is None:
            self._fail(
                f"{leg} has no IK-feasible escape direction in the configured "
                "search."
            )
        else:
            clearance, residual, direction, _ = overall_best
            angle_deg = float(np.degrees(np.arctan2(direction[2], direction[0])))
            self._fail(
                f"{leg} has no escape candidate with the required preflight "
                f"reserve. Best predicted clearance={clearance:.6f} m at "
                f"{angle_deg:+.1f} deg; residual={residual:.6f} m; required="
                f"{required_preflight:.6f} m."
            )
        return False

    def _preflight_swing_targets(self, data: mujoco.MjData, leg: str) -> bool:
        """Sequential IK check for lift -> translate -> lower.

        This check uses the XML-derived rung translation and the measured
        source palm pose.  Geometric clearance is diagnostic only in V8; the
        dynamic release gate is based on the actual old-rung contact count and
        world-space lift progress.
        """
        base_qpos = self._current_base_qpos(data)
        checks = (
            ("vertical_source", self._source_transfer_clear),
            ("vertical_target", self._target_transfer_clear),
            ("target", self._swing_goal),
        )
        warm_qpos = np.asarray(data.qpos, dtype=float).copy()
        residuals: dict[str, float] = {}

        print(f"Direct-vertical preflight {leg} (sequential warm-start):")
        for name, target in checks:
            desired = {
                item: np.asarray(
                    data.xpos[self._palm_body_ids[item]], dtype=float
                ).copy()
                for item in self.LEG_JOINTS
            }
            desired[leg] = np.asarray(target, dtype=float).copy()
            solved, solved_residuals = self._solve_leg_targets(
                self.model,
                data,
                desired_palms=desired,
                virtual_base_qpos=base_qpos,
                legs_to_solve=(leg,),
                initial_qpos=warm_qpos,
                iteration_limit=self.preflight_ik_iterations,
            )
            warm_qpos[self._leg_qpos_adrs[leg]] = solved[leg]
            residual = float(solved_residuals[leg])
            residuals[name] = residual
            solved_actual = np.asarray(
                self._ik_data.xpos[self._palm_body_ids[leg]], dtype=float
            ).copy()
            delta = np.asarray(target, dtype=float) - solved_actual
            print(
                f"  {name:15s}: residual={residual:.4f} m | "
                f"delta=({delta[0]:+.4f}, {delta[1]:+.4f}, "
                f"{delta[2]:+.4f}) | solved_palm=("
                f"{solved_actual[0]:+.4f}, {solved_actual[1]:+.4f}, "
                f"{solved_actual[2]:+.4f})"
            )

        worst = max(residuals.values())
        self._last_ik_residuals[leg] = residuals["target"]
        if worst > self.preflight_ik_tolerance:
            self._fail(
                f"{leg} direct vertical path is outside the configured IK "
                f"tolerance; worst residual={worst:.4f} m > "
                f"{self.preflight_ik_tolerance:.4f} m."
            )
            return False

        print(
            f"Direct-vertical preflight {leg} passed: worst residual="
            f"{worst:.4f} m; lift={self.transfer_lift_height:.4f} m; "
            f"rung spacing={self._rung_translation[0]:+.4f} m along x."
        )
        return True

    def _check_swing_motion_safety(self, data: mujoco.MjData) -> None:
        if self._swing_leg is None or self._phase not in (
            GaitPhase.OPENING_SWING_GRIPPER,
            GaitPhase.RETRACTING_SWING_LEG,
            GaitPhase.POST_RELEASE_SETTLE,
            GaitPhase.LIFTING_TO_TRANSFER_CLEARANCE,
            GaitPhase.TRANSFERRING_SWING_LEG,
            GaitPhase.APPROACHING_TARGET_RUNG,
            GaitPhase.CLOSING_SWING_GRIPPER,
            GaitPhase.VERIFYING_REGRASP,
        ):
            self._tracking_error_since = None
            return

        current_base = self._current_base_qpos(data)
        base_drift = float(
            np.linalg.norm(current_base[:3] - self._swing_base_reference_qpos[:3])
        )
        if base_drift > self.max_swing_base_drift:
            self._fail(
                f"Base drift exceeded the swing safety limit: "
                f"{base_drift:.4f} m > {self.max_swing_base_drift:.4f} m."
            )
            return

        desired = self._last_desired_palms.get(self._swing_leg)
        if desired is None:
            return
        actual = np.asarray(
            data.xpos[self._palm_body_ids[self._swing_leg]],
            dtype=float,
        )
        tracking_error = float(np.linalg.norm(desired - actual))
        if tracking_error <= self.max_swing_tracking_error:
            self._tracking_error_since = None
            return

        if self._tracking_error_since is None:
            self._tracking_error_since = float(data.time)
        elif float(data.time) - self._tracking_error_since >= self.tracking_error_timeout:
            self._fail(
                f"{self._swing_leg} palm tracking error persisted above the "
                f"safety limit: {tracking_error:.4f} m > "
                f"{self.max_swing_tracking_error:.4f} m."
            )

    # ------------------------------------------------------------------
    # Contact safety and regrasp verification
    # ------------------------------------------------------------------

    def _check_support_contacts(
        self,
        data: mujoco.MjData,
        contacts: Mapping[str, object],
    ) -> None:
        if self._phase in (GaitPhase.COMPLETE, GaitPhase.FAILED):
            return

        raw_grippers = contacts.get("grippers", set())
        if not isinstance(raw_grippers, set):
            return
        contacting = {str(item) for item in raw_grippers}

        if self._swing_leg is not None and self._phase in (
            GaitPhase.OPENING_SWING_GRIPPER,
            GaitPhase.RETRACTING_SWING_LEG,
            GaitPhase.POST_RELEASE_SETTLE,
            GaitPhase.LIFTING_TO_TRANSFER_CLEARANCE,
            GaitPhase.TRANSFERRING_SWING_LEG,
            GaitPhase.APPROACHING_TARGET_RUNG,
            GaitPhase.CLOSING_SWING_GRIPPER,
            GaitPhase.VERIFYING_REGRASP,
        ):
            required = set(self.LEG_JOINTS) - {self._swing_leg}
        else:
            required = set(self.LEG_JOINTS)

        support_ok = required.issubset(contacting)
        if support_ok:
            self._support_loss_since = None
            return

        if self._support_loss_since is None:
            self._support_loss_since = float(data.time)
            return

        if float(data.time) - self._support_loss_since >= self.support_loss_timeout:
            missing = sorted(required - contacting)
            self._fail(
                "Persistent supporting-gripper contact loss: "
                f"missing {missing}."
            )

    def _swing_finger_open_error(self, data: mujoco.MjData) -> float:
        if self._swing_leg is None:
            return float("nan")
        actual = np.asarray(
            data.qpos[self._finger_qpos_adrs[self._swing_leg]], dtype=float
        )
        return float(np.max(np.abs(actual - self.open_finger_pose)))

    def _minimum_leg_rung_clearance_details(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        leg: str | None,
        rung_name: str | None,
        *,
        distance_limit: float = 0.10,
    ) -> tuple[float, str]:
        if leg is None or rung_name is None:
            return float("nan"), ""
        rung_geom_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, rung_name
        )
        if rung_geom_id < 0:
            return float("nan"), ""
        best_distance = float("inf")
        best_name = ""
        from_to = np.zeros(6, dtype=float)
        for finger_geom_id in self._finger_geom_ids_by_leg.get(leg, ()):
            from_to[:] = 0.0
            distance = float(
                mujoco.mj_geomDistance(
                    model, data, int(finger_geom_id), int(rung_geom_id),
                    float(distance_limit), from_to
                )
            )
            if distance < best_distance:
                best_distance = distance
                name = mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_GEOM, int(finger_geom_id)
                )
                best_name = name or f"geom_{int(finger_geom_id)}"
        if not np.isfinite(best_distance):
            return float("nan"), ""
        return best_distance, best_name

    def _minimum_leg_rung_clearance(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        leg: str | None,
        rung_name: str | None,
        *,
        distance_limit: float = 0.10,
    ) -> float:
        """Return minimum signed surface distance from a gripper to one rung.

        Positive values mean separation; zero means touching; negative values
        mean penetration.  MuJoCo's mj_geomDistance computes the smallest
        signed distance between a pair of geoms.
        """
        distance, _ = self._minimum_leg_rung_clearance_details(
            model, data, leg, rung_name, distance_limit=distance_limit
        )
        return distance

    @staticmethod
    def _count_leg_rung_contacts(
        contacts: Mapping[str, object],
        leg: str | None,
        rung_name: str | None,
    ) -> int:
        if leg is None or rung_name is None:
            return 0
        details = contacts.get("contact_details", [])
        if not isinstance(details, list):
            return 0
        return sum(
            1
            for item in details
            if isinstance(item, dict)
            and item.get("leg") == leg
            and item.get("rung_geom_name") == rung_name
        )

    def _regrasp_is_valid(self, contacts: Mapping[str, object]) -> bool:
        if self._swing_leg is None or self._target_rung_name is None:
            return False

        grippers = contacts.get("grippers", set())
        details = contacts.get("contact_details", [])
        if not isinstance(grippers, set) or not isinstance(details, list):
            return False
        if set(self.LEG_JOINTS) != {str(item) for item in grippers}:
            return False

        matching_details = [
            item
            for item in details
            if isinstance(item, dict)
            and item.get("leg") == self._swing_leg
            and item.get("rung_geom_name") == self._target_rung_name
        ]
        matching_segments = {
            str(item.get("segment"))
            for item in matching_details
            if item.get("segment") is not None
        }

        return (
            len(matching_details) >= self.minimum_regrasp_contacts
            and len(matching_segments) >= 1
        )

    # ------------------------------------------------------------------
    # IK and actuator control
    # ------------------------------------------------------------------

    def _refresh_leg_targets_if_due(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        desired_palms: Mapping[str, FloatArray],
        virtual_base_qpos: FloatArray | None,
    ) -> None:
        now = float(data.time)
        if now - self._last_ik_update_time < self.ik_update_period:
            return

        if self._phase in (
            GaitPhase.RETRACTING_SWING_LEG,
            GaitPhase.POST_RELEASE_SETTLE,
            GaitPhase.LIFTING_TO_TRANSFER_CLEARANCE,
            GaitPhase.TRANSFERRING_SWING_LEG,
            GaitPhase.APPROACHING_TARGET_RUNG,
            GaitPhase.CLOSING_SWING_GRIPPER,
            GaitPhase.VERIFYING_REGRASP,
        ) and self._swing_leg is not None:
            # V8 actively holds all three supporting palms at their measured
            # world anchors.  This is joint-space support compensation, not a
            # floating-base pin.
            legs_to_solve = tuple(self.LEG_JOINTS)
        elif self._phase is GaitPhase.SHIFTING_BODY:
            legs_to_solve = tuple(self.LEG_JOINTS)
        else:
            return

        solved, residuals = self._solve_leg_targets(
            model,
            data,
            desired_palms=desired_palms,
            virtual_base_qpos=virtual_base_qpos,
            legs_to_solve=legs_to_solve,
        )

        update_dt = (
            self.ik_update_period
            if not np.isfinite(self._last_ik_update_time)
            else max(now - self._last_ik_update_time, float(model.opt.timestep))
        )
        maximum_change = self.max_joint_target_rate * update_dt

        for leg in legs_to_solve:
            previous = self._leg_target_qpos.get(
                leg,
                data.qpos[self._leg_qpos_adrs[leg]].copy(),
            )
            delta = np.clip(
                solved[leg] - previous,
                -maximum_change,
                maximum_change,
            )
            self._leg_target_qpos[leg] = previous + delta

        self._last_ik_residuals = residuals
        self._last_ik_update_time = now

    def _solve_leg_targets(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        desired_palms: Mapping[str, FloatArray],
        virtual_base_qpos: FloatArray | None,
        legs_to_solve: Sequence[str] | None = None,
        initial_qpos: FloatArray | None = None,
        iteration_limit: int | None = None,
    ) -> tuple[dict[str, FloatArray], dict[str, float]]:
        scratch = self._ik_data
        if initial_qpos is None:
            scratch.qpos[:] = data.qpos
        else:
            initial = np.asarray(initial_qpos, dtype=float)
            if initial.shape != data.qpos.shape:
                raise ValueError(
                    "initial_qpos must have the same shape as data.qpos."
                )
            scratch.qpos[:] = initial
        scratch.qvel[:] = 0.0
        scratch.ctrl[:] = 0.0

        if virtual_base_qpos is not None:
            if np.asarray(virtual_base_qpos).shape != (7,):
                raise ValueError("virtual_base_qpos must contain seven values.")
            qadr = self._base_qpos_adr
            scratch.qpos[qadr:qadr + 7] = virtual_base_qpos

        mujoco.mj_forward(model, scratch)

        active_legs = tuple(self.LEG_JOINTS) if legs_to_solve is None else tuple(legs_to_solve)
        invalid = set(active_legs) - set(self.LEG_JOINTS)
        if invalid:
            raise ValueError(f"Unknown IK leg names: {sorted(invalid)}")

        residuals = {leg: np.inf for leg in self.LEG_JOINTS}
        regulariser = self.ik_damping_gait**2 * np.eye(3)
        iterations = (
            self.ik_iterations_per_update
            if iteration_limit is None
            else int(iteration_limit)
        )
        if iterations < 1:
            raise ValueError("iteration_limit must be at least 1.")

        for _ in range(iterations):
            maximum_error = 0.0

            for leg in active_legs:
                target = np.asarray(desired_palms[leg], dtype=float)
                actual = np.asarray(
                    scratch.xpos[self._palm_body_ids[leg]],
                    dtype=float,
                )
                error = target - actual
                error_norm = float(np.linalg.norm(error))
                residuals[leg] = error_norm
                maximum_error = max(maximum_error, error_norm)

                if error_norm <= self.ik_tolerance_gait:
                    continue

                jacp = np.zeros((3, model.nv), dtype=float)
                jacr = np.zeros((3, model.nv), dtype=float)
                mujoco.mj_jacBody(
                    model,
                    scratch,
                    jacp,
                    jacr,
                    self._palm_body_ids[leg],
                )

                dof_addresses = self._leg_dof_adrs[leg]
                jacobian = jacp[:, dof_addresses]
                system = jacobian @ jacobian.T + regulariser

                try:
                    delta = jacobian.T @ np.linalg.solve(system, error)
                except np.linalg.LinAlgError:
                    delta = np.linalg.pinv(jacobian) @ error

                delta = np.clip(
                    delta,
                    -self.ik_step_limit_gait,
                    self.ik_step_limit_gait,
                )
                scratch.qpos[self._leg_qpos_adrs[leg]] += delta
                self._clip_joint_positions(scratch, self._leg_joint_ids[leg])
                mujoco.mj_forward(model, scratch)

            if maximum_error <= self.ik_tolerance_gait:
                break

        solved = {
            leg: scratch.qpos[self._leg_qpos_adrs[leg]].copy()
            for leg in self.LEG_JOINTS
        }

        for leg in self.LEG_JOINTS:
            target = np.asarray(desired_palms[leg], dtype=float)
            actual = np.asarray(
                scratch.xpos[self._palm_body_ids[leg]],
                dtype=float,
            )
            residuals[leg] = float(np.linalg.norm(target - actual))

        return solved, residuals

    def _apply_per_leg_finger_pd(
        self,
        data: mujoco.MjData,
        targets: Mapping[str, FloatArray],
    ) -> None:
        for leg in self.LEG_JOINTS:
            target = np.asarray(targets[leg], dtype=float)
            if target.shape != (6,):
                raise ValueError(f"Finger target for {leg} must have six values.")

            q = data.qpos[self._finger_qpos_adrs[leg]]
            qd = data.qvel[self._finger_dof_adrs[leg]]

            opening_phases = (
                GaitPhase.OPENING_SWING_GRIPPER,
                GaitPhase.RETRACTING_SWING_LEG,
                GaitPhase.POST_RELEASE_SETTLE,
                GaitPhase.LIFTING_TO_TRANSFER_CLEARANCE,
                GaitPhase.TRANSFERRING_SWING_LEG,
                GaitPhase.APPROACHING_TARGET_RUNG,
            )
            kp_scale = (
                self.swing_open_kp_scale
                if leg == self._swing_leg and self._phase in opening_phases
                else 1.0
            )
            torque = (
                self.finger_kp * kp_scale * (target - q)
                - self.finger_kd * qd
            )
            self._write_clipped_control(
                data,
                self._finger_actuator_ids[leg],
                torque,
            )

    # ------------------------------------------------------------------
    # Status and numerical helpers
    # ------------------------------------------------------------------

    def _make_status(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> GaitStatus:
        contacts = self._detect_contacts(model, data)
        grippers = contacts.get("grippers", set())
        contacting = (
            tuple(sorted(str(item) for item in grippers))
            if isinstance(grippers, set)
            else ()
        )
        old_rung_contacts = self._count_leg_rung_contacts(
            contacts,
            self._swing_leg,
            self._source_rung_name,
        )
        target_rung_contacts = self._count_leg_rung_contacts(
            contacts,
            self._swing_leg,
            self._target_rung_name,
        )

        swing_tracking_error = float("nan")
        if self._swing_leg is not None and self._swing_leg in self._last_desired_palms:
            actual_palm = np.asarray(
                data.xpos[self._palm_body_ids[self._swing_leg]],
                dtype=float,
            )
            desired_palm = self._last_desired_palms[self._swing_leg]
            swing_tracking_error = float(np.linalg.norm(desired_palm - actual_palm))

        source_clearance = self._minimum_leg_rung_clearance(
            model, data, self._swing_leg, self._source_rung_name
        )
        base = self._current_base_qpos(data)[:3]
        base_linear_speed, base_angular_speed = self._base_speed_norms(data)

        return GaitStatus(
            phase=self._phase.name,
            swing_leg=self._swing_leg,
            cycles_completed=self._cycles_completed,
            completed_regrasps=self._completed_regrasps,
            contacting_grippers=contacting,
            physical_contacts=int(contacts.get("physical_count", 0)),
            old_rung_contacts=old_rung_contacts,
            target_rung_contacts=target_rung_contacts,
            swing_tracking_error=swing_tracking_error,
            source_clearance=source_clearance,
            selected_release_displacement=self._selected_release_displacement,
            base_position=(float(base[0]), float(base[1]), float(base[2])),
            base_linear_speed=base_linear_speed,
            base_angular_speed=base_angular_speed,
            complete=self._phase is GaitPhase.COMPLETE,
            failed=self._phase is GaitPhase.FAILED,
            message=self._message,
        )

    def _base_speed_norms(self, data: mujoco.MjData) -> tuple[float, float]:
        dadr = self._base_dof_adr
        base_velocity = np.asarray(data.qvel[dadr:dadr + 6], dtype=float)
        return (
            float(np.linalg.norm(base_velocity[:3])),
            float(np.linalg.norm(base_velocity[3:6])),
        )

    def diagnostic_snapshot(
        self, model: mujoco.MjModel, data: mujoco.MjData
    ) -> dict[str, object]:
        """Return one flat gait-development telemetry sample."""
        status = self._make_status(model, data)
        base_qpos = self._current_base_qpos(data)
        dadr = self._base_dof_adr
        base_qvel = np.asarray(data.qvel[dadr:dadr + 6], dtype=float).copy()
        actual_palm = np.full(3, np.nan, dtype=float)
        desired_palm = np.full(3, np.nan, dtype=float)
        if self._swing_leg is not None:
            actual_palm = np.asarray(
                data.xpos[self._palm_body_ids[self._swing_leg]], dtype=float
            ).copy()
            desired = self._last_desired_palms.get(self._swing_leg)
            if desired is not None:
                desired_palm = np.asarray(desired, dtype=float).copy()

        row: dict[str, object] = {
            "time": float(data.time),
            "phase": status.phase,
            "swing_leg": status.swing_leg or "",
            "message": status.message,
            "cycles_completed": status.cycles_completed,
            "completed_regrasps": status.completed_regrasps,
            "physical_contacts": status.physical_contacts,
            "contacting_grippers": ";".join(status.contacting_grippers),
            "old_rung_contacts": status.old_rung_contacts,
            "target_rung_contacts": status.target_rung_contacts,
            "source_clearance": status.source_clearance,
            "selected_release_displacement": (
                status.selected_release_displacement
            ),
            "release_command_displacement": self._release_command_displacement,
            "release_direction_x": float(self._release_direction[0]),
            "release_direction_z": float(self._release_direction[2]),
            "closest_source_segment": self._closest_source_segment,
            "finger_open_error": self._swing_finger_open_error(data),
            "swing_tracking_error": status.swing_tracking_error,
            "base_x": float(base_qpos[0]),
            "base_y": float(base_qpos[1]),
            "base_z": float(base_qpos[2]),
            "base_qw": float(base_qpos[3]),
            "base_qx": float(base_qpos[4]),
            "base_qy": float(base_qpos[5]),
            "base_qz": float(base_qpos[6]),
            "base_vx": float(base_qvel[0]),
            "base_vy": float(base_qvel[1]),
            "base_vz": float(base_qvel[2]),
            "base_wx": float(base_qvel[3]),
            "base_wy": float(base_qvel[4]),
            "base_wz": float(base_qvel[5]),
            "base_linear_speed": status.base_linear_speed,
            "base_angular_speed": status.base_angular_speed,
            "swing_actual_x": float(actual_palm[0]),
            "swing_actual_y": float(actual_palm[1]),
            "swing_actual_z": float(actual_palm[2]),
            "swing_desired_x": float(desired_palm[0]),
            "swing_desired_y": float(desired_palm[1]),
            "swing_desired_z": float(desired_palm[2]),
        }
        for leg in self.LEG_JOINTS:
            actual_q = np.asarray(
                data.qpos[self._leg_qpos_adrs[leg]], dtype=float
            )
            target_q = np.asarray(
                self._leg_target_qpos.get(leg, actual_q), dtype=float
            )
            for index, joint_label in enumerate(("hip", "thigh", "calf")):
                row[f"{leg}_{joint_label}_q"] = float(actual_q[index])
                row[f"{leg}_{joint_label}_target"] = float(target_q[index])
            finger_q = np.asarray(
                data.qpos[self._finger_qpos_adrs[leg]], dtype=float
            )
            finger_target = self._finger_targets(data)[leg]
            for index, label in enumerate(("L1", "L2", "L3", "R1", "R2", "R3")):
                row[f"{leg}_finger_{label}_q"] = float(finger_q[index])
                row[f"{leg}_finger_{label}_target"] = float(finger_target[index])
        return row

    def _current_base_qpos(self, data: mujoco.MjData) -> FloatArray:
        qadr = self._base_qpos_adr
        return np.asarray(data.qpos[qadr:qadr + 7], dtype=float).copy()

    def _phase_elapsed(self, data: mujoco.MjData) -> float:
        return max(0.0, float(data.time) - self._phase_start_time)

    @staticmethod
    def _smooth_progress(elapsed: float, duration: float) -> float:
        raw = float(np.clip(elapsed / duration, 0.0, 1.0))
        return raw * raw * (3.0 - 2.0 * raw)

    @staticmethod
    def _lerp(start: FloatArray, end: FloatArray, alpha: float) -> FloatArray:
        return (1.0 - alpha) * np.asarray(start) + alpha * np.asarray(end)

    @staticmethod
    def _interpolate_base_qpos(
        start: FloatArray,
        end: FloatArray,
        alpha: float,
    ) -> FloatArray:
        result = np.asarray(start, dtype=float).copy()
        result[:3] = (1.0 - alpha) * np.asarray(start[:3]) + alpha * np.asarray(end[:3])

        # The gait currently keeps the desired body attitude constant.  The
        # start and goal quaternions are therefore identical, but normalising
        # here makes the helper safe if that changes later.
        quaternion = (1.0 - alpha) * np.asarray(start[3:7]) + alpha * np.asarray(end[3:7])
        norm = float(np.linalg.norm(quaternion))
        if norm > 1.0e-12:
            quaternion = quaternion / norm
        result[3:7] = quaternion
        return result
