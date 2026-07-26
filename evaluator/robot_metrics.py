"""
Robot-level stability metrics for the modified Unitree Go2 climbing robot.

The module evaluates only whole-body motion and the 12 leg actuators.
Finger contacts, finger actuator effort, grasp quality, and task success
belong to ContactMetrics, GraspMetrics, and TaskMetrics respectively.

Coordinate convention used by the climbing scene
-------------------------------------------------
x-axis: forward motion along the ladder
y-axis: lateral motion across the ladder
z-axis: normal displacement away from/towards the ladder

The class calculates the current metric state only. Complete time-series
storage is the responsibility of DataLogger.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Sequence

import mujoco
import numpy as np
from numpy.typing import NDArray

from .base_metric import BaseMetric, MetricDict


FloatArray = NDArray[np.float64]


class RobotMetrics(BaseMetric):
    """Calculate robot-body and leg-level stability metrics."""

    DEFAULT_BASE_BODY_NAME = "base"

    DEFAULT_LEG_ACTUATOR_NAMES = (
        "FL_hip",
        "FL_thigh",
        "FL_calf",
        "FR_hip",
        "FR_thigh",
        "FR_calf",
        "RL_hip",
        "RL_thigh",
        "RL_calf",
        "RR_hip",
        "RR_thigh",
        "RR_calf",
    )

    DEFAULT_LEG_JOINT_NAMES = (
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
    )

    def __init__(
        self,
        base_body_name: str = DEFAULT_BASE_BODY_NAME,
        leg_actuator_names: Sequence[str] = DEFAULT_LEG_ACTUATOR_NAMES,
        leg_joint_names: Sequence[str] = DEFAULT_LEG_JOINT_NAMES,
        *,
        epsilon: float = 1e-9,
    ) -> None:
        """
        Parameters
        ----------
        base_body_name:
            Name of the floating-base body in the MuJoCo model.

        leg_actuator_names:
            Names of the 12 leg actuators. Finger actuators must not be
            included here.

        leg_joint_names:
            Names of the 12 leg joints used for joint-limit evaluation.

        epsilon:
            Small positive value used to avoid division by zero.
        """
        super().__init__(name="robot")

        if not base_body_name.strip():
            raise ValueError("base_body_name must be a non-empty string.")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive.")

        self.base_body_name = base_body_name.strip()
        self.leg_actuator_names = self._validate_unique_names(
            leg_actuator_names, "leg_actuator_names"
        )
        self.leg_joint_names = self._validate_unique_names(
            leg_joint_names, "leg_joint_names"
        )
        self.epsilon = float(epsilon)

        self._base_body_id: int | None = None
        self._leg_actuator_ids: Dict[str, int] = {}
        self._leg_joint_ids: Dict[str, int] = {}

        self._initial_position: FloatArray | None = None
        self._initial_euler: FloatArray | None = None
        self._previous_linear_velocity: FloatArray | None = None
        self._previous_time: float | None = None

    def reset(self, model: Any | None = None, data: Any | None = None) -> None:
        """
        Reset references and finite-difference state for a new trial.

        When model and data are supplied, the current base pose becomes the
        zero reference for progress, drift, and orientation change.
        """
        self._clear_metrics()

        self._base_body_id = None
        self._leg_actuator_ids = {}
        self._leg_joint_ids = {}

        self._initial_position = None
        self._initial_euler = None
        self._previous_linear_velocity = None
        self._previous_time = None

        if model is None and data is None:
            return
        if model is None or data is None:
            raise ValueError("reset() requires both model and data, or neither.")

        self._resolve_model_ids(model)

        position = self._get_base_position(data)
        quaternion = self._get_base_quaternion(data)
        euler = self._quaternion_to_euler(quaternion)
        linear_velocity, _ = self._get_base_velocities(model, data)

        self._initial_position = position.copy()
        self._initial_euler = euler.copy()
        self._previous_linear_velocity = linear_velocity.copy()
        self._previous_time = float(data.time)
        self._is_initialised = True

    def update(self, model: Any, data: Any) -> MetricDict:
        """Calculate robot metrics for the current MuJoCo simulation state."""
        if model is None or data is None:
            raise ValueError("update() requires valid MuJoCo model and data objects.")

        if self._base_body_id is None:
            self._resolve_model_ids(model)

        position = self._get_base_position(data)
        quaternion = self._get_base_quaternion(data)
        euler = self._quaternion_to_euler(quaternion)
        linear_velocity, angular_velocity = self._get_base_velocities(model, data)

        if self._initial_position is None:
            self._initial_position = position.copy()
        if self._initial_euler is None:
            self._initial_euler = euler.copy()

        displacement = position - self._initial_position
        orientation_change = self._wrap_angles(euler - self._initial_euler)

        acceleration = self._calculate_linear_acceleration(
            current_velocity=linear_velocity,
            current_time=float(data.time),
        )

        leg_effort = self._calculate_leg_actuator_effort(model, data)
        joint_limit = self._calculate_joint_limit_utilisation(model, data)

        metrics: MetricDict = {
            "time": float(data.time),
            "position": self._vector_dict(position),
            "progress": {
                "forward": float(displacement[0]),
                "lateral_drift": float(displacement[1]),
                "normal_drift": float(displacement[2]),
                "lateral_drift_abs": float(abs(displacement[1])),
                "normal_drift_abs": float(abs(displacement[2])),
            },
            "orientation": {
                "roll": float(euler[0]),
                "pitch": float(euler[1]),
                "yaw": float(euler[2]),
                "roll_change": float(orientation_change[0]),
                "pitch_change": float(orientation_change[1]),
                "yaw_change": float(orientation_change[2]),
            },
            "linear_velocity": self._vector_with_magnitude(linear_velocity),
            "angular_velocity": self._vector_with_magnitude(angular_velocity),
            "base_acceleration": self._vector_with_magnitude(acceleration),
            "leg_actuator_effort": leg_effort,
            "joint_limit": joint_limit,
        }

        return self._set_metrics(metrics)

    def _resolve_model_ids(self, model: Any) -> None:
        """Resolve and validate all model object names once per trial."""
        self._base_body_id = self._name_to_id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            self.base_body_name,
            "body",
        )

        self._leg_actuator_ids = {
            name: self._name_to_id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, name, "actuator"
            )
            for name in self.leg_actuator_names
        }

        self._leg_joint_ids = {
            name: self._name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, name, "joint")
            for name in self.leg_joint_names
        }

        for name, joint_id in self._leg_joint_ids.items():
            joint_type = int(model.jnt_type[joint_id])
            if joint_type not in (
                int(mujoco.mjtJoint.mjJNT_HINGE),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
            ):
                raise ValueError(
                    f"Leg joint {name!r} must be hinge or slide, "
                    f"but model type is {joint_type}."
                )

    @staticmethod
    def _name_to_id(
        model: Any,
        object_type: mujoco.mjtObj,
        name: str,
        object_label: str,
    ) -> int:
        object_id = int(mujoco.mj_name2id(model, object_type, name))
        if object_id < 0:
            raise ValueError(
                f"MuJoCo {object_label} {name!r} was not found in the model."
            )
        return object_id

    def _get_base_position(self, data: Any) -> FloatArray:
        assert self._base_body_id is not None
        return np.asarray(data.xpos[self._base_body_id], dtype=np.float64).copy()

    def _get_base_quaternion(self, data: Any) -> FloatArray:
        assert self._base_body_id is not None
        quaternion = np.asarray(
            data.xquat[self._base_body_id], dtype=np.float64
        ).copy()

        norm = float(np.linalg.norm(quaternion))
        if norm <= self.epsilon:
            raise ValueError("Base quaternion has near-zero magnitude.")

        return quaternion / norm

    def _get_base_velocities(
        self,
        model: Any,
        data: Any,
    ) -> tuple[FloatArray, FloatArray]:
        """
        Return base linear and angular velocity in world coordinates.

        MuJoCo's mj_objectVelocity writes a six-dimensional vector ordered as
        angular velocity followed by linear velocity.
        """
        assert self._base_body_id is not None

        spatial_velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            model,
            data,
            mujoco.mjtObj.mjOBJ_BODY,
            self._base_body_id,
            spatial_velocity,
            0,  # world-aligned coordinates
        )

        angular_velocity = spatial_velocity[0:3].copy()
        linear_velocity = spatial_velocity[3:6].copy()
        return linear_velocity, angular_velocity

    def _calculate_linear_acceleration(
        self,
        current_velocity: FloatArray,
        current_time: float,
    ) -> FloatArray:
        """Estimate world-frame base acceleration by finite difference."""
        if (
            self._previous_linear_velocity is None
            or self._previous_time is None
        ):
            acceleration = np.zeros(3, dtype=np.float64)
        else:
            dt = current_time - self._previous_time
            if dt <= self.epsilon:
                acceleration = np.zeros(3, dtype=np.float64)
            else:
                acceleration = (
                    current_velocity - self._previous_linear_velocity
                ) / dt

        self._previous_linear_velocity = current_velocity.copy()
        self._previous_time = current_time
        return acceleration

    def _calculate_leg_actuator_effort(
        self,
        model: Any,
        data: Any,
    ) -> MetricDict:
        """
        Calculate leg actuator effort and normalised utilisation.

        For the XML used in this project, the leg actuators are direct-drive
        motor actuators with unit gear. Their scalar actuator force therefore
        corresponds numerically to the commanded joint torque. Utilisation is
        normalised by the largest absolute value in each actuator's ctrlrange.

        This method deliberately excludes all 24 finger actuators.
        """
        per_actuator: Dict[str, MetricDict] = {}
        utilisations: list[float] = []
        efforts: list[float] = []

        for name, actuator_id in self._leg_actuator_ids.items():
            effort = float(data.actuator_force[actuator_id])
            control = float(data.ctrl[actuator_id])

            control_range = np.asarray(
                model.actuator_ctrlrange[actuator_id], dtype=np.float64
            )
            control_limited = bool(model.actuator_ctrllimited[actuator_id])
            denominator = (
                float(np.max(np.abs(control_range)))
                if control_limited
                else float("nan")
            )

            if control_limited and denominator > self.epsilon:
                utilisation = abs(effort) / denominator
            else:
                utilisation = float("nan")

            per_actuator[name] = {
                "effort": effort,
                "control": control,
                "limit": denominator,
                "utilisation": float(utilisation),
                "over_limit": bool(
                    np.isfinite(utilisation) and utilisation > 1.0 + 1e-6
                ),
            }

            efforts.append(abs(effort))
            if np.isfinite(utilisation):
                utilisations.append(float(utilisation))

        utilisation_array = np.asarray(utilisations, dtype=np.float64)
        effort_array = np.asarray(efforts, dtype=np.float64)

        return {
            "maximum_absolute_effort": self._safe_max(effort_array),
            "mean_absolute_effort": self._safe_mean(effort_array),
            "maximum_utilisation": self._safe_max(utilisation_array),
            "mean_utilisation": self._safe_mean(utilisation_array),
            "over_limit_count": int(
                np.sum(utilisation_array > 1.0 + 1e-6)
            ) if utilisation_array.size else 0,
            "per_actuator": per_actuator,
        }

    def _calculate_joint_limit_utilisation(
        self,
        model: Any,
        data: Any,
    ) -> MetricDict:
        """
        Measure how close each leg joint is to its nearest configured limit.

        Utilisation is defined as the absolute distance from the centre of the
        joint range divided by the half-range:

            utilisation = |q - midpoint| / half_range

        Thus 0 means the joint is at the centre of its range, 1 means it is
        exactly at a limit, and values above 1 indicate a limit violation.
        """
        per_joint: Dict[str, MetricDict] = {}
        utilisations: list[float] = []
        limit_margins: list[float] = []

        for name, joint_id in self._leg_joint_ids.items():
            qpos_address = int(model.jnt_qposadr[joint_id])
            position = float(data.qpos[qpos_address])

            limited = bool(model.jnt_limited[joint_id])
            joint_range = np.asarray(model.jnt_range[joint_id], dtype=np.float64)
            lower = float(joint_range[0])
            upper = float(joint_range[1])

            if limited:
                half_range = 0.5 * (upper - lower)
                midpoint = 0.5 * (upper + lower)

                if half_range <= self.epsilon:
                    raise ValueError(
                        f"Joint {name!r} has an invalid range "
                        f"[{lower}, {upper}]."
                    )

                utilisation = abs(position - midpoint) / half_range
                lower_margin = position - lower
                upper_margin = upper - position
                nearest_margin = min(lower_margin, upper_margin)
                violation = position < lower or position > upper
            else:
                utilisation = float("nan")
                lower_margin = float("nan")
                upper_margin = float("nan")
                nearest_margin = float("nan")
                violation = False

            per_joint[name] = {
                "position": position,
                "lower_limit": lower if limited else float("nan"),
                "upper_limit": upper if limited else float("nan"),
                "utilisation": float(utilisation),
                "nearest_limit_margin": float(nearest_margin),
                "lower_limit_margin": float(lower_margin),
                "upper_limit_margin": float(upper_margin),
                "limit_violation": bool(violation),
            }

            if np.isfinite(utilisation):
                utilisations.append(float(utilisation))
            if np.isfinite(nearest_margin):
                limit_margins.append(float(nearest_margin))

        utilisation_array = np.asarray(utilisations, dtype=np.float64)
        margin_array = np.asarray(limit_margins, dtype=np.float64)

        return {
            "maximum_utilisation": self._safe_max(utilisation_array),
            "mean_utilisation": self._safe_mean(utilisation_array),
            "minimum_limit_margin": self._safe_min(margin_array),
            "violation_count": sum(
                int(values["limit_violation"])
                for values in per_joint.values()
            ),
            "per_joint": per_joint,
        }

    @staticmethod
    def _quaternion_to_euler(quaternion: FloatArray) -> FloatArray:
        """
        Convert MuJoCo quaternion [w, x, y, z] to XYZ roll-pitch-yaw.

        Returned angles are in radians.
        """
        w, x, y, z = quaternion

        sin_roll_cos_pitch = 2.0 * (w * x + y * z)
        cos_roll_cos_pitch = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sin_roll_cos_pitch, cos_roll_cos_pitch)

        sin_pitch = 2.0 * (w * y - z * x)
        pitch = np.arcsin(np.clip(sin_pitch, -1.0, 1.0))

        sin_yaw_cos_pitch = 2.0 * (w * z + x * y)
        cos_yaw_cos_pitch = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(sin_yaw_cos_pitch, cos_yaw_cos_pitch)

        return np.asarray([roll, pitch, yaw], dtype=np.float64)

    @staticmethod
    def _wrap_angles(angles: FloatArray) -> FloatArray:
        """Wrap angles to the interval [-pi, pi)."""
        return (angles + np.pi) % (2.0 * np.pi) - np.pi

    @staticmethod
    def _vector_dict(vector: FloatArray) -> Dict[str, float]:
        return {
            "x": float(vector[0]),
            "y": float(vector[1]),
            "z": float(vector[2]),
        }

    @classmethod
    def _vector_with_magnitude(cls, vector: FloatArray) -> Dict[str, float]:
        values = cls._vector_dict(vector)
        values["magnitude"] = float(np.linalg.norm(vector))
        return values

    @staticmethod
    def _validate_unique_names(
        names: Iterable[str],
        argument_name: str,
    ) -> tuple[str, ...]:
        cleaned = tuple(str(name).strip() for name in names)

        if not cleaned or any(not name for name in cleaned):
            raise ValueError(
                f"{argument_name} must contain one or more non-empty names."
            )
        if len(set(cleaned)) != len(cleaned):
            raise ValueError(f"{argument_name} contains duplicate names.")

        return cleaned

    @staticmethod
    def _safe_max(values: FloatArray) -> float:
        return float(np.max(values)) if values.size else float("nan")

    @staticmethod
    def _safe_min(values: FloatArray) -> float:
        return float(np.min(values)) if values.size else float("nan")

    @staticmethod
    def _safe_mean(values: FloatArray) -> float:
        return float(np.mean(values)) if values.size else float("nan")
