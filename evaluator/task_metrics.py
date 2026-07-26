"""
Task-level metrics for the modified Unitree Go2 ladder-climbing robot.

This module evaluates trial progress and task completion. It consumes the
latest outputs from RobotMetrics, ContactMetrics, and GraspMetrics rather than
recalculating lower-level quantities.

Responsibilities
----------------
- elapsed traversal time
- forward and path distance
- average and peak forward speed
- maximum rung reached and rung advancement
- confirmed regrasp events
- actuator work and energy per rung
- configurable task-success detection

The module stores only the small amount of state required to integrate work
and recognise events. Complete time-series storage remains the responsibility
of DataLogger.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

import mujoco
import numpy as np

from .base_metric import BaseMetric, MetricDict


@dataclass
class _RegraspCandidate:
    """Potential fingertip-rung attachment waiting for confirmation."""

    rung: str
    first_seen_time: float


class TaskMetrics(BaseMetric):
    """Evaluate traversal-level performance and task completion."""

    DEFAULT_BASE_BODY_NAME = "base"

    def __init__(
        self,
        robot_metric: BaseMetric,
        contact_metric: BaseMetric,
        grasp_metric: BaseMetric,
        *,
        base_body_name: str = DEFAULT_BASE_BODY_NAME,
        target_rung_name: str | None = "rung_11",
        target_forward_progress: float | None = None,
        minimum_successful_fingertips: int = 2,
        success_hold_time: float = 0.25,
        regrasp_confirmation_time: float = 0.05,
        count_backward_regrasps: bool = False,
        actuator_names: Sequence[str] | None = None,
        epsilon: float = 1e-9,
    ) -> None:
        """
        Parameters
        ----------
        robot_metric:
            RobotMetrics instance or compatible metric provider.

        contact_metric:
            ContactMetrics instance or compatible metric provider.

        grasp_metric:
            GraspMetrics instance or compatible metric provider.

        base_body_name:
            Floating-base body used to calculate travelled path length.

        target_rung_name:
            Rung whose sustained contact can complete the task. Set to None
            to disable rung-based completion.

        target_forward_progress:
            Optional required forward displacement in metres. When both a
            target rung and target progress are configured, both conditions
            must be satisfied.

        minimum_successful_fingertips:
            Minimum number of distinct fingertips required during the final
            success hold.

        success_hold_time:
            Duration for which all completion conditions must remain true.

        regrasp_confirmation_time:
            New fingertip-rung contact duration required before it is counted
            as a confirmed regrasp.

        count_backward_regrasps:
            If False, a change to a lower-numbered rung is recorded as a
            backward transition but not counted as a successful regrasp.

        actuator_names:
            Actuators included in work integration. None includes every model
            actuator, covering both locomotion and gripper effort.

        epsilon:
            Small positive numerical tolerance.
        """
        super().__init__(name="task")

        for metric, label in (
            (robot_metric, "robot_metric"),
            (contact_metric, "contact_metric"),
            (grasp_metric, "grasp_metric"),
        ):
            if not isinstance(metric, BaseMetric):
                raise TypeError(f"{label} must inherit from BaseMetric.")

        if target_rung_name is not None and not target_rung_name.strip():
            raise ValueError("target_rung_name must be non-empty or None.")
        if target_forward_progress is not None and target_forward_progress < 0.0:
            raise ValueError("target_forward_progress must be non-negative.")
        if minimum_successful_fingertips <= 0:
            raise ValueError("minimum_successful_fingertips must be positive.")
        if success_hold_time < 0.0:
            raise ValueError("success_hold_time must be non-negative.")
        if regrasp_confirmation_time < 0.0:
            raise ValueError("regrasp_confirmation_time must be non-negative.")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive.")

        self.robot_metric = robot_metric
        self.contact_metric = contact_metric
        self.grasp_metric = grasp_metric

        self.base_body_name = base_body_name.strip()
        self.target_rung_name = (
            target_rung_name.strip() if target_rung_name is not None else None
        )
        self.target_forward_progress = target_forward_progress
        self.minimum_successful_fingertips = int(minimum_successful_fingertips)
        self.success_hold_time = float(success_hold_time)
        self.regrasp_confirmation_time = float(regrasp_confirmation_time)
        self.count_backward_regrasps = bool(count_backward_regrasps)
        self.actuator_names = (
            tuple(str(name).strip() for name in actuator_names)
            if actuator_names is not None
            else None
        )
        self.epsilon = float(epsilon)

        if self.actuator_names is not None:
            if not self.actuator_names or any(not name for name in self.actuator_names):
                raise ValueError("actuator_names must contain non-empty names.")
            if len(set(self.actuator_names)) != len(self.actuator_names):
                raise ValueError("actuator_names contains duplicates.")

        self._base_body_id: int | None = None
        self._actuator_ids: Dict[str, int] = {}

        self._start_time: float | None = None
        self._previous_time: float | None = None
        self._initial_base_position: np.ndarray | None = None
        self._previous_base_position: np.ndarray | None = None

        self._path_distance = 0.0
        self._positive_mechanical_work = 0.0
        self._absolute_mechanical_work = 0.0
        self._net_mechanical_work = 0.0
        self._peak_forward_speed = 0.0

        self._initial_rung_index: int | None = None
        self._maximum_rung_index: int | None = None

        self._confirmed_rung_by_fingertip: Dict[str, str] = {}
        self._regrasp_candidates: Dict[str, _RegraspCandidate] = {}
        self._successful_regrasp_count = 0
        self._backward_regrasp_count = 0
        self._lateral_regrasp_count = 0

        self._success_condition_start_time: float | None = None
        self._task_success = False
        self._completion_time: float | None = None

    def reset(self, model: Any | None = None, data: Any | None = None) -> None:
        """Reset all trial-level state."""
        self._clear_metrics()

        self._base_body_id = None
        self._actuator_ids = {}

        self._start_time = None
        self._previous_time = None
        self._initial_base_position = None
        self._previous_base_position = None

        self._path_distance = 0.0
        self._positive_mechanical_work = 0.0
        self._absolute_mechanical_work = 0.0
        self._net_mechanical_work = 0.0
        self._peak_forward_speed = 0.0

        self._initial_rung_index = None
        self._maximum_rung_index = None

        self._confirmed_rung_by_fingertip = {}
        self._regrasp_candidates = {}
        self._successful_regrasp_count = 0
        self._backward_regrasp_count = 0
        self._lateral_regrasp_count = 0

        self._success_condition_start_time = None
        self._task_success = False
        self._completion_time = None

        if model is None and data is None:
            return
        if model is None or data is None:
            raise ValueError("reset() requires both model and data, or neither.")

        self._resolve_model_ids(model)

        current_time = float(data.time)
        base_position = self._get_base_position(data)

        self._start_time = current_time
        self._previous_time = current_time
        self._initial_base_position = base_position.copy()
        self._previous_base_position = base_position.copy()
        self._is_initialised = True

    def update(self, model: Any, data: Any) -> MetricDict:
        """
        Update trial metrics after Robot, Contact, and Grasp metrics.

        Required per-step order:

            robot_metric.update(model, data)
            contact_metric.update(model, data)
            grasp_metric.update(model, data)
            task_metric.update(model, data)
        """
        if model is None or data is None:
            raise ValueError("update() requires valid MuJoCo model and data.")

        if self._base_body_id is None:
            self._resolve_model_ids(model)

        current_time = float(data.time)
        robot = self._get_current_metric(
            self.robot_metric, "RobotMetrics", current_time
        )
        contact = self._get_current_metric(
            self.contact_metric, "ContactMetrics", current_time
        )
        grasp = self._get_current_metric(
            self.grasp_metric, "GraspMetrics", current_time
        )

        if self._start_time is None:
            self._start_time = current_time

        dt = self._calculate_dt(current_time)
        base_position = self._get_base_position(data)
        self._update_path_distance(base_position)
        power = self._update_mechanical_work(model, data, dt)

        forward_progress = float(robot["progress"]["forward"])
        forward_speed = float(robot["linear_velocity"]["x"])
        self._peak_forward_speed = max(
            self._peak_forward_speed,
            max(forward_speed, 0.0),
        )

        contacts = list(contact.get("contacts", []))
        active_rung_indices = self._active_rung_indices(contacts)
        self._update_rung_progress(active_rung_indices)
        self._update_regrasp_events(contacts, current_time)

        elapsed_time = max(current_time - self._start_time, 0.0)
        rung_advancement = self._rung_advancement()
        self._update_success_state(
            current_time=current_time,
            forward_progress=forward_progress,
            active_rung_indices=active_rung_indices,
            unique_fingertip_count=int(
                grasp.get("unique_fingertip_count", 0)
            ),
        )

        average_forward_speed = (
            forward_progress / elapsed_time
            if elapsed_time > self.epsilon
            else 0.0
        )
        average_path_speed = (
            self._path_distance / elapsed_time
            if elapsed_time > self.epsilon
            else 0.0
        )
        energy_per_rung = (
            self._absolute_mechanical_work / rung_advancement
            if rung_advancement > 0
            else float("nan")
        )
        energy_per_forward_metre = (
            self._absolute_mechanical_work / forward_progress
            if forward_progress > self.epsilon
            else float("nan")
        )

        metrics: MetricDict = {
            "time": current_time,
            "elapsed_time": float(elapsed_time),
            "traversal": {
                "forward_progress": forward_progress,
                "path_distance": float(self._path_distance),
                "average_forward_speed": float(average_forward_speed),
                "average_path_speed": float(average_path_speed),
                "current_forward_speed": forward_speed,
                "peak_forward_speed": float(self._peak_forward_speed),
            },
            "rung_progress": {
                "initial_rung_index": self._initial_rung_index,
                "maximum_rung_index": self._maximum_rung_index,
                "rung_advancement": int(rung_advancement),
                "active_rung_indices": active_rung_indices,
            },
            "regrasp": {
                "successful_regrasp_count": int(
                    self._successful_regrasp_count
                ),
                "backward_regrasp_count": int(
                    self._backward_regrasp_count
                ),
                "same_rung_transition_count": int(
                    self._lateral_regrasp_count
                ),
                "confirmed_rung_by_fingertip": dict(
                    self._confirmed_rung_by_fingertip
                ),
            },
            "energy": {
                "instantaneous_net_power": float(power["net"]),
                "instantaneous_absolute_power": float(power["absolute"]),
                "instantaneous_positive_power": float(power["positive"]),
                "net_mechanical_work": float(self._net_mechanical_work),
                "positive_mechanical_work": float(
                    self._positive_mechanical_work
                ),
                "absolute_mechanical_work": float(
                    self._absolute_mechanical_work
                ),
                "energy_per_rung": float(energy_per_rung),
                "energy_per_forward_metre": float(
                    energy_per_forward_metre
                ),
            },
            "success": {
                "task_success": bool(self._task_success),
                "completion_time": self._completion_time,
                "target_rung_name": self.target_rung_name,
                "target_forward_progress": self.target_forward_progress,
                "minimum_successful_fingertips": (
                    self.minimum_successful_fingertips
                ),
                "success_hold_time": self.success_hold_time,
                "condition_hold_duration": self._current_success_hold_duration(
                    current_time
                ),
            },
        }

        self._previous_time = current_time
        self._previous_base_position = base_position.copy()

        return self._set_metrics(metrics)

    def _resolve_model_ids(self, model: Any) -> None:
        base_body_id = int(
            mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                self.base_body_name,
            )
        )
        if base_body_id < 0:
            raise ValueError(
                f"MuJoCo base body {self.base_body_name!r} was not found."
            )
        self._base_body_id = base_body_id

        if self.actuator_names is None:
            self._actuator_ids = {
                self._actuator_name(model, actuator_id): actuator_id
                for actuator_id in range(int(model.nu))
            }
        else:
            actuator_ids: Dict[str, int] = {}
            for name in self.actuator_names:
                actuator_id = int(
                    mujoco.mj_name2id(
                        model,
                        mujoco.mjtObj.mjOBJ_ACTUATOR,
                        name,
                    )
                )
                if actuator_id < 0:
                    raise ValueError(
                        f"MuJoCo actuator {name!r} was not found."
                    )
                actuator_ids[name] = actuator_id
            self._actuator_ids = actuator_ids

    @staticmethod
    def _actuator_name(model: Any, actuator_id: int) -> str:
        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            actuator_id,
        )
        return str(name) if name is not None else f"actuator_{actuator_id}"

    def _get_current_metric(
        self,
        metric: BaseMetric,
        label: str,
        current_time: float,
    ) -> Mapping[str, Any]:
        values = metric.get_metrics()

        if not metric.is_initialised or not values:
            raise RuntimeError(f"{label} must be updated before TaskMetrics.")

        metric_time = values.get("time")
        if metric_time is None:
            raise KeyError(f"{label} output does not contain 'time'.")

        if not np.isclose(
            float(metric_time),
            current_time,
            rtol=0.0,
            atol=max(self.epsilon, 1e-12),
        ):
            raise RuntimeError(
                f"{label} contains stale data: metric time={metric_time}, "
                f"current time={current_time}."
            )

        return values

    def _get_base_position(self, data: Any) -> np.ndarray:
        assert self._base_body_id is not None
        return np.asarray(
            data.xpos[self._base_body_id],
            dtype=np.float64,
        ).copy()

    def _calculate_dt(self, current_time: float) -> float:
        if self._previous_time is None:
            return 0.0

        dt = current_time - self._previous_time
        return float(dt) if dt > self.epsilon else 0.0

    def _update_path_distance(self, base_position: np.ndarray) -> None:
        if self._previous_base_position is None:
            self._previous_base_position = base_position.copy()
            if self._initial_base_position is None:
                self._initial_base_position = base_position.copy()
            return

        self._path_distance += float(
            np.linalg.norm(base_position - self._previous_base_position)
        )

    def _update_mechanical_work(
        self,
        model: Any,
        data: Any,
        dt: float,
    ) -> Dict[str, float]:
        """
        Integrate actuator mechanical work.

        MuJoCo actuator power is calculated as scalar actuator force multiplied
        by scalar actuator velocity. Absolute work is the primary energy proxy
        because it does not cancel positive and negative work.
        """
        if not self._actuator_ids:
            return {"net": 0.0, "positive": 0.0, "absolute": 0.0}

        ids = np.asarray(
            list(self._actuator_ids.values()),
            dtype=np.int64,
        )
        actuator_force = np.asarray(
            data.actuator_force,
            dtype=np.float64,
        )[ids]
        actuator_velocity = np.asarray(
            data.actuator_velocity,
            dtype=np.float64,
        )[ids]

        individual_power = actuator_force * actuator_velocity
        net_power = float(np.sum(individual_power))
        positive_power = float(np.sum(np.maximum(individual_power, 0.0)))
        absolute_power = float(np.sum(np.abs(individual_power)))

        if dt > 0.0:
            self._net_mechanical_work += net_power * dt
            self._positive_mechanical_work += positive_power * dt
            self._absolute_mechanical_work += absolute_power * dt

        return {
            "net": net_power,
            "positive": positive_power,
            "absolute": absolute_power,
        }

    def _active_rung_indices(
        self,
        contacts: Sequence[Mapping[str, Any]],
    ) -> list[int]:
        indices = {
            index
            for contact in contacts
            if (
                index := self._parse_rung_index(
                    str(contact.get("rung", ""))
                )
            )
            is not None
        }
        return sorted(indices)

    def _update_rung_progress(self, active_indices: Sequence[int]) -> None:
        if not active_indices:
            return

        current_maximum = max(active_indices)

        if self._initial_rung_index is None:
            self._initial_rung_index = current_maximum
        if (
            self._maximum_rung_index is None
            or current_maximum > self._maximum_rung_index
        ):
            self._maximum_rung_index = current_maximum

    def _rung_advancement(self) -> int:
        if (
            self._initial_rung_index is None
            or self._maximum_rung_index is None
        ):
            return 0
        return max(
            int(self._maximum_rung_index - self._initial_rung_index),
            0,
        )

    def _update_regrasp_events(
        self,
        contacts: Sequence[Mapping[str, Any]],
        current_time: float,
    ) -> None:
        """
        Confirm fingertip attachment changes after a short dwell period.

        The first confirmed rung for a fingertip establishes its baseline and
        is not counted as a regrasp. A later confirmed change to a higher rung
        is counted as successful.
        """
        active_by_fingertip: Dict[str, Mapping[str, Any]] = {}

        for contact in contacts:
            fingertip = str(contact["fingertip"])
            existing = active_by_fingertip.get(fingertip)

            # If one fingertip simultaneously contacts more than one rung,
            # use the contact with the greatest normal force.
            if (
                existing is None
                or float(contact["normal_force"])
                > float(existing["normal_force"])
            ):
                active_by_fingertip[fingertip] = contact

        inactive_fingertips = (
            set(self._regrasp_candidates) - set(active_by_fingertip)
        )
        for fingertip in inactive_fingertips:
            del self._regrasp_candidates[fingertip]

        for fingertip, contact in active_by_fingertip.items():
            rung = str(contact["rung"])
            duration = float(contact.get("contact_duration", 0.0))

            candidate = self._regrasp_candidates.get(fingertip)
            if candidate is None or candidate.rung != rung:
                candidate = _RegraspCandidate(
                    rung=rung,
                    first_seen_time=current_time,
                )
                self._regrasp_candidates[fingertip] = candidate

            dwell_time = max(
                duration,
                current_time - candidate.first_seen_time,
            )
            if dwell_time + self.epsilon < self.regrasp_confirmation_time:
                continue

            previous_rung = self._confirmed_rung_by_fingertip.get(fingertip)
            if previous_rung == rung:
                continue

            if previous_rung is not None:
                previous_index = self._parse_rung_index(previous_rung)
                current_index = self._parse_rung_index(rung)

                if (
                    previous_index is not None
                    and current_index is not None
                ):
                    if current_index > previous_index:
                        self._successful_regrasp_count += 1
                    elif current_index < previous_index:
                        self._backward_regrasp_count += 1
                        if self.count_backward_regrasps:
                            self._successful_regrasp_count += 1
                    else:
                        self._lateral_regrasp_count += 1
                else:
                    self._successful_regrasp_count += 1

            self._confirmed_rung_by_fingertip[fingertip] = rung

    def _update_success_state(
        self,
        *,
        current_time: float,
        forward_progress: float,
        active_rung_indices: Sequence[int],
        unique_fingertip_count: int,
    ) -> None:
        if self._task_success:
            return

        target_rung_satisfied = True
        if self.target_rung_name is not None:
            target_index = self._parse_rung_index(self.target_rung_name)
            target_rung_satisfied = (
                target_index is not None
                and target_index in active_rung_indices
            )

        progress_satisfied = (
            self.target_forward_progress is None
            or forward_progress + self.epsilon
            >= self.target_forward_progress
        )
        support_satisfied = (
            unique_fingertip_count
            >= self.minimum_successful_fingertips
        )

        all_conditions = (
            target_rung_satisfied
            and progress_satisfied
            and support_satisfied
        )

        if not all_conditions:
            self._success_condition_start_time = None
            return

        if self._success_condition_start_time is None:
            self._success_condition_start_time = current_time

        held_duration = (
            current_time - self._success_condition_start_time
        )
        if held_duration + self.epsilon >= self.success_hold_time:
            self._task_success = True
            assert self._start_time is not None
            self._completion_time = float(
                current_time - self._start_time
            )

    def _current_success_hold_duration(self, current_time: float) -> float:
        if self._success_condition_start_time is None:
            return 0.0
        return float(
            max(current_time - self._success_condition_start_time, 0.0)
        )

    @staticmethod
    def _parse_rung_index(rung_name: str) -> int | None:
        try:
            return int(rung_name.rsplit("_", maxsplit=1)[1])
        except (IndexError, ValueError):
            return None
