"""
Top-level orchestrator for the quantitative stability evaluation framework.

StabilityEvaluator owns the metric modules and enforces the required update
order:

    RobotMetrics
        ↓
    ContactMetrics
        ↓
    GraspMetrics
        ↓
    TaskMetrics

The class performs no metric calculations. It only resets modules, triggers
their updates, validates outputs, and returns one merged dictionary.

Complete time-series storage remains the responsibility of DataLogger.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping

import numpy as np

from .base_metric import BaseMetric, MetricDict
from .contact_metrics import ContactMetrics
from .grasp_metrics import GraspMetrics
from .robot_metrics import RobotMetrics
from .task_metrics import TaskMetrics


class StabilityEvaluator:
    """
    Coordinate all metric modules for one simulation trial.

    Parameters
    ----------
    robot_metrics:
        Optional preconfigured RobotMetrics instance.

    contact_metrics:
        Optional preconfigured ContactMetrics instance.

    grasp_metrics:
        Optional preconfigured GraspMetrics instance. When omitted, it is
        constructed from the selected ContactMetrics instance.

    task_metrics:
        Optional preconfigured TaskMetrics instance. When omitted, it is
        constructed from the selected RobotMetrics, ContactMetrics, and
        GraspMetrics instances.

    strict_dependency_check:
        When True, verify that supplied GraspMetrics and TaskMetrics reference
        the same lower-level module instances used by this evaluator.
    """

    MODULE_ORDER = ("robot", "contact", "grasp", "task")

    def __init__(
        self,
        robot_metrics: RobotMetrics | None = None,
        contact_metrics: ContactMetrics | None = None,
        grasp_metrics: GraspMetrics | None = None,
        task_metrics: TaskMetrics | None = None,
        *,
        strict_dependency_check: bool = True,
    ) -> None:
        self.robot_metrics = robot_metrics or RobotMetrics()
        self.contact_metrics = contact_metrics or ContactMetrics()

        self.grasp_metrics = grasp_metrics or GraspMetrics(
            contact_metric=self.contact_metrics
        )

        self.task_metrics = task_metrics or TaskMetrics(
            robot_metric=self.robot_metrics,
            contact_metric=self.contact_metrics,
            grasp_metric=self.grasp_metrics,
        )

        self.strict_dependency_check = bool(strict_dependency_check)

        self._validate_metric_types()
        if self.strict_dependency_check:
            self._validate_dependencies()

        self._latest_metrics: MetricDict = {}
        self._is_initialised = False
        self._last_update_time: float | None = None
        self._step_count = 0

    @property
    def is_initialised(self) -> bool:
        """Return whether reset() or update() has initialised the evaluator."""
        return self._is_initialised

    @property
    def step_count(self) -> int:
        """Return the number of successful update calls in the current trial."""
        return self._step_count

    @property
    def last_update_time(self) -> float | None:
        """Return the MuJoCo time of the most recent successful update."""
        return self._last_update_time

    def reset(self, model: Any | None = None, data: Any | None = None) -> None:
        """
        Reset every metric module for a new trial.

        Parameters
        ----------
        model:
            Optional MuJoCo model.

        data:
            Optional MuJoCo data containing the trial's initial state.

        Notes
        -----
        Supply both model and data for normal simulation use. Calling reset()
        without either argument only clears internal state.
        """
        if (model is None) != (data is None):
            raise ValueError("reset() requires both model and data, or neither.")

        self._latest_metrics = {}
        self._last_update_time = None
        self._step_count = 0
        self._is_initialised = False

        # Dependency order is retained during reset so lower-level reference
        # states are available before dependent modules are initialised.
        self.robot_metrics.reset(model, data)
        self.contact_metrics.reset(model, data)
        self.grasp_metrics.reset(model, data)
        self.task_metrics.reset(model, data)

        self._is_initialised = model is not None and data is not None

    def update(self, model: Any, data: Any) -> MetricDict:
        """
        Update all metric modules in the required dependency order.

        Returns
        -------
        dict
            Nested metric output with top-level keys:

            - metadata
            - robot
            - contact
            - grasp
            - task
        """
        if model is None or data is None:
            raise ValueError("update() requires valid MuJoCo model and data.")

        current_time = float(data.time)

        if (
            self._last_update_time is not None
            and current_time < self._last_update_time
        ):
            raise RuntimeError(
                "Simulation time moved backwards. Call reset(model, data) "
                "before starting a new trial or restoring an earlier state."
            )

        robot_output = self.robot_metrics.update(model, data)
        contact_output = self.contact_metrics.update(model, data)
        grasp_output = self.grasp_metrics.update(model, data)
        task_output = self.task_metrics.update(model, data)

        outputs = {
            "robot": robot_output,
            "contact": contact_output,
            "grasp": grasp_output,
            "task": task_output,
        }

        self._validate_update_times(outputs, current_time)

        self._step_count += 1
        self._last_update_time = current_time
        self._is_initialised = True

        self._latest_metrics = {
            "metadata": {
                "time": current_time,
                "step_count": self._step_count,
                "module_order": list(self.MODULE_ORDER),
            },
            **outputs,
        }

        return self.get_metrics()

    def get_metrics(self) -> MetricDict:
        """Return a deep copy of the latest merged metric output."""
        return deepcopy(self._latest_metrics)

    def get_module_metrics(self, module_name: str) -> MetricDict:
        """
        Return the latest output for one module.

        Parameters
        ----------
        module_name:
            One of ``robot``, ``contact``, ``grasp``, or ``task``.
        """
        cleaned_name = module_name.strip().lower()
        if cleaned_name not in self.MODULE_ORDER:
            raise KeyError(
                f"Unknown module {module_name!r}. "
                f"Expected one of {self.MODULE_ORDER}."
            )

        if cleaned_name not in self._latest_metrics:
            raise RuntimeError(
                "No metric output is available. Call update() first."
            )

        return deepcopy(self._latest_metrics[cleaned_name])

    def get_summary(self) -> MetricDict:
        """
        Return a compact set of high-level scalar indicators.

        This method only selects values already calculated by metric modules;
        it does not perform new metric calculations.
        """
        if not self._latest_metrics:
            raise RuntimeError("No metric output is available. Call update() first.")

        robot = self._latest_metrics["robot"]
        contact = self._latest_metrics["contact"]
        grasp = self._latest_metrics["grasp"]
        task = self._latest_metrics["task"]

        return {
            "time": float(self._latest_metrics["metadata"]["time"]),
            "step_count": int(self._latest_metrics["metadata"]["step_count"]),
            "forward_progress": float(robot["progress"]["forward"]),
            "lateral_drift_abs": float(
                robot["progress"]["lateral_drift_abs"]
            ),
            "normal_drift_abs": float(
                robot["progress"]["normal_drift_abs"]
            ),
            "base_acceleration_magnitude": float(
                robot["base_acceleration"]["magnitude"]
            ),
            "maximum_leg_effort_utilisation": float(
                robot["leg_actuator_effort"]["maximum_utilisation"]
            ),
            "maximum_joint_limit_utilisation": float(
                robot["joint_limit"]["maximum_utilisation"]
            ),
            "active_contact_count": int(
                contact["active_contact_count"]
            ),
            "slipping_contact_count": int(
                contact["summary"]["slipping_contact_count"]
            ),
            "maximum_friction_ratio": float(
                contact["summary"]["maximum_friction_ratio"]
            ),
            "minimum_grasp_singular_value": float(
                grasp["grasp_matrix"]["minimum_singular_value"]
            ),
            "grasp_isotropy": float(
                grasp["grasp_matrix"]["isotropy"]
            ),
            "grasp_full_wrench_rank": bool(
                grasp["grasp_matrix"]["full_wrench_rank"]
            ),
            "normalised_load_share_entropy": float(
                grasp["force_distribution"][
                    "normalised_load_share_entropy"
                ]
            ),
            "rung_advancement": int(
                task["rung_progress"]["rung_advancement"]
            ),
            "successful_regrasp_count": int(
                task["regrasp"]["successful_regrasp_count"]
            ),
            "absolute_mechanical_work": float(
                task["energy"]["absolute_mechanical_work"]
            ),
            "task_success": bool(
                task["success"]["task_success"]
            ),
            "completion_time": task["success"]["completion_time"],
        }

    def module_map(self) -> Dict[str, BaseMetric]:
        """Return the metric modules keyed by their framework names."""
        return {
            "robot": self.robot_metrics,
            "contact": self.contact_metrics,
            "grasp": self.grasp_metrics,
            "task": self.task_metrics,
        }

    def _validate_metric_types(self) -> None:
        expected_types = (
            (self.robot_metrics, RobotMetrics, "robot_metrics"),
            (self.contact_metrics, ContactMetrics, "contact_metrics"),
            (self.grasp_metrics, GraspMetrics, "grasp_metrics"),
            (self.task_metrics, TaskMetrics, "task_metrics"),
        )

        for metric, expected_type, argument_name in expected_types:
            if not isinstance(metric, expected_type):
                raise TypeError(
                    f"{argument_name} must be an instance of "
                    f"{expected_type.__name__}."
                )

    def _validate_dependencies(self) -> None:
        """
        Ensure dependent modules reference this evaluator's module instances.
        """
        if self.grasp_metrics.contact_metric is not self.contact_metrics:
            raise ValueError(
                "grasp_metrics references a different ContactMetrics "
                "instance from the evaluator."
            )

        if self.task_metrics.robot_metric is not self.robot_metrics:
            raise ValueError(
                "task_metrics references a different RobotMetrics "
                "instance from the evaluator."
            )

        if self.task_metrics.contact_metric is not self.contact_metrics:
            raise ValueError(
                "task_metrics references a different ContactMetrics "
                "instance from the evaluator."
            )

        if self.task_metrics.grasp_metric is not self.grasp_metrics:
            raise ValueError(
                "task_metrics references a different GraspMetrics "
                "instance from the evaluator."
            )

    @staticmethod
    def _validate_update_times(
        outputs: Mapping[str, Mapping[str, Any]],
        current_time: float,
    ) -> None:
        for module_name, output in outputs.items():
            if not isinstance(output, Mapping):
                raise TypeError(
                    f"{module_name} metric output must be a mapping."
                )

            if "time" not in output:
                raise KeyError(
                    f"{module_name} metric output does not contain 'time'."
                )

            if not np.isclose(
                float(output["time"]),
                current_time,
                rtol=0.0,
                atol=1e-12,
            ):
                raise RuntimeError(
                    f"{module_name} output time {output['time']} does not "
                    f"match simulation time {current_time}."
                )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"initialised={self._is_initialised}, "
            f"step_count={self._step_count}, "
            f"last_update_time={self._last_update_time}"
            f")"
        )
