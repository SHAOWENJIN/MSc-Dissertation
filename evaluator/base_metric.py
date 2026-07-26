"""
Base interface for all evaluation metric modules.

This module defines the common behaviour required by RobotMetrics,
ContactMetrics, GraspMetrics, and TaskMetrics.

Metric classes calculate only the current simulation state. They do not
store complete time histories. Time-series storage is the responsibility
of DataLogger.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict


MetricDict = Dict[str, Any]


class BaseMetric(ABC):
    """
    Abstract base class for all metric modules.

    Every metric module must implement:

    - update(model, data)
    - reset(model, data)

    The current metric values are stored in ``self._metrics`` and can be
    retrieved through ``get_metrics()``.

    Notes
    -----
    This class deliberately does not import MuJoCo. Keeping the interface
    independent of MuJoCo's concrete Python types makes the framework
    easier to test and extend.
    """

    def __init__(self, name: str) -> None:
        """
        Initialise a metric module.

        Parameters
        ----------
        name:
            Human-readable module name, such as ``"robot"`` or
            ``"contact"``.
        """
        if not name or not name.strip():
            raise ValueError("Metric module name must be a non-empty string.")

        self._name = name.strip()
        self._metrics: MetricDict = {}
        self._is_initialised = False

    @property
    def name(self) -> str:
        """Return the metric module name."""
        return self._name

    @property
    def is_initialised(self) -> bool:
        """
        Return whether the module has been reset or updated successfully.
        """
        return self._is_initialised

    @abstractmethod
    def update(self, model: Any, data: Any) -> MetricDict:
        """
        Calculate metrics for the current simulation step.

        Parameters
        ----------
        model:
            MuJoCo model object.

        data:
            MuJoCo data object containing the current simulation state.

        Returns
        -------
        dict
            Metrics calculated for the current simulation step.

        Notes
        -----
        Implementations should update ``self._metrics`` and return the
        resulting dictionary.

        This method must not store the full simulation history.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self, model: Any | None = None, data: Any | None = None) -> None:
        """
        Reset the metric module before a new experimental trial.

        Parameters
        ----------
        model:
            Optional MuJoCo model object.

        data:
            Optional MuJoCo data object representing the initial state.

        Notes
        -----
        Stateful metrics may use the initial state to establish reference
        quantities. For example:

        - RobotMetrics records the initial base position.
        - ContactMetrics clears contact-duration counters.
        - GraspMetrics clears its current contact set.
        - TaskMetrics records the trial start position and time.
        """
        raise NotImplementedError

    def get_metrics(self) -> MetricDict:
        """
        Return a copy of the current metric values.

        A deep copy is returned so that Logger or other external modules
        cannot accidentally modify the internal metric state.
        """
        return deepcopy(self._metrics)

    def _set_metrics(self, metrics: MetricDict) -> MetricDict:
        """
        Store and return the current metric dictionary.

        Subclasses should normally call this method at the end of
        ``update()``.

        Parameters
        ----------
        metrics:
            Current-step metric values.

        Returns
        -------
        dict
            A copy of the stored metric values.
        """
        if not isinstance(metrics, dict):
            raise TypeError(
                f"{self.__class__.__name__} metrics must be provided as a dict."
            )

        self._metrics = metrics
        self._is_initialised = True

        return self.get_metrics()

    def _clear_metrics(self) -> None:
        """
        Clear current metric values.

        Subclasses should normally call this method inside ``reset()``.
        """
        self._metrics = {}
        self._is_initialised = False

    def __repr__(self) -> str:
        """Return a concise representation for debugging."""
        return (
            f"{self.__class__.__name__}("
            f"name={self._name!r}, "
            f"initialised={self._is_initialised}, "
            f"metric_count={len(self._metrics)}"
            f")"
        )