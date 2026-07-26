"""
Grasp-level metrics for the modified Unitree Go2 ladder-climbing robot.

This module consumes the logical fingertip-rung contacts already extracted by
ContactMetrics. It does not read or reconstruct MuJoCo contacts independently.

For n active contacts, a point-contact grasp matrix is assembled as

    G_i = [ I ]
          [ [r_i]x ]

where r_i is the vector from the robot base reference point to contact i,
expressed in world coordinates. The complete matrix G has shape (6, 3n).

The resulting singular values describe instantaneous geometric grasp quality.
They are evaluation indicators, not a proof of force closure; friction-cone
constraints and actuator limits must be considered separately.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Sequence

import mujoco
import numpy as np
from numpy.typing import NDArray

from .base_metric import BaseMetric, MetricDict


FloatArray = NDArray[np.float64]


class GraspMetrics(BaseMetric):
    """Evaluate multi-contact grasp geometry using ContactMetrics output."""

    DEFAULT_BASE_BODY_NAME = "base"
    DEFAULT_EXPECTED_FINGERTIP_COUNT = 8

    def __init__(
        self,
        contact_metric: BaseMetric,
        *,
        base_body_name: str = DEFAULT_BASE_BODY_NAME,
        expected_fingertip_count: int = DEFAULT_EXPECTED_FINGERTIP_COUNT,
        singular_value_tolerance: float = 1e-8,
        epsilon: float = 1e-12,
    ) -> None:
        """
        Parameters
        ----------
        contact_metric:
            ContactMetrics instance, or another metric object returning the
            same ``contacts`` data structure.

        base_body_name:
            Robot body used as the grasp-wrench reference point.

        expected_fingertip_count:
            Number of available fingertips, used only to calculate contact
            coverage.

        singular_value_tolerance:
            Relative tolerance used when estimating grasp-matrix rank.

        epsilon:
            Small positive numerical tolerance.
        """
        super().__init__(name="grasp")

        if not isinstance(contact_metric, BaseMetric):
            raise TypeError("contact_metric must inherit from BaseMetric.")
        if not base_body_name.strip():
            raise ValueError("base_body_name must be non-empty.")
        if expected_fingertip_count <= 0:
            raise ValueError("expected_fingertip_count must be positive.")
        if singular_value_tolerance <= 0.0:
            raise ValueError("singular_value_tolerance must be positive.")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive.")

        self.contact_metric = contact_metric
        self.base_body_name = base_body_name.strip()
        self.expected_fingertip_count = int(expected_fingertip_count)
        self.singular_value_tolerance = float(singular_value_tolerance)
        self.epsilon = float(epsilon)

        self._base_body_id: int | None = None

    def reset(self, model: Any | None = None, data: Any | None = None) -> None:
        """Reset grasp state before a new experimental trial."""
        self._clear_metrics()
        self._base_body_id = None

        if model is None and data is None:
            return
        if model is None or data is None:
            raise ValueError("reset() requires both model and data, or neither.")

        self._resolve_base_body(model)
        self._is_initialised = True

    def update(self, model: Any, data: Any) -> MetricDict:
        """
        Calculate current grasp metrics from ContactMetrics' latest output.

        ContactMetrics must be updated before this method at each simulation
        step. A time mismatch raises an error rather than silently combining
        stale contact data with the current robot state.
        """
        if model is None or data is None:
            raise ValueError("update() requires valid MuJoCo model and data.")

        if self._base_body_id is None:
            self._resolve_base_body(model)

        contact_metrics = self.contact_metric.get_metrics()
        self._validate_contact_metrics(contact_metrics, float(data.time))

        contacts = list(contact_metrics.get("contacts", []))
        base_reference = np.asarray(
            data.xipos[self._base_body_id],
            dtype=np.float64,
        ).copy()

        contact_positions = self._extract_contact_positions(contacts)
        contact_names = [
            f"{contact['fingertip']}@{contact['rung']}"
            for contact in contacts
        ]

        grasp_matrix = self._build_grasp_matrix(
            contact_positions,
            base_reference,
        )
        matrix_metrics = self._analyse_grasp_matrix(grasp_matrix)

        geometry_metrics = self._calculate_contact_geometry(
            contacts=contacts,
            contact_positions=contact_positions,
            base_reference=base_reference,
        )
        distribution_metrics = self._calculate_force_distribution(contacts)

        unique_fingertips = sorted(
            {str(contact["fingertip"]) for contact in contacts}
        )
        unique_rungs = sorted(
            {str(contact["rung"]) for contact in contacts}
        )

        metrics: MetricDict = {
            "time": float(data.time),
            "reference_point": self._vector_dict(base_reference),
            "contact_count": len(contacts),
            "unique_fingertip_count": len(unique_fingertips),
            "unique_rung_count": len(unique_rungs),
            "contact_coverage": float(
                len(unique_fingertips) / self.expected_fingertip_count
            ),
            "active_fingertips": unique_fingertips,
            "active_rungs": unique_rungs,
            "contact_labels": contact_names,
            "contact_geometry": geometry_metrics,
            "force_distribution": distribution_metrics,
            "grasp_matrix": {
                **matrix_metrics,
                "shape": [
                    int(grasp_matrix.shape[0]),
                    int(grasp_matrix.shape[1]),
                ],
                "matrix": grasp_matrix.tolist(),
            },
        }

        return self._set_metrics(metrics)

    def _resolve_base_body(self, model: Any) -> None:
        body_id = int(
            mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                self.base_body_name,
            )
        )
        if body_id < 0:
            raise ValueError(
                f"MuJoCo base body {self.base_body_name!r} was not found."
            )
        self._base_body_id = body_id

    def _validate_contact_metrics(
        self,
        metrics: Mapping[str, Any],
        current_time: float,
    ) -> None:
        if not self.contact_metric.is_initialised or not metrics:
            raise RuntimeError(
                "ContactMetrics must be updated before GraspMetrics."
            )

        if "contacts" not in metrics:
            raise KeyError(
                "Contact metric output does not contain a 'contacts' field."
            )

        contact_time = metrics.get("time")
        if contact_time is None:
            raise KeyError(
                "Contact metric output does not contain a 'time' field."
            )

        if not np.isclose(
            float(contact_time),
            current_time,
            rtol=0.0,
            atol=max(self.epsilon, 1e-12),
        ):
            raise RuntimeError(
                "ContactMetrics contains stale data: "
                f"contact time={contact_time}, current time={current_time}."
            )

    @staticmethod
    def _extract_contact_positions(
        contacts: Sequence[Mapping[str, Any]],
    ) -> FloatArray:
        if not contacts:
            return np.empty((0, 3), dtype=np.float64)

        positions = []
        for contact in contacts:
            position = contact.get("position")
            if not isinstance(position, Mapping):
                raise TypeError(
                    "Each contact must contain a position mapping."
                )

            positions.append(
                [
                    float(position["x"]),
                    float(position["y"]),
                    float(position["z"]),
                ]
            )

        return np.asarray(positions, dtype=np.float64)

    @staticmethod
    def _build_grasp_matrix(
        contact_positions: FloatArray,
        reference_point: FloatArray,
    ) -> FloatArray:
        """
        Construct a 6 x 3n point-contact grasp matrix in world coordinates.
        """
        contact_count = int(contact_positions.shape[0])
        if contact_count == 0:
            return np.empty((6, 0), dtype=np.float64)

        grasp_matrix = np.zeros(
            (6, 3 * contact_count),
            dtype=np.float64,
        )

        for index, position in enumerate(contact_positions):
            moment_arm = position - reference_point
            block = np.vstack(
                (
                    np.eye(3, dtype=np.float64),
                    GraspMetrics._skew(moment_arm),
                )
            )
            start = 3 * index
            grasp_matrix[:, start : start + 3] = block

        return grasp_matrix

    def _analyse_grasp_matrix(
        self,
        grasp_matrix: FloatArray,
    ) -> MetricDict:
        if grasp_matrix.size == 0:
            return {
                "rank": 0,
                "nullity": 0,
                "singular_values": [],
                "minimum_singular_value": 0.0,
                "maximum_singular_value": 0.0,
                "condition_number": float("inf"),
                "isotropy": 0.0,
                "manipulability": 0.0,
                "full_wrench_rank": False,
            }

        singular_values = np.linalg.svd(
            grasp_matrix,
            compute_uv=False,
        )
        maximum = float(singular_values[0]) if singular_values.size else 0.0
        minimum = float(singular_values[-1]) if singular_values.size else 0.0

        tolerance = (
            self.singular_value_tolerance
            * max(grasp_matrix.shape)
            * maximum
        )
        rank = int(np.sum(singular_values > tolerance))
        nullity = int(grasp_matrix.shape[1] - rank)

        if minimum <= self.epsilon:
            condition_number = float("inf")
            isotropy = 0.0
        else:
            condition_number = maximum / minimum
            isotropy = minimum / maximum if maximum > self.epsilon else 0.0

        # sqrt(det(GG^T)) equals the product of the six singular values when
        # G has full row rank. SVD avoids the numerical instability of a
        # direct determinant.
        if rank == 6 and singular_values.size >= 6:
            manipulability = float(np.prod(singular_values[:6]))
        else:
            manipulability = 0.0

        return {
            "rank": rank,
            "nullity": nullity,
            "singular_values": [
                float(value) for value in singular_values
            ],
            "minimum_singular_value": minimum,
            "maximum_singular_value": maximum,
            "condition_number": float(condition_number),
            "isotropy": float(isotropy),
            "manipulability": manipulability,
            "full_wrench_rank": bool(rank == 6),
        }

    def _calculate_contact_geometry(
        self,
        contacts: Sequence[Mapping[str, Any]],
        contact_positions: FloatArray,
        base_reference: FloatArray,
    ) -> MetricDict:
        if not contacts:
            return {
                "centroid": self._vector_dict(base_reference),
                "centroid_offset": self._vector_dict(
                    np.zeros(3, dtype=np.float64)
                ),
                "centroid_offset_magnitude": 0.0,
                "maximum_pairwise_distance": 0.0,
                "mean_pairwise_distance": 0.0,
                "x_span": 0.0,
                "y_span": 0.0,
                "z_span": 0.0,
                "rung_index_span": 0,
            }

        centroid = np.mean(contact_positions, axis=0)
        centroid_offset = centroid - base_reference
        pairwise_distances = self._pairwise_distances(contact_positions)

        rung_indices = [
            self._parse_rung_index(str(contact["rung"]))
            for contact in contacts
        ]
        valid_rung_indices = [
            index for index in rung_indices if index is not None
        ]
        rung_index_span = (
            max(valid_rung_indices) - min(valid_rung_indices)
            if valid_rung_indices
            else 0
        )

        coordinate_span = (
            np.max(contact_positions, axis=0)
            - np.min(contact_positions, axis=0)
        )

        return {
            "centroid": self._vector_dict(centroid),
            "centroid_offset": self._vector_dict(centroid_offset),
            "centroid_offset_magnitude": float(
                np.linalg.norm(centroid_offset)
            ),
            "maximum_pairwise_distance": (
                float(np.max(pairwise_distances))
                if pairwise_distances.size
                else 0.0
            ),
            "mean_pairwise_distance": (
                float(np.mean(pairwise_distances))
                if pairwise_distances.size
                else 0.0
            ),
            "x_span": float(coordinate_span[0]),
            "y_span": float(coordinate_span[1]),
            "z_span": float(coordinate_span[2]),
            "rung_index_span": int(rung_index_span),
        }

    def _calculate_force_distribution(
        self,
        contacts: Sequence[Mapping[str, Any]],
    ) -> MetricDict:
        if not contacts:
            return {
                "total_normal_force": 0.0,
                "mean_normal_force": 0.0,
                "standard_deviation": 0.0,
                "coefficient_of_variation": 0.0,
                "load_share_entropy": 0.0,
                "normalised_load_share_entropy": 0.0,
                "maximum_load_share": 0.0,
                "minimum_load_share": 0.0,
                "per_contact_load_share": {},
            }

        normal_forces = np.asarray(
            [
                max(float(contact["normal_force"]), 0.0)
                for contact in contacts
            ],
            dtype=np.float64,
        )
        total = float(np.sum(normal_forces))
        mean = float(np.mean(normal_forces))
        standard_deviation = float(np.std(normal_forces))

        coefficient_of_variation = (
            standard_deviation / mean
            if mean > self.epsilon
            else 0.0
        )

        if total > self.epsilon:
            load_shares = normal_forces / total
        else:
            load_shares = np.zeros_like(normal_forces)

        positive_shares = load_shares[load_shares > self.epsilon]
        entropy = (
            float(-np.sum(positive_shares * np.log(positive_shares)))
            if positive_shares.size
            else 0.0
        )
        maximum_entropy = (
            float(np.log(len(contacts))) if len(contacts) > 1 else 0.0
        )
        normalised_entropy = (
            entropy / maximum_entropy
            if maximum_entropy > self.epsilon
            else 1.0 if len(contacts) == 1 else 0.0
        )

        labels = [
            f"{contact['fingertip']}@{contact['rung']}"
            for contact in contacts
        ]

        return {
            "total_normal_force": total,
            "mean_normal_force": mean,
            "standard_deviation": standard_deviation,
            "coefficient_of_variation": float(coefficient_of_variation),
            "load_share_entropy": entropy,
            "normalised_load_share_entropy": float(normalised_entropy),
            "maximum_load_share": float(np.max(load_shares)),
            "minimum_load_share": float(np.min(load_shares)),
            "per_contact_load_share": {
                label: float(share)
                for label, share in zip(labels, load_shares)
            },
        }

    @staticmethod
    def _pairwise_distances(points: FloatArray) -> FloatArray:
        if points.shape[0] < 2:
            return np.empty(0, dtype=np.float64)

        differences = points[:, None, :] - points[None, :, :]
        distance_matrix = np.linalg.norm(differences, axis=2)
        upper_indices = np.triu_indices(points.shape[0], k=1)
        return distance_matrix[upper_indices]

    @staticmethod
    def _parse_rung_index(rung_name: str) -> int | None:
        try:
            return int(rung_name.rsplit("_", maxsplit=1)[1])
        except (IndexError, ValueError):
            return None

    @staticmethod
    def _skew(vector: FloatArray) -> FloatArray:
        x, y, z = vector
        return np.asarray(
            [
                [0.0, -z, y],
                [z, 0.0, -x],
                [-y, x, 0.0],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _vector_dict(vector: FloatArray) -> Dict[str, float]:
        return {
            "x": float(vector[0]),
            "y": float(vector[1]),
            "z": float(vector[2]),
        }
