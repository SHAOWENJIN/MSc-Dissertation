"""
Contact-level metrics for the modified Unitree Go2 ladder-climbing robot.

The module extracts physical contacts between the eight fingertip bodies and
ladder rung geoms. It reports current contact forces, friction utilisation,
relative slip velocity, accumulated slip distance, and contact duration.

The fingertip collision geoms in the current go2.xml are unnamed. Therefore
fingertip identification is performed through each geom's owning body:

    fl_finger_L3, fl_finger_R3
    fr_finger_L3, fr_finger_R3
    rl_finger_L3, rl_finger_R3
    rr_finger_L3, rr_finger_R3

Ladder contacts are identified from geom names beginning with ``rung_``.

Only the current metric state is returned. Small state variables required to
calculate duration and accumulated slip are retained internally, but complete
time-series storage remains the responsibility of DataLogger.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Sequence, Tuple

import mujoco
import numpy as np
from numpy.typing import NDArray

from .base_metric import BaseMetric, MetricDict


FloatArray = NDArray[np.float64]
ContactKey = Tuple[str, str]


@dataclass
class _PersistentContactState:
    """State required across simulation steps for one fingertip-rung pair."""

    duration: float = 0.0
    slip_distance: float = 0.0
    last_seen_time: float | None = None


class ContactMetrics(BaseMetric):
    """Evaluate fingertip-to-rung contact behaviour."""

    DEFAULT_FINGERTIP_BODY_NAMES = (
        "fl_finger_L3",
        "fl_finger_R3",
        "fr_finger_L3",
        "fr_finger_R3",
        "rl_finger_L3",
        "rl_finger_R3",
        "rr_finger_L3",
        "rr_finger_R3",
    )

    def __init__(
        self,
        fingertip_body_names: Sequence[str] = DEFAULT_FINGERTIP_BODY_NAMES,
        *,
        rung_geom_prefix: str = "rung_",
        minimum_normal_force: float = 1e-6,
        slip_speed_threshold: float = 1e-4,
        epsilon: float = 1e-9,
    ) -> None:
        """
        Parameters
        ----------
        fingertip_body_names:
            Names of the eight distal fingertip bodies.

        rung_geom_prefix:
            Prefix used to identify ladder rung collision geoms.

        minimum_normal_force:
            Contacts below this normal-force magnitude are ignored.

        slip_speed_threshold:
            Tangential relative speed above which a contact is flagged as
            slipping.

        epsilon:
            Small positive numerical tolerance.
        """
        super().__init__(name="contact")

        self.fingertip_body_names = self._validate_unique_names(
            fingertip_body_names,
            "fingertip_body_names",
        )

        if not rung_geom_prefix:
            raise ValueError("rung_geom_prefix must be non-empty.")
        if minimum_normal_force < 0.0:
            raise ValueError("minimum_normal_force must be non-negative.")
        if slip_speed_threshold < 0.0:
            raise ValueError("slip_speed_threshold must be non-negative.")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive.")

        self.rung_geom_prefix = rung_geom_prefix
        self.minimum_normal_force = float(minimum_normal_force)
        self.slip_speed_threshold = float(slip_speed_threshold)
        self.epsilon = float(epsilon)

        self._fingertip_body_ids: Dict[str, int] = {}
        self._fingertip_body_id_to_name: Dict[int, str] = {}
        self._persistent_contacts: Dict[ContactKey, _PersistentContactState] = {}
        self._previous_time: float | None = None

    def reset(self, model: Any | None = None, data: Any | None = None) -> None:
        """Clear all active-contact state before a new experimental trial."""
        self._clear_metrics()

        self._fingertip_body_ids = {}
        self._fingertip_body_id_to_name = {}
        self._persistent_contacts = {}
        self._previous_time = None

        if model is None and data is None:
            return
        if model is None or data is None:
            raise ValueError("reset() requires both model and data, or neither.")

        self._resolve_model_ids(model)
        self._previous_time = float(data.time)
        self._is_initialised = True

    def update(self, model: Any, data: Any) -> MetricDict:
        """
        Extract and aggregate fingertip-rung contacts for the current step.

        Multiple MuJoCo contact points can exist for the same fingertip-rung
        pair. These are aggregated into one logical contact record.
        """
        if model is None or data is None:
            raise ValueError("update() requires valid MuJoCo model and data.")

        if not self._fingertip_body_ids:
            self._resolve_model_ids(model)

        current_time = float(data.time)
        dt = self._calculate_dt(current_time)

        raw_records = self._extract_raw_contacts(model, data)
        aggregated = self._aggregate_contacts(raw_records)

        active_keys = set(aggregated)
        self._update_persistent_state(aggregated, active_keys, current_time, dt)

        contacts = []
        for key in sorted(aggregated):
            record = aggregated[key]
            state = self._persistent_contacts[key]

            record["contact_duration"] = float(state.duration)
            record["slip_distance"] = float(state.slip_distance)
            record["is_slipping"] = bool(
                record["tangential_speed"] > self.slip_speed_threshold
            )
            contacts.append(record)

        summary = self._build_summary(contacts)

        metrics: MetricDict = {
            "time": current_time,
            "active_contact_count": len(contacts),
            "active_fingertip_count": len(
                {record["fingertip"] for record in contacts}
            ),
            "active_rung_count": len({record["rung"] for record in contacts}),
            "contacts": contacts,
            "summary": summary,
        }

        self._previous_time = current_time
        return self._set_metrics(metrics)

    def _resolve_model_ids(self, model: Any) -> None:
        """Resolve fingertip body names and validate their presence."""
        body_ids: Dict[str, int] = {}

        for name in self.fingertip_body_names:
            body_id = int(
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            )
            if body_id < 0:
                raise ValueError(
                    f"MuJoCo fingertip body {name!r} was not found."
                )
            body_ids[name] = body_id

        self._fingertip_body_ids = body_ids
        self._fingertip_body_id_to_name = {
            body_id: name for name, body_id in body_ids.items()
        }

    def _extract_raw_contacts(
        self,
        model: Any,
        data: Any,
    ) -> list[MetricDict]:
        """Extract all physical fingertip-rung contact points."""
        records: list[MetricDict] = []

        for contact_index in range(int(data.ncon)):
            contact = data.contact[contact_index]

            geom1_id = int(contact.geom1)
            geom2_id = int(contact.geom2)

            classification = self._classify_contact(
                model,
                geom1_id,
                geom2_id,
            )
            if classification is None:
                continue

            (
                fingertip_name,
                fingertip_body_id,
                rung_name,
                rung_body_id,
                fingertip_is_geom1,
            ) = classification

            contact_force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(
                model,
                data,
                contact_index,
                contact_force,
            )

            normal_force = abs(float(contact_force[0]))
            if normal_force < self.minimum_normal_force:
                continue

            tangential_force = float(np.linalg.norm(contact_force[1:3]))
            friction_ratio = tangential_force / max(
                normal_force,
                self.epsilon,
            )

            contact_position = np.asarray(
                contact.pos,
                dtype=np.float64,
            ).copy()

            frame = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
            contact_normal = frame[0].copy()

            # Orient the reported normal consistently from the rung towards
            # the fingertip, regardless of MuJoCo geom ordering.
            if fingertip_is_geom1:
                contact_normal = -contact_normal

            relative_velocity = self._relative_point_velocity(
                model=model,
                data=data,
                body1_id=fingertip_body_id,
                body2_id=rung_body_id,
                point=contact_position,
            )

            normal_speed_signed = float(
                np.dot(relative_velocity, contact_normal)
            )
            tangential_velocity = (
                relative_velocity
                - normal_speed_signed * contact_normal
            )
            tangential_speed = float(np.linalg.norm(tangential_velocity))

            records.append(
                {
                    "contact_index": contact_index,
                    "fingertip": fingertip_name,
                    "rung": rung_name,
                    "position": self._vector_dict(contact_position),
                    "normal": self._vector_dict(contact_normal),
                    "normal_force": normal_force,
                    "tangential_force": tangential_force,
                    "friction_ratio": float(friction_ratio),
                    "normal_speed": normal_speed_signed,
                    "tangential_speed": tangential_speed,
                    "relative_velocity": self._vector_dict(
                        relative_velocity
                    ),
                    "tangential_velocity": self._vector_dict(
                        tangential_velocity
                    ),
                }
            )

        return records

    def _classify_contact(
        self,
        model: Any,
        geom1_id: int,
        geom2_id: int,
    ) -> tuple[str, int, str, int, bool] | None:
        """
        Return fingertip/rung metadata when a geom pair is relevant.

        The final boolean indicates whether the fingertip is geom1.
        """
        body1_id = int(model.geom_bodyid[geom1_id])
        body2_id = int(model.geom_bodyid[geom2_id])

        tip1 = self._fingertip_body_id_to_name.get(body1_id)
        tip2 = self._fingertip_body_id_to_name.get(body2_id)

        geom1_name = self._id_to_name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            geom1_id,
        )
        geom2_name = self._id_to_name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            geom2_id,
        )

        geom1_is_rung = bool(
            geom1_name and geom1_name.startswith(self.rung_geom_prefix)
        )
        geom2_is_rung = bool(
            geom2_name and geom2_name.startswith(self.rung_geom_prefix)
        )

        if tip1 is not None and geom2_is_rung:
            return tip1, body1_id, geom2_name, body2_id, True

        if tip2 is not None and geom1_is_rung:
            return tip2, body2_id, geom1_name, body1_id, False

        return None

    def _aggregate_contacts(
        self,
        raw_records: list[MetricDict],
    ) -> Dict[ContactKey, MetricDict]:
        """
        Aggregate multiple physical contact points per fingertip-rung pair.

        Force quantities are summed. Position, normal, and velocity quantities
        are normal-force-weighted averages.
        """
        grouped: Dict[ContactKey, list[MetricDict]] = defaultdict(list)

        for record in raw_records:
            key = (record["fingertip"], record["rung"])
            grouped[key].append(record)

        aggregated: Dict[ContactKey, MetricDict] = {}

        for key, records in grouped.items():
            normal_forces = np.asarray(
                [record["normal_force"] for record in records],
                dtype=np.float64,
            )
            weights = normal_forces / max(
                float(np.sum(normal_forces)),
                self.epsilon,
            )

            position = self._weighted_vector(
                records,
                "position",
                weights,
            )
            normal = self._weighted_vector(
                records,
                "normal",
                weights,
            )
            normal_norm = float(np.linalg.norm(normal))
            if normal_norm > self.epsilon:
                normal /= normal_norm

            relative_velocity = self._weighted_vector(
                records,
                "relative_velocity",
                weights,
            )
            tangential_velocity = self._weighted_vector(
                records,
                "tangential_velocity",
                weights,
            )

            normal_force = float(
                sum(record["normal_force"] for record in records)
            )
            tangential_force = float(
                sum(record["tangential_force"] for record in records)
            )
            friction_ratio = tangential_force / max(
                normal_force,
                self.epsilon,
            )

            aggregated[key] = {
                "fingertip": key[0],
                "rung": key[1],
                "physical_contact_count": len(records),
                "position": self._vector_dict(position),
                "normal": self._vector_dict(normal),
                "normal_force": normal_force,
                "tangential_force": tangential_force,
                "friction_ratio": float(friction_ratio),
                "normal_speed": float(
                    np.dot(relative_velocity, normal)
                ),
                "tangential_speed": float(
                    np.linalg.norm(tangential_velocity)
                ),
                "relative_velocity": self._vector_dict(
                    relative_velocity
                ),
                "tangential_velocity": self._vector_dict(
                    tangential_velocity
                ),
            }

        return aggregated

    def _update_persistent_state(
        self,
        aggregated: Dict[ContactKey, MetricDict],
        active_keys: set[ContactKey],
        current_time: float,
        dt: float,
    ) -> None:
        """Update duration/slip state and discard contacts that ended."""
        ended_keys = set(self._persistent_contacts) - active_keys
        for key in ended_keys:
            del self._persistent_contacts[key]

        for key, record in aggregated.items():
            state = self._persistent_contacts.setdefault(
                key,
                _PersistentContactState(),
            )

            if state.last_seen_time is None:
                state.duration = 0.0
            elif dt > 0.0:
                state.duration += dt

            if dt > 0.0:
                state.slip_distance += (
                    float(record["tangential_speed"]) * dt
                )

            state.last_seen_time = current_time

    def _calculate_dt(self, current_time: float) -> float:
        if self._previous_time is None:
            return 0.0

        dt = current_time - self._previous_time
        if dt <= self.epsilon:
            return 0.0

        return float(dt)

    @staticmethod
    def _relative_point_velocity(
        model: Any,
        data: Any,
        body1_id: int,
        body2_id: int,
        point: FloatArray,
    ) -> FloatArray:
        """
        Return velocity of body1's contact point relative to body2's point.

        Body spatial velocities are requested in world coordinates and then
        shifted from each body's origin to the world-space contact point.
        """
        velocity1 = ContactMetrics._body_point_velocity(
            model,
            data,
            body1_id,
            point,
        )
        velocity2 = ContactMetrics._body_point_velocity(
            model,
            data,
            body2_id,
            point,
        )
        return velocity1 - velocity2

    @staticmethod
    def _body_point_velocity(
        model: Any,
        data: Any,
        body_id: int,
        point: FloatArray,
    ) -> FloatArray:
        spatial_velocity = np.zeros(6, dtype=np.float64)

        mujoco.mj_objectVelocity(
            model,
            data,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
            spatial_velocity,
            0,
        )

        angular_velocity = spatial_velocity[0:3]
        linear_velocity = spatial_velocity[3:6]
        body_position = np.asarray(
            data.xpos[body_id],
            dtype=np.float64,
        )

        return linear_velocity + np.cross(
            angular_velocity,
            point - body_position,
        )

    def _build_summary(self, contacts: list[MetricDict]) -> MetricDict:
        if not contacts:
            return {
                "total_normal_force": 0.0,
                "total_tangential_force": 0.0,
                "maximum_normal_force": 0.0,
                "maximum_tangential_force": 0.0,
                "maximum_friction_ratio": 0.0,
                "mean_friction_ratio": 0.0,
                "maximum_tangential_speed": 0.0,
                "mean_tangential_speed": 0.0,
                "slipping_contact_count": 0,
                "maximum_contact_duration": 0.0,
                "maximum_slip_distance": 0.0,
            }

        normal_forces = np.asarray(
            [contact["normal_force"] for contact in contacts],
            dtype=np.float64,
        )
        tangential_forces = np.asarray(
            [contact["tangential_force"] for contact in contacts],
            dtype=np.float64,
        )
        friction_ratios = np.asarray(
            [contact["friction_ratio"] for contact in contacts],
            dtype=np.float64,
        )
        tangential_speeds = np.asarray(
            [contact["tangential_speed"] for contact in contacts],
            dtype=np.float64,
        )
        durations = np.asarray(
            [contact["contact_duration"] for contact in contacts],
            dtype=np.float64,
        )
        slip_distances = np.asarray(
            [contact["slip_distance"] for contact in contacts],
            dtype=np.float64,
        )

        return {
            "total_normal_force": float(np.sum(normal_forces)),
            "total_tangential_force": float(np.sum(tangential_forces)),
            "maximum_normal_force": float(np.max(normal_forces)),
            "maximum_tangential_force": float(np.max(tangential_forces)),
            "maximum_friction_ratio": float(np.max(friction_ratios)),
            "mean_friction_ratio": float(np.mean(friction_ratios)),
            "maximum_tangential_speed": float(np.max(tangential_speeds)),
            "mean_tangential_speed": float(np.mean(tangential_speeds)),
            "slipping_contact_count": sum(
                int(contact["is_slipping"]) for contact in contacts
            ),
            "maximum_contact_duration": float(np.max(durations)),
            "maximum_slip_distance": float(np.max(slip_distances)),
        }

    @staticmethod
    def _id_to_name(
        model: Any,
        object_type: mujoco.mjtObj,
        object_id: int,
    ) -> str | None:
        name = mujoco.mj_id2name(
            model,
            object_type,
            object_id,
        )
        return str(name) if name is not None else None

    @staticmethod
    def _weighted_vector(
        records: list[MetricDict],
        field: str,
        weights: FloatArray,
    ) -> FloatArray:
        vectors = np.asarray(
            [
                [
                    record[field]["x"],
                    record[field]["y"],
                    record[field]["z"],
                ]
                for record in records
            ],
            dtype=np.float64,
        )
        return np.sum(vectors * weights[:, None], axis=0)

    @staticmethod
    def _vector_dict(vector: FloatArray) -> Dict[str, float]:
        return {
            "x": float(vector[0]),
            "y": float(vector[1]),
            "z": float(vector[2]),
        }

    @staticmethod
    def _validate_unique_names(
        names: Iterable[str],
        argument_name: str,
    ) -> tuple[str, ...]:
        cleaned = tuple(str(name).strip() for name in names)

        if not cleaned or any(not name for name in cleaned):
            raise ValueError(
                f"{argument_name} must contain non-empty names."
            )
        if len(set(cleaned)) != len(cleaned):
            raise ValueError(f"{argument_name} contains duplicates.")

        return cleaned
