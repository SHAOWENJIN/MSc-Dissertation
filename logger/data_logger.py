"""
Data logging utilities for the quantitative stability evaluation framework.

DataLogger stores:
1. a compact scalar time series suitable for CSV analysis; and
2. optional complete nested metric snapshots suitable for JSON inspection.

The logger performs no metric calculations. It only validates, flattens,
stores, and exports data produced by StabilityEvaluator.
"""

from __future__ import annotations

import csv
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


Scalar = int | float | bool | str | None
FlatRecord = Dict[str, Scalar]


class DataLogger:
    """
    Store and export evaluation results for one or more simulation trials.

    Parameters
    ----------
    output_directory:
        Default directory used by export methods.

    store_full_metrics:
        If True, retain a deep copy of every complete nested evaluator output.
        This is useful for debugging but can consume substantial memory.

    scalar_fields:
        Optional explicit dotted paths to store in the compact time series.
        When omitted, ``StabilityEvaluator.get_summary()`` should be supplied
        to ``log_step`` and all scalar fields in it are recorded.

    float_precision:
        Number of decimal places used when writing finite floating-point values
        to CSV. Internal stored values are not rounded.

    strict_time_order:
        If True, reject samples whose time is earlier than the preceding sample.
    """

    def __init__(
        self,
        output_directory: str | Path = "results",
        *,
        store_full_metrics: bool = False,
        scalar_fields: Sequence[str] | None = None,
        float_precision: int = 10,
        strict_time_order: bool = True,
    ) -> None:
        if float_precision < 0:
            raise ValueError("float_precision must be non-negative.")

        self.output_directory = Path(output_directory)
        self.store_full_metrics = bool(store_full_metrics)
        self.scalar_fields = (
            tuple(str(field).strip() for field in scalar_fields)
            if scalar_fields is not None
            else None
        )
        self.float_precision = int(float_precision)
        self.strict_time_order = bool(strict_time_order)

        if self.scalar_fields is not None:
            if not self.scalar_fields or any(not field for field in self.scalar_fields):
                raise ValueError("scalar_fields must contain non-empty paths.")
            if len(set(self.scalar_fields)) != len(self.scalar_fields):
                raise ValueError("scalar_fields contains duplicates.")

        self._trial_name: str | None = None
        self._trial_metadata: Dict[str, Any] = {}
        self._scalar_records: list[FlatRecord] = []
        self._full_records: list[Dict[str, Any]] = []
        self._last_time: float | None = None

    @property
    def trial_name(self) -> str | None:
        return self._trial_name

    @property
    def sample_count(self) -> int:
        return len(self._scalar_records)

    @property
    def is_empty(self) -> bool:
        return not self._scalar_records

    def reset(
        self,
        trial_name: str | None = None,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Clear all records and begin a new trial."""
        if trial_name is not None and not trial_name.strip():
            raise ValueError("trial_name must be non-empty or None.")

        self._trial_name = trial_name.strip() if trial_name else None
        self._trial_metadata = deepcopy(dict(metadata or {}))
        self._scalar_records = []
        self._full_records = []
        self._last_time = None

    def log_step(
        self,
        *,
        summary: Mapping[str, Any],
        full_metrics: Mapping[str, Any] | None = None,
        controller_name: str | None = None,
        trial_id: str | int | None = None,
    ) -> FlatRecord:
        """
        Store one simulation-step sample.

        ``summary`` should normally be the result of
        ``StabilityEvaluator.get_summary()``. If ``scalar_fields`` was supplied,
        dotted paths are extracted from ``full_metrics`` when available, or
        from ``summary`` otherwise.
        """
        if not isinstance(summary, Mapping):
            raise TypeError("summary must be a mapping.")
        if full_metrics is not None and not isinstance(full_metrics, Mapping):
            raise TypeError("full_metrics must be a mapping or None.")

        source: Mapping[str, Any]
        if self.scalar_fields is None:
            record = self._flatten_scalars(summary)
        else:
            source = full_metrics if full_metrics is not None else summary
            record = {
                field: self._normalise_scalar(
                    self._get_dotted_value(source, field)
                )
                for field in self.scalar_fields
            }

        if "time" not in record:
            # Prefer summary time, then evaluator metadata time.
            if "time" in summary:
                record["time"] = self._normalise_scalar(summary["time"])
            elif full_metrics is not None:
                metadata = full_metrics.get("metadata", {})
                if isinstance(metadata, Mapping) and "time" in metadata:
                    record["time"] = self._normalise_scalar(metadata["time"])

        if "time" not in record:
            raise KeyError("Each logged sample must contain a 'time' value.")

        current_time = float(record["time"])
        self._validate_time(current_time)

        if controller_name is not None:
            record["controller_name"] = str(controller_name)
        if trial_id is not None:
            record["trial_id"] = str(trial_id)
        if self._trial_name is not None:
            record["trial_name"] = self._trial_name

        self._scalar_records.append(dict(record))

        if self.store_full_metrics:
            if full_metrics is None:
                raise ValueError(
                    "store_full_metrics=True requires full_metrics in log_step()."
                )
            self._full_records.append(deepcopy(dict(full_metrics)))

        self._last_time = current_time
        return dict(record)

    def log_evaluator(
        self,
        evaluator: Any,
        *,
        controller_name: str | None = None,
        trial_id: str | int | None = None,
    ) -> FlatRecord:
        """
        Convenience wrapper around a StabilityEvaluator-like object.

        The object must implement ``get_summary()`` and ``get_metrics()``.
        """
        if not hasattr(evaluator, "get_summary") or not hasattr(
            evaluator, "get_metrics"
        ):
            raise TypeError(
                "evaluator must provide get_summary() and get_metrics()."
            )

        summary = evaluator.get_summary()
        full_metrics = evaluator.get_metrics()

        return self.log_step(
            summary=summary,
            full_metrics=full_metrics,
            controller_name=controller_name,
            trial_id=trial_id,
        )

    def get_scalar_records(self) -> list[FlatRecord]:
        """Return a copy of the compact scalar time series."""
        return deepcopy(self._scalar_records)

    def get_full_records(self) -> list[Dict[str, Any]]:
        """Return a copy of complete nested metric snapshots."""
        if not self.store_full_metrics:
            raise RuntimeError(
                "Full metrics were not enabled for this DataLogger."
            )
        return deepcopy(self._full_records)

    def latest(self) -> FlatRecord:
        """Return the latest compact record."""
        if not self._scalar_records:
            raise RuntimeError("No logged samples are available.")
        return dict(self._scalar_records[-1])

    def export_csv(
        self,
        filename: str | Path | None = None,
    ) -> Path:
        """Write compact scalar records to CSV."""
        if not self._scalar_records:
            raise RuntimeError("No scalar records are available to export.")

        path = self._resolve_output_path(
            filename,
            default_suffix="_metrics.csv",
        )
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = self._ordered_fieldnames(self._scalar_records)

        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
                extrasaction="ignore",
            )
            writer.writeheader()

            for record in self._scalar_records:
                writer.writerow(
                    {
                        field: self._format_csv_value(record.get(field))
                        for field in fieldnames
                    }
                )

        return path

    def export_full_json(
        self,
        filename: str | Path | None = None,
        *,
        indent: int = 2,
    ) -> Path:
        """Write complete nested snapshots and trial metadata to JSON."""
        if not self.store_full_metrics:
            raise RuntimeError(
                "Full metrics were not enabled for this DataLogger."
            )
        if not self._full_records:
            raise RuntimeError("No full metric records are available to export.")

        path = self._resolve_output_path(
            filename,
            default_suffix="_full_metrics.json",
        )
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "trial_name": self._trial_name,
            "metadata": deepcopy(self._trial_metadata),
            "sample_count": len(self._full_records),
            "records": self._full_records,
        }

        with path.open("w", encoding="utf-8") as file:
            json.dump(
                payload,
                file,
                indent=indent,
                ensure_ascii=False,
                allow_nan=True,
                default=self._json_default,
            )

        return path

    def export_metadata_json(
        self,
        filename: str | Path | None = None,
        *,
        extra_metadata: Mapping[str, Any] | None = None,
        indent: int = 2,
    ) -> Path:
        """Write trial configuration and logging metadata to JSON."""
        path = self._resolve_output_path(
            filename,
            default_suffix="_metadata.json",
        )
        path.parent.mkdir(parents=True, exist_ok=True)

        metadata = deepcopy(self._trial_metadata)
        if extra_metadata:
            metadata.update(deepcopy(dict(extra_metadata)))

        payload = {
            "trial_name": self._trial_name,
            "sample_count": self.sample_count,
            "start_time": (
                self._scalar_records[0].get("time")
                if self._scalar_records
                else None
            ),
            "end_time": (
                self._scalar_records[-1].get("time")
                if self._scalar_records
                else None
            ),
            "store_full_metrics": self.store_full_metrics,
            "scalar_fields": list(self.scalar_fields)
            if self.scalar_fields is not None
            else None,
            "metadata": metadata,
        }

        with path.open("w", encoding="utf-8") as file:
            json.dump(
                payload,
                file,
                indent=indent,
                ensure_ascii=False,
                allow_nan=True,
                default=self._json_default,
            )

        return path

    def export_all(
        self,
        *,
        basename: str | None = None,
        include_metadata: bool = True,
    ) -> Dict[str, Path]:
        """Export all enabled formats and return their paths."""
        stem = basename or self._trial_name or "trial"

        outputs = {
            "csv": self.export_csv(f"{stem}_metrics.csv"),
        }

        if self.store_full_metrics:
            outputs["full_json"] = self.export_full_json(
                f"{stem}_full_metrics.json"
            )

        if include_metadata:
            outputs["metadata_json"] = self.export_metadata_json(
                f"{stem}_metadata.json"
            )

        return outputs

    def _validate_time(self, current_time: float) -> None:
        if not math.isfinite(current_time):
            raise ValueError("Logged time must be finite.")

        if (
            self.strict_time_order
            and self._last_time is not None
            and current_time < self._last_time
        ):
            raise RuntimeError(
                "Logged time moved backwards. Call reset() before a new trial."
            )

    def _resolve_output_path(
        self,
        filename: str | Path | None,
        *,
        default_suffix: str,
    ) -> Path:
        if filename is None:
            stem = self._trial_name or "trial"
            filename = f"{stem}{default_suffix}"

        path = Path(filename)
        if not path.is_absolute():
            path = self.output_directory / path

        return path

    @classmethod
    def _flatten_scalars(
        cls,
        values: Mapping[str, Any],
        *,
        prefix: str = "",
    ) -> FlatRecord:
        """Flatten nested mappings using dotted field names."""
        flattened: FlatRecord = {}

        for key, value in values.items():
            field = f"{prefix}.{key}" if prefix else str(key)

            if isinstance(value, Mapping):
                flattened.update(
                    cls._flatten_scalars(value, prefix=field)
                )
            elif cls._is_scalar(value):
                flattened[field] = cls._normalise_scalar(value)
            # Lists, arrays, and matrices are intentionally excluded from the
            # compact CSV stream.

        return flattened

    @staticmethod
    def _get_dotted_value(
        values: Mapping[str, Any],
        dotted_path: str,
    ) -> Any:
        current: Any = values

        for part in dotted_path.split("."):
            if not isinstance(current, Mapping) or part not in current:
                raise KeyError(
                    f"Metric field {dotted_path!r} was not found."
                )
            current = current[part]

        if not DataLogger._is_scalar(current):
            raise TypeError(
                f"Metric field {dotted_path!r} is not scalar."
            )

        return current

    @staticmethod
    def _is_scalar(value: Any) -> bool:
        return (
            value is None
            or isinstance(value, (str, bool, int, float))
            or hasattr(value, "item")
        )

    @staticmethod
    def _normalise_scalar(value: Any) -> Scalar:
        if value is None or isinstance(value, (str, bool, int, float)):
            return value

        if hasattr(value, "item"):
            converted = value.item()
            if isinstance(converted, (str, bool, int, float)):
                return converted

        raise TypeError(f"Unsupported scalar type: {type(value).__name__}.")

    @staticmethod
    def _ordered_fieldnames(records: Iterable[FlatRecord]) -> list[str]:
        fields: list[str] = []
        seen: set[str] = set()

        preferred = (
            "trial_name",
            "trial_id",
            "controller_name",
            "time",
            "step_count",
        )

        all_fields = {
            field for record in records for field in record
        }

        for field in preferred:
            if field in all_fields:
                fields.append(field)
                seen.add(field)

        for field in sorted(all_fields):
            if field not in seen:
                fields.append(field)

        return fields

    def _format_csv_value(self, value: Scalar) -> Scalar:
        if value is None:
            return ""

        if isinstance(value, float):
            if math.isnan(value):
                return "nan"
            if math.isinf(value):
                return "inf" if value > 0 else "-inf"
            return f"{value:.{self.float_precision}f}"

        return value

    @staticmethod
    def _json_default(value: Any) -> Any:
        if hasattr(value, "tolist"):
            return value.tolist()
        if hasattr(value, "item"):
            return value.item()
        if isinstance(value, Path):
            return str(value)

        raise TypeError(
            f"Object of type {type(value).__name__} is not JSON serialisable."
        )

    def __len__(self) -> int:
        return self.sample_count

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"trial_name={self._trial_name!r}, "
            f"sample_count={self.sample_count}, "
            f"store_full_metrics={self.store_full_metrics}"
            f")"
        )
