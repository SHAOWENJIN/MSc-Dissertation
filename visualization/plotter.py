"""
Plotting utilities for the quantitative stability evaluation framework.

The plotter reads compact CSV files exported by DataLogger and creates
publication-ready figures without performing new physical metric calculations.

Supported workflows
-------------------
- single-trial time-series plots
- multi-controller time-series comparison
- trial-level scalar comparison
- success-rate and completion-time summaries
- automatic export to PNG, PDF, or SVG

The implementation uses only pandas and matplotlib.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class EvaluationPlotter:
    """
    Load evaluation CSV files and generate comparison figures.

    Parameters
    ----------
    output_directory:
        Default destination for exported figures.

    dpi:
        Raster export resolution.

    figure_size:
        Default figure size in inches.

    font_size:
        Base matplotlib font size.

    strict_columns:
        If True, missing requested columns raise errors. If False, plotting
        methods skip unavailable optional columns where possible.
    """

    DEFAULT_TIME_COLUMN = "time"
    DEFAULT_CONTROLLER_COLUMN = "controller_name"
    DEFAULT_TRIAL_COLUMN = "trial_id"

    def __init__(
        self,
        output_directory: str | Path = "figures",
        *,
        dpi: int = 300,
        figure_size: tuple[float, float] = (8.0, 5.0),
        font_size: float = 11.0,
        strict_columns: bool = True,
    ) -> None:
        if dpi <= 0:
            raise ValueError("dpi must be positive.")
        if len(figure_size) != 2 or any(value <= 0 for value in figure_size):
            raise ValueError("figure_size must contain two positive values.")
        if font_size <= 0:
            raise ValueError("font_size must be positive.")

        self.output_directory = Path(output_directory)
        self.dpi = int(dpi)
        self.figure_size = tuple(float(value) for value in figure_size)
        self.font_size = float(font_size)
        self.strict_columns = bool(strict_columns)

    def load_csv(
        self,
        paths: str | Path | Sequence[str | Path],
    ) -> pd.DataFrame:
        """
        Load one or more DataLogger CSV files into a single dataframe.

        A ``source_file`` column is added to preserve provenance. Existing
        trial/controller columns are retained.
        """
        path_list = self._normalise_paths(paths)
        frames: list[pd.DataFrame] = []

        for path in path_list:
            if not path.exists():
                raise FileNotFoundError(path)

            frame = pd.read_csv(path)
            frame["source_file"] = path.name
            frames.append(frame)

        if not frames:
            raise ValueError("No CSV files were provided.")

        data = pd.concat(frames, ignore_index=True, sort=False)
        return self._coerce_known_types(data)

    def plot_time_series(
        self,
        data: pd.DataFrame,
        metric: str,
        *,
        time_column: str = DEFAULT_TIME_COLUMN,
        group_columns: Sequence[str] | None = None,
        title: str | None = None,
        ylabel: str | None = None,
        xlabel: str = "Time (s)",
        show_legend: bool = True,
        save_as: str | Path | None = None,
    ) -> plt.Figure:
        """
        Plot one metric against time.

        When group_columns are supplied, one line is drawn for each unique
        group. Typical grouping is ``("controller_name", "trial_id")``.
        """
        self._require_columns(data, [time_column, metric])

        figure, axis = plt.subplots(figsize=self.figure_size)

        valid_data = data[[time_column, metric] + list(group_columns or [])].copy()
        valid_data = valid_data.dropna(subset=[time_column, metric])

        if group_columns:
            self._require_columns(valid_data, group_columns)

            group_key: str | list[str]
            group_key = (
                group_columns[0]
                if len(group_columns) == 1
                else list(group_columns)
            )

            for group_value, group in valid_data.groupby(
                group_key,
                dropna=False,
                sort=True,
            ):
                ordered = group.sort_values(time_column)
                label = self._format_group_label(
                    group_columns,
                    group_value,
                )
                axis.plot(
                    ordered[time_column],
                    ordered[metric],
                    label=label,
                )
        else:
            ordered = valid_data.sort_values(time_column)
            axis.plot(ordered[time_column], ordered[metric])

        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel or self._humanise(metric))
        axis.set_title(title or self._humanise(metric))
        axis.grid(True, alpha=0.3)

        if show_legend and group_columns:
            axis.legend()

        figure.tight_layout()
        self._save_figure(figure, save_as)
        return figure

    def plot_controller_mean_time_series(
        self,
        data: pd.DataFrame,
        metric: str,
        *,
        controller_column: str = DEFAULT_CONTROLLER_COLUMN,
        trial_column: str = DEFAULT_TRIAL_COLUMN,
        time_column: str = DEFAULT_TIME_COLUMN,
        time_bin_width: float | None = None,
        uncertainty: str = "std",
        title: str | None = None,
        ylabel: str | None = None,
        save_as: str | Path | None = None,
    ) -> plt.Figure:
        """
        Plot controller mean trajectories with optional uncertainty bands.

        Parameters
        ----------
        time_bin_width:
            Optional width for binning asynchronous trial timestamps. If None,
            exact recorded timestamps are grouped.

        uncertainty:
            ``"std"``, ``"sem"``, or ``"none"``.
        """
        self._require_columns(
            data,
            [controller_column, trial_column, time_column, metric],
        )

        if time_bin_width is not None and time_bin_width <= 0.0:
            raise ValueError("time_bin_width must be positive.")
        if uncertainty not in {"std", "sem", "none"}:
            raise ValueError("uncertainty must be 'std', 'sem', or 'none'.")

        working = data[
            [controller_column, trial_column, time_column, metric]
        ].dropna(subset=[controller_column, time_column, metric]).copy()

        if time_bin_width is None:
            working["_plot_time"] = working[time_column].astype(float)
        else:
            working["_plot_time"] = (
                np.round(
                    working[time_column].astype(float) / time_bin_width
                )
                * time_bin_width
            )

        # Average repeated samples within the same trial/time bin first.
        per_trial = (
            working.groupby(
                [controller_column, trial_column, "_plot_time"],
                as_index=False,
                dropna=False,
            )[metric]
            .mean()
        )

        figure, axis = plt.subplots(figsize=self.figure_size)

        for controller, group in per_trial.groupby(
            controller_column,
            dropna=False,
            sort=True,
        ):
            summary = (
                group.groupby("_plot_time")[metric]
                .agg(["mean", "std", "count"])
                .reset_index()
                .sort_values("_plot_time")
            )

            axis.plot(
                summary["_plot_time"],
                summary["mean"],
                label=str(controller),
            )

            if uncertainty == "none":
                continue

            spread = summary["std"].fillna(0.0)
            if uncertainty == "sem":
                spread = spread / np.sqrt(
                    summary["count"].clip(lower=1)
                )

            axis.fill_between(
                summary["_plot_time"].to_numpy(dtype=float),
                (summary["mean"] - spread).to_numpy(dtype=float),
                (summary["mean"] + spread).to_numpy(dtype=float),
                alpha=0.2,
            )

        axis.set_xlabel("Time (s)")
        axis.set_ylabel(ylabel or self._humanise(metric))
        axis.set_title(
            title
            or f"{self._humanise(metric)} by controller"
        )
        axis.grid(True, alpha=0.3)
        axis.legend()
        figure.tight_layout()

        self._save_figure(figure, save_as)
        return figure

    def plot_trial_metric_comparison(
        self,
        data: pd.DataFrame,
        metric: str,
        *,
        aggregation: str = "last",
        controller_column: str = DEFAULT_CONTROLLER_COLUMN,
        trial_column: str = DEFAULT_TRIAL_COLUMN,
        time_column: str = DEFAULT_TIME_COLUMN,
        kind: str = "box",
        title: str | None = None,
        ylabel: str | None = None,
        save_as: str | Path | None = None,
    ) -> plt.Figure:
        """
        Compare one trial-level metric across controllers.

        ``aggregation`` determines how each trial is reduced:
        ``last``, ``mean``, ``max``, ``min``, ``sum``, or ``peak_abs``.
        """
        self._require_columns(
            data,
            [controller_column, trial_column, time_column, metric],
        )

        trial_values = self.aggregate_trials(
            data,
            metric,
            aggregation=aggregation,
            controller_column=controller_column,
            trial_column=trial_column,
            time_column=time_column,
        )

        controllers = list(
            trial_values[controller_column].dropna().unique()
        )
        if not controllers:
            raise ValueError("No controller groups are available.")

        grouped_values = [
            trial_values.loc[
                trial_values[controller_column] == controller,
                metric,
            ].dropna().to_numpy(dtype=float)
            for controller in controllers
        ]

        figure, axis = plt.subplots(figsize=self.figure_size)

        if kind == "box":
            axis.boxplot(grouped_values, tick_labels=[str(c) for c in controllers])
        elif kind == "bar":
            means = [
                float(np.mean(values)) if values.size else math.nan
                for values in grouped_values
            ]
            errors = [
                float(np.std(values, ddof=1)) if values.size > 1 else 0.0
                for values in grouped_values
            ]
            positions = np.arange(len(controllers))
            axis.bar(positions, means, yerr=errors, capsize=4)
            axis.set_xticks(positions)
            axis.set_xticklabels([str(c) for c in controllers])
        elif kind == "scatter":
            for index, values in enumerate(grouped_values):
                x_values = np.full(values.shape, index, dtype=float)
                axis.scatter(x_values, values)
            axis.set_xticks(np.arange(len(controllers)))
            axis.set_xticklabels([str(c) for c in controllers])
        else:
            raise ValueError("kind must be 'box', 'bar', or 'scatter'.")

        axis.set_xlabel("Controller")
        axis.set_ylabel(ylabel or self._humanise(metric))
        axis.set_title(
            title
            or f"{self._humanise(metric)} ({aggregation} per trial)"
        )
        axis.grid(True, axis="y", alpha=0.3)
        figure.tight_layout()

        self._save_figure(figure, save_as)
        return figure

    def plot_success_rate(
        self,
        data: pd.DataFrame,
        *,
        success_column: str = "task_success",
        controller_column: str = DEFAULT_CONTROLLER_COLUMN,
        trial_column: str = DEFAULT_TRIAL_COLUMN,
        time_column: str = DEFAULT_TIME_COLUMN,
        title: str = "Task success rate",
        save_as: str | Path | None = None,
    ) -> plt.Figure:
        """Plot final task-success rate for each controller."""
        self._require_columns(
            data,
            [
                success_column,
                controller_column,
                trial_column,
                time_column,
            ],
        )

        final_samples = self._last_trial_samples(
            data,
            controller_column=controller_column,
            trial_column=trial_column,
            time_column=time_column,
        )
        final_samples[success_column] = self._coerce_boolean_series(
            final_samples[success_column]
        )

        success = (
            final_samples.groupby(controller_column)[success_column]
            .agg(["mean", "count"])
            .reset_index()
        )

        figure, axis = plt.subplots(figsize=self.figure_size)
        positions = np.arange(len(success))
        axis.bar(positions, success["mean"].to_numpy(dtype=float))
        axis.set_xticks(positions)
        axis.set_xticklabels(success[controller_column].astype(str))
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel("Controller")
        axis.set_ylabel("Success rate")
        axis.set_title(title)
        axis.grid(True, axis="y", alpha=0.3)

        for position, row in zip(positions, success.itertuples()):
            axis.text(
                position,
                float(row.mean),
                f"{float(row.mean):.2f}\n(n={int(row.count)})",
                ha="center",
                va="bottom",
            )

        figure.tight_layout()
        self._save_figure(figure, save_as)
        return figure

    def plot_completion_time(
        self,
        data: pd.DataFrame,
        *,
        completion_column: str = "completion_time",
        success_column: str = "task_success",
        controller_column: str = DEFAULT_CONTROLLER_COLUMN,
        trial_column: str = DEFAULT_TRIAL_COLUMN,
        time_column: str = DEFAULT_TIME_COLUMN,
        kind: str = "box",
        title: str = "Completion time for successful trials",
        save_as: str | Path | None = None,
    ) -> plt.Figure:
        """Compare completion times using successful trials only."""
        self._require_columns(
            data,
            [
                completion_column,
                success_column,
                controller_column,
                trial_column,
                time_column,
            ],
        )

        final_samples = self._last_trial_samples(
            data,
            controller_column=controller_column,
            trial_column=trial_column,
            time_column=time_column,
        )
        success_mask = self._coerce_boolean_series(
            final_samples[success_column]
        )
        successful = final_samples.loc[success_mask].copy()

        if successful.empty:
            raise ValueError("No successful trials contain completion times.")

        return self.plot_trial_metric_comparison(
            successful,
            completion_column,
            aggregation="last",
            controller_column=controller_column,
            trial_column=trial_column,
            time_column=time_column,
            kind=kind,
            title=title,
            ylabel="Completion time (s)",
            save_as=save_as,
        )

    def plot_standard_report(
        self,
        data: pd.DataFrame,
        *,
        prefix: str = "evaluation",
        controller_column: str = DEFAULT_CONTROLLER_COLUMN,
        trial_column: str = DEFAULT_TRIAL_COLUMN,
        time_column: str = DEFAULT_TIME_COLUMN,
        close_figures: bool = True,
    ) -> dict[str, Path]:
        """
        Generate a standard set of dissertation comparison figures.

        Missing optional metric columns are skipped when ``strict_columns`` is
        False. Returned paths correspond only to successfully created figures.
        """
        specifications = [
            (
                "forward_progress",
                "Forward progress",
                "Forward progress (m)",
                "mean_time",
            ),
            (
                "lateral_drift_abs",
                "Absolute lateral drift",
                "Lateral drift (m)",
                "mean_time",
            ),
            (
                "base_acceleration_magnitude",
                "Base acceleration magnitude",
                "Acceleration (m/s²)",
                "mean_time",
            ),
            (
                "maximum_friction_ratio",
                "Maximum friction ratio",
                "Tangential / normal force",
                "mean_time",
            ),
            (
                "minimum_grasp_singular_value",
                "Minimum grasp singular value",
                "Minimum singular value",
                "mean_time",
            ),
            (
                "grasp_isotropy",
                "Grasp isotropy",
                "Isotropy",
                "mean_time",
            ),
            (
                "absolute_mechanical_work",
                "Absolute mechanical work",
                "Mechanical work (J)",
                "trial_last",
            ),
            (
                "successful_regrasp_count",
                "Successful regrasp count",
                "Count",
                "trial_last",
            ),
        ]

        created: dict[str, Path] = {}

        for metric, title, ylabel, mode in specifications:
            if metric not in data.columns:
                if self.strict_columns:
                    raise KeyError(
                        f"Required plotting column {metric!r} is missing."
                    )
                continue

            filename = f"{prefix}_{metric}.png"

            if mode == "mean_time":
                figure = self.plot_controller_mean_time_series(
                    data,
                    metric,
                    controller_column=controller_column,
                    trial_column=trial_column,
                    time_column=time_column,
                    uncertainty="std",
                    title=title,
                    ylabel=ylabel,
                    save_as=filename,
                )
            else:
                figure = self.plot_trial_metric_comparison(
                    data,
                    metric,
                    aggregation="last",
                    controller_column=controller_column,
                    trial_column=trial_column,
                    time_column=time_column,
                    kind="box",
                    title=title,
                    ylabel=ylabel,
                    save_as=filename,
                )

            created[metric] = self.output_directory / filename
            if close_figures:
                plt.close(figure)

        if "task_success" in data.columns:
            filename = f"{prefix}_success_rate.png"
            figure = self.plot_success_rate(
                data,
                controller_column=controller_column,
                trial_column=trial_column,
                time_column=time_column,
                save_as=filename,
            )
            created["success_rate"] = self.output_directory / filename
            if close_figures:
                plt.close(figure)

        if (
            "completion_time" in data.columns
            and "task_success" in data.columns
        ):
            try:
                filename = f"{prefix}_completion_time.png"
                figure = self.plot_completion_time(
                    data,
                    controller_column=controller_column,
                    trial_column=trial_column,
                    time_column=time_column,
                    save_as=filename,
                )
                created["completion_time"] = (
                    self.output_directory / filename
                )
                if close_figures:
                    plt.close(figure)
            except ValueError:
                # A controller comparison can legitimately contain no
                # successful trials during early development.
                pass

        return created

    def aggregate_trials(
        self,
        data: pd.DataFrame,
        metric: str,
        *,
        aggregation: str = "last",
        controller_column: str = DEFAULT_CONTROLLER_COLUMN,
        trial_column: str = DEFAULT_TRIAL_COLUMN,
        time_column: str = DEFAULT_TIME_COLUMN,
    ) -> pd.DataFrame:
        """Reduce a time-series metric to one value per controller/trial."""
        self._require_columns(
            data,
            [controller_column, trial_column, time_column, metric],
        )

        working = data[
            [controller_column, trial_column, time_column, metric]
        ].dropna(subset=[controller_column, trial_column, time_column]).copy()

        grouped = working.groupby(
            [controller_column, trial_column],
            dropna=False,
            sort=True,
        )

        if aggregation == "last":
            ordered = working.sort_values(time_column)
            result = (
                ordered.groupby(
                    [controller_column, trial_column],
                    as_index=False,
                    dropna=False,
                )
                .tail(1)
                [[controller_column, trial_column, metric]]
                .reset_index(drop=True)
            )
        elif aggregation in {"mean", "max", "min", "sum"}:
            result = (
                grouped[metric]
                .agg(aggregation)
                .reset_index()
            )
        elif aggregation == "peak_abs":
            result = (
                grouped[metric]
                .apply(
                    lambda values: float(
                        np.nanmax(np.abs(values.to_numpy(dtype=float)))
                    )
                    if values.notna().any()
                    else math.nan
                )
                .reset_index(name=metric)
            )
        else:
            raise ValueError(
                "aggregation must be one of: last, mean, max, min, sum, "
                "peak_abs."
            )

        return result

    def close_all(self) -> None:
        """Close all matplotlib figures."""
        plt.close("all")

    def _save_figure(
        self,
        figure: plt.Figure,
        save_as: str | Path | None,
    ) -> Path | None:
        if save_as is None:
            return None

        path = Path(save_as)
        if not path.is_absolute():
            path = self.output_directory / path

        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            path,
            dpi=self.dpi,
            bbox_inches="tight",
        )
        return path

    def _require_columns(
        self,
        data: pd.DataFrame,
        columns: Iterable[str],
    ) -> None:
        missing = [column for column in columns if column not in data.columns]
        if missing:
            raise KeyError(
                "Missing dataframe columns: " + ", ".join(missing)
            )

    @staticmethod
    def _normalise_paths(
        paths: str | Path | Sequence[str | Path],
    ) -> list[Path]:
        if isinstance(paths, (str, Path)):
            return [Path(paths)]

        normalised = [Path(path) for path in paths]
        if not normalised:
            raise ValueError("At least one CSV path is required.")
        return normalised

    @staticmethod
    def _format_group_label(
        columns: Sequence[str],
        group_value: Any,
    ) -> str:
        values = (
            group_value
            if isinstance(group_value, tuple)
            else (group_value,)
        )
        return ", ".join(
            f"{column}={value}"
            for column, value in zip(columns, values)
        )

    @staticmethod
    def _humanise(field_name: str) -> str:
        return field_name.replace(".", " ").replace("_", " ").strip().title()

    @classmethod
    def _coerce_known_types(cls, data: pd.DataFrame) -> pd.DataFrame:
        converted = data.copy()

        boolean_columns = (
            "task_success",
            "grasp_full_wrench_rank",
        )
        for column in boolean_columns:
            if column in converted.columns:
                converted[column] = cls._coerce_boolean_series(
                    converted[column]
                )

        numeric_exclusions = {
            "trial_name",
            "trial_id",
            "controller_name",
            "source_file",
        }
        for column in converted.columns:
            if column in numeric_exclusions:
                continue
            if converted[column].dtype == object:
                converted[column] = pd.to_numeric(
                    converted[column],
                    errors="ignore",
                )

        return converted

    @staticmethod
    def _coerce_boolean_series(series: pd.Series) -> pd.Series:
        if pd.api.types.is_bool_dtype(series):
            return series.fillna(False)

        truthy = {"true", "1", "yes", "y", "t"}
        falsy = {"false", "0", "no", "n", "f", ""}

        def convert(value: Any) -> bool:
            if pd.isna(value):
                return False
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(value)

            cleaned = str(value).strip().lower()
            if cleaned in truthy:
                return True
            if cleaned in falsy:
                return False
            raise ValueError(f"Cannot interpret {value!r} as boolean.")

        return series.map(convert)

    @staticmethod
    def _last_trial_samples(
        data: pd.DataFrame,
        *,
        controller_column: str,
        trial_column: str,
        time_column: str,
    ) -> pd.DataFrame:
        ordered = data.sort_values(time_column)
        return (
            ordered.groupby(
                [controller_column, trial_column],
                as_index=False,
                dropna=False,
            )
            .tail(1)
            .reset_index(drop=True)
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"output_directory={str(self.output_directory)!r}, "
            f"dpi={self.dpi}, "
            f"strict_columns={self.strict_columns}"
            f")"
        )
