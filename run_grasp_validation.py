#!/usr/bin/env python3
"""
Static ladder-grasp validation and metrics collection.

This runner:
1. loads the MuJoCo scene,
2. resets StaticLadderGraspController,
3. acquires and verifies a four-gripper grasp,
4. records metrics only after the controller enters HOLDING,
5. saves CSV, summary text, and diagnostic plots.

Run from the project root, for example:

    python run_grasp_validation.py \
        --scene scene.xml \
        --duration 30 \
        --real-time \
        --trial-name static_grasp_trial_001

The output directory is:

    results/<trial-name>/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np

from controller.grasp_controller import StaticLadderGraspController


LEGS = ("FL", "FR", "RL", "RR")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a static ladder grasp and record stability metrics."
    )
    parser.add_argument(
        "--scene",
        type=Path,
        default=Path("scene.xml"),
        help="Path to the MuJoCo scene XML. Default: scene.xml",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Total simulation duration in seconds. Default: 30",
    )
    parser.add_argument(
        "--real-time",
        action="store_true",
        help="Pace the simulation approximately in real time.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without the MuJoCo viewer.",
    )
    parser.add_argument(
        "--trial-name",
        type=str,
        default="static_grasp_trial_001",
        help="Name used for the results folder and output files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Parent directory for results. Default: results",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=100.0,
        help="Metrics sampling frequency in Hz during HOLDING. Default: 100",
    )
    parser.add_argument(
        "--status-rate",
        type=float,
        default=1.0,
        help="Terminal status printing frequency in Hz. Default: 1",
    )
    parser.add_argument(
        "--print-contact-details",
        action="store_true",
        help="Print the controller's detailed contact report at each status update.",
    )
    parser.add_argument(
        "--minimum-holding-time",
        type=float,
        default=5.0,
        help="Minimum recorded HOLDING duration required for a successful trial.",
    )
    return parser.parse_args()


def quaternion_angular_distance(q0: np.ndarray, q1: np.ndarray) -> float:
    """Return the shortest angular distance between wxyz quaternions, in radians."""
    a = np.asarray(q0, dtype=float)
    b = np.asarray(q1, dtype=float)

    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a <= 1.0e-12 or norm_b <= 1.0e-12:
        return float("nan")

    a = a / norm_a
    b = b / norm_b
    dot = float(np.clip(abs(np.dot(a, b)), 0.0, 1.0))
    return 2.0 * math.acos(dot)


def cumulative_trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """NumPy-only cumulative trapezoidal integration."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    result = np.zeros_like(y, dtype=float)
    if len(y) < 2:
        return result

    dt = np.diff(x)
    interval_area = 0.5 * (y[1:] + y[:-1]) * dt
    result[1:] = np.cumsum(interval_area)
    return result


def collect_static_grasp_metrics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller: StaticLadderGraspController,
    *,
    trial_time: float,
    holding_reference_position: np.ndarray,
    holding_reference_quaternion: np.ndarray,
) -> dict[str, float | int]:
    """
    Collect one metrics sample.

    mj_contactForce returns a 6-vector in the contact frame:
        [normal force, tangent-1 force, tangent-2 force, torque-1, torque-2, torque-3]
    """

    row: dict[str, float | int] = {
        "time": float(trial_time),
        "physical_contacts": 0,
        "distinct_contacting_grippers": 0,
        "normal_force_total": 0.0,
        "tangential_force_total": 0.0,
        "global_force_ratio": 0.0,
        "maximum_contact_friction_utilization": 0.0,
    }

    normal_by_gripper = {leg: 0.0 for leg in LEGS}
    tangential_by_gripper = {leg: 0.0 for leg in LEGS}
    contact_count_by_gripper = {leg: 0 for leg in LEGS}
    max_utilization_by_gripper = {leg: 0.0 for leg in LEGS}

    contact_force = np.zeros(6, dtype=float)
    epsilon = 1.0e-12

    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        geom1_id = int(contact.geom1)
        geom2_id = int(contact.geom2)

        geom1_is_rung = geom1_id in controller._rung_geom_ids
        geom2_is_rung = geom2_id in controller._rung_geom_ids

        # Keep only contacts containing exactly one rung.
        if geom1_is_rung == geom2_is_rung:
            continue

        finger_geom_id = geom2_id if geom1_is_rung else geom1_id
        finger_body_id = int(model.geom_bodyid[finger_geom_id])
        leg = controller._finger_body_to_leg.get(finger_body_id)

        if leg is None:
            continue

        contact_force[:] = 0.0
        mujoco.mj_contactForce(
            model,
            data,
            contact_index,
            contact_force,
        )

        normal_force = abs(float(contact_force[0]))
        tangential_force = float(np.linalg.norm(contact_force[1:3]))

        # MuJoCo stores the primary sliding friction coefficient here.
        friction_coefficient = max(float(contact.friction[0]), epsilon)
        utilization = tangential_force / max(
            friction_coefficient * normal_force,
            epsilon,
        )

        row["physical_contacts"] = int(row["physical_contacts"]) + 1
        normal_by_gripper[leg] += normal_force
        tangential_by_gripper[leg] += tangential_force
        contact_count_by_gripper[leg] += 1
        max_utilization_by_gripper[leg] = max(
            max_utilization_by_gripper[leg],
            utilization,
        )

        row["maximum_contact_friction_utilization"] = max(
            float(row["maximum_contact_friction_utilization"]),
            utilization,
        )

    normal_force_total = float(sum(normal_by_gripper.values()))
    tangential_force_total = float(sum(tangential_by_gripper.values()))
    contacting_grippers = sum(
        count > 0 for count in contact_count_by_gripper.values()
    )

    row["distinct_contacting_grippers"] = int(contacting_grippers)
    row["normal_force_total"] = normal_force_total
    row["tangential_force_total"] = tangential_force_total
    row["global_force_ratio"] = (
        tangential_force_total / max(normal_force_total, epsilon)
    )

    for leg in LEGS:
        row[f"{leg}_contact_count"] = int(contact_count_by_gripper[leg])
        row[f"{leg}_normal_force"] = float(normal_by_gripper[leg])
        row[f"{leg}_tangential_force"] = float(tangential_by_gripper[leg])
        row[f"{leg}_force_ratio"] = (
            tangential_by_gripper[leg]
            / max(normal_by_gripper[leg], epsilon)
        )
        row[f"{leg}_maximum_friction_utilization"] = float(
            max_utilization_by_gripper[leg]
        )

    qpos_adr = controller._base_qpos_adr
    dof_adr = controller._base_dof_adr

    base_position = np.asarray(
        data.qpos[qpos_adr:qpos_adr + 3],
        dtype=float,
    ).copy()
    base_quaternion = np.asarray(
        data.qpos[qpos_adr + 3:qpos_adr + 7],
        dtype=float,
    ).copy()
    base_linear_velocity = np.asarray(
        data.qvel[dof_adr:dof_adr + 3],
        dtype=float,
    ).copy()
    base_angular_velocity = np.asarray(
        data.qvel[dof_adr + 3:dof_adr + 6],
        dtype=float,
    ).copy()

    position_delta = base_position - holding_reference_position

    row.update(
        {
            "base_x": float(base_position[0]),
            "base_y": float(base_position[1]),
            "base_z": float(base_position[2]),
            "base_position_drift": float(np.linalg.norm(position_delta)),
            "base_orientation_drift_rad": quaternion_angular_distance(
                holding_reference_quaternion,
                base_quaternion,
            ),
            "base_linear_speed": float(np.linalg.norm(base_linear_velocity)),
            "base_angular_speed": float(np.linalg.norm(base_angular_velocity)),
        }
    )

    actuator_force = np.asarray(data.actuator_force, dtype=float)
    actuator_velocity = np.asarray(data.actuator_velocity, dtype=float)
    actuator_power = actuator_force * actuator_velocity

    row["signed_mechanical_power"] = float(np.sum(actuator_power))
    row["absolute_mechanical_power"] = float(np.sum(np.abs(actuator_power)))
    row["peak_absolute_actuator_power"] = float(
        np.max(np.abs(actuator_power)) if actuator_power.size else 0.0
    )

    return row


def save_csv(rows: list[dict[str, float | int]], path: Path) -> None:
    if not rows:
        raise ValueError("Cannot save an empty metrics table.")

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def column(
    rows: list[dict[str, float | int]],
    name: str,
) -> np.ndarray:
    return np.asarray([float(row[name]) for row in rows], dtype=float)


def save_line_plot(
    *,
    x: np.ndarray,
    series: dict[str, np.ndarray],
    xlabel: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    plt.figure(figsize=(9, 5.5))
    for label, values in series.items():
        plt.plot(x, values, label=label)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)

    if len(series) > 1:
        plt.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def save_step_plot(
    *,
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    plt.figure(figsize=(9, 5.5))
    plt.step(x, y, where="post")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def save_plots(
    rows: list[dict[str, float | int]],
    results_dir: Path,
    trial_name: str,
) -> dict[str, Path]:
    time_values = column(rows, "time")
    absolute_power = column(rows, "absolute_mechanical_power")
    cumulative_absolute_work = cumulative_trapezoid(
        absolute_power,
        time_values,
    )

    paths: dict[str, Path] = {}

    paths["normal_force"] = (
        results_dir / f"{trial_name}_normal_force.png"
    )
    save_line_plot(
        x=time_values,
        series={
            leg: column(rows, f"{leg}_normal_force")
            for leg in LEGS
        },
        xlabel="Holding time (s)",
        ylabel="Normal force (N)",
        title="Static Grasp: Normal Force by Gripper",
        path=paths["normal_force"],
    )

    paths["tangential_force"] = (
        results_dir / f"{trial_name}_tangential_force.png"
    )
    save_line_plot(
        x=time_values,
        series={
            leg: column(rows, f"{leg}_tangential_force")
            for leg in LEGS
        },
        xlabel="Holding time (s)",
        ylabel="Tangential force (N)",
        title="Static Grasp: Tangential Force by Gripper",
        path=paths["tangential_force"],
    )

    paths["friction_utilization"] = (
        results_dir / f"{trial_name}_friction_utilization.png"
    )
    save_line_plot(
        x=time_values,
        series={
            leg: column(
                rows,
                f"{leg}_maximum_friction_utilization",
            )
            for leg in LEGS
        },
        xlabel="Holding time (s)",
        ylabel="Friction utilization ratio",
        title="Static Grasp: Maximum Contact Friction Utilization",
        path=paths["friction_utilization"],
    )

    paths["contact_count"] = (
        results_dir / f"{trial_name}_contact_count.png"
    )
    save_step_plot(
        x=time_values,
        y=column(rows, "physical_contacts"),
        xlabel="Holding time (s)",
        ylabel="Physical contact count",
        title="Static Grasp: Contact Count",
        path=paths["contact_count"],
    )

    paths["base_motion"] = (
        results_dir / f"{trial_name}_base_motion.png"
    )
    save_line_plot(
        x=time_values,
        series={
            "linear speed": column(rows, "base_linear_speed"),
            "angular speed": column(rows, "base_angular_speed"),
            "position drift": column(rows, "base_position_drift"),
        },
        xlabel="Holding time (s)",
        ylabel="Magnitude",
        title="Static Grasp: Base Motion and Drift",
        path=paths["base_motion"],
    )

    paths["absolute_work"] = (
        results_dir
        / f"{trial_name}_absolute_mechanical_work.png"
    )
    save_line_plot(
        x=time_values,
        series={"cumulative absolute work": cumulative_absolute_work},
        xlabel="Holding time (s)",
        ylabel="Absolute mechanical work (J)",
        title="Static Grasp: Cumulative Absolute Mechanical Work",
        path=paths["absolute_work"],
    )

    paths["absolute_power"] = (
        results_dir
        / f"{trial_name}_absolute_mechanical_power.png"
    )
    save_line_plot(
        x=time_values,
        series={"absolute mechanical power": absolute_power},
        xlabel="Holding time (s)",
        ylabel="Absolute mechanical power (W)",
        title="Static Grasp: Absolute Mechanical Power",
        path=paths["absolute_power"],
    )

    return paths


def build_summary(
    rows: list[dict[str, float | int]],
    *,
    trial_name: str,
    scene_path: Path,
    model_timestep: float,
) -> dict[str, Any]:
    time_values = column(rows, "time")
    absolute_power = column(rows, "absolute_mechanical_power")
    signed_power = column(rows, "signed_mechanical_power")

    cumulative_absolute_work = cumulative_trapezoid(
        absolute_power,
        time_values,
    )
    cumulative_net_work = cumulative_trapezoid(
        signed_power,
        time_values,
    )

    summary: dict[str, Any] = {
        "trial_name": trial_name,
        "scene": str(scene_path),
        "model_timestep_s": model_timestep,
        "samples": len(rows),
        "recorded_holding_duration_s": float(time_values[-1]),
        "absolute_mechanical_work_J": float(
            cumulative_absolute_work[-1]
        ),
        "net_mechanical_work_J": float(cumulative_net_work[-1]),
        "mean_physical_contacts": float(
            np.mean(column(rows, "physical_contacts"))
        ),
        "minimum_physical_contacts": int(
            np.min(column(rows, "physical_contacts"))
        ),
        "minimum_contacting_grippers": int(
            np.min(column(rows, "distinct_contacting_grippers"))
        ),
        "mean_total_normal_force_N": float(
            np.mean(column(rows, "normal_force_total"))
        ),
        "mean_total_tangential_force_N": float(
            np.mean(column(rows, "tangential_force_total"))
        ),
        "maximum_contact_friction_utilization": float(
            np.max(
                column(
                    rows,
                    "maximum_contact_friction_utilization",
                )
            )
        ),
        "maximum_base_position_drift_m": float(
            np.max(column(rows, "base_position_drift"))
        ),
        "maximum_base_orientation_drift_rad": float(
            np.max(column(rows, "base_orientation_drift_rad"))
        ),
        "maximum_base_linear_speed_m_per_s": float(
            np.max(column(rows, "base_linear_speed"))
        ),
        "maximum_base_angular_speed_rad_per_s": float(
            np.max(column(rows, "base_angular_speed"))
        ),
    }

    for leg in LEGS:
        summary[f"{leg}_mean_normal_force_N"] = float(
            np.mean(column(rows, f"{leg}_normal_force"))
        )
        summary[f"{leg}_mean_tangential_force_N"] = float(
            np.mean(column(rows, f"{leg}_tangential_force"))
        )
        summary[
            f"{leg}_maximum_friction_utilization"
        ] = float(
            np.max(
                column(
                    rows,
                    f"{leg}_maximum_friction_utilization",
                )
            )
        )

    return summary


def write_summary(
    summary: dict[str, Any],
    json_path: Path,
    text_path: Path,
) -> None:
    json_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    lines = [
        f"{key}: {value}"
        for key, value in summary.items()
    ]
    text_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def print_status(
    *,
    data: mujoco.MjData,
    status: Any,
    controller: StaticLadderGraspController,
    model: mujoco.MjModel,
    print_contact_details: bool,
) -> None:
    print(
        f"t={data.time:7.3f} s | "
        f"state={status.state:11s} | "
        f"pinned={status.base_pinned} | "
        f"contacts={status.physical_contacts} | "
        f"finger_segments={status.distinct_fingertips} | "
        f"grippers={status.distinct_grippers} | "
        f"verified={status.verified}"
    )

    if print_contact_details:
        print(controller.contact_report(model, data))


def run_simulation(
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller: StaticLadderGraspController,
    args: argparse.Namespace,
) -> list[dict[str, float | int]]:
    controller.reset(model, data)

    print("\nStarting static grasp validation and metrics collection")
    print("Metrics are recorded only after state=HOLDING.")
    print(f"Scene: {args.scene}")
    print(f"Duration: {args.duration:.3f} s")
    print(f"Metrics sample rate: {args.sample_rate:.3f} Hz")

    metrics_rows: list[dict[str, float | int]] = []

    holding_start_time: float | None = None
    holding_reference_position: np.ndarray | None = None
    holding_reference_quaternion: np.ndarray | None = None

    next_sample_time = 0.0
    next_status_time = 0.0
    sample_period = 1.0 / args.sample_rate
    status_period = 1.0 / args.status_rate

    wall_start = time.perf_counter()

    def simulation_active(viewer: Any | None) -> bool:
        if data.time >= args.duration:
            return False
        if viewer is not None and not viewer.is_running():
            return False
        return True

    def step_once(viewer: Any | None) -> None:
        nonlocal holding_start_time
        nonlocal holding_reference_position
        nonlocal holding_reference_quaternion
        nonlocal next_sample_time
        nonlocal next_status_time

        status = controller.update(model, data)

        mujoco.mj_step(model, data)
        controller.post_step(model, data)

        # Re-read status after the step. The controller stores its latest state.
        status = controller.status

        if data.time + 1.0e-12 >= next_status_time:
            print_status(
                data=data,
                status=status,
                controller=controller,
                model=model,
                print_contact_details=args.print_contact_details,
            )
            next_status_time += status_period

        if status.state == "HOLDING":
            qadr = controller._base_qpos_adr

            if holding_start_time is None:
                holding_start_time = float(data.time)
                holding_reference_position = np.asarray(
                    data.qpos[qadr:qadr + 3],
                    dtype=float,
                ).copy()
                holding_reference_quaternion = np.asarray(
                    data.qpos[qadr + 3:qadr + 7],
                    dtype=float,
                ).copy()
                next_sample_time = holding_start_time

                print(
                    "\nVerified four-gripper grasp acquired. "
                    f"Metrics recording started at t={holding_start_time:.3f} s."
                )

            assert holding_reference_position is not None
            assert holding_reference_quaternion is not None

            if data.time + 1.0e-12 >= next_sample_time:
                metrics_rows.append(
                    collect_static_grasp_metrics(
                        model,
                        data,
                        controller,
                        trial_time=float(data.time - holding_start_time),
                        holding_reference_position=(
                            holding_reference_position
                        ),
                        holding_reference_quaternion=(
                            holding_reference_quaternion
                        ),
                    )
                )

                # Prevent drift if the physics timestep is larger than the
                # requested sample period.
                while next_sample_time <= data.time + 1.0e-12:
                    next_sample_time += sample_period

        if viewer is not None:
            viewer.sync()

        if args.real_time:
            target_wall_time = wall_start + float(data.time)
            sleep_time = target_wall_time - time.perf_counter()
            if sleep_time > 0.0:
                time.sleep(sleep_time)

    if args.headless:
        while simulation_active(None):
            step_once(None)
    else:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while simulation_active(viewer):
                step_once(viewer)

    return metrics_rows


def main() -> None:
    args = parse_args()

    if args.duration <= 0.0:
        raise ValueError("--duration must be positive.")
    if args.sample_rate <= 0.0:
        raise ValueError("--sample-rate must be positive.")
    if args.status_rate <= 0.0:
        raise ValueError("--status-rate must be positive.")
    if args.minimum_holding_time < 0.0:
        raise ValueError("--minimum-holding-time cannot be negative.")
    if not args.scene.exists():
        raise FileNotFoundError(f"Scene XML not found: {args.scene}")

    results_dir = args.output_dir / args.trial_name
    results_dir.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)

    controller = StaticLadderGraspController(model)

    metrics_rows = run_simulation(
        model=model,
        data=data,
        controller=controller,
        args=args,
    )

    print("\nFinal grasp status")
    print("-" * 72)
    print(f"State:               {controller.status.state}")
    print(f"Base pinned:         {controller.status.base_pinned}")
    print(f"Physical contacts:   {controller.status.physical_contacts}")
    print(f"Distinct grippers:   {controller.status.distinct_grippers}")
    print(f"Verified:            {controller.status.verified}")
    print(f"Message:             {controller.status.message}")
    print("-" * 72)

    if not controller.grasp_verified:
        raise RuntimeError(
            "A verified grasp was not established. "
            "No valid HOLDING metrics trial can be produced."
        )

    if not metrics_rows:
        raise RuntimeError(
            "The grasp was verified, but no HOLDING metrics were sampled."
        )

    holding_duration = float(metrics_rows[-1]["time"])
    if holding_duration < args.minimum_holding_time:
        raise RuntimeError(
            "The recorded HOLDING interval was too short: "
            f"{holding_duration:.3f} s < "
            f"{args.minimum_holding_time:.3f} s."
        )

    csv_path = (
        results_dir / f"{args.trial_name}_metrics.csv"
    )
    summary_json_path = (
        results_dir / f"{args.trial_name}_summary.json"
    )
    summary_text_path = (
        results_dir / f"{args.trial_name}_summary.txt"
    )

    save_csv(metrics_rows, csv_path)

    summary = build_summary(
        metrics_rows,
        trial_name=args.trial_name,
        scene_path=args.scene,
        model_timestep=float(model.opt.timestep),
    )
    write_summary(
        summary,
        summary_json_path,
        summary_text_path,
    )

    plot_paths = save_plots(
        metrics_rows,
        results_dir,
        args.trial_name,
    )

    print("\nMetrics trial completed")
    print("-" * 72)
    print(f"CSV:                     {csv_path}")
    print(f"Summary JSON:            {summary_json_path}")
    print(f"Summary text:            {summary_text_path}")
    print(
        "Absolute mechanical work: "
        f"{summary['absolute_mechanical_work_J']:.9f} J"
    )
    print(
        "Recorded HOLDING duration: "
        f"{summary['recorded_holding_duration_s']:.3f} s"
    )
    for name, path in plot_paths.items():
        print(f"Plot ({name}): {path}")
    print("-" * 72)


if __name__ == "__main__":
    main()
