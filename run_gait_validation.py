"""Run a direct vertical-lift one-leg ladder-step development trial (V8.0).

This runner keeps the dissertation evaluator disabled.  It automatically
exports gait-development telemetry (CSV + JSON + optional diagnostic plots)
so release, base drift, tracking error and contact transitions can be analysed
without manually extracting values from the MuJoCo viewer.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mujoco

from controller.gait_controller import GaitController


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SCENE = PROJECT_ROOT / "scene.xml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "gait_debug"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate one RR vertical-lift ladder step (V8.0)."
    )
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--duration", type=float, default=45.0)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--trial-name", type=str, default="rr_vertical_step_v8")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-rate", type=float, default=100.0)
    parser.add_argument("--no-plots", action="store_true")

    parser.add_argument(
        "--release-clearance",
        "--lift-height",
        dest="release_clearance",
        type=float,
        default=0.035,
        help="Compatibility value; V8 uses --transfer-lift-height for the nominal +z lift.",
    )
    parser.add_argument(
        "--release-clearance-step",
        type=float,
        default=0.005,
        help="Compatibility option retained for CSV/config continuity; unused in direct vertical mode.",
    )
    parser.add_argument(
        "--release-clearance-max",
        type=float,
        default=0.085,
        help="Maximum upward lift allowed if the nominal +z lift does not clear the source rung.",
    )
    parser.add_argument(
        "--minimum-source-clearance",
        type=float,
        default=0.010,
        help="Diagnostic clearance line for plots; V8 release acceptance uses actual old-rung contact loss.",
    )
    parser.add_argument(
        "--clearance-comparison-tolerance",
        type=float,
        default=0.0005,
        help="Numerical tolerance used only when comparing signed clearances.",
    )
    parser.add_argument("--release-pose-tolerance", type=float, default=0.050)
    parser.add_argument("--release-hold-time", type=float, default=0.20)
    parser.add_argument("--release-timeout", type=float, default=8.0)
    parser.add_argument("--source-recontact-timeout", type=float, default=0.15)

    parser.add_argument("--post-release-settle-time", type=float, default=0.50)
    parser.add_argument("--post-release-settle-timeout", type=float, default=2.0)
    parser.add_argument("--settle-hold-time", type=float, default=0.20)
    parser.add_argument("--max-settle-linear-speed", type=float, default=0.025)
    parser.add_argument("--max-settle-angular-speed", type=float, default=0.20)
    parser.add_argument(
        "--stop-after-release-settle",
        action="store_true",
        help="Stop successfully after one gripper is geometrically released and settled.",
    )

    parser.add_argument("--opening-time", type=float, default=1.0)
    parser.add_argument("--opening-timeout", type=float, default=3.0)
    parser.add_argument(
        "--finger-open-tolerance",
        type=float,
        default=0.10,
        help=(
            "Strict maximum finger-joint error required after geometric "
            "release, before transfer or a release-settle success is accepted."
        ),
    )
    parser.add_argument(
        "--release-start-finger-tolerance",
        type=float,
        default=0.35,
        help=(
            "Diagnostic only in V8; the direct vertical lift starts after the configured preload time."
        ),
    )
    parser.add_argument(
        "--swing-open-kp-scale",
        type=float,
        default=1.75,
        help="PD proportional-gain multiplier for the opening swing gripper only.",
    )
    parser.add_argument(
        "--peel-start-time",
        type=float,
        default=0.70,
        help="Opening preload time before the direct world +z lift begins.",
    )
    parser.add_argument(
        "--peel-start-max-contacts",
        type=int,
        default=1,
        help="Legacy compatibility option; V8 does not gate the vertical lift on this count.",
    )
    parser.add_argument(
        "--peel-open-time",
        type=float,
        default=1.20,
        help="Blend time from the loaded finger pose to the open pose during vertical lift.",
    )
    parser.add_argument(
        "--transport-finger-tolerance",
        type=float,
        default=0.40,
        help="Diagnostic finger error retained for telemetry; V8 does not block transfer on this value.",
    )
    parser.add_argument("--retract-time", type=float, default=4.0, help="Duration of the nominal vertical +z lift.")
    parser.add_argument(
        "--release-extension-speed",
        type=float,
        default=0.008,
        help="Additional upward speed if source contact remains after the nominal lift.",
    )
    parser.add_argument(
        "--preflight-clearance-margin",
        type=float,
        default=0.010,
        help="Extra predicted clearance required above the dynamic clearance threshold.",
    )
    parser.add_argument(
        "--escape-direction-samples",
        type=int,
        default=16,
        help="Compatibility option retained from V7; unused in direct vertical mode.",
    )
    parser.add_argument(
        "--transfer-lift-height",
        type=float,
        default=0.080,
        help="Nominal direct world +z lift before translating one XML rung spacing in +x.",
    )
    parser.add_argument("--clearance-lift-time", type=float, default=3.0)
    parser.add_argument("--clearance-hold-time", type=float, default=0.20)
    parser.add_argument("--clearance-timeout", type=float, default=5.0)
    parser.add_argument("--transfer-time", type=float, default=6.0)
    parser.add_argument("--approach-time", type=float, default=4.0)
    parser.add_argument("--closing-time", type=float, default=1.5)
    parser.add_argument("--body-shift-time", type=float, default=5.0)
    parser.add_argument("--body-shift-scale", type=float, default=0.50)
    parser.add_argument(
        "--swing-sequence",
        type=str,
        default="RR,RL,FR,FL",
        help="Comma-separated order containing RR, RL, FR and FL exactly once.",
    )
    parser.add_argument("--max-joint-target-rate", type=float, default=0.45)
    parser.add_argument("--max-swing-base-drift", type=float, default=0.18)
    parser.add_argument("--max-swing-tracking-error", type=float, default=0.08)
    parser.add_argument("--tracking-error-timeout", type=float, default=0.40)
    parser.add_argument("--preflight-ik-tolerance", type=float, default=0.025)
    parser.add_argument("--preflight-ik-iterations", type=int, default=180)
    parser.add_argument("--minimum-regrasp-contacts", type=int, default=1)
    parser.add_argument(
        "--early-close-min-time",
        type=float,
        default=0.40,
        help="Minimum approach/closing time before an existing target contact triggers early verification.",
    )
    parser.add_argument("--support-loss-timeout", type=float, default=0.25)
    parser.add_argument(
        "--stop-after-regrasps",
        type=int,
        default=1,
        help="Stop successfully after this many verified target-rung regrasps.",
    )
    parser.add_argument("--real-time", action="store_true")

    viewer_group = parser.add_mutually_exclusive_group()
    viewer_group.add_argument("--viewer", dest="viewer", action="store_true")
    viewer_group.add_argument("--no-viewer", dest="viewer", action="store_false")
    parser.set_defaults(viewer=True)
    parser.add_argument(
        "--hold-viewer-on-exit",
        action="store_true",
        help="Keep the MuJoCo window open after COMPLETE/FAILED until closed manually.",
    )
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> tuple[str, ...]:
    if not args.scene.exists():
        raise FileNotFoundError(f"Scene XML not found: {args.scene}")
    if args.duration <= 0.0:
        raise ValueError("--duration must be positive.")
    if args.cycles < 1:
        raise ValueError("--cycles must be at least 1.")
    if args.sample_rate <= 0.0:
        raise ValueError("--sample-rate must be positive.")
    if not args.trial_name.strip():
        raise ValueError("--trial-name must not be empty.")

    positive = (
        "release_clearance",
        "release_clearance_step",
        "release_clearance_max",
        "minimum_source_clearance",
        "release_pose_tolerance",
        "release_hold_time",
        "release_timeout",
        "source_recontact_timeout",
        "post_release_settle_time",
        "post_release_settle_timeout",
        "settle_hold_time",
        "max_settle_linear_speed",
        "max_settle_angular_speed",
        "opening_time",
        "opening_timeout",
        "finger_open_tolerance",
        "release_start_finger_tolerance",
        "peel_start_time",
        "peel_open_time",
        "transport_finger_tolerance",
        "swing_open_kp_scale",
        "retract_time",
        "release_extension_speed",
        "transfer_lift_height",
        "clearance_lift_time",
        "clearance_hold_time",
        "clearance_timeout",
        "transfer_time",
        "approach_time",
        "closing_time",
        "body_shift_time",
        "body_shift_scale",
        "max_joint_target_rate",
        "max_swing_base_drift",
        "max_swing_tracking_error",
        "tracking_error_timeout",
        "preflight_ik_tolerance",
        "support_loss_timeout",
    )
    for name in positive:
        if getattr(args, name) <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.clearance_comparison_tolerance < 0.0:
        raise ValueError("--clearance-comparison-tolerance must be non-negative.")
    if args.preflight_clearance_margin < 0.0:
        raise ValueError("--preflight-clearance-margin must be non-negative.")
    if args.opening_timeout <= args.opening_time:
        raise ValueError("--opening-timeout must be greater than --opening-time.")
    if args.release_start_finger_tolerance < args.finger_open_tolerance:
        raise ValueError(
            "--release-start-finger-tolerance must be >= --finger-open-tolerance."
        )
    if args.swing_open_kp_scale < 1.0:
        raise ValueError("--swing-open-kp-scale must be at least 1.0.")
    if args.peel_start_max_contacts < 0:
        raise ValueError("--peel-start-max-contacts must be non-negative.")
    if args.early_close_min_time < 0.0:
        raise ValueError("--early-close-min-time must be non-negative.")
    if args.escape_direction_samples < 8:
        raise ValueError("--escape-direction-samples must be at least 8.")
    if args.release_clearance_max < args.release_clearance:
        raise ValueError("--release-clearance-max must be >= --release-clearance.")
    if args.clearance_timeout <= args.clearance_lift_time:
        raise ValueError("--clearance-timeout must be greater than lift time.")
    if args.post_release_settle_timeout <= args.post_release_settle_time:
        raise ValueError(
            "--post-release-settle-timeout must be greater than settle time."
        )
    if args.preflight_ik_iterations < 1:
        raise ValueError("--preflight-ik-iterations must be at least 1.")
    if args.minimum_regrasp_contacts < 1:
        raise ValueError("--minimum-regrasp-contacts must be at least 1.")
    if args.stop_after_regrasps is not None and args.stop_after_regrasps < 1:
        raise ValueError("--stop-after-regrasps must be at least 1.")

    sequence = tuple(
        item.strip().upper() for item in args.swing_sequence.split(",") if item.strip()
    )
    if len(sequence) != 4 or set(sequence) != {"FL", "FR", "RL", "RR"}:
        raise ValueError("--swing-sequence must contain FL, FR, RL and RR once.")
    return sequence


def _print_periodic(snapshot: dict[str, object]) -> None:
    tracking = float(snapshot["swing_tracking_error"])
    clearance = float(snapshot["source_clearance"])
    commanded = float(snapshot.get("release_command_displacement", float("nan")))
    open_error = float(snapshot.get("finger_open_error", float("nan")))
    closest = str(snapshot.get("closest_source_segment", "") or "-")
    tracking_text = "nan" if not math.isfinite(tracking) else f"{tracking:.4f}"
    clearance_text = "nan" if not math.isfinite(clearance) else f"{clearance:+.5f}"
    commanded_text = "nan" if not math.isfinite(commanded) else f"{commanded:.4f}"
    open_text = "nan" if not math.isfinite(open_error) else f"{open_error:.3f}"
    print(
        f"t={float(snapshot['time']):7.3f} | "
        f"phase={str(snapshot['phase']):28s} | "
        f"swing={str(snapshot['swing_leg'] or 'None'):4s} | "
        f"contacts={int(snapshot['physical_contacts']):2d} | "
        f"old={int(snapshot['old_rung_contacts']):1d} | "
        f"target={int(snapshot['target_rung_contacts']):1d} | "
        f"palm_err={tracking_text:>6s} m | "
        f"clear={clearance_text:>8s} m | "
        f"cmd={commanded_text:>6s} m | open_err={open_text:>5s} rad | "
        f"closest={closest:>10.10s} | "
        f"|v_b|={float(snapshot['base_linear_speed']):.4f} m/s | "
        f"|w_b|={float(snapshot['base_angular_speed']):.4f} rad/s | "
        f"grippers={str(snapshot['contacting_grippers']).split(';') if snapshot['contacting_grippers'] else []} | "
        f"base=({float(snapshot['base_x']):+.4f}, "
        f"{float(snapshot['base_y']):+.4f}, "
        f"{float(snapshot['base_z']):+.4f})"
    )


def run_loop(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller: GaitController,
    *,
    duration: float,
    passive_viewer: object | None,
    real_time: bool,
    sample_rate: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    phase_events: list[dict[str, object]] = []
    previous_phase: str | None = None
    next_report = 0.0
    next_sample = 0.0
    sample_period = 1.0 / sample_rate

    while float(data.time) < duration:
        wall_start = time.perf_counter()

        controller.update(model, data)
        mujoco.mj_step(model, data)
        controller.post_step(model, data)

        if passive_viewer is not None:
            passive_viewer.sync()

        snapshot = controller.diagnostic_snapshot(model, data)
        now = float(data.time)
        phase = str(snapshot["phase"])

        if phase != previous_phase:
            phase_events.append(
                {
                    "time": now,
                    "phase": phase,
                    "message": str(snapshot["message"]),
                }
            )

        if now + 1.0e-12 >= next_sample:
            rows.append(snapshot)
            while next_sample <= now + 1.0e-12:
                next_sample += sample_period

        if phase != previous_phase or now >= next_report:
            _print_periodic(snapshot)
            next_report = now + 0.5
            previous_phase = phase

        if controller.gait_complete or controller.gait_failed:
            break

        if real_time:
            elapsed = time.perf_counter() - wall_start
            remaining = float(model.opt.timestep) - elapsed
            if remaining > 0.0:
                time.sleep(remaining)

    if not rows or float(rows[-1]["time"]) < float(data.time):
        rows.append(controller.diagnostic_snapshot(model, data))
    return rows, phase_events


def save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, ensure_ascii=False)


def save_plots(
    output_dir: Path,
    trial_name: str,
    rows: list[dict[str, object]],
    *,
    minimum_source_clearance: float,
    release_pose_tolerance: float,
) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is unavailable; diagnostic plots were skipped.")
        return []

    if len(rows) < 2:
        return []

    time_values = [float(row["time"]) for row in rows]
    output_paths: list[Path] = []

    def save_current(suffix: str) -> None:
        path = output_dir / f"{trial_name}_{suffix}.png"
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        output_paths.append(path)

    reference = rows[0]
    plt.figure(figsize=(10, 5))
    for axis in ("x", "y", "z"):
        values = [
            float(row[f"base_{axis}"]) - float(reference[f"base_{axis}"])
            for row in rows
        ]
        plt.plot(time_values, values, label=f"base d{axis}")
    plt.xlabel("Time [s]")
    plt.ylabel("Base displacement [m]")
    plt.grid(True)
    plt.legend()
    save_current("base_displacement")

    plt.figure(figsize=(10, 5))
    plt.plot(
        time_values,
        [float(row["base_linear_speed"]) for row in rows],
        label="base linear speed",
    )
    plt.plot(
        time_values,
        [float(row["base_angular_speed"]) for row in rows],
        label="base angular speed",
    )
    plt.xlabel("Time [s]")
    plt.ylabel("Speed magnitude")
    plt.grid(True)
    plt.legend()
    save_current("base_speed")

    plt.figure(figsize=(10, 5))
    plt.plot(
        time_values,
        [float(row["source_clearance"]) for row in rows],
        label="source clearance",
    )
    plt.axhline(minimum_source_clearance, linestyle="--", label="required clearance")
    plt.plot(
        time_values,
        [float(row["swing_tracking_error"]) for row in rows],
        label="swing palm tracking error",
    )
    plt.axhline(release_pose_tolerance, linestyle=":", label="release pose tolerance")
    plt.xlabel("Time [s]")
    plt.ylabel("Distance [m]")
    plt.grid(True)
    plt.legend()
    save_current("clearance_tracking")

    if "release_command_displacement" in rows[0]:
        plt.figure(figsize=(10, 5))
        plt.plot(
            time_values,
            [float(row.get("release_command_displacement", float("nan"))) for row in rows],
            label="commanded vertical lift",
        )
        plt.plot(
            time_values,
            [float(row["source_clearance"]) for row in rows],
            label="measured source clearance",
        )
        plt.xlabel("Time [s]")
        plt.ylabel("Distance [m]")
        plt.grid(True)
        plt.legend()
        save_current("release_command")

    plt.figure(figsize=(10, 5))
    plt.step(
        time_values,
        [int(row["old_rung_contacts"]) for row in rows],
        where="post",
        label="old-rung contacts",
    )
    plt.step(
        time_values,
        [int(row["target_rung_contacts"]) for row in rows],
        where="post",
        label="target-rung contacts",
    )
    plt.step(
        time_values,
        [int(row["physical_contacts"]) for row in rows],
        where="post",
        label="all physical contacts",
    )
    plt.xlabel("Time [s]")
    plt.ylabel("Contact count")
    plt.grid(True)
    plt.legend()
    save_current("contact_counts")

    return output_paths


def print_final_result(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller: GaitController,
) -> None:
    final_status = controller.gait_status
    print("\n" + "=" * 72)
    print("GAIT VALIDATION RESULT")
    print("=" * 72)
    print(f"Final phase:          {final_status.phase}")
    print(f"Completed cycles:     {final_status.cycles_completed}")
    print(f"Verified regrasps:    {final_status.completed_regrasps}")
    print(f"Contacting grippers:  {list(final_status.contacting_grippers)}")
    print(f"Physical contacts:    {final_status.physical_contacts}")
    print(f"Old-rung contacts:    {final_status.old_rung_contacts}")
    print(f"Target-rung contacts: {final_status.target_rung_contacts}")
    print(f"Swing tracking error: {final_status.swing_tracking_error}")
    print(f"Source clearance:     {final_status.source_clearance}")
    print(
        "Selected vertical lift: "
        f"{final_status.selected_release_displacement}"
    )
    print(f"Base linear speed:    {final_status.base_linear_speed}")
    print(f"Base angular speed:   {final_status.base_angular_speed}")
    print(f"Base position:        {final_status.base_position}")
    print(f"Message:              {final_status.message}")
    print("\nFinal contact report")
    print(controller.contact_report(model, data))

    print("\nFinal gait IK residuals")
    for leg, residual in controller.last_ik_residuals.items():
        print(f"  {leg}: {residual:.6f} m")
    print("=" * 72)


def main() -> None:
    # Keep progress visible even when stdout is piped through `tee`.
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    args = parse_arguments()
    swing_sequence = validate_arguments(args)

    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    model.opt.gravity[:] = [0.0, 0.0, 0.0]
    data = mujoco.MjData(model)

    controller = GaitController(
        model,
        max_cycles=args.cycles,
        swing_sequence=swing_sequence,
        opening_time=args.opening_time,
        opening_timeout=args.opening_timeout,
        finger_open_tolerance=args.finger_open_tolerance,
        release_start_finger_tolerance=(
            args.release_start_finger_tolerance
        ),
        peel_start_time=args.peel_start_time,
        peel_start_max_contacts=args.peel_start_max_contacts,
        peel_open_time=args.peel_open_time,
        transport_finger_tolerance=args.transport_finger_tolerance,
        swing_open_kp_scale=args.swing_open_kp_scale,
        retract_time=args.retract_time,
        release_extension_speed=args.release_extension_speed,
        preflight_clearance_margin=args.preflight_clearance_margin,
        escape_direction_samples=args.escape_direction_samples,
        clearance_lift_time=args.clearance_lift_time,
        clearance_hold_time=args.clearance_hold_time,
        clearance_timeout=args.clearance_timeout,
        transfer_time=args.transfer_time,
        approach_time=args.approach_time,
        closing_time_gait=args.closing_time,
        body_shift_time=args.body_shift_time,
        release_clearance=args.release_clearance,
        release_clearance_step=args.release_clearance_step,
        release_clearance_max=args.release_clearance_max,
        minimum_source_clearance=args.minimum_source_clearance,
        clearance_comparison_tolerance=args.clearance_comparison_tolerance,
        release_pose_tolerance=args.release_pose_tolerance,
        post_release_settle_time=args.post_release_settle_time,
        post_release_settle_timeout=args.post_release_settle_timeout,
        settle_hold_time=args.settle_hold_time,
        max_settle_linear_speed=args.max_settle_linear_speed,
        max_settle_angular_speed=args.max_settle_angular_speed,
        transfer_lift_height=args.transfer_lift_height,
        release_hold_time=args.release_hold_time,
        release_timeout=args.release_timeout,
        source_recontact_timeout=args.source_recontact_timeout,
        body_shift_scale=args.body_shift_scale,
        max_joint_target_rate=args.max_joint_target_rate,
        max_swing_base_drift=args.max_swing_base_drift,
        max_swing_tracking_error=args.max_swing_tracking_error,
        tracking_error_timeout=args.tracking_error_timeout,
        preflight_ik_tolerance=args.preflight_ik_tolerance,
        preflight_ik_iterations=args.preflight_ik_iterations,
        early_close_min_time=args.early_close_min_time,
        minimum_regrasp_contacts=args.minimum_regrasp_contacts,
        support_loss_timeout=args.support_loss_timeout,
        stop_after_regrasps=args.stop_after_regrasps,
        stop_after_release_settle=args.stop_after_release_settle,
    )
    controller.reset(model, data)

    trial_dir = (args.output_dir / args.trial_name).resolve()
    trial_dir.mkdir(parents=True, exist_ok=True)
    csv_path = trial_dir / f"{args.trial_name}_telemetry.csv"
    json_path = trial_dir / f"{args.trial_name}_summary.json"

    print("\nStarting direct vertical-step validation V8.0")
    print("-" * 72)
    print(f"Scene:                       {args.scene.resolve()}")
    print(f"Trial output:                {trial_dir}")
    print(f"Duration limit:              {args.duration:.3f} s")
    print(f"Swing sequence:              {list(swing_sequence)}")
    print("Trajectory:                  open + vertical +z lift -> +x rung spacing -> vertical lower -> close")
    print("Support control:             active world-space palm hold; floating base remains free")
    print(
        "Vertical lift limits:        "
        f"{args.release_clearance:.3f}..{args.release_clearance_max:.3f} m "
        f"in {args.release_clearance_step:.3f} m steps"
    )
    print(f"Required source clearance:   {args.minimum_source_clearance:.4f} m")
    print(f"Preflight clearance margin:  {args.preflight_clearance_margin:.4f} m")
    print(f"Legacy direction samples:    {args.escape_direction_samples}")
    print(f"Extra upward speed:          {args.release_extension_speed:.4f} m/s")
    print(f"Strict finger-open tol.:     {args.finger_open_tolerance:.4f} rad")
    print(
        "Release finger diagnostic:  "
        f"{args.release_start_finger_tolerance:.4f} rad (diagnostic)"
    )
    print(f"Lift start time:             {args.peel_start_time:.3f} s")
    print(f"Legacy peel contacts:        {args.peel_start_max_contacts}")
    print(f"Finger-open blend time:      {args.peel_open_time:.3f} s")
    print(f"Transport finger tolerance:  {args.transport_finger_tolerance:.3f} rad")
    print(f"Swing opening Kp scale:      {args.swing_open_kp_scale:.3f}")
    print(
        "Clearance comparison tol.:   "
        f"{args.clearance_comparison_tolerance:.6f} m"
    )
    print(f"Post-release settle time:    {args.post_release_settle_time:.3f} s")
    print(f"Direct +z lift height:       {args.transfer_lift_height:.4f} m")
    print(f"Stop after release settle:   {args.stop_after_release_settle}")
    print(f"Stop after regrasps:         {args.stop_after_regrasps}")
    print(f"Telemetry sample rate:       {args.sample_rate:.1f} Hz")
    print("Dissertation metrics:        disabled")
    print("-" * 72)

    rows: list[dict[str, object]]
    phase_events: list[dict[str, object]]

    if args.viewer:
        from mujoco import viewer as mujoco_viewer

        with mujoco_viewer.launch_passive(model, data) as passive_viewer:
            rows, phase_events = run_loop(
                model,
                data,
                controller,
                duration=args.duration,
                passive_viewer=passive_viewer,
                real_time=args.real_time,
                sample_rate=args.sample_rate,
            )
            print_final_result(model, data, controller)

            save_csv(csv_path, rows)
            plot_paths = [] if args.no_plots else save_plots(
                trial_dir,
                args.trial_name,
                rows,
                minimum_source_clearance=args.minimum_source_clearance,
                release_pose_tolerance=args.release_pose_tolerance,
            )
            save_json(
                json_path,
                {
                    "version": "V8.0",
                    "arguments": vars(args),
                    "final_status": asdict(controller.gait_status),
                    "phase_events": phase_events,
                    "sample_count": len(rows),
                    "telemetry_csv": csv_path,
                    "plots": plot_paths,
                    "note": (
                        "These are gait-development diagnostics, not the final "
                        "dissertation metrics framework."
                    ),
                },
            )

            print(f"\nTelemetry CSV: {csv_path}")
            print(f"Summary JSON:  {json_path}")
            for path in plot_paths:
                print(f"Plot:          {path}")

            if args.hold_viewer_on_exit and passive_viewer.is_running():
                print("\nViewer is being held open. Close the MuJoCo window to exit.")
                while passive_viewer.is_running():
                    passive_viewer.sync()
                    time.sleep(0.05)
    else:
        rows, phase_events = run_loop(
            model,
            data,
            controller,
            duration=args.duration,
            passive_viewer=None,
            real_time=args.real_time,
            sample_rate=args.sample_rate,
        )
        print_final_result(model, data, controller)
        save_csv(csv_path, rows)
        plot_paths = [] if args.no_plots else save_plots(
            trial_dir,
            args.trial_name,
            rows,
            minimum_source_clearance=args.minimum_source_clearance,
            release_pose_tolerance=args.release_pose_tolerance,
        )
        save_json(
            json_path,
            {
                "version": "V8.0",
                "arguments": vars(args),
                "final_status": asdict(controller.gait_status),
                "phase_events": phase_events,
                "sample_count": len(rows),
                "telemetry_csv": csv_path,
                "plots": plot_paths,
                "note": (
                    "These are gait-development diagnostics, not the final "
                    "dissertation metrics framework."
                ),
            },
        )
        print(f"\nTelemetry CSV: {csv_path}")
        print(f"Summary JSON:  {json_path}")
        for path in plot_paths:
            print(f"Plot:          {path}")


if __name__ == "__main__":
    main()
