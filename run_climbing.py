"""
Run one MuJoCo climbing trial using config.py as the project-wide default
configuration source.

Configuration priority
----------------------
1. Command-line arguments
2. config.py
3. Class defaults

The XML still defines model geometry, joints, actuators, and contact
properties. After loading the XML, simulation-wide options such as gravity
and timestep are explicitly overwritten from config.py so the effective
experiment settings are unambiguous and are saved in trial metadata.
"""

from __future__ import annotations

import argparse
import inspect
import time
from pathlib import Path
from typing import Any, Callable

import mujoco
import numpy as np

from config import GAIT, LADDER, METRICS, PATHS, SIMULATION
from controller.gait_controller import GaitController
from evaluator import (
    ContactMetrics,
    GraspMetrics,
    RobotMetrics,
    StabilityEvaluator,
    TaskMetrics,
)
from logger import DataLogger
from visualization import EvaluationPlotter


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SCENE_PATH = PROJECT_ROOT / "scene.xml"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / PATHS["results"]
DEFAULT_FIGURES_DIR = PROJECT_ROOT / PATHS["figures"]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and evaluate one quadruped climbing trial."
    )
    parser.add_argument(
        "--scene",
        type=Path,
        default=DEFAULT_SCENE_PATH,
        help="Path to the MuJoCo scene XML.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=float(SIMULATION["simulation_time"]),
        help="Maximum simulation duration in seconds.",
    )
    parser.add_argument(
        "--timestep",
        type=float,
        default=float(SIMULATION["dt"]),
        help="MuJoCo integration timestep in seconds.",
    )
    parser.add_argument(
        "--controller-name",
        default="gait",
        help="Controller label written to the experiment log.",
    )
    parser.add_argument(
        "--trial-id",
        default="1",
        help="Trial identifier written to the experiment log.",
    )
    parser.add_argument(
        "--trial-name",
        default=None,
        help="Output stem. Defaults to '<controller>_trial_<id>'.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory for CSV and JSON outputs.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help="Directory for generated figures.",
    )

    viewer_group = parser.add_mutually_exclusive_group()
    viewer_group.add_argument(
        "--viewer",
        dest="viewer",
        action="store_true",
        help="Launch the passive MuJoCo viewer.",
    )
    viewer_group.add_argument(
        "--no-viewer",
        dest="viewer",
        action="store_false",
        help="Run without the MuJoCo viewer.",
    )
    parser.set_defaults(viewer=bool(SIMULATION["viewer"]))

    parser.add_argument(
        "--real-time",
        action="store_true",
        help="Sleep to approximately match wall-clock simulation time.",
    )
    parser.add_argument(
        "--full-metrics",
        action="store_true",
        help="Store complete nested metric snapshots in JSON.",
    )
    parser.add_argument(
        "--stop-on-success",
        action="store_true",
        help="End the trial when TaskMetrics confirms success.",
    )
    parser.add_argument(
        "--initial-x",
        type=float,
        default=-0.50,
        help="Initial floating-base x coordinate.",
    )
    parser.add_argument(
        "--initial-y",
        type=float,
        default=0.005,
        help="Initial floating-base y coordinate.",
    )
    parser.add_argument(
        "--initial-z",
        type=float,
        default=0.25,
        help="Initial floating-base z coordinate.",
    )
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    if args.duration <= 0.0:
        raise ValueError("--duration must be positive.")
    if args.timestep <= 0.0:
        raise ValueError("--timestep must be positive.")
    if not args.scene.exists():
        raise FileNotFoundError(f"Scene XML was not found: {args.scene}")


def apply_simulation_configuration(
    model: mujoco.MjModel,
    *,
    timestep: float,
) -> None:
    """
    Apply experiment-level settings after XML loading.

    This makes config.py/CLI the authoritative source for gravity and
    integration timestep, even if the XML contains different values.
    """
    gravity = np.asarray(SIMULATION["gravity"], dtype=float)
    if gravity.shape != (3,):
        raise ValueError("SIMULATION['gravity'] must contain three values.")

    model.opt.gravity[:] = gravity
    model.opt.timestep = float(timestep)


def create_evaluator() -> StabilityEvaluator:
    """
    Construct all metric modules explicitly from one shared configuration.

    Explicit construction guarantees that GraspMetrics and TaskMetrics use
    the exact same lower-level metric instances.
    """
    robot_metric = RobotMetrics()

    contact_metric = ContactMetrics(
        minimum_normal_force=float(METRICS["minimum_contact_force"]),
        slip_speed_threshold=float(METRICS["slip_speed_threshold"]),
    )

    grasp_metric = GraspMetrics(
        contact_metric=contact_metric,
        expected_fingertip_count=8,
    )

    task_metric = TaskMetrics(
        robot_metric=robot_metric,
        contact_metric=contact_metric,
        grasp_metric=grasp_metric,
        target_rung_name=str(LADDER["target_rung"]),
        success_hold_time=float(METRICS["success_hold_time"]),
        regrasp_confirmation_time=float(
            METRICS["contact_loss_timeout"]
        ),
    )

    return StabilityEvaluator(
        robot_metrics=robot_metric,
        contact_metrics=contact_metric,
        grasp_metrics=grasp_metric,
        task_metrics=task_metric,
        strict_dependency_check=True,
    )


def create_controller(model: mujoco.MjModel, data: mujoco.MjData) -> Any:
    """Construct GaitController with a supported constructor signature."""
    signature = inspect.signature(GaitController)

    for arguments in ((model, data), (model,), ()):
        try:
            signature.bind(*arguments)
        except TypeError:
            continue
        return GaitController(*arguments)

    raise TypeError(
        "GaitController must accept (model, data), (model), or no arguments."
    )


def resolve_controller_step(controller: Any) -> Callable[[Any, Any], Any]:
    for method_name in ("update", "step"):
        method = getattr(controller, method_name, None)
        if callable(method):
            return method

    raise AttributeError(
        "GaitController must provide update(model, data) or step(model, data)."
    )


def reset_controller(
    controller: Any,
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> None:
    reset_method = getattr(controller, "reset", None)
    if not callable(reset_method):
        return

    signature = inspect.signature(reset_method)
    for arguments in ((model, data), (model,), ()):
        try:
            signature.bind(*arguments)
        except TypeError:
            continue
        reset_method(*arguments)
        return

    raise TypeError(
        "Controller reset method must accept (model, data), (model), or ()."
    )


def initialise_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    initial_x: float,
    initial_y: float,
    initial_z: float,
) -> None:
    mujoco.mj_resetData(model, data)

    if model.nq < 7:
        raise RuntimeError(
            "Expected a free-joint robot with at least seven qpos values."
        )

    data.qpos[0:3] = [initial_x, initial_y, initial_z]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)


def run_simulation(
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller: Any,
    evaluator: StabilityEvaluator,
    logger: DataLogger,
    duration: float,
    controller_name: str,
    trial_id: str,
    viewer: Any | None,
    real_time: bool,
    stop_on_success: bool,
) -> None:
    controller_step = resolve_controller_step(controller)

    while data.time < duration:
        wall_start = time.perf_counter()

        controller_step(model, data)
        mujoco.mj_step(model, data)

        evaluator.update(model, data)
        logger.log_evaluator(
            evaluator,
            controller_name=controller_name,
            trial_id=trial_id,
        )

        if viewer is not None:
            viewer.sync()

        if (
            stop_on_success
            and evaluator.get_module_metrics("task")["success"]["task_success"]
        ):
            break

        if real_time:
            elapsed = time.perf_counter() - wall_start
            remaining = float(model.opt.timestep) - elapsed
            if remaining > 0.0:
                time.sleep(remaining)


def print_effective_configuration(
    model: mujoco.MjModel,
    args: argparse.Namespace,
) -> None:
    print("\nEffective experiment configuration")
    print("-" * 48)
    print(f"Scene:                 {args.scene.resolve()}")
    print(f"Duration:              {args.duration:.4f} s")
    print(f"Timestep:              {model.opt.timestep:.6f} s")
    print(
        "Gravity:               "
        f"{np.asarray(model.opt.gravity, dtype=float).tolist()}"
    )
    print(
        "Minimum contact force: "
        f"{METRICS['minimum_contact_force']} N"
    )
    print(
        "Slip-speed threshold:  "
        f"{METRICS['slip_speed_threshold']} m/s"
    )
    print(f"Target rung:            {LADDER['target_rung']}")
    print(
        "Success hold time:     "
        f"{METRICS['success_hold_time']} s"
    )
    print("-" * 48)


def print_final_summary(evaluator: StabilityEvaluator) -> None:
    summary = evaluator.get_summary()

    print("\n" + "=" * 68)
    print("FINAL QUANTITATIVE EVALUATION SUMMARY")
    print("=" * 68)
    print(f"Simulation time:                  {summary['time']:.4f} s")
    print(f"Forward progress:                 {summary['forward_progress']:.4f} m")
    print(f"Absolute lateral drift:           {summary['lateral_drift_abs']:.4f} m")
    print(f"Active contacts:                  {summary['active_contact_count']}")
    print(f"Slipping contacts:                {summary['slipping_contact_count']}")
    print(
        "Minimum grasp singular value:   "
        f"{summary['minimum_grasp_singular_value']:.6f}"
    )
    print(f"Grasp isotropy:                   {summary['grasp_isotropy']:.6f}")
    print(f"Rung advancement:                 {summary['rung_advancement']}")
    print(
        "Successful regrasp count:        "
        f"{summary['successful_regrasp_count']}"
    )
    print(
        "Absolute mechanical work:        "
        f"{summary['absolute_mechanical_work']:.4f} J"
    )
    print(f"Task success:                     {summary['task_success']}")
    print(f"Completion time:                  {summary['completion_time']}")
    print("=" * 68)


def main() -> None:
    args = parse_arguments()
    validate_arguments(args)

    trial_name = (
        args.trial_name
        or f"{args.controller_name}_trial_{args.trial_id}"
    )

    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    apply_simulation_configuration(model, timestep=args.timestep)
    data = mujoco.MjData(model)

    initialise_state(
        model,
        data,
        initial_x=args.initial_x,
        initial_y=args.initial_y,
        initial_z=args.initial_z,
    )

    controller = create_controller(model, data)
    reset_controller(controller, model, data)

    evaluator = create_evaluator()
    evaluator.reset(model, data)

    logger = DataLogger(
        output_directory=args.results_dir,
        store_full_metrics=args.full_metrics,
    )
    logger.reset(
        trial_name=trial_name,
        metadata={
            "scene": str(args.scene.resolve()),
            "controller": args.controller_name,
            "trial_id": str(args.trial_id),
            "duration_limit": float(args.duration),
            "effective_simulation": {
                "timestep": float(model.opt.timestep),
                "gravity": np.asarray(
                    model.opt.gravity,
                    dtype=float,
                ).tolist(),
            },
            "gait_configuration": dict(GAIT),
            "metric_configuration": dict(METRICS),
            "ladder_configuration": dict(LADDER),
            "initial_position": {
                "x": args.initial_x,
                "y": args.initial_y,
                "z": args.initial_z,
            },
        },
    )

    print_effective_configuration(model, args)

    if args.viewer:
        from mujoco import viewer as mujoco_viewer

        with mujoco_viewer.launch_passive(model, data) as passive_viewer:
            run_simulation(
                model=model,
                data=data,
                controller=controller,
                evaluator=evaluator,
                logger=logger,
                duration=args.duration,
                controller_name=args.controller_name,
                trial_id=str(args.trial_id),
                viewer=passive_viewer,
                real_time=args.real_time,
                stop_on_success=args.stop_on_success,
            )
    else:
        run_simulation(
            model=model,
            data=data,
            controller=controller,
            evaluator=evaluator,
            logger=logger,
            duration=args.duration,
            controller_name=args.controller_name,
            trial_id=str(args.trial_id),
            viewer=None,
            real_time=args.real_time,
            stop_on_success=args.stop_on_success,
        )

    output_paths = logger.export_all(basename=trial_name)

    plotter = EvaluationPlotter(
        output_directory=args.figures_dir,
        strict_columns=False,
    )
    logged_data = plotter.load_csv(output_paths["csv"])
    figure_paths = plotter.plot_standard_report(
        logged_data,
        prefix=trial_name,
    )

    print_final_summary(evaluator)

    print("\nSaved data:")
    for label, path in output_paths.items():
        print(f"  {label}: {path}")

    if figure_paths:
        print("\nSaved figures:")
        for label, path in figure_paths.items():
            print(f"  {label}: {path}")


if __name__ == "__main__":
    main()
