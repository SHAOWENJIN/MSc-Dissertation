"""
Minimal visual validation runner for the contact-aware grasp controller.

This file deliberately excludes the dissertation evaluator, logger and
plotter.  Its only purpose is to prove that the model can establish and
hold a real ladder grasp before climbing metrics are collected.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
from pathlib import Path

import numpy as np
import pandas as pd


from controller.grasp_controller import StaticLadderGraspController


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene",
        type=Path,
        default=PROJECT_ROOT / "scene.xml",
    )
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--timestep", type=float, default=0.002)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--real-time", action="store_true")
    return parser.parse_args()


def run_loop(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller: StaticLadderGraspController,
    duration: float,
    *,
    viewer=None,
    real_time: bool,
) -> None:
    last_print_second = -1

    while data.time < duration:
        if viewer is not None and not viewer.is_running():
            break

        wall_start = time.perf_counter()

        status = controller.update(model, data)
        mujoco.mj_step(model, data)
        controller.post_step(model, data)

        current_second = int(data.time)
        if current_second != last_print_second:
            last_print_second = current_second

            print(
                f"t={data.time:6.2f} s | "
                f"state={status.state:11s} | "
                f"pinned={status.base_pinned} | "
                f"contacts={status.physical_contacts} | "
                f"finger_segments={status.distinct_fingertips} | "
                f"grippers={status.distinct_grippers}"
            )

            print(
                controller.contact_report(
                    model,
                    data,
                )
            )

        if viewer is not None:
            viewer.sync()

        if real_time:
            remaining = model.opt.timestep - (
                time.perf_counter() - wall_start
            )
            if remaining > 0.0:
                time.sleep(remaining)


def main() -> None:
    args = parse_arguments()
    if not args.scene.exists():
        raise FileNotFoundError(args.scene)
    if args.duration <= 0.0:
        raise ValueError("--duration must be positive.")

    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    data = mujoco.MjData(model)

    model.opt.gravity[:] = [0.0, 0.0, 0.0]
    model.opt.timestep = args.timestep

    controller = StaticLadderGraspController(model)
    controller.reset(model, data)

    print("\nStarting static grasp validation")
    print("The base remains pinned until persistent real contacts are verified.")

    if args.no_viewer:
        run_loop(
            model,
            data,
            controller,
            args.duration,
            real_time=args.real_time,
        )
    else:
        from mujoco import viewer as mujoco_viewer

        with mujoco_viewer.launch_passive(model, data) as passive_viewer:
            run_loop(
                model,
                data,
                controller,
                args.duration,
                viewer=passive_viewer,
                real_time=args.real_time,
            )

    final = controller.status
    print("\nFinal grasp status")
    print("-" * 72)
    print(f"State:                 {final.state}")
    print(f"Base pinned:           {final.base_pinned}")
    print(f"Physical contacts:     {final.physical_contacts}")
    print(f"Distinct fingertips:   {final.distinct_fingertips}")
    print(f"Distinct grippers:     {final.distinct_grippers}")
    print(f"Verified:              {final.verified}")
    print(f"Message:               {final.message}")
    print("-" * 72)

    if not final.verified:
        raise RuntimeError(
            "A verified grasp was not established. "
            "Use the printed IK and contact reports for diagnosis."
        )


if __name__ == "__main__":
    main()
