#!/usr/bin/env python3
"""Run the Hill-frame gait experiment for 0.8g, 0.7g, 0.6g and 0.5g."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


GRAVITY_LEVELS = (0.8, 0.7, 0.6, 0.5)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--sample-rate", type=float, default=50.0)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--real-time", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    here = Path(__file__).resolve().parent
    runner = here / "run_hill_frame_gait.py"
    scene = here / "scene_hill_frame.xml"
    project_root = here.parent
    output_root = project_root / "results" / "experiment_4_hill_frame"

    for gravity_scale in GRAVITY_LEVELS:
        gravity_tag = str(gravity_scale).replace(".", "_")
        for trial in range(1, args.trials + 1):
            trial_name = f"hill_g_{gravity_tag}_trial_{trial:02d}"
            cmd = [
                sys.executable,
                "-u",
                str(runner),
                "--scene",
                str(scene),
                "--gravity-scale",
                str(gravity_scale),
                "--cycles",
                str(args.cycles),
                "--sample-rate",
                str(args.sample_rate),
                "--trial-name",
                trial_name,
                "--output-dir",
                str(output_root / f"g_{gravity_tag}"),
                "--headless",
            ]
            if args.real_time:
                cmd.append("--real-time")

            print("\n" + "=" * 78)
            print("Running:", " ".join(cmd))
            print("=" * 78)
            subprocess.run(cmd, cwd=project_root, check=True)

    print("\nAll Hill-frame gravity-level trials completed.")


if __name__ == "__main__":
    main()
