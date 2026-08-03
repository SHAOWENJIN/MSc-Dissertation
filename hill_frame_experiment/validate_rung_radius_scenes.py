#!/usr/bin/env python3
"""Validate the three Hill-frame rung-radius scene variants."""

from pathlib import Path
import mujoco


EXPECTED = {
    "baseline": 0.0100,
    "r125": 0.0125,
    "r150": 0.0150,
}


def main():
    here = Path(__file__).resolve().parent

    for label, expected_radius in EXPECTED.items():
        scene = here / f"scene_hill_frame_rung_{label}.xml"
        model = mujoco.MjModel.from_xml_path(str(scene))

        rung_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"rung_{i}")
            for i in range(1, 12)
        ]
        if any(i < 0 for i in rung_ids):
            raise RuntimeError(f"{label}: one or more rung geoms are missing")

        radii = [float(model.geom_size[i, 0]) for i in rung_ids]
        max_error = max(abs(r - expected_radius) for r in radii)

        print(
            f"{label:8s} | scene={scene.name} | "
            f"radius={radii[0]:.4f} m | "
            f"gravity={model.opt.gravity.tolist()} | "
            f"rungs={len(rung_ids)}"
        )

        if max_error > 1e-9:
            raise RuntimeError(
                f"{label}: expected radius {expected_radius}, got {radii}"
            )

    print("\nAll rung-radius scenes loaded and validated successfully.")


if __name__ == "__main__":
    main()
