#!/usr/bin/env python3
"""Load the Hill-frame scene and print the important model properties."""

from pathlib import Path
import mujoco


def main():
    scene = Path(__file__).resolve().parent / "scene_hill_frame.xml"
    model = mujoco.MjModel.from_xml_path(str(scene))

    print("Scene loaded successfully")
    print("Scene:", scene)
    print("Gravity from XML:", model.opt.gravity.tolist())
    print("Bodies:", model.nbody)
    print("Geometries:", model.ngeom)

    required = (
        ("earth_reference", mujoco.mjtObj.mjOBJ_BODY),
        ("orbital_station_visual", mujoco.mjtObj.mjOBJ_BODY),
        ("ladder_structure", mujoco.mjtObj.mjOBJ_BODY),
        ("rung_5", mujoco.mjtObj.mjOBJ_GEOM),
        ("rung_8", mujoco.mjtObj.mjOBJ_GEOM),
    )
    for name, obj_type in required:
        obj_id = mujoco.mj_name2id(model, obj_type, name)
        print(f"{name}: id={obj_id}")
        if obj_id < 0:
            raise RuntimeError(f"Required object not found: {name}")


if __name__ == "__main__":
    main()
