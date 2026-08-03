#!/usr/bin/env python3
"""
Hill-frame gravity-level report runner for the assisted cyclic crawl baseline.

This script reuses the motion logic from run_assisted_cyclic_crawl_v3.py and adds:
- CSV time-series logging
- JSON summary export
- stage-report plots focused on contact / friction / support metrics

Important:
- This is for the CURRENT ASSISTED gait-demonstration baseline.
- It is suitable for stage reporting and visual update slides.
- It is NOT a claim of fully free-floating dynamic climbing.
- Mechanical work / actuator power are intentionally omitted here because the
  assisted crawler directly sets qpos along an IK path, so those quantities
  would be misleading at this stage.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
import sys
from pathlib import Path

# This file lives in code/hill_frame_experiment/. Add code/ to sys.path
# so the verified controller and assisted crawler can be reused without copies.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from controller.grasp_controller import StaticLadderGraspController
from run_assisted_cyclic_crawl_v3 import AssistedCyclicCrawler, DEFAULT_SCENE, LEGS


PHASE_ORDER = [
    "WAITING_FOR_STATIC_GRASP",
    "OPEN_FL", "WITHDRAW_FL", "TRANSFER_FL", "APPROACH_FL", "CLOSE_FL", "HOLD_FL",
    "OPEN_FR", "WITHDRAW_FR", "TRANSFER_FR", "APPROACH_FR", "CLOSE_FR", "HOLD_FR",
    "OPEN_RL", "WITHDRAW_RL", "TRANSFER_RL", "APPROACH_RL", "CLOSE_RL", "HOLD_RL",
    "OPEN_RR", "WITHDRAW_RR", "TRANSFER_RR", "APPROACH_RR", "CLOSE_RR", "HOLD_RR",
    "BODY_SHIFT",
    "COMPLETE",
]
PHASE_TO_CODE = {name: i for i, name in enumerate(PHASE_ORDER)}


def geom_name(model: mujoco.MjModel, geom_id: int) -> str:
    return (
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id))
        or f"geom_{int(geom_id)}"
    )


def moving_leg_code(leg: str) -> int:
    mapping = {"": 0, None: 0, "FL": 1, "FR": 2, "RL": 3, "RR": 4}
    return mapping.get(leg, 0)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_contact_metrics(model, data, controller):
    rung_ids = set(int(x) for x in controller._rung_geom_ids)
    metrics = {
        leg: {
            "contact_count": 0,
            "normal_force": 0.0,
            "tangential_force": 0.0,
            "friction_utilization_max": 0.0,
            "rung_counts": {},
        }
        for leg in LEGS
    }

    for contact_id in range(int(data.ncon)):
        contact = data.contact[contact_id]
        g1 = int(contact.geom1)
        g2 = int(contact.geom2)
        b1 = int(model.geom_bodyid[g1])
        b2 = int(model.geom_bodyid[g2])

        leg1 = controller._finger_body_to_leg.get(b1)
        leg2 = controller._finger_body_to_leg.get(b2)
        rung1 = g1 in rung_ids
        rung2 = g2 in rung_ids

        if leg1 is not None and rung2:
            leg = leg1
            rung_geom = g2
        elif leg2 is not None and rung1:
            leg = leg2
            rung_geom = g1
        else:
            continue

        wrench = np.zeros(6, dtype=float)
        mujoco.mj_contactForce(model, data, contact_id, wrench)
        fn = max(0.0, float(wrench[0]))
        ft = float(np.linalg.norm(wrench[1:3]))
        mu = max(float(contact.friction[0]), 1e-8)
        util = ft / (mu * fn) if fn > 1e-8 else 0.0

        entry = metrics[leg]
        entry["contact_count"] += 1
        entry["normal_force"] += fn
        entry["tangential_force"] += ft
        entry["friction_utilization_max"] = max(entry["friction_utilization_max"], util)

        rung = geom_name(model, rung_geom)
        entry["rung_counts"][rung] = entry["rung_counts"].get(rung, 0) + 1

    return metrics


class MetricsRecorder:
    def __init__(self, model, data, controller, crawler, output_dir: Path, sample_rate: float):
        self.model = model
        self.data = data
        self.controller = controller
        self.crawler = crawler
        self.output_dir = output_dir
        self.sample_dt = 1.0 / max(sample_rate, 1e-6)
        self.next_sample_time = 0.0
        self.rows = []
        self.base_ref = None
        self.static_grasp_time = None

    def maybe_record(self):
        if float(self.data.time) + 1e-12 < self.next_sample_time:
            return
        self.next_sample_time = float(self.data.time) + self.sample_dt

        if self.base_ref is None and self.crawler.started:
            qadr = self.controller._base_qpos_adr
            self.base_ref = self.data.qpos[qadr : qadr + 3].copy()
            self.static_grasp_time = float(self.data.time)

        status = self.controller.status
        base_qadr = self.controller._base_qpos_adr
        base_dadr = self.controller._base_dof_adr

        base_pos = self.data.qpos[base_qadr : base_qadr + 3].copy()
        base_lin = self.data.qvel[base_dadr : base_dadr + 3].copy()
        base_ang = self.data.qvel[base_dadr + 3 : base_dadr + 6].copy()

        if self.base_ref is None:
            base_disp = np.zeros(3, dtype=float)
        else:
            base_disp = base_pos - self.base_ref

        metrics = get_contact_metrics(self.model, self.data, self.controller)
        total_physical_contacts = sum(metrics[leg]["contact_count"] for leg in LEGS)
        distinct_grippers = sum(1 for leg in LEGS if metrics[leg]["contact_count"] > 0)

        phase = getattr(self.crawler, "phase", "WAITING_FOR_STATIC_GRASP")
        moving_leg = getattr(self.crawler, "moving_leg", "") or ""
        source_rung = getattr(self.crawler, "source_rung", "") or ""
        target_rung = getattr(self.crawler, "target_rung", "") or ""

        source_contact_count = 0
        target_contact_count = 0
        moving_palm = [math.nan, math.nan, math.nan]
        if moving_leg in LEGS:
            rung_counts = metrics[moving_leg]["rung_counts"]
            source_contact_count = int(rung_counts.get(source_rung, 0))
            target_contact_count = int(rung_counts.get(target_rung, 0))
            palm_body = self.controller._palm_body_ids[moving_leg]
            moving_palm = list(np.asarray(self.data.xpos[palm_body], dtype=float))

        row = {
            "time": float(self.data.time),
            "gravity_scale": float(getattr(self.crawler, "_hill_gravity_scale", math.nan)),
            "gravity_z": float(self.model.opt.gravity[2]),
            "phase": phase,
            "phase_code": PHASE_TO_CODE.get(phase, -1),
            "cycle_index": int(getattr(self.crawler, "cycle_index", 0)),
            "completed_steps": int(getattr(self.crawler, "completed_steps", 0)),
            "moving_leg": moving_leg,
            "moving_leg_code": moving_leg_code(moving_leg),
            "source_rung": source_rung,
            "target_rung": target_rung,
            "source_contact_count": source_contact_count,
            "target_contact_count": target_contact_count,
            "status_state": status.state,
            "status_verified": int(bool(status.verified)),
            "physical_contact_count": int(total_physical_contacts),
            "distinct_grippers": int(distinct_grippers),
            "base_x": float(base_pos[0]),
            "base_y": float(base_pos[1]),
            "base_z": float(base_pos[2]),
            "base_dx": float(base_disp[0]),
            "base_dy": float(base_disp[1]),
            "base_dz": float(base_disp[2]),
            "base_linear_speed": float(np.linalg.norm(base_lin)),
            "base_angular_speed": float(np.linalg.norm(base_ang)),
            "moving_palm_x": float(moving_palm[0]),
            "moving_palm_y": float(moving_palm[1]),
            "moving_palm_z": float(moving_palm[2]),
        }

        for leg in LEGS:
            entry = metrics[leg]
            row[f"{leg}_contact_count"] = int(entry["contact_count"])
            row[f"{leg}_normal_force"] = float(entry["normal_force"])
            row[f"{leg}_tangential_force"] = float(entry["tangential_force"])
            row[f"{leg}_friction_utilization"] = float(entry["friction_utilization_max"])

        self.rows.append(row)

    def write_outputs(self, trial_name: str):
        ensure_dir(self.output_dir)
        csv_path = self.output_dir / f"{trial_name}_metrics.csv"
        json_path = self.output_dir / f"{trial_name}_summary.json"
        txt_path = self.output_dir / f"{trial_name}_summary.txt"

        if not self.rows:
            summary = {
                "trial_name": trial_name,
                "message": "No samples were recorded.",
            }
            json_path.write_text(json.dumps(summary, indent=2))
            txt_path.write_text("No samples were recorded.\n")
            return {
                "csv": str(csv_path),
                "json": str(json_path),
                "txt": str(txt_path),
                "plots": [],
            }

        fieldnames = list(self.rows[0].keys())
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)

        summary = self._build_summary(trial_name)
        json_path.write_text(json.dumps(summary, indent=2))

        summary_lines = [
            f"Trial: {trial_name}",
            f"Type: assisted cyclic crawl report",
            f"Static grasp confirmed at: {summary.get('static_grasp_time', 'N/A')}",
            f"Final phase: {summary['final_phase']}",
            f"Cycles completed: {summary['cycles_completed']}",
            f"Leg steps completed: {summary['leg_steps_completed']}",
            f"Recorded duration: {summary['recorded_duration_s']:.3f} s",
            "",
            "Important note:",
            "  This is an assisted gait baseline used for stage reporting.",
            "  It is not a free-floating dynamic climbing validation run.",
            "  Contact/friction/support metrics are meaningful for presentation,",
            "  but actuator work/power are intentionally omitted at this stage.",
            "",
            "Per-leg summary:",
        ]
        for leg in LEGS:
            leg_s = summary["per_leg"][leg]
            summary_lines.extend(
                [
                    f"  {leg}:",
                    f"    mean normal force: {leg_s['mean_normal_force']:.6f}",
                    f"    peak normal force: {leg_s['peak_normal_force']:.6f}",
                    f"    mean tangential force: {leg_s['mean_tangential_force']:.6f}",
                    f"    peak tangential force: {leg_s['peak_tangential_force']:.6f}",
                    f"    peak friction utilization: {leg_s['peak_friction_utilization']:.6f}",
                    f"    contact duty ratio: {leg_s['contact_duty_ratio']:.6f}",
                ]
            )
        txt_path.write_text("\n".join(summary_lines) + "\n")

        plots = self._make_plots(trial_name)
        return {
            "csv": str(csv_path),
            "json": str(json_path),
            "txt": str(txt_path),
            "plots": plots,
        }

    def _build_summary(self, trial_name: str):
        rows = self.rows
        times = np.asarray([row["time"] for row in rows], dtype=float)
        recorded_duration = float(times[-1] - times[0]) if len(times) > 1 else 0.0

        summary = {
            "trial_name": trial_name,
            "type": "hill_frame_assisted_cyclic_crawl_report",
            "gravity_scale": float(getattr(self.crawler, "_hill_gravity_scale", math.nan)),
            "gravity_vector": [float(x) for x in self.model.opt.gravity],
            "static_grasp_time": self.static_grasp_time,
            "final_phase": rows[-1]["phase"],
            "cycles_completed": int(getattr(self.crawler, "cycle_index", 0)),
            "leg_steps_completed": int(getattr(self.crawler, "completed_steps", 0)),
            "recorded_duration_s": recorded_duration,
            "max_base_displacement_xyz": {
                "x": float(np.max(np.abs([row["base_dx"] for row in rows]))),
                "y": float(np.max(np.abs([row["base_dy"] for row in rows]))),
                "z": float(np.max(np.abs([row["base_dz"] for row in rows]))),
            },
            "max_base_linear_speed": float(np.max([row["base_linear_speed"] for row in rows])),
            "max_base_angular_speed": float(np.max([row["base_angular_speed"] for row in rows])),
            "max_distinct_grippers": int(np.max([row["distinct_grippers"] for row in rows])),
            "max_physical_contact_count": int(np.max([row["physical_contact_count"] for row in rows])),
            "per_leg": {},
        }

        n = max(len(rows), 1)
        for leg in LEGS:
            normal = np.asarray([row[f"{leg}_normal_force"] for row in rows], dtype=float)
            tang = np.asarray([row[f"{leg}_tangential_force"] for row in rows], dtype=float)
            util = np.asarray([row[f"{leg}_friction_utilization"] for row in rows], dtype=float)
            cnts = np.asarray([row[f"{leg}_contact_count"] for row in rows], dtype=float)

            summary["per_leg"][leg] = {
                "mean_normal_force": float(np.mean(normal)),
                "peak_normal_force": float(np.max(normal)),
                "mean_tangential_force": float(np.mean(tang)),
                "peak_tangential_force": float(np.max(tang)),
                "peak_friction_utilization": float(np.max(util)),
                "contact_duty_ratio": float(np.sum(cnts > 0) / n),
            }

        return summary

    def _make_plots(self, trial_name: str):
        ensure_dir(self.output_dir)
        plots = []
        t = np.asarray([row["time"] for row in self.rows], dtype=float)

        fig = plt.figure(figsize=(10, 6))
        for leg in LEGS:
            y = np.asarray([row[f"{leg}_normal_force"] for row in self.rows], dtype=float)
            plt.plot(t, y, label=leg)
        plt.xlabel("Time [s]")
        plt.ylabel("Summed normal contact force")
        plt.title("Normal contact force per gripper")
        plt.legend()
        plt.grid(True)
        path = self.output_dir / f"{trial_name}_normal_force.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        plots.append(str(path))

        fig = plt.figure(figsize=(10, 6))
        for leg in LEGS:
            y = np.asarray([row[f"{leg}_tangential_force"] for row in self.rows], dtype=float)
            plt.plot(t, y, label=leg)
        plt.xlabel("Time [s]")
        plt.ylabel("Summed tangential contact force")
        plt.title("Tangential contact force per gripper")
        plt.legend()
        plt.grid(True)
        path = self.output_dir / f"{trial_name}_tangential_force.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        plots.append(str(path))

        fig = plt.figure(figsize=(10, 6))
        for leg in LEGS:
            y = np.asarray([row[f"{leg}_friction_utilization"] for row in self.rows], dtype=float)
            plt.plot(t, y, label=leg)
        plt.xlabel("Time [s]")
        plt.ylabel(r"max $|f_t| / (\mu f_n)$")
        plt.title("Friction utilization per gripper")
        plt.legend()
        plt.grid(True)
        path = self.output_dir / f"{trial_name}_friction_utilization.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        plots.append(str(path))

        fig = plt.figure(figsize=(10, 6))
        total_contacts = np.asarray([row["physical_contact_count"] for row in self.rows], dtype=float)
        distinct_grippers = np.asarray([row["distinct_grippers"] for row in self.rows], dtype=float)
        src_contacts = np.asarray([row["source_contact_count"] for row in self.rows], dtype=float)
        tgt_contacts = np.asarray([row["target_contact_count"] for row in self.rows], dtype=float)
        plt.plot(t, total_contacts, label="physical_contact_count")
        plt.plot(t, distinct_grippers, label="distinct_grippers")
        plt.plot(t, src_contacts, label="moving_leg_source_rung_contacts")
        plt.plot(t, tgt_contacts, label="moving_leg_target_rung_contacts")
        plt.xlabel("Time [s]")
        plt.ylabel("Count")
        plt.title("Support / contact transition metrics")
        plt.legend()
        plt.grid(True)
        path = self.output_dir / f"{trial_name}_contact_support.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        plots.append(str(path))

        fig = plt.figure(figsize=(10, 6))
        dx = np.asarray([row["base_dx"] for row in self.rows], dtype=float)
        dy = np.asarray([row["base_dy"] for row in self.rows], dtype=float)
        dz = np.asarray([row["base_dz"] for row in self.rows], dtype=float)
        plt.plot(t, dx, label="base_dx")
        plt.plot(t, dy, label="base_dy")
        plt.plot(t, dz, label="base_dz")
        plt.xlabel("Time [s]")
        plt.ylabel("Displacement [m]")
        plt.title("Base displacement from grasp-confirmed reference")
        plt.legend()
        plt.grid(True)
        path = self.output_dir / f"{trial_name}_base_motion.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        plots.append(str(path))

        fig = plt.figure(figsize=(12, 6))
        phase_codes = np.asarray([row["phase_code"] for row in self.rows], dtype=float)
        plt.step(t, phase_codes, where="post")
        yticks = sorted(set(int(v) for v in phase_codes if v >= 0))
        ylabels = [PHASE_ORDER[i] if i < len(PHASE_ORDER) else str(i) for i in yticks]
        plt.yticks(yticks, ylabels)
        plt.xlabel("Time [s]")
        plt.ylabel("Phase")
        plt.title("Phase timeline")
        plt.grid(True)
        path = self.output_dir / f"{trial_name}_phase_timeline.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        plots.append(str(path))

        fig = plt.figure(figsize=(10, 6))
        mx = np.asarray([row["moving_palm_x"] for row in self.rows], dtype=float)
        mz = np.asarray([row["moving_palm_z"] for row in self.rows], dtype=float)
        valid = np.isfinite(mx) & np.isfinite(mz)
        if np.any(valid):
            plt.plot(mx[valid], mz[valid])
        plt.xlabel("Moving-palm x [m]")
        plt.ylabel("Moving-palm z [m]")
        plt.title("Moving-leg palm x-z path")
        plt.grid(True)
        path = self.output_dir / f"{trial_name}_moving_palm_path_xz.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        plots.append(str(path))

        return plots


def parse_args():
    parser = argparse.ArgumentParser(description="Hill-frame assisted cyclic crawl report runner")
    parser.add_argument("--scene", type=Path, default=Path(__file__).resolve().parent / "scene_hill_frame.xml")
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--gravity-scale", type=float, choices=(0.8, 0.7, 0.6, 0.5), required=True)
    parser.add_argument("--sequence", default="RR,RL,FR,FL")
    parser.add_argument("--approach-clearance", type=float, default=0.055)
    parser.add_argument("--open-time", type=float, default=1.5)
    parser.add_argument("--lift-time", type=float, default=2.0)
    parser.add_argument("--transfer-time", type=float, default=3.0)
    parser.add_argument("--lower-time", type=float, default=2.0)
    parser.add_argument("--close-time", type=float, default=1.5)
    parser.add_argument("--hold-time", type=float, default=1.0)
    parser.add_argument("--body-shift-time", type=float, default=3.0)
    parser.add_argument("--duration", type=float, default=180.0)
    parser.add_argument("--status-rate", type=float, default=2.0)
    parser.add_argument("--sample-rate", type=float, default=50.0)
    parser.add_argument("--trial-name", default="assisted_cyclic_report")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "experiment_4_hill_frame")
    parser.add_argument("--real-time", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--hold-viewer-on-exit", action="store_true")
    return parser.parse_args()


def run(args):
    scene = args.scene.expanduser().resolve()
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    gravity_magnitude = 9.81 * args.gravity_scale
    model.opt.gravity[:] = [0.0, 0.0, -gravity_magnitude]
    print(
        f"Hill-frame gravity: scale={args.gravity_scale:.1f}g, "
        f"vector={model.opt.gravity.tolist()} m/s^2"
    )

    static = StaticLadderGraspController(model)
    static.reset(model, data)

    sequence = tuple(part.strip().upper() for part in args.sequence.split(","))
    if sorted(sequence) != sorted(LEGS):
        raise ValueError("--sequence must contain FL, FR, RL and RR exactly once.")

    crawler = AssistedCyclicCrawler(
        model,
        data,
        static,
        sequence=sequence,
        cycles=max(1, args.cycles),
        approach_clearance=args.approach_clearance,
        open_time=args.open_time,
        lift_time=args.lift_time,
        transfer_time=args.transfer_time,
        lower_time=args.lower_time,
        close_time=args.close_time,
        hold_time=args.hold_time,
        body_shift_time=args.body_shift_time,
    )

    crawler._hill_gravity_scale = float(args.gravity_scale)

    output_dir = args.output_dir.expanduser().resolve() / args.trial_name
    ensure_dir(output_dir)
    recorder = MetricsRecorder(model, data, static, crawler, output_dir, args.sample_rate)

    print("\nStarting Hill-frame assisted cyclic crawl report run")
    print("-" * 72)
    print(f"Scene:       {scene}")
    print(f"Trial name:  {args.trial_name}")
    print(f"Output dir:  {output_dir}")
    print(f"Cycles:      {args.cycles}")
    print(f"Sequence:    {sequence}")
    print(f"Sample rate: {args.sample_rate:.1f} Hz")
    print("-" * 72)

    start_wall = time.perf_counter()
    next_report = 0.0

    def step_once():
        nonlocal next_report
        if not crawler.started:
            static.update(model, data)
            mujoco.mj_step(model, data)
            static.post_step(model, data)

            status = static.status
            if status.state == "HOLDING":
                print(
                    f"\nStatic grasp confirmed at t={data.time:.3f} s: "
                    f"contacts={status.physical_contacts}, "
                    f"grippers={status.distinct_grippers}. "
                    "Starting assisted cyclic crawl with metrics."
                )
                crawler.begin()
        else:
            crawler.apply()
            mujoco.mj_step(model, data)
            crawler.post_step()

        recorder.maybe_record()

        if data.time >= next_report:
            if crawler.started:
                crawler.report()
            else:
                st = static.status
                print(
                    f"t={data.time:7.3f} | static_state={st.state:10s} | "
                    f"contacts={st.physical_contacts} | grippers={st.distinct_grippers}"
                )
            next_report = data.time + 1.0 / max(args.status_rate, 0.1)

        if args.real_time:
            target = start_wall + data.time
            delay = target - time.perf_counter()
            if delay > 0:
                time.sleep(delay)

    if args.headless:
        while data.time < args.duration and not crawler.finished:
            step_once()
    else:
        from mujoco import viewer as mj_viewer
        with mj_viewer.launch_passive(model, data) as viewer:
            # Start with a wide Hill-frame overview. The user can switch back
            # to the free camera from the MuJoCo viewer camera menu.
            overview_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_CAMERA, "hill_overview"
            )
            if overview_id >= 0:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = overview_id
            else:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                viewer.cam.lookat[:] = np.array([0.0, 0.0, -0.6])
                viewer.cam.distance = 5.5
                viewer.cam.azimuth = 135.0
                viewer.cam.elevation = -20.0

            while viewer.is_running() and data.time < args.duration and not crawler.finished:
                step_once()
                viewer.sync()
            if args.hold_viewer_on_exit:
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(0.02)

    crawler.final_result()
    outputs = recorder.write_outputs(args.trial_name)

    print("\nReport outputs")
    print("-" * 72)
    print(f"CSV:          {outputs['csv']}")
    print(f"Summary JSON: {outputs['json']}")
    print(f"Summary text: {outputs['txt']}")
    for plot_path in outputs["plots"]:
        print(f"Plot:         {plot_path}")
    print("-" * 72)


def main():
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
