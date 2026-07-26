"""Central configuration for the climbing simulation and evaluation."""

from __future__ import annotations

SIMULATION = {
    "gravity": (0.0, 0.0, 0.0),
    "dt": 0.002,
    "simulation_time": 30.0,
    "viewer": True,
}

ROBOT = {
    "name": "Unitree Go2",
    "mass": 15.0,
    "number_of_legs": 4,
    "joints_per_leg": 3,
    "leg_actuators": 12,
    "finger_actuators": 24,
    "total_actuators": 36,
}

LADDER = {
    "rung_spacing": 0.25,
    "ladder_width": 0.30,
    "friction": 0.6,
    "target_rung": "rung_11",
}

GAIT = {
    "frequency": 0.5,
    "swing_amplitude": 0.20,
    "push_amplitude": 0.15,
    "stance_force": 5.0,
    "release_force": -2.0,
    "initialisation_duration": 1.5,
}

CONTROL = {
    "kp": 100.0,
    "kd": 2.0,
}

METRICS = {
    "minimum_contact_force": 1e-6,
    "slip_speed_threshold": 1e-4,
    "maximum_friction_ratio": 0.8,
    "maximum_pose_drift": 0.05,
    "maximum_torque_ratio": 0.90,
    "contact_loss_timeout": 0.05,
    "success_hold_time": 0.25,
}

PATHS = {
    "results": "results",
    "figures": "figures",
}
