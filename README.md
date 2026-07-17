# Zero-Gravity Quadrupedal Ladder Climbing Simulation via Pure Force-Control

An academic-grade MuJoCo physics simulation repository investigating the locomotion dynamics and kinematic stability of a quadrupedal robot (Unitree Go2) modified with specialized micro-grippers for space-station ladder climbing under zero-gravity environments ($g = [0, 0, 0]^T$).

---

## Project Overview

Locomotion in microgravity eliminates the stabilizing effects of a support polygon governed by weight. Instead, the robot must rely entirely on **Closed-Loop Kinematic Chains**, where active gripping forces (Gripping Wrench) and contact constraints govern systemic stability. 

This project implements a multi-body dynamic simulation framework in MuJoCo that completely bypasses rigid kinematic position/velocity pinning during locomotion. Instead, it features a **two-phase controller**:
1. **0.0s – 1.5s (Safe Initialization):** Smooth gripper ramping combined with numerical velocity damping to dissipate contact impact energy and avoid penetration explosions.
2. **>= 1.5s (Force-Controlled Trot Gait):** A purely contact-driven gait where stance limbs firmly squeeze the rungs while generating backward thrusting forces to dynamically translate the system Center of Mass (CoM).

---

## Key Academic Metrics

To validate systemic stability, singularity avoidance, and contact efficiency, this framework extracts and logs **10 distinct physical indices** in real-time, categorized into four core academic panels:

1. **Net Forward Progress & Forward Velocity ($V_x$):** Quantifies steady-state translation velocity and tracks the continuous acceleration/deceleration profiles across periodic gait phases.
2. **Lateral Pose Drift (Self-Alignment Error):** Monitors state deviation along the lateral $Y$-axis. Verifies whether the passive/active gripper mechanics can damp out exponential drift under zero-g.
3. **Task-Oriented Wrench Margin:** Maps the proximity of the CoM relative to the structural boundaries of the space ladder. A safety index bounded between $[0, 1]$, where values $>0$ mathematically guarantee zero risk of system toppling or detaching.
4. **Robust Yoshikawa Manipulability Index:** Evaluates the active leg Jacobian matrix ($J_{\text{active}} \in \mathbb{R}^{6 \times 12}$) via **Singular Value Decomposition (SVD)** to explicitly prove that the workspace trajectories bypass any kinematic singularity boundaries.

---

## Repository Structure

```directory
.
├── run_climbing.py          # Master control script managing simulation loops and gait scheduling
├── metrics_calculator.py    # Core physics engine extracting contact forces, SVD Jacobians, and Wrench margins
├── data_logger.py           # Central telemetry buffer managing in-memory logging and .npy serialization
├── plotter.py               # Double-engine visualization suite rendering academic-grade evaluation figures
└── unitree_go2/             # Model description assets directory
    ├── scene.xml            # Combined world body file defining space station module and ladder rails
    ├── go2.xml              # Main robot asset containing joint constraints and motor ranges
    └── assets/              # Underlying .stl meshes and surface textures for rendering
```

---

## Installation & Dependencies
```
Bash

# Clone this repository
git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
cd your-repository-name
```

---

## Install core dependencies
```
pip install numpy matplotlib mujoco
Note: If you are executing the simulation inside an SSH terminal or a headless server container, remember to configure your X11 forwarding window configurations accordingly.
```

---

## How to Run & Results
Execute the main simulation pipeline via the following command:
```
python run_climbing.py
```
1. Terminal Telemetry Summary
Upon termination, the script automatically parses the telemetry dictionary and prints out an academic Data Sheet for rapid paper/slides reporting:

```
======================================================================
             KEY QUANTITATIVE METRICS FOR ACADEMIC REPORT
======================================================================
1. Net Forward Progress:               0.3947 m
2. Peak Forward Velocity (Vx_max):     1.0205 m/s
3. Maximum Lateral Drift (Peak Error):  0.0456 m
4. Minimum Wrench Margin (Stability):   0.6957
5. Mean Yoshikawa Manipulability:       0.012542
======================================================================
```

2. Automated Figure Generation
The pipeline triggers a post-processing rendering process that automatically generates two high-resolution (300 DPI) analytical figures in your workspace directory:

my_grasp_report.png: Evaluates contact-level parameters including Normal Interaction Forces, Friction Conical Surface Utilization (Slip Thresholds), and Actuator Core Torque Utilization limits.

climbing_metrics_timeline.png: A professional 4-panel time-series figure detailing Net Progress, Lateral Self-alignment Drift, Dynamic Wrench Balance Bounds, and Singularity Avoidance Indices over the simulation timeline—perfect for insertion directly into progress report slides or dessertation appendices.
