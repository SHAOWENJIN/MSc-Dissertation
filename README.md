# MSc-Dissertation
# Zero-Gravity Quadrupedal Ladder Climbing Simulation via Pure Force-Control

An academic-grade MuJoCo physics simulation repository investigating the locomotion dynamics and kinematic stability of a quadrupedal robot (Unitree Go2) modified with specialized micro-grippers for space-station ladder climbing under zero-gravity environments ($g = [0, 0, 0]^T$).

---

## 🌌 Project Overview

Locomotion in microgravity eliminates the stabilizing effects of a support polygon governed by weight. Instead, the robot must rely entirely on **Closed-Loop Kinematic Chains**, where active gripping forces (Gripping Wrench) and contact constraints govern systemic stability. 

This project implements a multi-body dynamic simulation framework in MuJoCo that completely bypasses rigid kinematic position/velocity pinning during locomotion. Instead, it features a **two-phase controller**:
1. **0.0s – 1.5s (Safe Initialization):** Smooth gripper ramping combined with numerical velocity damping to dissipate contact impact energy and avoid penetration explosions.
2. **>= 1.5s (Force-Controlled Trot Gait):** A purely contact-driven gait where stance limbs firmly squeeze the rungs while generating backward thrusting forces to dynamically translate the system Center of Mass (CoM).

---

## 📊 Key Academic Metrics

To validate systemic stability, singularity avoidance, and contact efficiency, this framework extracts and logs **10 distinct physical indices** in real-time, categorized into four core academic panels:

1. **Net Forward Progress & Forward Velocity ($V_x$):** Quantifies steady-state translation velocity and tracks the continuous acceleration/deceleration profiles across periodic gait phases.
2. **Lateral Pose Drift (Self-Alignment Error):** Monitors state deviation along the lateral $Y$-axis. Verifies whether the passive/active gripper mechanics can damp out exponential drift under zero-g.
3. **Task-Oriented Wrench Margin:** Maps the proximity of the CoM relative to the structural boundaries of the space ladder. A safety index bounded between $[0, 1]$, where values $>0$ mathematically guarantee zero risk of system toppling or detaching.
4. **Robust Yoshikawa Manipulability Index:** Evaluates the active leg Jacobian matrix ($J_{\text{active}} \in \mathbb{R}^{6 \times 12}$) via **Singular Value Decomposition (SVD)** to explicitly prove that the workspace trajectories bypass any kinematic singularity boundaries.

---

## 📂 Repository Structure

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
