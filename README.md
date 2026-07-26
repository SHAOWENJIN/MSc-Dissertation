# Investigating the Influence of Grasp Configuration on the Stability of a Multi-Limbed Robotic Platform in MuJoCo

**MSc Robotics Dissertation Project**  
**University of Manchester**

## Overview

This project investigates the influence of grasp configuration on the stability of a multi-limbed robotic platform in a microgravity environment using the MuJoCo physics simulator.

The robot is a modified Unitree Go2 quadruped equipped with four custom robotic grippers. Before dynamic climbing is investigated, the current stage focuses on establishing a stable four-gripper static grasp and quantitatively evaluating grasp stability through a set of contact-based metrics.

The framework separates robot control from performance evaluation, allowing different grasp configurations to be compared using consistent stability metrics.

---

## Project Objectives

The current implementation aims to:

- Establish a stable four-gripper grasp
- Release the floating base after contact verification
- Evaluate grasp stability during free-base holding
- Record contact and stability metrics
- Generate quantitative plots for analysis

Future work will extend the framework to dynamic climbing and motion planning.

---

## Project Structure

```text
code/
│
├── config.py
│
├── run_grasp_validation.py        # Main simulation entry
│
├── controller/
│   ├── grasp_controller.py        # Static grasp controller
│   └── state_machine.py           # Grasp state transitions
│
├── evaluator/
│   ├── contact_metrics.py         # Contact-force metrics
│   └── stability_evaluator.py     # Global stability metrics
│
├── logger/
│   └── data_logger.py             # Save experiment data
│
├── visualization/
│   └── plotter.py                 # Generate figures
│
└── results/
```

---

## Framework

The experimental workflow is

```text
MuJoCo Simulation
        │
        ▼
Static Grasp Controller
        │
        ▼
Contact Verification
        │
        ▼
Floating Base Release
        │
        ▼
Holding Phase
        │
        ▼
Metrics Evaluation
        │
        ▼
CSV + Summary + Figures
```

Metrics are recorded **only after the robot successfully enters the HOLDING state**, ensuring that all evaluations correspond to genuine free-base grasp stability.

---

## Stability Metrics

The implemented metrics include:

### Contact Persistence

- Active grippers
- Physical contact count
- Contact persistence

### Contact Forces

- Normal contact force
- Tangential contact force
- Force distribution

### Friction Stability

- Friction utilization ratio

### Robot Stability

- Base position drift
- Base orientation drift
- Linear velocity
- Angular velocity

### Control Effort

- Absolute mechanical power
- Absolute mechanical work

---

## Outputs

Each experiment automatically generates

```text
results/
│
├── metrics.csv
├── summary.json
├── summary.txt
├── normal_force.png
├── tangential_force.png
├── friction_utilization.png
├── contact_count.png
├── base_motion.png
├── absolute_mechanical_power.png
└── absolute_mechanical_work.png
```

---

## Requirements

- Python 3.10+
- MuJoCo
- NumPy
- Matplotlib

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Simulation

```bash
python run_grasp_validation.py
```

The simulation will

1. Initialise the robot.
2. Execute the static grasp controller.
3. Verify four-gripper contact.
4. Release the floating base.
5. Record stability metrics.
6. Save results automatically.

---

## Current Progress

Current achievements include:

- Stable four-gripper static grasp
- Floating-base validation
- Contact-based stability evaluation
- Automatic metrics logging
- Automatic figure generation

The next stage of the project will investigate dynamic climbing under microgravity conditions.

---

## Disclaimer

This repository is part of an academic research project. The implementation is under continuous development and primarily intended for research and educational purposes.
