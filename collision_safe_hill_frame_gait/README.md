# Collision-safe Hill-frame grasp gait

This folder is the corrected, student-facing MuJoCo simulation. It contains a
free-flying Unitree Go2 model performing the gait `RR -> RL -> FR -> FL` on a
truss fixed to a space-structure reference in zero gravity.

The code does not accept a grasp merely because the hand looks close to a
rung. A latch can engage only after MuJoCo detects both hook chains contacting
opposite faces of the same rung. The visible thigh and calf meshes also
participate in collision checking, preventing the rendered limbs from passing
through rods while smaller hidden proxies remain clear.

## Prerequisites

- Python 3.11 or 3.12
- macOS, Linux, or Windows with OpenGL support for the interactive viewer
- A complete clone of this repository. The robot mesh files are shared from
  the repository-level `assets/` directory through `go2_hill.xml`.

## Linux quick start

From the repository root:

```bash
bash collision_safe_hill_frame_gait/setup_linux.sh
bash collision_safe_hill_frame_gait/run_linux.sh
```

The setup script creates the repository-level `.venv` and installs the exact
Python packages required by this folder. The run script checks the scene first
and then opens the interactive MuJoCo simulation in zero gravity.

To run the complete regression separately:

```bash
source .venv/bin/activate
python collision_safe_hill_frame_gait/validate_simulation.py
```

## Manual setup

From the repository root:

### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r collision_safe_hill_frame_gait/requirements.txt
```

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r collision_safe_hill_frame_gait\requirements.txt
```

## Validate before running

From the repository root:

```bash
python collision_safe_hill_frame_gait/validate_simulation.py
```

The validation takes several minutes. A successful run ends with `PASS` and
checks the complete four-leg gait, real two-sided contacts, visible hook
enclosure, forward body displacement, and transient visible-mesh clearance.

## Run the interactive simulation

On macOS, use `mjpython` so MuJoCo can open its native viewer:

```bash
mjpython collision_safe_hill_frame_gait/run_simulation.py \
  --gravity-scale 0.0 \
  --cycles 1 \
  --sequence RR,RL,FR,FL \
  --real-time \
  --hold-viewer-on-exit \
  --trial-name collision_safe_demo
```

On Linux or Windows, use:

```bash
python collision_safe_hill_frame_gait/run_simulation.py \
  --gravity-scale 0.0 \
  --cycles 1 \
  --sequence RR,RL,FR,FL \
  --real-time \
  --hold-viewer-on-exit \
  --trial-name collision_safe_demo
```

Viewer controls:

- Drag: orbit
- Shift-drag: pan
- Mouse wheel or trackpad scroll: zoom

Results are written inside `collision_safe_hill_frame_gait/results/`.

## Important modelling boundary

The robot and structure are separate systems: the robot base has a free joint,
while the truss is fixed to the world. The grasp welds represent idealized
mechanical latches and activate only after verified physical contact. The gait
is trajectory-assisted and should not be presented as an autonomous climbing
controller.

If rung dimensions, robot meshes, gripper geometry, or timing are changed,
rerun `validate_simulation.py`. Do not weaken or remove its clearance and
two-sided-contact assertions.
