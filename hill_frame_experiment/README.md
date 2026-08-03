# Rung-Radius Comparison Experiment

This package adds a controlled comparison of ladder-rung radius while preserving:

- the working `go2_hill.xml` asset path;
- the Hill-frame Earth/station visual scene;
- rung positions and spacing;
- robot model and controller;
- gravity level;
- gait sequence and phase timing.

## Conditions

| Condition | Radius | Diameter | Relative to baseline |
|---|---:|---:|---:|
| baseline | 0.0100 m | 0.0200 m | 1.00× |
| r125 | 0.0125 m | 0.0250 m | 1.25× |
| r150 | 0.0150 m | 0.0300 m | 1.50× |

Only the rung-cylinder radius is changed. Rung half-length remains 0.15 m.

## Files to copy

Copy these files into:

```text
~/00Dissertation/code/hill_frame_experiment/
```

Files:

```text
scene_hill_frame_rung_baseline.xml
scene_hill_frame_rung_r125.xml
scene_hill_frame_rung_r150.xml
validate_rung_radius_scenes.py
run_rung_radius_batch.py
```

Do not replace:

```text
scene_hill_frame.xml
go2_hill.xml
run_hill_frame_gait.py
```

## Validate

```bash
cd ~/00Dissertation/code
python3 hill_frame_experiment/validate_rung_radius_scenes.py
```

## Visual check for each radius at 0.8g

Baseline:

```bash
python3 -u hill_frame_experiment/run_hill_frame_gait.py   --scene hill_frame_experiment/scene_hill_frame_rung_baseline.xml   --gravity-scale 0.8   --cycles 1   --trial-name rung_baseline_visual   --sample-rate 50   --real-time   --hold-viewer-on-exit
```

1.25×:

```bash
python3 -u hill_frame_experiment/run_hill_frame_gait.py   --scene hill_frame_experiment/scene_hill_frame_rung_r125.xml   --gravity-scale 0.8   --cycles 1   --trial-name rung_r125_visual   --sample-rate 50   --real-time   --hold-viewer-on-exit
```

1.50×:

```bash
python3 -u hill_frame_experiment/run_hill_frame_gait.py   --scene hill_frame_experiment/scene_hill_frame_rung_r150.xml   --gravity-scale 0.8   --cycles 1   --trial-name rung_r150_visual   --sample-rate 50   --real-time   --hold-viewer-on-exit
```

## Batch comparison

One trial per radius at 0.8g:

```bash
python3 -u hill_frame_experiment/run_rung_radius_batch.py   --gravity-scale 0.8   --trials 1   --cycles 1   --sample-rate 50
```

Five trials per radius:

```bash
python3 -u hill_frame_experiment/run_rung_radius_batch.py   --gravity-scale 0.8   --trials 5   --cycles 1   --sample-rate 50
```

Results are written to:

```text
~/00Dissertation/code/results/experiment_rung_radius/g_0_8/
```
