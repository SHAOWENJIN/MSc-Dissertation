# run_climbing.py
import os
os.environ["XDG_SESSION_TYPE"] = "x11"
os.environ["GDK_BACKEND"] = "x11"

import mujoco
import mujoco.viewer
import numpy as np
import time

# Importing structured submodules
import metrics_calculator as mc
from data_logger import DataLogger
import plotter

print("==============================================================")
print("Starting Zero-Gravity Safe Initialization & Force-Controlled Climbing...")
print("==============================================================")

# Load XML core model
model = mujoco.MjModel.from_xml_path('/home/student49/00Dissertation/mujoco_menagerie/unitree_go2/scene.xml')
data = mujoco.MjData(model)

# Strictly enforce zero gravity environment configurations
model.opt.gravity = [0.0, 0.0, 0.0]  

logger = DataLogger()

# Direct actuator mapping identifiers
leg_actuators = [
    "FL_hip", "FL_thigh", "FL_calf",
    "FR_hip", "FR_thigh", "FR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "RR_hip", "RR_thigh", "RR_calf"
]
leg_actuator_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in leg_actuators]

gripper_actuators = [
    "fl_gripper_L1", "fl_gripper_L2", "fl_gripper_L3", "fl_gripper_R1", "fl_gripper_R2", "fl_gripper_R3",
    "fr_gripper_L1", "fr_gripper_L2", "fr_gripper_L3", "fr_gripper_R1", "fr_gripper_R2", "fr_gripper_R3",
    "rl_gripper_L1", "rl_gripper_L2", "rl_gripper_L3", "rl_gripper_R1", "rl_gripper_R2", "rl_gripper_R3",
    "rr_gripper_L1", "rr_gripper_L2", "rr_gripper_L3", "rr_gripper_R1", "rr_gripper_R2", "rr_gripper_R3"
]
gripper_actuator_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in gripper_actuators]

# Locomotion gait profile constants
GAIT_FREQ = 0.5        
SWING_AMP = 0.20       
PUSH_AMP = 0.15        

with mujoco.viewer.launch_passive(model, data) as viewer:
    # Kinematic spatial pre-alignment initialization
    data.qpos[0] = -0.50   # Center of Mass starting coordinate (X-axis)
    data.qpos[1] = 0.005   # Lateral centering offset (Y-axis)
    data.qpos[2] = 0.25    # Vertical clearance boundary (Z-axis)
    data.qpos[3:7] = [1, 0, 0, 0] 
    
    init_angles = {
        "FL": [-0.29, 0.85, -1.65], "FR": [0.15, 0.85, -1.65],
        "RL": [-0.29, 0.85, -1.65], "RR": [0.15, 0.85, -1.65]
    }
    
    initial_x = data.qpos[0]
    
    while viewer.is_running():
        step_start = time.time()
        time_now = data.time
        
        # ------------------------------------------------------------
        # Phase A: Nominal Smooth Gripper Engaging Phase (0.0s - 1.5s)
        # ------------------------------------------------------------
        if time_now < 1.5:
            angles = init_angles
            clamp_force = min(5.0, (time_now / 1.5) * 5.0)
            gripper_cmds = {"FL": clamp_force, "FR": clamp_force, "RL": clamp_force, "RR": clamp_force}
            
            # Attenuate early shock velocities to eliminate penetrations
            data.qvel[0:3] *= 0.8  
            data.qvel[3:6] *= 0.8  
            
        # ------------------------------------------------------------
        # Phase B: Direct Contact Force Locomotion Phase (>= 1.5s)
        # ------------------------------------------------------------
        else:
            climb_time = time_now - 1.5
            phase = np.sin(2 * np.pi * GAIT_FREQ * climb_time)
            
            angles = {
                "FL": [-0.29, 0.85, -1.65], "FR": [0.15, 0.85, -1.65],
                "RL": [-0.29, 0.85, -1.65], "RR": [0.15, 0.85, -1.65]
            }
            gripper_cmds = {"FL": 0.0, "FR": 0.0, "RL": 0.0, "RR": 0.0}
            
            # Trot Kinematic Phase 1: FL & RR active stance, FR & RL active swing
            if phase > 0:
                angles["FR"][1] -= phase * SWING_AMP
                angles["FR"][2] += phase * 0.15
                angles["RL"][1] -= phase * SWING_AMP
                angles["RL"][2] += phase * 0.15
                gripper_cmds["FR"] = -2.0  
                gripper_cmds["RL"] = -2.0
                
                angles["FL"][1] += phase * PUSH_AMP
                angles["FL"][2] -= phase * 0.05
                angles["RR"][1] += phase * PUSH_AMP
                angles["RR"][2] -= phase * 0.05
                gripper_cmds["FL"] = 5.0   
                gripper_cmds["RR"] = 5.0
                
            # Trot Kinematic Phase 2: FR & RL active stance, FL & RR active swing
            else:
                angles["FL"][1] -= abs(phase) * SWING_AMP
                angles["FL"][2] += abs(phase) * 0.15
                angles["RR"][1] -= abs(phase) * SWING_AMP
                angles["RR"][2] += abs(phase) * 0.15
                gripper_cmds["FL"] = -2.0
                gripper_cmds["RR"] = -2.0
                
                angles["FR"][1] += abs(phase) * PUSH_AMP
                angles["FR"][2] -= abs(phase) * 0.05
                angles["RL"][1] += abs(phase) * PUSH_AMP
                angles["RL"][2] -= abs(phase) * 0.05
                gripper_cmds["FR"] = 5.0
                gripper_cmds["RL"] = 5.0

        # ------------------------------------------------------------
        # Multi-Joint Closed-loop Actuation (PD Position Servo)
        # ------------------------------------------------------------
        for idx, act_name in enumerate(leg_actuators):
            leg_prefix = act_name.split("_")[0]
            joint_type = act_name.split("_")[1]
            sub_id = 0 if joint_type == "hip" else (1 if joint_type == "thigh" else 2)
            
            target_angle = angles[leg_prefix][sub_id]
            actuator_id = leg_actuator_ids[idx]
            joint_id = model.actuator_trnid[actuator_id, 0]
            current_angle = data.qpos[model.jnt_qposadr[joint_id]]
            
            data.ctrl[actuator_id] = 100.0 * (target_angle - current_angle)

        for g_idx, g_act_name in enumerate(gripper_actuators):
            leg_prefix = g_act_name.split("_")[0].upper()
            data.ctrl[gripper_actuator_ids[g_idx]] = gripper_cmds[leg_prefix]

        # ------------------------------------------------------------
        # Module-Based Real-time Telemetry Calculations & Logging
        # ------------------------------------------------------------
        fn, f_util = mc.get_contact_metrics(model, data)
        sing_val = mc.get_grasp_singular_value(model, data)
        t_util = mc.get_torque_utilisation(model, data)
        
        # Pull high-level advanced metrics from calculator
        progress, vel_x, drift, w_margin, manip = mc.calculate_advanced_metrics(model, data, initial_x)
        
        # Log all 10 tracked metrics into centralized data logger
        logger.log(time_now, fn, f_util, sing_val, t_util, progress, vel_x, drift, w_margin, manip)

        # Step full multi-body system equations of motion
        mujoco.mj_step(model, data)
        viewer.sync()
        
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

# ------------------------------------------------------------
# Post-Simulation Triggers: File Export, Rendering, Data Sheet Printing
# ------------------------------------------------------------
print("\nSimulation loops completed.")
logger.save_to_file("simulation_data.npy")

# Trigger plot calls from standalone plotter module
plotter.plot_metrics("simulation_data.npy", "my_grasp_report.png")
plotter.plot_timeline_metrics("simulation_data.npy", "climbing_metrics_timeline.png")

# Fetch terminal logging array for end-of-run data logging summary
data_dict = logger.data_dict
print("\n" + "="*70)
print("             KEY QUANTITATIVE METRICS FOR ACADEMIC REPORT")
print("="*70)
print(f"1. Net Forward Progress:               {data_dict['net_progress'][-1]:.4f} m")
print(f"2. Peak Forward Velocity (Vx_max):     {np.max(data_dict['velocity_x']):.4f} m/s")
print(f"3. Maximum Lateral Drift (Peak Error):  {np.max(np.abs(data_dict['lateral_drift'])):.4f} m")
print(f"4. Minimum Wrench Margin (Stability):   {np.min(data_dict['wrench_margin']):.4f}")
print(f"5. Mean Yoshikawa Manipulability:       {np.mean(data_dict['manipulability']):.6f}")
print("="*70 + "\n")