# plotter.py
import numpy as np
import matplotlib.pyplot as plt

def plot_metrics(data_filepath="simulation_data.npy", save_img_path="my_grasp_report.png"):
    """
    Renders standard contact and actuator tracking diagnostic profiles.
    """
    try:
        data = np.load(data_filepath, allow_pickle=True).item()
    except FileNotFoundError:
        print(f"Error: Telemetry file '{data_filepath}' not found.")
        return

    time_history = data["time"]
    plt.figure(figsize=(10, 8))
    
    # 1. Contact Normal Force Curve
    plt.subplot(2, 2, 1)
    plt.plot(time_history, data["normal_force"], color='blue', lw=1.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Normal Force (N)')
    plt.title('Individual Contact: Normal Force')
    plt.grid(True, linestyle=':')
    
    # 2. Tangential Friction Utilization
    plt.subplot(2, 2, 2)
    plt.plot(time_history, data["friction_util"], color='red', lw=1.5)
    plt.axhline(y=1.0, color='darkred', linestyle='--', label='Slip Boundary')
    plt.xlabel('Time (s)')
    plt.ylabel('Utilisation Ratio')
    plt.title('Friction Utilisation Profile')
    plt.legend()
    plt.grid(True, linestyle=':')
    
    # 3. Nominal Grasp Matrix SVD
    plt.subplot(2, 2, 3)
    plt.plot(time_history, data["singular_value"], color='purple', lw=1.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Min Singular Value')
    plt.title('Grasp-Matrix Minimal SVD Singularities')
    plt.grid(True, linestyle=':')
    
    # 4. Global Actuator Torque Utilization
    plt.subplot(2, 2, 4)
    plt.plot(time_history, data["torque_util"], color='orange', lw=1.5)
    plt.axhline(y=1.0, color='darkorange', linestyle='--', label='Hardware Bound')
    plt.xlabel('Time (s)')
    plt.ylabel('Max Utilisation Ratio')
    plt.title('Actuator Core Torque Utilisation')
    plt.legend()
    plt.grid(True, linestyle=':')
    
    plt.suptitle("Grasp-Contact Diagnostics Report", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_img_path, dpi=300)
    print(f"-> Successfully saved Grasp-Contact report figure to: '{save_img_path}'")


def plot_timeline_metrics(data_filepath="simulation_data.npy", save_img_path="climbing_metrics_timeline.png"):
    """
    Renders high-quality multi-panel academic figures tracing global trajectories, 
    lateral drift errors, stability indexes, and singularity indices.
    """
    try:
        data = np.load(data_filepath, allow_pickle=True).item()
    except FileNotFoundError:
        return

    t_arr = np.array(data["time"])
    progress_arr = np.array(data["net_progress"])
    vel_arr = np.array(data["velocity_x"])
    drift_arr = np.array(data["lateral_drift"])
    wm_arr = np.array(data["wrench_margin"])
    mani_arr = np.array(data["manipulability"])

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Panel 1: Progress vs Velocity Tracking Profile
    color = 'tab:blue'
    axs[0, 0].set_xlabel('Simulation Time (s)', fontweight='bold')
    axs[0, 0].set_ylabel('Net Forward Progress (m)', color=color, fontweight='bold')
    axs[0, 0].plot(t_arr, progress_arr, color=color, lw=2.5, label='Progress')
    axs[0, 0].tick_params(axis='y', labelcolor=color)
    axs[0, 0].grid(True, linestyle=':')

    ax_vel = axs[0, 0].twinx()
    color = 'tab:red'
    ax_vel.set_ylabel('Forward Velocity Vx (m/s)', color=color, fontweight='bold')
    ax_vel.plot(t_arr, vel_arr, color=color, lw=1.5, alpha=0.8, linestyle='--', label='Velocity')
    ax_vel.tick_params(axis='y', labelcolor=color)
    axs[0, 0].set_title('1. Net Progress and Forward Velocity', fontsize=11, fontweight='bold')

    # Panel 2: Lateral Pose Drift Tracking
    axs[0, 1].plot(t_arr, drift_arr, color='forestgreen', lw=2, label='Lateral Drift')
    axs[0, 1].axhline(y=0.0, color='black', linestyle=':', alpha=0.5)
    axs[0, 1].set_xlabel('Simulation Time (s)', fontweight='bold')
    axs[0, 1].set_ylabel('Lateral Position Y (m)', fontweight='bold')
    axs[0, 1].set_title('2. Lateral Pose Drift (Relative Alignment)', fontsize=11, fontweight='bold')
    axs[0, 1].grid(True, linestyle=':')
    axs[0, 1].legend()

    # Panel 3: Dynamic Wrench Margin Timeline
    axs[1, 0].plot(t_arr, wm_arr, color='darkorange', lw=2, label='Wrench Margin')
    axs[1, 0].set_xlabel('Simulation Time (s)', fontweight='bold')
    axs[1, 0].set_ylabel('Safety Index [0 - 1]', fontweight='bold')
    axs[1, 0].set_title('3. Task-Oriented Wrench Margin', fontsize=11, fontweight='bold')
    axs[1, 0].grid(True, linestyle=':')
    axs[1, 0].legend()

    # Panel 4: Yoshikawa Manipulability Indices Tracking
    axs[1, 1].plot(t_arr, mani_arr, color='purple', lw=2, label='Yoshikawa Index')
    axs[1, 1].set_xlabel('Simulation Time (s)', fontweight='bold')
    axs[1, 1].set_ylabel('Manipulability Measure', fontweight='bold')
    axs[1, 1].set_title('4. Joint Manipulability (Singularity Avoidance)', fontsize=11, fontweight='bold')
    axs[1, 1].grid(True, linestyle=':')
    axs[1, 1].legend()

    plt.suptitle("Zero-Gravity Robotic Ladder Climbing: Dynamic Metrics Timeline", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_img_path, dpi=300)
    print(f"-> Successfully saved Academic Timeline figure to: '{save_img_path}'")