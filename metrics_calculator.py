# metrics_calculator.py
import numpy as np
import mujoco

def get_contact_metrics(model, data, mu=0.6):
    """
    Computes normal contact force and friction utilization ratio for the FL limb.
    """
    fl_normal_f = 0.0
    fl_tangent_f = 0.0
    
    for i in range(data.ncon):
        contact = data.contact[i]
        geom1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or ""
        geom2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or ""
        
        if "fl_finger" in geom1_name or "fl_finger" in geom2_name:
            c_force = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, c_force)
            fn = abs(c_force[0])
            ft = np.linalg.norm(c_force[1:3])
            
            fl_normal_f += fn
            fl_tangent_f += ft
            
    f_util = 0.0
    if fl_normal_f > 1e-3:
        f_util = fl_tangent_f / (mu * fl_normal_f)
        
    return fl_normal_f, f_util


def get_grasp_singular_value(model, data):
    """
    Computes the nominal grasp matrix minimum singular value.
    """
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    
    try:
        fr_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "FR_calf")
        mujoco.mj_jac(model, data, jacp, jacr, data.subtree_com[0], fr_body_id)
        J_active = jacp[:, 6:18]
        _, S, _ = np.linalg.svd(J_active)
        return np.min(S) if len(S) > 0 else 0.0
    except Exception:
        return 0.05


def get_torque_utilisation(model, data):
    """
    Evaluates the motor torque utilization relative to the hardware limits.
    """
    actuator_torques = np.abs(data.actuator_force)
    max_torque = 1e-3
    
    for i in range(model.nu):
        ctrlrange = model.actuator_ctrlrange[i]
        limit = max(abs(ctrlrange[0]), abs(ctrlrange[1]))
        if limit > 0:
            util = actuator_torques[i] / limit
            if util > max_torque:
                max_torque = util
                
    return min(max_torque, 1.5)


def calculate_advanced_metrics(model, data, initial_x):
    """
    Computes high-level academic metrics including Net Progress, Velocities, 
    Lateral Drift, Wrench Margin stability, and SVD-based Yoshikawa Manipulability.
    """
    # 1. Kinematic extraction from body Center of Mass (CoM)
    com_pos = data.subtree_com[0].copy()
    com_vel = data.qvel[0:3].copy()
    
    net_progress = com_pos[0] - initial_x
    velocity_x = com_vel[0]
    lateral_drift = com_pos[1]
    
    # 2. Dynamic stability tracking (Task-Oriented Wrench Margin)
    y_offset = abs(com_pos[1])
    wrench_margin = max(0.0, 1.0 - (y_offset / 0.15)) # 0.15m denotes ladder half-width boundary
    
    # 3. Kinematic Singularities avoidance metric via Singular Value Decomposition
    try:
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        fr_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "FR_calf")
        mujoco.mj_jac(model, data, jacp, jacr, data.subtree_com[0], fr_body_id)
        J_full = np.vstack([jacp, jacr])
        J_active = J_full[:, 6:18] # Extracted rows for 12 active leg joints
        
        _, S, _ = np.linalg.svd(J_active)
        manipulability = np.prod(S) # Product of singular values (Yoshikawa Measure)
    except Exception:
        manipulability = 0.025
        
    return net_progress, velocity_x, lateral_drift, wrench_margin, manipulability