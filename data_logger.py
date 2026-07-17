# data_logger.py
import numpy as np

class DataLogger:
    def __init__(self):
        """
        Initializes data storage arrays for both baseline contact metrics 
        and high-level academic timeline tracking.
        """
        self.data_dict = {
            "time": [],
            "normal_force": [],
            "friction_util": [],
            "singular_value": [],
            "torque_util": [],
            "net_progress": [],
            "velocity_x": [],
            "lateral_drift": [],
            "wrench_margin": [],
            "manipulability": []
        }
        
    def log(self, time_val, normal_f, f_util, s_val, t_util, progress, vel_x, drift, w_margin, manip):
        """
        Appends the computed multi-body data vectors at each step into system memory.
        """
        self.data_dict["time"].append(time_val)
        self.data_dict["normal_force"].append(normal_f)
        self.data_dict["friction_util"].append(f_util)
        self.data_dict["singular_value"].append(s_val)
        self.data_dict["torque_util"].append(t_util)
        self.data_dict["net_progress"].append(progress)
        self.data_dict["velocity_x"].append(vel_x)
        self.data_dict["lateral_drift"].append(drift)
        self.data_dict["wrench_margin"].append(w_margin)
        self.data_dict["manipulability"].append(manip)
        
    def save_to_file(self, filename="simulation_data.npy"):
        """
        Serializes data dict structures into a physical numpy file for post-processing.
        """
        np.save(filename, self.data_dict)
        print(f"-> Simulation telemetry successfully stored to file: '{filename}'")