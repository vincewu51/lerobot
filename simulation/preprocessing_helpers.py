import numpy as np

def normalize_data(data, mean=None, std=None):
    """Normalize data to zero mean and unit variance."""
    data = np.array(data, dtype=np.float32)
    if mean is None:
        mean = data.mean(axis=0)
    if std is None:
        std = data.std(axis=0) + 1e-8
    return (data - mean) / std, mean, std

def clip_values(data, min_val=-1.0, max_val=1.0):
    """Clip data to a given range."""
    return np.clip(data, min_val, max_val)

def resample_trajectory(traj, num_points):
    """Resample a trajectory to a fixed number of points."""
    traj = np.array(traj)
    idxs = np.linspace(0, len(traj) - 1, num_points).astype(int)
    return traj[idxs]