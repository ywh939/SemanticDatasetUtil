import open3d
import numpy as np

def load_lidar_data(lidar_filename):
    return np.asarray(open3d.io.read_point_cloud(str(lidar_filename)).points, dtype=np.float32)

def load_lidar_bin_data(lidar_filename):
    return np.fromfile(str(lidar_filename), dtype=np.float32).reshape(-1, 3)

def convert_pcd_to_bin(lidar_filename):
    pcd = load_lidar_data(lidar_filename)
    bin_file_name = str(lidar_filename).replace('.pcd', '.bin')
    pcd.tofile(bin_file_name)