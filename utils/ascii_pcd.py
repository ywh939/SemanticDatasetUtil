import numpy as np

def read_ascii_pcd(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    points = []
    header = True
    for line in lines:
        if header:
            if line.startswith("DATA"):
                header = False
        else:
            points.append([float(value) for value in line.strip().split()])

    return np.array(points)