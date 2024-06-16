import open3d as o3d
import numpy as np


def visualize_labeled_pcd(pcd, colors):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pcd)
    point_cloud.colors = o3d.utility.Vector3dVector(colors / 255.0)  # 归一化颜色值

    o3d.visualization.draw_geometries([point_cloud])

def get_point_cloud_visualizer(pcd, paint_color=False):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    vis.add_geometry(axis_pcd)
    pts = o3d.geometry.PointCloud()
    pts.points = o3d.utility.Vector3dVector(pcd)
    vis.add_geometry(pts)
    if not paint_color:
        pts.colors = o3d.utility.Vector3dVector(np.ones((pcd.shape[0], 3)))
    return vis

def _run_open3d_visualizer(visualizer):
    visualizer.run()
    visualizer.destroy_window()
    
def visualize_commont_point_cloud(pcd, paint_color=False):
    vis = get_point_cloud_visualizer(pcd, paint_color)
    _run_open3d_visualizer(vis)