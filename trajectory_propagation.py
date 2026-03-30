import numpy as np
import pandas as pd
import torch
import h5py
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation as R

import os
import sys
PATH = os.getcwd()
module_dir = os.path.abspath(f"{PATH}/../sam-3d-objects/notebook/")
if module_dir not in sys.path:
    sys.path.append(module_dir)

from inference import (
    Inference, load_image, load_masks, display_image
)
from alignment_pipeline import apply_moge_transform

def get_motion_matrices(csv_path: str, object_id: int):
    """
    Extracts relative motion matrices from the DynoSAM motion log for a specific object.
    
    Args:
        csv_path (str): Path to 'frontend_object_motion_log.csv'.
        object_id (int): The target object ID.
        
    Returns:
        list: A list of 4x4 numpy arrays representing the relative motion frame-to-frame.
    """
    df_motion = pd.read_csv(csv_path)
    df_obj = df_motion[df_motion["object_id"] == object_id].sort_values("frame_id")
    
    matrices = []
    for _, row in df_obj.iterrows():
        # Extract Translation
        t = np.array([row['tx'], row['ty'], row['tz']])
        
        # Extract Rotation from Quaternion [x, y, z, w]
        quat = [row['qx'], row['qy'], row['qz'], row['qw']]
        r_mat = R.from_quat(quat).as_matrix()
        
        # Construct 4x4 Matrix
        T = np.eye(4)
        T[:3, :3] = r_mat
        T[:3, 3] = t
        matrices.append(T)
        
    return matrices

def get_pose_trajectories(csv_path: str, object_id: int):
    """
    Extracts the absolute Ground Truth and Estimated trajectories for the object.
    Assumes columns: 'tx', 'ty', 'tz' (Estimated) and 'gt_tx', 'gt_ty', 'gt_tz' (Ground Truth).
    """
    df_pose = pd.read_csv(csv_path)
    
    # Filter by object_id if applicable, otherwise use the whole file
    if "object_id" in df_pose.columns:
        df_pose = df_pose[df_pose["object_id"] == object_id].sort_values("frame_id")
    
    try:
        est_traj = df_pose[['tx', 'ty', 'tz']].values
        gt_traj = df_pose[['gt_tx', 'gt_ty', 'gt_tz']].values
    except KeyError as e:
        print(f"Warning: Missing expected pose columns in {csv_path}: {e}")
        # Fallback to zeros if ground truth is missing to prevent crashing
        est_traj = np.zeros((len(df_pose), 3))
        gt_traj = np.zeros((len(df_pose), 3))
        
    return est_traj, gt_traj


def get_sparse_features(csv_path: str, object_id: int, num_frames: int):
    """
    Extracts DynoSAM sparse feature points frame-by-frame for a specific object.
    """
    df_features = pd.read_csv(csv_path)
    df_obj = df_features[df_features["object_id"] == object_id]
    
    features = []
    for i in range(num_frames):
        # Assuming DynoSAM frames are 1-indexed
        frame_id = i + 1 
        df_frm = df_obj[df_obj["frame_id"] == frame_id]
        
        pts = df_frm[["x_world", "y_world", "z_world"]].values
        
        # Open3D needs an explicit empty array of shape (0, 3) if no features exist in a frame
        if len(pts) == 0:
            pts = np.zeros((0, 3))
            
        features.append(pts)
        
    return features


def propagate_trajectory(aligned_mesh, motion_matrices):
    """
    Propagates the aligned mesh using dynoSAM motion matrices across time.
    
    Args:
        aligned_mesh (trimesh.Trimesh): The base mesh after 2-stage alignment.
        motion_matrices (list): List of 4x4 relative motion matrices from DynoSAM.
        
    Returns:
        np.ndarray: A 3D numpy array of shape (Frames, N_points, 3) containing the mesh history.
    """
    num_frames = len(motion_matrices) + 1
    num_vertices = len(aligned_mesh.vertices)
    
    # Storage for all frame poses: Shape (Frames, N_points, 3)
    history = np.zeros((num_frames, num_vertices, 3))
    
    # Frame 0: The initially aligned mesh
    history[0] = aligned_mesh.vertices
    
    # T_cum represents the current pose relative to the World frame
    T_cum = np.eye(4)
    
    for i, T_delta in enumerate(motion_matrices):
        # Update cumulative transformation: T_new = T_delta * T_old
        T_cum = T_delta @ T_cum
        
        # Apply to a FRESH copy of the original aligned mesh 
        # (This prevents compounding floating-point errors over hundreds of frames)
        next_mesh = aligned_mesh.copy()
        next_mesh.apply_transform(T_cum)
        
        history[i+1] = next_mesh.vertices

    return history

def save_trajectory_h5(history: np.ndarray, filename="propagated_mesh.h5"):
    """
    Saves the 4D mesh data to an HDF5 file for efficient storage and later rendering.
    """
    with h5py.File(filename, 'w') as f:
        f.create_dataset('mesh_history', data=history, compression="gzip")
        print(f"Successfully saved 4D mesh data shape {history.shape} to {filename}")



# ==========================================
# Video Visualization Module
# ==========================================

def create_lineset(points, color):
    """Helper to create an Open3D LineSet from a sequence of 3D points."""
    lineset = o3d.geometry.LineSet()
    if len(points) > 1:
        lineset.points = o3d.utility.Vector3dVector(points)
        lines = [[i, i+1] for i in range(len(points)-1)]
        lineset.lines = o3d.utility.Vector2iVector(lines)
        colors = [color for _ in range(len(lines))]
        lineset.colors = o3d.utility.Vector3dVector(colors)
    return lineset

def create_axes_with_ticks(size=5.0, step=1.0, tick_size=0.1):
    """Generates X (Red), Y (Green), Z (Blue) axes with cross-ticks."""
    points, lines, colors = [], [], []
    idx = 0

    def add_line(pt1, pt2, color):
        nonlocal idx
        points.extend([pt1, pt2])
        lines.append([idx, idx+1])
        colors.append(color)
        idx += 2

    # X axis (Red)
    add_line([-size, 0, 0], [size, 0, 0], [1, 0, 0])
    for x in np.arange(-size, size + step, step):
        add_line([x, -tick_size, 0], [x, tick_size, 0], [1, 0, 0])
        add_line([x, 0, -tick_size], [x, 0, tick_size], [1, 0, 0])

    # Y axis (Green)
    add_line([0, -size, 0], [0, size, 0], [0, 1, 0])
    for y in np.arange(-size, size + step, step):
        add_line([-tick_size, y, 0], [tick_size, y, 0], [0, 1, 0])
        add_line([0, y, -tick_size], [0, y, tick_size], [0, 1, 0])

    # Z axis (Blue)
    add_line([0, 0, -size], [0, 0, size], [0, 0, 1])
    for z in np.arange(-size, size + step, step):
        add_line([-tick_size, 0, z], [tick_size, 0, z], [0, 0, 1])
        add_line([0, -tick_size, z], [0, tick_size, z], [0, 0, 1])

    axes = o3d.geometry.LineSet()
    axes.points = o3d.utility.Vector3dVector(points)
    axes.lines = o3d.utility.Vector2iVector(lines)
    axes.colors = o3d.utility.Vector3dVector(colors)
    return axes

def render_trajectory_video(mesh_history, faces, est_traj, gt_traj, sparse_features, output_mp4, fps=10):
    """Renders an MP4 video of the propagating mesh, its bounding box, trajectories, and features."""
    print(f"Rendering video to {output_mp4}...")
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Trajectory Renderer", width=1280, height=720, visible=False)

    # 1. Dynamically Scale Axes Based on Trajectory Bounds
    max_bound = max(
        np.max(np.abs(gt_traj)) if len(gt_traj) > 0 else 0,
        np.max(np.abs(est_traj)) if len(est_traj) > 0 else 0,
        np.max(np.abs(mesh_history[0]))
    )
    axis_size = np.ceil(max_bound) + 1.0
    axis_step = max(0.5, np.round(axis_size / 5.0, 1)) 
    
    axes = create_axes_with_ticks(size=axis_size, step=axis_step, tick_size=axis_step*0.1)
    # vis.add_geometry(axes)

    # 2. Add Ground Truth (Red) & Estimated (Green) Trajectories
    gt_lineset = create_lineset(gt_traj, color=[1, 0, 0])
    est_lineset = create_lineset(est_traj, color=[0, 1, 0])
    vis.add_geometry(gt_lineset)
    vis.add_geometry(est_lineset)

    # 3. Dynamic Geometries (Mesh, Box, Centroid Line, Sparse Features)
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh_history[0])
    mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
    mesh_o3d.compute_vertex_normals()
    mesh_o3d.paint_uniform_color([0.7, 0.7, 0.7]) # Gray mesh
    vis.add_geometry(mesh_o3d)
    
    box_o3d = mesh_o3d.get_axis_aligned_bounding_box()
    box_o3d.color = [0, 0, 1] # Blue box
    vis.add_geometry(box_o3d)
    
    centroid_lineset = o3d.geometry.LineSet()
    vis.add_geometry(centroid_lineset)

    # PointCloud for dynamic sparse features (Green Dots)
    feature_pcd = o3d.geometry.PointCloud()
    if len(sparse_features[0]) > 0:
        feature_pcd.points = o3d.utility.Vector3dVector(sparse_features[0])
        feature_pcd.paint_uniform_color([0, 1, 0]) # Green
    vis.add_geometry(feature_pcd)

    # Variables for rendering
    video_writer = None
    centroids = []

    for frame_idx in range(len(mesh_history)):
        # Update Mesh
        mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh_history[frame_idx])
        mesh_o3d.compute_vertex_normals()
        
        # Update Blue Bounding Box
        new_box = mesh_o3d.get_axis_aligned_bounding_box()
        new_box.color = [0, 0, 1]
        box_o3d.min_bound = new_box.min_bound
        box_o3d.max_bound = new_box.max_bound
        
        # Update Centroid Blue Line
        centroids.append(mesh_o3d.get_center())
        if len(centroids) > 1:
            centroid_lineset.points = o3d.utility.Vector3dVector(centroids)
            lines = [[i, i+1] for i in range(len(centroids)-1)]
            centroid_lineset.lines = o3d.utility.Vector2iVector(lines)
            centroid_lineset.colors = o3d.utility.Vector3dVector([[0, 0, 1] for _ in lines]) # Blue Line
        
        # Update Sparse Features
        pts = sparse_features[frame_idx]
        if len(pts) > 0:
            feature_pcd.points = o3d.utility.Vector3dVector(pts)
            feature_pcd.paint_uniform_color([0, 1, 0])
        else:
            # Provide an empty payload to make them disappear on frames with no features
            feature_pcd.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))

        # Refresh renderer safely
        vis.update_geometry(mesh_o3d)
        vis.update_geometry(box_o3d)
        vis.update_geometry(centroid_lineset)
        vis.update_geometry(feature_pcd)
        
        if frame_idx == 0:
            vis.reset_view_point(True) # Auto-focus camera on first frame
            
            # --- NEW: Shift the camera to an angled 3D view ---
            ctr = vis.get_view_control()
            
            # 'front' is the vector pointing FROM the object TO the camera. 
            # [1, 1, 1] puts the camera in the positive X, Y, Z quadrant looking at the origin.
            ctr.set_front([1.0, 1.0, 0.8]) 
            
            # 'up' defines which way is "up" for the camera. Assuming Z is up.
            ctr.set_up([0.0, 0.0, 1.0]) 
            
            # Zoom out a little bit (values > 1 zoom out, < 1 zoom in) so the axes don't get clipped
            ctr.set_zoom(1.2)
            
        vis.poll_events()
        vis.update_renderer()

        # Capture Frame for Video
        img_buffer = vis.capture_screen_float_buffer(do_render=True)
        img_array = (np.asarray(img_buffer) * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Initialize Video Writer on first frame
        if video_writer is None:
            height, width, _ = img_bgr.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_mp4, fourcc, fps, (width, height))
            
        video_writer.write(img_bgr)

    video_writer.release()
    vis.destroy_window()
    print(f"Video {output_mp4} saved successfully.")

# ==========================================
# Main Execution Pipeline
# ==========================================

def run_propagation(sam3d_output_path, motion_csv, pose_csv, feature_csv, alignment_matrix, object_id):
    """Master function to load, align, propagate, and render."""
    print(f"\n[Object {object_id}] Loading SAM3D outputs...")
    outputs = torch.load(sam3d_output_path)
    out = outputs[object_id - 1] 
    raw_mesh = out['glb'].copy()
    faces = np.asarray(raw_mesh.faces) 
    
    # 2. Align Mesh
    mesh_moge_aligned = apply_moge_transform(raw_mesh, out)
    # mesh_moge_aligned.apply_transform(alignment_matrix)
    aligned_mesh = mesh_moge_aligned
    
    # 3. Extract Motion & Poses
    motion_matrices = get_motion_matrices(motion_csv, object_id)
    est_traj, gt_traj = get_pose_trajectories(pose_csv, object_id)
    
    # 4. Propagate Trajectory
    history = propagate_trajectory(aligned_mesh, motion_matrices)
    num_frames = len(history)

    # 5. Extract Sparse Features matching the exact number of frames
    sparse_features = get_sparse_features(feature_csv, object_id, num_frames)
    
    # 6. Render Video
    output_mp4 = f"object_{object_id}_propagation.mp4"
    render_trajectory_video(
        mesh_history=history, 
        faces=faces, 
        est_traj=est_traj, 
        gt_traj=gt_traj, 
        sparse_features=sparse_features,
        output_mp4=output_mp4
    )


if __name__ == "__main__":
    MOTION_CSV = "../results/exp1/frontend_object_motion_log.csv"
    POSE_CSV = "../results/exp1/frontend_object_pose_log.csv"
    FEATURE_CSV = "../results/exp1/frontend_map_points_log.csv"
    SAM3D_PT = "sam3d_object_cache_ptmp_input.pt"
    
    print("Starting 4-Object Propagation and Rendering Pipeline...")
    
    for target_object_id in range(1, 5):
        try:
            alignment_matrix = np.load(f'alignment_matrix_{target_object_id}.npy')
        except FileNotFoundError:
            print(f"Alignment matrix for Obj {target_object_id} not found, using Identity matrix.")
            alignment_matrix = np.eye(4) 
        
        run_propagation(
            sam3d_output_path=SAM3D_PT,
            motion_csv=MOTION_CSV,
            pose_csv=POSE_CSV,
            feature_csv=FEATURE_CSV,
            alignment_matrix=alignment_matrix,
            object_id=target_object_id
        )