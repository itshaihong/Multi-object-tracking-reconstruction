import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import trimesh
import open3d as o3d
import matplotlib.pyplot as plt

from sklearn.neighbors import LocalOutlierFactor
from skimage.measure import ransac
from skimage.transform import SimilarityTransform
from pytorch3d.transforms import quaternion_to_matrix, Transform3d

# --- Configuration & Imports ---
# Setup sys.path if relying on local modules from sam-3d-objects
PATH = os.getcwd()
MODULE_DIR = os.path.abspath(os.path.join(PATH, "../sam-3d-objects/notebook/"))
if MODULE_DIR not in sys.path:
    sys.path.append(MODULE_DIR)

# Constants: Coordinate System Transformations
R_YUP_TO_ZUP = torch.tensor([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=torch.float32)
R_FLIP_Z = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=torch.float32)
R_PYTORCH3D_TO_CAM = torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=torch.float32)


# ==========================================
# Module 1: SAM3D / MOGE 3D Transformations
# ==========================================

def transform_mesh_vertices(vertices: np.ndarray, rotation: torch.Tensor, translation: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Transforms mesh vertices using Scale, Rotation, and Translation in PyTorch3D."""
    if isinstance(vertices, np.ndarray):
        vertices = torch.tensor(vertices, dtype=torch.float32)

    vertices = vertices.unsqueeze(0)  # batch dimension [1, N, 3]
    
    # Coordinate system flips
    vertices = vertices @ R_FLIP_Z.to(vertices.device) 
    vertices = vertices @ R_YUP_TO_ZUP.to(vertices.device)
    
    # Construct Transformation Matrix
    r_mat = quaternion_to_matrix(rotation.to(vertices.device))
    tfm = Transform3d(dtype=vertices.dtype, device=vertices.device)
    tfm = tfm.scale(scale).rotate(r_mat).translate(translation[0], translation[1], translation[2])
    
    # Apply transformation
    vertices_world = tfm.transform_points(vertices)
    vertices_world = vertices_world @ R_PYTORCH3D_TO_CAM.to(vertices_world.device)
    
    return vertices_world[0]  # Return [N, 3] by removing batch dimension


def apply_moge_transform(mesh: trimesh.Trimesh, out: dict) -> trimesh.Trimesh:
    """Extracts poses from SAM3D output dictionary and applies them to a trimesh."""
    vertices = mesh.vertices

    scale = out["scale"][0].cpu().float()
    translation = out["translation"][0].cpu().float()
    rotation = out["rotation"].squeeze().cpu().float()

    vertices_transformed = transform_mesh_vertices(vertices, rotation, translation, scale)
    mesh.vertices = vertices_transformed.cpu().numpy().astype(np.float32)
    mesh.fix_normals()
    
    return mesh


def transform_mesh_data(mesh: trimesh.Trimesh, model_matrix: np.ndarray) -> trimesh.Trimesh:
    """Applies a 4x4 homogenous transformation matrix to a mesh dictionary."""
    mesh.apply_transform(model_matrix)
    mesh.fix_normals()
    
    return mesh


# ==========================================
# Module 2: RANSAC Similarity Alignment
# ==========================================
def parse_sam3d_output(output_dict, binary_mask, align_coord=True):
    """
    Visualizes the pointmap and pointmap_colors from SAM3D meta output.
    """
    # 1. Extract data
    points = output_dict.get('pointmap')
    colors = output_dict.get('pointmap_colors')

    if points is None:
        print("Error: 'pointmap' not found in dictionary.")
        return None, None, None

    if isinstance(binary_mask, np.ndarray):
        binary_mask = torch.from_numpy(binary_mask)

    # Resize mask to match pointmap dimensions
    mask_rescaled = F.interpolate(
        binary_mask.unsqueeze(0).unsqueeze(0).float(), 
        size=(points.shape[0], points.shape[1]), 
        mode='nearest'
    ).squeeze()
    
    y_indices, x_indices = torch.where(mask_rescaled > 0)
    y_indices = y_indices.to(points.device)
    x_indices = x_indices.to(points.device)
    
    filtered_points = points[y_indices, x_indices]
    
    # Handle colors safely (some models might not output colors)
    if colors is not None:
        filtered_colors = colors[y_indices, x_indices]
    else:
        filtered_colors = None

    # Apply coordinate inversion alignment
    if align_coord:
        filtered_points[:, 0] = -filtered_points[:, 0]
        filtered_points[:, 1] = -filtered_points[:, 1]

    # create dataframe with pixel-coordinate correspondance
    pts_np = filtered_points.detach().cpu().numpy()
    u_np = x_indices.detach().cpu().numpy()
    v_np = y_indices.detach().cpu().numpy()
    
    data = {
        'x_mesh': pts_np[:, 0],
        'y_mesh': pts_np[:, 1],
        'z_mesh': pts_np[:, 2],
        'u': u_np,
        'v': v_np
    }

    df_mesh = pd.DataFrame(data)

    # scale the coordinate properly by scale factor = 2
    df_mesh['u'] = df_mesh['u'] * 2
    df_mesh['v'] = df_mesh['v'] * 2
    
    # 2. Reshape/Flatten the grids
    flat_points = filtered_points.reshape(-1, 3)
    
    # 3. Handle Colors
    if filtered_colors is not None:
        flat_colors = filtered_colors.reshape(-1, 3)
        if flat_colors.max() > 1.0:
            flat_colors = flat_colors / 255.0
    else:
        # Default to gray if no colors provided
        flat_colors = torch.ones_like(flat_points) * 0.5

    # 4. Filter out invalid points (zeros or NaNs)
    valid_idx = ~(flat_points == 0).all(dim=1)
    final_points = flat_points[valid_idx]
    final_colors = flat_colors[valid_idx]

    return final_points, final_colors, df_mesh

def extract_corresponding_points(sam3d_output, pixel_coords_csv, map_points_csv, mask, frame_id=1, object_id=1):
    """
    Extracts corresponding world coordinates (DynoSAM) and mesh coordinates (SAM3D),
    utilizing the scaled and inverted df_mesh from parse_sam3d_output.
    """
    # 1. Get the parsed and filtered mesh DataFrame from SAM3D
    final_points, final_colors, df_mesh = parse_sam3d_output(
        output_dict=sam3d_output, 
        binary_mask=mask, 
        align_coord=True
    )
    
    if df_mesh is None or df_mesh.empty:
        return pd.DataFrame() # Return empty if no points found

    # 2. Load DynoSAM 2D Tracking Data
    df_pixel_coord = pd.read_csv(pixel_coords_csv, header=None)
    df_pixel_coord.columns = ['frame_id', 'object_id', 'tracklet_id', 'u', 'v']
    
    # Ensure DynoSAM u, v are integers so they can cleanly merge with df_mesh
    df_pixel_coord['u'] = df_pixel_coord['u'].astype(int)
    df_pixel_coord['v'] = df_pixel_coord['v'].astype(int)
    
    # 3. Load DynoSAM 3D Map Points
    df_3d_coord = pd.read_csv(map_points_csv)
    
    # 4. Link DynoSAM 2D pixels with DynoSAM 3D World coordinates
    df_dynosam = pd.merge(
        df_pixel_coord, 
        df_3d_coord, 
        on=['frame_id', 'object_id', 'tracklet_id'], 
        how='inner'
    )
    
    # Filter for the specific frame and object
    df_dynosam = df_dynosam[(df_dynosam['frame_id'] == frame_id) & (df_dynosam['object_id'] == object_id)]
    
    # 5. Find Correspondences by matching (u, v) from DynoSAM to (u, v) from SAM3D
    df_final = pd.merge(
        df_dynosam,
        df_mesh,
        on=['u', 'v'],
        how='inner'
    )
    
    # Drop any potential NaNs generated during transformations
    df_final = df_final.dropna(subset=['x_mesh', 'y_mesh', 'z_mesh']).reset_index(drop=True)
    
    print(f"Extracted {len(df_final)} corresponding points strictly within the mask for Object {object_id}.")
    
    return df_final[['u', 'v', 'x_world', 'y_world', 'z_world', 'x_mesh', 'y_mesh', 'z_mesh']], final_points

def apply_transformation_matrix(points: np.ndarray, model: SimilarityTransform) -> np.ndarray:
    """Applies a SimilarityTransform model to a point cloud."""
    n = points.shape[0]
    ones = np.ones((n, 1))
    points_homo = np.hstack([points, ones])
    
    aligned_homo = points_homo @ model.params.T
    aligned_points = aligned_homo[:, :3]
    
    return aligned_points


def estimate_transform(df: pd.DataFrame, visualize=False):
    """Estimates the transform between mesh points and world points using RANSAC."""
    src = df[['x_mesh', 'y_mesh', 'z_mesh']].values
    dst = df[['x_world', 'y_world', 'z_world']].values

    # Preliminary Outlier Removal
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
    good_mask = lof.fit_predict(src) == 1
    src_clean = src[good_mask]
    dst_clean = dst[good_mask]

    # RANSAC with SimilarityTransform (Umeyama)
    model, inliers = ransac(
        (src_clean, dst_clean),
        SimilarityTransform, 
        min_samples=3,
        residual_threshold=0.1,  # Max error allowed (in meters)
        max_trials=1000
    )

    print(f"Kept {sum(inliers)} inliers out of {len(src_clean)} points.")
    
    # Apply transformations
    all_mesh_pts = src_clean
    aligned_mesh_pts = apply_transformation_matrix(all_mesh_pts, model)
    world_pts = dst_clean
    
    # Calculate RMSE error for the inliers
    diff = aligned_mesh_pts[inliers] - world_pts[inliers]
    rmse = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    
    print(f"Alignment RMSE: {rmse:.4f} meters")
    if rmse < 0.05:
        print("Excellent alignment! Suitable for precise satellite trajectory prediction.")
    else:
        print("Alignment has some jitter. Check for non-rigid motion or sensor noise.")

    # Only visualize if explicitly asked (we skip this in the loop now)
    if visualize:
        visualize_overlay(aligned_mesh_pts, all_mesh_pts, world_pts)
        
    return model, inliers, aligned_mesh_pts, all_mesh_pts, world_pts


# ==========================================
# Module 3: Visualizations
# ==========================================


def visualize_all_overlays(all_objects_data):
    """Visualizes the 2D point cloud overlay for all objects in ONE Matplotlib figure."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate distinct colors for each object
    colors = plt.cm.get_cmap('tab10', max(10, len(all_objects_data)))
    
    for i, data in enumerate(all_objects_data):
        obj_id = data['id']
        aligned_pts = data['aligned_pts']
        world_pts = data['world_pts']
        
        # Plot Aligned Mesh Points (Circles)
        ax.scatter(aligned_pts[:, 0], aligned_pts[:, 1], aligned_pts[:, 2], 
                   label=f"Obj {obj_id}: Aligned Mesh", color=colors(i), marker='o', s=10, alpha=0.8)
        # Plot Target World Points (Crosses)
        ax.scatter(world_pts[:, 0], world_pts[:, 1], world_pts[:, 2], 
                   label=f"Obj {obj_id}: DynoSAM World", color=colors(i), marker='x', s=15, alpha=0.8)
        
    ax.set_xlabel('$x_{world}$')
    ax.set_ylabel('$y_{world}$')
    ax.set_zlabel('$z_{world}$')
    ax.set_title('3D World Coordinates - All Objects')
    
    # Move legend outside the plot so it doesn't block data
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def get_o3d_geometries(pointmap: np.ndarray, feature: np.ndarray, aligned_mesh: trimesh.Trimesh, aligned_mesh_2: trimesh.Trimesh, color: list):
    """Creates and returns Open3D geometries for a single object WITHOUT rendering."""
    geometries = []
    base_color = np.array(color[:3])         
    bright_color = np.clip(base_color, 0.0, 1.0)     
    dark_color = np.clip(base_color * 0.4, 0.0, 1.0)

    # 1. Pointmap
    if pointmap is not None and len(pointmap) > 0:
        pcd_masked = o3d.geometry.PointCloud()
        pcd_masked.points = o3d.utility.Vector3dVector(pointmap)
        pcd_masked.paint_uniform_color([0, 0, 1])  # Blue
        geometries.append(pcd_masked)

    # 2. DynoSAM Features (Always red for visibility)
    if feature is not None and len(feature) > 0:
        feature_pcd = o3d.geometry.PointCloud()
        feature_pcd.points = o3d.utility.Vector3dVector(feature)
        feature_pcd.paint_uniform_color([1, 0, 0])  # Red
        geometries.append(feature_pcd)

    # 4. Aligned GLB Mesh
    aligned_mesh_o3d = o3d.geometry.TriangleMesh()
    aligned_mesh_o3d.vertices = o3d.utility.Vector3dVector(np.asarray(aligned_mesh.vertices))
    aligned_mesh_o3d.triangles = o3d.utility.Vector3iVector(np.asarray(aligned_mesh.faces))
    aligned_mesh_o3d.compute_vertex_normals()
    aligned_mesh_o3d.paint_uniform_color(dark_color)
    geometries.append(aligned_mesh_o3d)

    # 5. Visualize 2nd aligned GLB Mesh
    aligned_mesh_2_o3d = o3d.geometry.TriangleMesh()
    aligned_mesh_2_o3d.vertices = o3d.utility.Vector3dVector(np.asarray(aligned_mesh_2.vertices))
    aligned_mesh_2_o3d.triangles = o3d.utility.Vector3iVector(np.asarray(aligned_mesh_2.faces))
    aligned_mesh_2_o3d.compute_vertex_normals()
    aligned_mesh_2_o3d.paint_uniform_color(bright_color)
    geometries.append(aligned_mesh_2_o3d)

    
    # 4. Add a Coordinate Frame
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    geometries.append(origin)

    return geometries