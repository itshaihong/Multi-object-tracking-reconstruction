import numpy as np
import pandas as pd
import torch
import trimesh
import sys
import os
import matplotlib.pyplot as plt

# --- Setup Paths & Imports ---
PATH = os.getcwd()
module_dir = os.path.abspath(f"{PATH}/../sam-3d-objects/notebook/")
if module_dir not in sys.path:
    sys.path.append(module_dir)

from inference import (
    Inference, load_image, load_masks, display_image
)

# Import our custom alignment pipeline
from alignment_pipeline import (
    apply_moge_transform, estimate_transform, 
    extract_corresponding_points, transform_mesh_data,
    get_o3d_geometries, visualize_all_overlays
)


if __name__ == "__main__":
    # --- 1. Define File Paths ---
    PIXEL_COORDS_PATH = "../results/exp1/pixel_coords_debug.csv"
    MAP_POINTS_PATH = "../results/exp1/frontend_map_points_log.csv"
    SAM3D_OUTPUT_PATH = "OMD_results/sam3d_object_cache_ptmp_input.pt"

    # --- 2. Load SAM3D Outputs ---
    print(f"Loading SAM3D outputs from: {SAM3D_OUTPUT_PATH}")
    outputs = torch.load(SAM3D_OUTPUT_PATH)
    
    # --- 3. Load Image and Object Masks ---
    IMAGE_PATH = f"{PATH}/../tracking_dataset/omd_mask/000001/image.png"
    IMAGE_NAME = os.path.basename(os.path.dirname(IMAGE_PATH))

    print(f"Loading image from: {IMAGE_PATH}")
    image = load_image(IMAGE_PATH)
    
    # Load specific masks using the indices provided
    mask_indices = [0, 2, 1, 4]
    print(f"Loading masks for indices: {mask_indices}")
    masks = load_masks(os.path.dirname(IMAGE_PATH), indices_list=mask_indices, extension=".png")
    
    print("Displaying original Image and Masks...")
    # (Close the matplotlib window to let the script continue)
    try:
        display_image(image, masks)
    except Exception as e:
        print(f"Could not display image/masks interactively: {e}")

    # --- 4. Process Each Object Independently ---
    print("\n--- Starting Pipeline for Multiple Objects ---")

    all_plot_data = []
    all_o3d_geometries = []
    cmap = plt.cm.get_cmap('tab10', 10)
    
    # Iterate over the 4 loaded objects and their corresponding masks
    for i, out in enumerate(outputs):
        object_id = i + 1 
        current_mask = masks[i]  # Get the mask for the current object
        obj_color = cmap(i)[:3]
        
        print(f"\n" + "="*45)
        print(f"       Processing Object {object_id} (Mask {mask_indices[i]})")
        print("="*45)
        
        mesh = out["glb"].copy()
        
        # Step 4a: Extract Correspondences (NOW WITH MASK FILTERING)
        df_corresp, clean_pointcloud_tensor = extract_corresponding_points(
            sam3d_output=out,
            pixel_coords_csv=PIXEL_COORDS_PATH,
            map_points_csv=MAP_POINTS_PATH,
            mask=current_mask,      # Pass the mask here!
            frame_id=1,
            object_id=object_id
        )
        
        if df_corresp.empty:
            print(f"Warning: No valid corresponding points found within the mask for Object {object_id}. Skipping.")
            continue

        # Step 4b: Base Alignment
        print(f"\n[Object {object_id}] 1. Applying Base SAM3D/MoGe Transformation...")
        aligned_mesh = apply_moge_transform(mesh, out)

        # Step 4c: 2D-3D Feature Alignment using RANSAC (Now purely object points!)
        print(f"\n[Object {object_id}] 2. Estimating Similarity Transform (Mesh -> DynoSAM World)...")
        model, inliers, aligned_pts, original_pts, target_pts = estimate_transform(df_corresp)
        np.save(f'alignment_matrix_{object_id}.npy', model.params)

        # Save data for the matplotlib 3D scatter plot
        all_plot_data.append({
            'id': object_id,
            'aligned_pts': aligned_pts,
            'world_pts': target_pts
        })
        
        print(f"Transformation Matrix (4x4):")
        print(model.params)

        aligned_mesh_2 = transform_mesh_data(aligned_mesh.copy(), model.params)

        # Step 4d: Masked Visualizations
        print(f"\n[Object {object_id}] 3. Visualizing Masked Outputs...")
        
        clean_pointmap_np = clean_pointcloud_tensor.detach().cpu().numpy()

        geometries = get_o3d_geometries(
            pointmap=clean_pointmap_np, 
            feature=df_corresp[['x_world', 'y_world', 'z_world']].values, 
            aligned_mesh=aligned_mesh,
            aligned_mesh_2=aligned_mesh_2,
            color=obj_color
        )
        # Append this object's geometries to the master list
        all_o3d_geometries.extend(geometries)

    # ==========================================
    # 5. Global Visualization (Post-Loop)
    # ==========================================
    print("\n" + "="*45)
    print("       Global Visualization")
    print("="*45)
    
    if all_plot_data:
        print("1. Opening Matplotlib Point Alignment Window...")
        # Close this window to trigger the Open3D window
        visualize_all_overlays(all_plot_data)
        
    if all_o3d_geometries:
        print("2. Opening Open3D Interactive Window...")
        import open3d as o3d
        # Add a world origin frame for reference
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        all_o3d_geometries.append(origin)
        
        o3d.visualization.draw_geometries(all_o3d_geometries)

    print("\nPipeline execution complete!")