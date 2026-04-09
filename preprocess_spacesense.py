import os
import glob
import shutil
from datetime import datetime
import numpy as np
import cv2
import pandas as pd

def parse_timestamp_to_seconds(ts_str):
    """
    Converts YYYYMMDDHHMMSSXXX (where XXX is ms) to a standard UNIX timestamp in seconds.
    """
    # Append '000' to the end to convert milliseconds to microseconds for strptime
    dt = datetime.strptime(ts_str + "000", "%Y%m%d%H%M%S%f")
    return dt.timestamp()

def preprocess_dataset(raw_dir, output_dir):
    # 1. Setup Input Paths
    raw_depth_dir = os.path.join(raw_dir, 'depth')
    raw_image_dir = os.path.join(raw_dir, 'image')
    raw_seg_dir = os.path.join(raw_dir, 'seg')
    csv_path = os.path.join(raw_dir, 'pose_ground_truth.csv')

    # 2. Setup Output Paths
    out_image_dir = os.path.join(output_dir, 'image')
    out_depth_dir = os.path.join(output_dir, 'depth')
    out_seg_dir = os.path.join(output_dir, 'seg')
    
    os.makedirs(out_image_dir, exist_ok=True)
    os.makedirs(out_depth_dir, exist_ok=True)
    os.makedirs(out_seg_dir, exist_ok=True)

    out_times_path = os.path.join(output_dir, 'times.txt')
    out_pose_path = os.path.join(output_dir, 'pose_ground_truth.txt')

    # 3. Load Ground Truth CSV
    print(f"Loading Ground Truth CSV: {csv_path}")
    # Read timestamp as string to avoid integer overflow or truncation issues
    df = pd.read_csv(csv_path, dtype={'timestamp': str})
    
    # 4. Gather and Sort Files
    raw_image_files = sorted(glob.glob(os.path.join(raw_image_dir, '*.png')))
    
    num_frames = len(raw_image_files)
    print(f"Found {num_frames} frames to process.")

    # --- Pre-compute Grid for Depth Unprojection ---
    # Since all images are 1024x1024, we only need to calculate the grid once
    fx = fy = 1097.99
    cx = cy = 512.0
    h, w = 1024, 1024
    
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    dx = (u - cx) / fx
    dy = (v - cy) / fy
    # Calculate the ray magnitude (hypotenuse) for every pixel
    ray_mag = np.sqrt(dx**2 + dy**2 + 1.0)

    with open(out_times_path, 'w') as f_times, open(out_pose_path, 'w') as f_pose:
        for frame_idx, raw_img_path in enumerate(raw_image_files):
            filename = os.path.basename(raw_img_path)
            ts_str = os.path.splitext(filename)[0]
            idx_str = f"{frame_idx:06d}"
            
            # --- A. Process Timestamp ---
            ts_sec = parse_timestamp_to_seconds(ts_str)
            f_times.write(f"{ts_sec:.6f}\n")
            
            # --- B. Process Pose ---
            row_match = df[df['timestamp'] == ts_str]
            if row_match.empty:
                raise ValueError(f"Timestamp {ts_str} not found in CSV!")
            row = row_match.iloc[0]
            
            tx = row['camera_in_world_x(m)']
            ty = row['camera_in_world_y(m)']
            tz = row['camera_in_world_z(m)']
            qw = row['camera_in_world_quat_w']
            qx = row['camera_in_world_quat_x']
            qy = row['camera_in_world_quat_y']
            qz = row['camera_in_world_quat_z']
            
            f_pose.write(f"{ts_sec:.6f} {tx} {ty} {tz} {qx} {qy} {qz} {qw}\n")

            # --- C. Process RGB Image ---
            out_img_path = os.path.join(out_image_dir, f"{idx_str}.png")
            shutil.copy2(raw_img_path, out_img_path)

            # --- D. Process Depth (Range to Planar) ---
            raw_depth_path = os.path.join(raw_depth_dir, f"{ts_str}.npz")
            out_depth_path = os.path.join(out_depth_dir, f"{idx_str}.tiff")
            
            npz_data = np.load(raw_depth_path)
            depth_mm = npz_data[npz_data.files[0]] 
            
            # Convert int32 mm to float32 meters
            depth_perspective_m = depth_mm.astype(np.float32) / 1000.0
            
            # Convert Perspective (Range) Depth to Planar Depth
            depth_planar_m = depth_perspective_m / ray_mag
    
            
            # Mask out the 10,000m void clipping plane. 
            # Setting to 0.0 prevents SAM3D/DynoSAM from tracking the "background wall".
            depth_planar_m[depth_perspective_m >= 9999.0] = 50
            
            
            # Save as 32-bit TIFF
            cv2.imwrite(out_depth_path, depth_planar_m)

            # --- E. Process Segmentation Mask ---
            raw_seg_path = os.path.join(raw_seg_dir, f"{ts_str}.png")
            out_seg_path = os.path.join(out_seg_dir, f"{idx_str}.png")
            
            seg_bgr = cv2.imread(raw_seg_path, cv2.IMREAD_COLOR)
            seg_gray = cv2.cvtColor(seg_bgr, cv2.COLOR_BGR2GRAY)
            
            mask = (seg_gray > 0).astype(np.uint8)
            cv2.imwrite(out_seg_path, mask)

            if frame_idx % 100 == 0 or frame_idx == num_frames - 1:
                print(f"Processed frame {frame_idx + 1}/{num_frames}")

    print("Preprocessing complete!")

if __name__ == "__main__":
    # --- UPDATE THESE PATHS ---
    PATH = os.getcwd()
    RAW_DATA_DIR = f"{PATH}/../tracking_dataset/Cheops_raw/Cheops/approach_top_p-45_y45/"
    OUTPUT_DIR =f"{PATH}/../tracking_dataset/Cheops/approach_top_p-45_y45/"
    
    preprocess_dataset(RAW_DATA_DIR, OUTPUT_DIR)