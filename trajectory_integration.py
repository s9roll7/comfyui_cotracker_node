import numpy as np
import cv2
from typing import Tuple, Optional, List
import os
import torch

def create_mask_from_tracks(forward_tracks: np.ndarray, 
                           forward_visibility: np.ndarray,
                           frame_shape: Tuple[int, int],
                           radius: int = 10,
                           frame_idx: Optional[int] = None) -> np.ndarray:
                           
    H, W = frame_shape
    
    if frame_idx is not None:
        mask = np.zeros((H, W), dtype=np.uint8)
        points = forward_tracks[frame_idx]  # shape (N, 2)
        visibility = forward_visibility[frame_idx]  # shape (N,)
        
        valid_points = points[visibility > 0]
        
        for point in valid_points:
            x, y = int(point[0]), int(point[1])
            if 0 <= x < W and 0 <= y < H:
                cv2.circle(mask, (x, y), radius, 1, -1)
        
        return mask
    else:
        T = forward_tracks.shape[0]
        masks = np.zeros((T, H, W), dtype=np.uint8)
        
        for t in range(T):
            points = forward_tracks[t]  # shape (N, 2)
            visibility = forward_visibility[t]  # shape (N,)
            
            valid_points = points[visibility > 0]
            
            for point in valid_points:
                x, y = int(point[0]), int(point[1])
                if 0 <= x < W and 0 <= y < H:
                    cv2.circle(masks[t], (x, y), radius, 1, -1)
        
        return masks

def detect_empty_regions(forward_tracks: np.ndarray,
                        forward_visibility: np.ndarray,
                        frame_shape: Tuple[int, int],
                        frame_idx: int,
                        radius: int = 10,
                        min_region_size: int = 100) -> List[Tuple[int, int, int, int]]:
                        
    mask = create_mask_from_tracks(forward_tracks, forward_visibility, frame_shape, radius, frame_idx)
    
    empty_mask = 1 - mask
    
    contours, _ = cv2.findContours(empty_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    empty_regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_region_size:
            x, y, w, h = cv2.boundingRect(contour)
            empty_regions.append((x, y, w, h))
    
    return empty_regions

def has_data_in_region(backward_tracks: np.ndarray,
                      backward_visibility: np.ndarray,
                      spatial_region: Tuple[int, int, int, int],
                      frame_idx: int,
                      min_points: int = 1) -> bool:
    x, y, w, h = spatial_region
    points = backward_tracks[frame_idx]  # shape (N, 2)
    visibility = backward_visibility[frame_idx]  # shape (N,)
    
    valid_points = points[visibility > 0]
    
    if len(valid_points) == 0:
        return False
    
    in_region = ((valid_points[:, 0] >= x) & (valid_points[:, 0] < x + w) &
                 (valid_points[:, 1] >= y) & (valid_points[:, 1] < y + h))
    
    return np.sum(in_region) >= min_points

def extract_trajectory_with_indices(tracks: np.ndarray,
                                   visibility: np.ndarray,
                                   spatial_region: Tuple[int, int, int, int],
                                   frame_idx: int) -> Tuple[np.ndarray, np.ndarray, List[int]]:
                                   
    x, y, w, h = spatial_region
    points = tracks[frame_idx]  # shape (N, 2)
    vis = visibility[frame_idx]  # shape (N,)
    
    valid_mask = vis > 0
    in_region_mask = ((points[:, 0] >= x) & (points[:, 0] < x + w) &
                      (points[:, 1] >= y) & (points[:, 1] < y + h))
    
    selected_indices = np.where(valid_mask & in_region_mask)[0]
    
    if len(selected_indices) == 0:
        return np.empty((tracks.shape[0], 0, 2)), np.empty((tracks.shape[0], 0)), []
    
    extracted_tracks = tracks[:, selected_indices, :]
    extracted_visibility = visibility[:, selected_indices]
    
    return extracted_tracks, extracted_visibility, selected_indices.tolist()


def time_reverse(tracks: np.ndarray, visibility: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return np.flip(tracks, axis=0), np.flip(visibility, axis=0)



def integrate_tracking_results(forward_tracks: np.ndarray,
                              forward_visibility: np.ndarray,
                              backward_tracks: np.ndarray,
                              backward_visibility: np.ndarray,
                              frame_shape: Tuple[int, int],
                              radius: int = 10,
                              min_region_size: int = 100,
                              min_points: int = 1,
                              output_dir: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
                              
    T = forward_tracks.shape[0]
    
    backward_tracks, backward_visibility = time_reverse(backward_tracks, backward_visibility)
    
    if output_dir is not None:
        import os
        os.makedirs(output_dir, exist_ok=True)
        print(f"Mask images will be saved to: {output_dir}")
    
    integrated_tracks_list = [forward_tracks]
    integrated_visibility_list = [forward_visibility]
    
    global_used_indices = set()
    
    for frame_idx in range(T):
        print(f"Processing frame {frame_idx}...")
        
        current_integrated_tracks = np.concatenate(integrated_tracks_list, axis=1)
        current_integrated_visibility = np.concatenate(integrated_visibility_list, axis=1)
        
        if output_dir is not None:
            save_mask_visualization(
                current_integrated_tracks, current_integrated_visibility,
                frame_shape, frame_idx, radius, output_dir, 
                suffix="before_integration"
            )
        
        empty_regions = detect_empty_regions(
            current_integrated_tracks, current_integrated_visibility, 
            frame_shape, frame_idx, radius, min_region_size
        )
        print(f"  Found {len(empty_regions)} empty regions")
        
        frame_new_tracks = []
        frame_new_visibility = []
        
        for region_idx, spatial_region in enumerate(empty_regions):
            available_indices = [i for i in range(backward_tracks.shape[1]) if i not in global_used_indices]
            
            if len(available_indices) == 0:
                print(f"  Region {region_idx}: No more available backward tracks")
                continue
                
            available_backward_tracks = backward_tracks[:, available_indices, :]
            available_backward_visibility = backward_visibility[:, available_indices]
            
            if has_data_in_region(available_backward_tracks, available_backward_visibility, spatial_region, frame_idx, min_points):
                backward_trajectory, backward_vis, local_extracted_indices = extract_trajectory_with_indices(
                    available_backward_tracks, available_backward_visibility, spatial_region, frame_idx
                )
                
                if backward_trajectory.shape[1] > 0:
                    global_extracted_indices = [available_indices[i] for i in local_extracted_indices]
                    
                    print(f"  Region {region_idx}: Extracted {len(global_extracted_indices)} tracks: {global_extracted_indices}")
                    
                    frame_new_tracks.append(backward_trajectory)
                    frame_new_visibility.append(backward_vis)
                    
                    global_used_indices.update(global_extracted_indices)
                    
                    print(f"  Total used indices so far: {len(global_used_indices)}")
            else:
                print(f"  Region {region_idx}: No backward data found")
        
        if frame_new_tracks:
            integrated_tracks_list.extend(frame_new_tracks)
            integrated_visibility_list.extend(frame_new_visibility)
        
        
        if output_dir is not None:
            final_integrated_tracks = np.concatenate(integrated_tracks_list, axis=1)
            final_integrated_visibility = np.concatenate(integrated_visibility_list, axis=1)
            
            save_mask_visualization(
                final_integrated_tracks, final_integrated_visibility,
                frame_shape, frame_idx, radius, output_dir, 
                suffix="after_integration"
            )
    
    integrated_tracks = np.concatenate(integrated_tracks_list, axis=1)
    integrated_visibility = np.concatenate(integrated_visibility_list, axis=1)
    
    
    if output_dir is not None:
        print(f"\nCreating comparison images...")
        for frame_idx in range(T):
            create_mask_comparison(
                forward_tracks, forward_visibility,
                integrated_tracks, integrated_visibility,
                frame_shape, frame_idx, radius, output_dir
            )
    
    print(f"\nFinal summary:")
    print(f"Used backward track indices: {sorted(global_used_indices)}")
    print(f"Total backward tracks used: {len(global_used_indices)}")
    print(f"Original backward tracks: {backward_tracks.shape[1]}")
    print(f"Final integrated tracks shape: {integrated_tracks.shape}")
    
    return integrated_tracks, integrated_visibility

def save_mask_visualization(tracks: np.ndarray,
                           visibility: np.ndarray,
                           frame_shape: Tuple[int, int],
                           frame_idx: int,
                           radius: int,
                           output_dir: str,
                           suffix: str = "") -> None:
                           
    mask = create_mask_from_tracks(tracks, visibility, frame_shape, radius, frame_idx)
    
    mask_vis = (mask * 255).astype(np.uint8)
    
    mask_colored = cv2.applyColorMap(mask_vis, cv2.COLORMAP_JET)
    
    mask_colored[mask == 0] = [0, 0, 0]
    
    if suffix:
        filename = f"mask_frame_{frame_idx:04d}_{suffix}.png"
    else:
        filename = f"mask_frame_{frame_idx:04d}.png"
    
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, mask_colored)
    print(f"    Saved mask: {filepath}")

def create_mask_comparison(forward_tracks: np.ndarray,
                          forward_visibility: np.ndarray,
                          integrated_tracks: np.ndarray,
                          integrated_visibility: np.ndarray,
                          frame_shape: Tuple[int, int],
                          frame_idx: int,
                          radius: int,
                          output_dir: str) -> None:
                          
    forward_mask = create_mask_from_tracks(forward_tracks, forward_visibility, frame_shape, radius, frame_idx)
    
    integrated_mask = create_mask_from_tracks(integrated_tracks, integrated_visibility, frame_shape, radius, frame_idx)
    
    added_mask = integrated_mask - forward_mask
    
    H, W = frame_shape
    comparison = np.zeros((H, W, 3), dtype=np.uint8)
    
    comparison[forward_mask > 0] = [255, 0, 0]
    
    comparison[added_mask > 0] = [0, 0, 255]
    
    filename = f"comparison_frame_{frame_idx:04d}.png"
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, comparison)
    print(f"    Saved comparison: {filepath}")


def trajectory_integration(forward_tracks, forward_visibility, backward_tracks, backward_visibility, frame_shape, grid_size):
    
    ft = forward_tracks.squeeze(0).cpu().numpy()  # (T, N, 2)
    fv = forward_visibility.squeeze(0).cpu().numpy()  # (T, N)
    
    bt = backward_tracks.squeeze(0).cpu().numpy()  # (T, N, 2)
    bv = backward_visibility.squeeze(0).cpu().numpy()  # (T, N)
    
    radius = min(frame_shape) / grid_size * 1.5
    radius = max(int(round(radius)), 3)
    min_region_size = radius ** 2
    
    t, v = integrate_tracking_results(ft,fv,bt,bv,frame_shape,
                                        radius=radius, min_region_size=min_region_size, min_points=1, output_dir=None)
    
    t = torch.from_numpy(t).float().unsqueeze(0)
    v = torch.from_numpy(v).float().unsqueeze(0)
    
    return t,v


