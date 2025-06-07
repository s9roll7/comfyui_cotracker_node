import torch
import numpy as np
import json
import cv2
from PIL import Image
import torchvision.transforms as transforms
import gc

import comfy.model_management as mm



class CoTrackerNode:
    
    def __init__(self):
        self.device = mm.get_torch_device()
        self.offload_device = mm.unet_offload_device()
        self.model = None
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "tracking_points": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Enter x and y coordinates separated by a newline. This is optional — normally not needed, as points with large motion are selected automatically. \nExample:\n500,300\n200,250"
                }),
                "grid_size": ("INT", {
                    "default": 20,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Number of divisions along both width and height to create a grid of tracking points."
                }),
                "max_num_of_points": ("INT", {
                    "default": 100,
                    "min": 1,
                    "max": 10000,
                    "step": 1
                }),
            },
            "optional": {
                "tracking_mask": ("MASK", {"tooltip": "Mask for grid coordinates"}),
                "confidence_threshold": ("FLOAT", {
                    "default": 0.90,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "min_distance": ("INT", {
                    "default": 30,
                    "min": 0,
                    "max": 500,
                    "step": 1,
                    "tooltip": "Minimum distance between tracking points"
                }),
                "force_offload": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("STRING","IMAGE")
    RETURN_NAMES = ("tracking_results","image_with_results")
    FUNCTION = "track_points"
    CATEGORY = "tracking"
    DESCRIPTION = "https://github.com/facebookresearch/co-tracker \nIf you get an OOM error, try lowering the `grid_size`."
    
    
    def load_model(self, model_type):
        try:
            if self.model is None:
                print(f"Loading CoTracker model: {model_type}")
                self.model = torch.hub.load("facebookresearch/co-tracker", model_type).to(self.device)
            self.model.to(self.device)
            self.model.eval()
            print("CoTracker model loaded successfully")
        except Exception as e:
            raise Exception(f"Failed to load CoTracker model: {str(e)}")
    
    def parse_tracking_points(self, tracking_points_str):
        points = []
        lines = tracking_points_str.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line and ',' in line:
                try:
                    x, y = line.split(',')
                    points.append([float(x.strip()), float(y.strip())])
                except ValueError:
                    print(f"parse_tracking_points : Invalid point format: {line}")
                    continue
        
        return np.array(points)
    
    def preprocess_images(self, images):
        # (B, H, W, C) -> (1, B, C, H, W)
        if len(images.shape) == 4:
            images = images.permute(0, 3, 1, 2)  # (B, C, H, W)
            images = images.unsqueeze(0)  # (1, B, C, H, W)
        
        images = images.float()
        images = images * 255
        
        return images.to(self.device)
        
    
    def prepare_query_points(self, points, video_shape):
        # video_shape:(1, B, C, H, W)
        
        # Set points on frame 0 (specify all points on the first frame)
        query_points_tensor = []
        for x, y in points:
            query_points_tensor.append([0, x, y])  # frame=0, x, y
        
        query_points_tensor = torch.tensor(query_points_tensor, dtype=torch.float32)
        
        # (1, N, 3) - (batch, points, [frame, x, y])
        query_points_tensor = query_points_tensor[None].to(self.device)
        
        return query_points_tensor
    
    def track_points(self, images, tracking_points, grid_size, max_num_of_points, tracking_mask=None, confidence_threshold=0.5, min_distance=60, force_offload=True):
        
        self.load_model("cotracker3_online")
        
        points = self.parse_tracking_points(tracking_points)
        if len(points) == 0:
            print("Info : No valid points found in tracking_points")
        
        if tracking_mask is not None:
            print(f"{tracking_mask.shape=}")
        
        images_np = images.cpu().numpy()
        images_np = (images_np * 255).astype(np.uint8)
        
        video = self.preprocess_images(images)
        
        queries = self.prepare_query_points(points, video.shape)
        
        
        if video.shape[1] <= self.model.step:
            print(f"{video.shape[1]=}")
            raise ValueError(f"At least {self.model.step+1} frames are required to perform tracking.")
        
        
        results = []
        
        if len(points) > 0:
            print(f"forward - queries")
            with torch.no_grad():
                self.model(
                    video_chunk=video,
                    is_first_step=True,
                    grid_size=0,
                    queries=queries,
                    add_support_grid=True
                )
                for ind in range(0, video.shape[1] - self.model.step, self.model.step):
                    pred_tracks, pred_visibility = self.model(
                        video_chunk=video[:, ind : ind + self.model.step * 2],
                        is_first_step=False,
                        grid_size=0,
                        queries=queries,
                        add_support_grid=True
                    )  # B T N 2,  B T N 1
            
            results, images_np = self.format_results(pred_tracks, pred_visibility, None, confidence_threshold, points, max_num_of_points, min_distance, images_np)
            
            print(f"{len(results)=}")
            
            if len(results) >= max_num_of_points:
                return (results,)
            
            max_num_of_points -= len(results)
        else:
            results = []
        
        if grid_size > 0:
            print(f"forward - grid")
            with torch.no_grad():
                self.model(
                    video_chunk=video,
                    is_first_step=True,
                    grid_size=grid_size,
                    queries=None,
                    add_support_grid=False
                )
                for ind in range(0, video.shape[1] - self.model.step, self.model.step):
                    pred_tracks, pred_visibility = self.model(
                        video_chunk=video[:, ind : ind + self.model.step * 2],
                        is_first_step=False,
                        grid_size=grid_size,
                        queries=None,
                        add_support_grid=False
                    )  # B T N 2,  B T N 1
            
            results2, images_np = self.format_results(pred_tracks, pred_visibility, tracking_mask, confidence_threshold, points, max_num_of_points, min_distance, images_np)
            print(f"{len(results2)=}")
            
            results = results + results2
        
        
        images_with_markers = torch.from_numpy(images_np)
        images_with_markers = images_with_markers.float() / 255.0
        
        if force_offload:
            self.model.to(self.offload_device)
            mm.soft_empty_cache()
            gc.collect()
        
        return (results,images_with_markers)
       
    
    
    def select_diverse_points(self, motion_sorted_indices, tracks, visibility, max_points, min_distance):
        """
        Selects spatially diverse points from among those with large motion.
        
        Args:
            motion_sorted_indices: Indices of points sorted in descending order of motion magnitude.
            tracks: Coordinate data of points across frames.
            visibility: Confidence data indicating the reliability of each point.(bool)
            max_points: Maximum number of points to select.
            min_distance: Minimum spatial distance required between selected points.
        
        Returns:
            selected_indices: A list of indices for the selected points.
        """
        if len(motion_sorted_indices) == 0:
            return []
        
        selected_indices = []
        
        # Compute the representative position of each point (average position over frames with high confidence)
        representative_positions = {}
        
        for point_idx in motion_sorted_indices:
            valid_frames = visibility[:, point_idx] == True
            if np.any(valid_frames):
                valid_positions = tracks[valid_frames, point_idx]
                representative_positions[point_idx] = np.mean(valid_positions, axis=0)
            else:
                # Fallback: average over all frames
                representative_positions[point_idx] = np.mean(tracks[:, point_idx], axis=0)
        
        # Select spatially dispersed points using a greedy algorithm
        for candidate_idx in motion_sorted_indices:
            if len(selected_indices) >= max_points:
                break
            
            candidate_pos = representative_positions[candidate_idx]
            
            # Check distance to points already selected
            too_close = False
            for selected_idx in selected_indices:
                selected_pos = representative_positions[selected_idx]
                distance = np.linalg.norm(candidate_pos - selected_pos)
                
                if distance < min_distance:
                    too_close = True
                    break
            
            # Select if sufficiently far apart
            if not too_close:
                selected_indices.append(candidate_idx)
        
        return selected_indices



    def select_points(self, tracks, visibility, vis_threshold=0.5, max_points=9, min_distance=60):
    
        n_frames, n_points, _ = tracks.shape
        
        # 1. Confidence filtering: calculate the average confidence for each point
        avg_visibility = np.mean(visibility, axis=0)
        valid_points = avg_visibility >= vis_threshold
        valid_indices = np.where(valid_points)[0]
        
        print(f"{len(valid_points)=}")
        print(f"{len(valid_indices)=}")
        
        if len(valid_indices) == 0:
            print("Warning: No points meet the confidence criteria")
            return []
        
        # 2. Calculate the magnitude of motion for each point (sum of movement distances across all frames)
        motion_magnitudes = []
        
        for point_idx in valid_indices:
            total_motion = 0.0
            valid_frame_count = 0
            
            for frame_idx in range(n_frames - 1):
                if (visibility[frame_idx, point_idx] == True and 
                    visibility[frame_idx + 1, point_idx] == True):
                    
                    pos1 = tracks[frame_idx, point_idx]
                    pos2 = tracks[frame_idx + 1, point_idx]
                    distance = np.linalg.norm(pos2 - pos1)
                    total_motion += distance
                    valid_frame_count += 1
            
            # Normalize by the number of frames (average movement distance)
            avg_motion = total_motion / max(valid_frame_count, 1)
            motion_magnitudes.append(avg_motion)
        
        motion_magnitudes = np.array(motion_magnitudes)
        
        # 3. Point selection
        selected_indices = []
        
        if len(valid_indices) <= max_points:
            selected_indices = valid_indices.tolist()
        else:
            # Sort points in descending order of motion magnitude
            motion_sorted_indices = valid_indices[np.argsort(motion_magnitudes)[::-1]]
            
            high_motion_indices = self.select_diverse_points(
                motion_sorted_indices, tracks, visibility, max_points=max_points-1, min_distance=min_distance
            )
            selected_indices.extend(high_motion_indices)            
            
            # Select only one point with the smallest motion (from points not yet selected)
            if len(selected_indices) < max_points:
                remaining_indices = [idx for idx in motion_sorted_indices if idx not in selected_indices]
                if len(remaining_indices) > 0:
                    # Use the previous coordinates
                    remaining_motions = [motion_magnitudes[np.where(valid_indices == idx)[0][0]] 
                                       for idx in remaining_indices]
                    min_motion_idx = remaining_indices[np.argmin(remaining_motions)]
                    selected_indices.append(min_motion_idx)            
        
        return selected_indices
    
    
    def format_results(self, tracks, visibility, mask, confidence_threshold, original_points, max_points, min_distance, images_np):
        # tracks : (B, T, N, 2) where B=batch, T=frames, N=points
        tracks = tracks.squeeze(0).cpu().numpy()  # (T, N, 2)
        visibility = visibility.squeeze(0).cpu().numpy()  # (T, N)
        
        num_frames, num_points, _ = tracks.shape
        
        def filter_by_mask(trs, vis, mask):
            if mask is not None:
                mask = mask.cpu().numpy()
                if len(mask.shape) == 3 and mask.shape[0] == 1:
                    mask = mask[0]
                
                initial_coords = trs[0]  # (N, 2)
                
                masked_indices = []
                
                for n in range(initial_coords.shape[0]):
                    x, y = initial_coords[n]
                    
                    if (0 <= int(x) < mask.shape[1] and 
                        0 <= int(y) < mask.shape[0] and 
                        mask[int(y), int(x)] > 0):
                        masked_indices.append(n)
                
                if len(masked_indices) > 0:
                    filtered_tracks = trs[:, masked_indices]  # (T, len(masked_indices), 2)
                    filtered_visibility = vis[:, masked_indices]  # (T, len(masked_indices))
                else:
                    # empty
                    filtered_tracks = np.empty((tracks.shape[0], 0, 2))
                    filtered_visibility = np.empty((visibility.shape[0], 0))
                
                return filtered_tracks, filtered_visibility
            else:
                return trs, vis
        
        
        tracks, visibility = filter_by_mask(tracks, visibility, mask)
        
        selected_indices = self.select_points(tracks, visibility, vis_threshold=confidence_threshold, max_points=max_points, min_distance=min_distance)
        
        
        marker_radius = 3
        marker_thickness = -1
        marker_color = (255, 0, 0)
        
        # Create tracking results for each point
        point_results = []
        
        for point_idx in selected_indices:
            point_track = []
            for frame_idx in range(num_frames):
                x, y = tracks[frame_idx, point_idx]
                vis = visibility[frame_idx, point_idx]
                
                if vis == True:
                    point_track.append({
                        "x": int(x),
                        "y": int(y),
                    })
                else:
                    # Use the previous coordinates
                    if len(point_track) > 0:
                        last_point = point_track[-1].copy()
                        point_track.append(last_point)
                        x = last_point["x"]
                        y = last_point["y"]
                    else:
                        point_track.append({
                            "x": int(x), 
                            "y": int(y),
                        })
                 
                if frame_idx < images_np.shape[0]:
                    cv2.circle(images_np[frame_idx], (int(x), int(y)), marker_radius, marker_color, marker_thickness)
            
            point_results += [json.dumps(point_track)]
        
        return point_results, images_np

def test():
    node = CoTrackerNode()
    
    tracks = np.array([[(50,50),(100,50),(50,100)],[(50,50),(100,50),(50,100)],[(50,50),(100,50),(50,100)]])
    visibility = np.array([[False,True,False],[False,True,False],[True,True,False]])
    max_points = 3
    min_distance = 10
    
    selected_indices = node.select_points(tracks, visibility, max_points=max_points, min_distance=min_distance)
    
    print(f"{selected_indices=}")
    
if __name__ == '__main__':
    test()

NODE_CLASS_MAPPINGS = {
    "CoTrackerNode": CoTrackerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CoTrackerNode": "CoTracker Point Tracking"
}

