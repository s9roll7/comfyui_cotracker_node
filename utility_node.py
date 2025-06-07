import numpy as np
import json
import torch
import cv2


class GridPointGeneratorNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"default": None}),
                "grid_size": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 1000,
                    "step": 1,
                    "tooltip": "Number of divisions along both width and height to create a grid of tracking points."
                }),
                "frame_count": ("INT", {
                    "default": 121,
                    "min": 1,
                    "max": 9999,
                    "step": 1,
                }),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Generate grid points only inside masked area"}),
                "existing_coordinates": ("STRING",),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("grid_coordinates","")
    FUNCTION = "generate_grid"
    CATEGORY = "tracking/utility"
    
    def generate_grid(self, image, grid_size=10, frame_count=121, mask=None, existing_coordinates=""):
        
        # (B, H, W, C)
        _, H, W, _ = image.shape
        
        if mask is not None:
            mask = mask.cpu().numpy()
            if len(mask.shape) == 3 and mask.shape[0] == 1:
                mask = mask[0]
        
        raw_data = []
        if existing_coordinates and len(existing_coordinates) > 0:        
             raw_data = [[(d["x"], d["y"]) for d in json.loads(s)] for s in existing_coordinates]
        
        # Generate grid points
        grid_points = []
        step_x = W / (grid_size + 1)  # +1 to avoid edge placement
        step_y = H / (grid_size + 1)
        
        for i in range(1, grid_size + 1):
            for j in range(1, grid_size + 1):
                x = int(i * step_x)
                y = int(j * step_y)
                
                # Check if point is within mask (if mask is provided)
                if mask is not None:
                    if y < mask.shape[0] and x < mask.shape[1]:
                        if mask[y, x] > 0:
                            grid_points.append((x, y))
                    else:
                        continue
                else:
                    grid_points.append((x, y))
        
        # Add grid points to raw_data (each grid point gets all frames)
        for grid_point in grid_points:
            point_frames = [grid_point for _ in range(frame_count)]
            raw_data.append(point_frames)
        
        
        result = [json.dumps([{"x": x, "y": y} for x, y in coords]) for coords in raw_data]
        
        return (result,)


class XYMotionAmplifierNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "coordinates": ("STRING",),
                "x_positive_amp": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                }),
                "x_negative_amp": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                }),
                "y_positive_amp": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                }),
                "y_negative_amp": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                }),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Modify points only inside masked area"}),
                "images_for_marker": ("IMAGE", {"default": None}),
            }
        }
    
    RETURN_TYPES = ("STRING","IMAGE")
    RETURN_NAMES = ("coordinates","image_with_results")
    FUNCTION = "amplify"
    CATEGORY = "tracking/utility"
    
    
    def amplify(self, coordinates, x_positive_amp, x_negative_amp, y_positive_amp, y_negative_amp, mask=None, images_for_marker=None):
        
        if mask is not None:
            mask = mask.cpu().numpy()
            if len(mask.shape) == 3 and mask.shape[0] == 1:
                mask = mask[0]
        
        raw_data = [[(d["x"], d["y"]) for d in json.loads(s)] for s in coordinates]
        
        amplified_data = []
        for point_idx, point_frames in enumerate(raw_data):
            
            should_amplify = True
            if mask is not None and len(point_frames) > 0:
                initial_x, initial_y = point_frames[0]
                # Convert to integer coordinates for mask indexing
                mask_x = int(round(initial_x))
                mask_y = int(round(initial_y))
                # Check bounds and mask value
                if (0 <= mask_y < mask.shape[0] and 0 <= mask_x < mask.shape[1]):
                    should_amplify = mask[mask_y, mask_x] > 0
                else:
                    should_amplify = False
            
            amplified_point_frames = []
            for frame_idx, (x, y) in enumerate(point_frames):
                if frame_idx == 0 or not should_amplify:
                    # First frame or point not in mask: no amplification
                    new_x, new_y = x, y
                else:
                    # Calculate movement from previous frame
                    prev_x, prev_y = point_frames[frame_idx - 1]
                    delta_x = x - prev_x
                    delta_y = y - prev_y
                    
                    # Amplify movement and add to previous amplified position
                    if delta_x > 0:
                        amplified_delta_x = delta_x * x_positive_amp
                    elif delta_x < 0:
                        amplified_delta_x = delta_x * x_negative_amp
                    else:
                        amplified_delta_x = 0
                    
                    if delta_y > 0:
                        amplified_delta_y = delta_y * y_positive_amp
                    elif delta_y < 0:
                        amplified_delta_y = delta_y * y_negative_amp
                    else:
                        amplified_delta_y = 0
                    
                    prev_amplified_x, prev_amplified_y = amplified_point_frames[frame_idx - 1]
                    new_x = prev_amplified_x + amplified_delta_x
                    new_y = prev_amplified_y + amplified_delta_y
                    
                amplified_point_frames.append((new_x, new_y))
            
            amplified_data.append(amplified_point_frames)
        
        
        if images_for_marker is not None:
            images_with_markers = self.apply_marker(amplified_data, images_for_marker)
        else:
            images_with_markers = None
        
        
        result = [json.dumps([{"x": x, "y": y} for x, y in coords]) for coords in amplified_data]
        
        return (result,images_with_markers)
        
    def apply_marker(self, amplified_data, images):
        
        images_np = images.cpu().numpy()
        images_np = (images_np * 255).astype(np.uint8)
        
        marker_radius = 3
        marker_thickness = -1
        marker_color = (0, 0, 255)
        
        for coords in amplified_data:
            for i,(x,y) in enumerate(coords):
                if i < images_np.shape[0]:
                    cv2.circle(images_np[i], (int(x), int(y)), marker_radius, marker_color, marker_thickness)
        
        images_with_markers = torch.from_numpy(images_np)
        images_with_markers = images_with_markers.float() / 255.0
        
        return images_with_markers


NODE_CLASS_MAPPINGS = {
    "GridPointGeneratorNode": GridPointGeneratorNode,
    "XYMotionAmplifierNode": XYMotionAmplifierNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GridPointGeneratorNode": "Grid Point Generator",
    "XYMotionAmplifierNode": "XY Motion Amplifier"
}

