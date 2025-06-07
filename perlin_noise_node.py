import numpy as np
import math
import json
import cv2
import torch

class PerlinNoise:
    """
    Simple Perlin noise implementation for coordinate randomization
    """
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        # Generate permutation table
        self.p = np.arange(256)
        np.random.shuffle(self.p)
        self.p = np.concatenate([self.p, self.p])  # Duplicate for overflow handling
    
    def fade(self, t):
        """Fade function for smooth interpolation"""
        return t * t * t * (t * (t * 6 - 15) + 10)
    
    def lerp(self, t, a, b):
        """Linear interpolation"""
        return a + t * (b - a)
    
    def grad(self, hash_val, x, y, z):
        """Gradient function"""
        h = hash_val & 15
        u = x if h < 8 else y
        v = y if h < 4 else (x if h == 12 or h == 14 else z)
        return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)
    
    def noise(self, x, y, z):
        """Generate 3D Perlin noise"""
        # Find unit cube containing point
        X = int(math.floor(x)) & 255
        Y = int(math.floor(y)) & 255
        Z = int(math.floor(z)) & 255
        
        # Find relative position in cube
        x -= math.floor(x)
        y -= math.floor(y)
        z -= math.floor(z)
        
        # Compute fade curves
        u = self.fade(x)
        v = self.fade(y)
        w = self.fade(z)
        
        # Hash coordinates of cube corners
        A = self.p[X] + Y
        AA = self.p[A] + Z
        AB = self.p[A + 1] + Z
        B = self.p[X + 1] + Y
        BA = self.p[B] + Z
        BB = self.p[B + 1] + Z
        
        # Interpolate between cube corners
        return self.lerp(w, 
                        self.lerp(v, 
                                 self.lerp(u, self.grad(self.p[AA], x, y, z),
                                          self.grad(self.p[BA], x-1, y, z)),
                                 self.lerp(u, self.grad(self.p[AB], x, y-1, z),
                                          self.grad(self.p[BB], x-1, y-1, z))),
                        self.lerp(v, 
                                 self.lerp(u, self.grad(self.p[AA+1], x, y, z-1),
                                          self.grad(self.p[BA+1], x-1, y, z-1)),
                                 self.lerp(u, self.grad(self.p[AB+1], x, y-1, z-1),
                                          self.grad(self.p[BB+1], x-1, y-1, z-1))))

def randomize_coordinates_with_perlin(coord_data, 
                                    spatial_scale=10.0, 
                                    time_scale=50.0, 
                                    intensity=1.0, 
                                    octaves=3,
                                    seed=None,
                                    mask=None):
    """
    Randomize coordinate data using 3D Perlin noise
    
    Parameters:
    coord_data: list of lists - [[(x1,y1), (x2,y2), ...], [(x1,y1), (x2,y2), ...], ...]
                Each inner list contains all frames for one coordinate point
    spatial_scale: float - spatial frequency of noise (larger = smoother in space)
    time_scale: float - temporal frequency of noise (larger = slower changes)
    intensity: float - amplitude of noise displacement
    octaves: int - number of noise octaves to combine (more = more detail)
    seed: int - random seed for reproducibility
    
    Returns:
    randomized_data: randomized coordinate data in the same format (with int coordinates)
    """
    
    # Initialize Perlin noise generator
    perlin = PerlinNoise(seed=seed)
    
    # Get data dimensions
    num_points = len(coord_data)
    num_frames = len(coord_data[0])
    
    print(f"Data shape: {num_points} coordinate points, {num_frames} frames each")
    print(f"Parameters: spatial_scale={spatial_scale}, time_scale={time_scale}, intensity={intensity}, octaves={octaves}")
    
    # Convert to numpy array for easier processing [point, frame, xy]
    coords_array = np.array(coord_data, dtype=float)
    
    def multi_octave_noise(x, y, z, octaves):
        """Generate multi-octave Perlin noise"""
        value = 0
        amplitude = 1
        frequency = 1
        max_value = 0
        
        for _ in range(octaves):
            value += perlin.noise(x * frequency, y * frequency, z * frequency) * amplitude
            max_value += amplitude
            amplitude *= 0.5
            frequency *= 2
        
        return value / max_value
    
    def is_masked(x, y):
        if mask is None:
            return True     # no mask
        return (0 <= int(x) < mask.shape[1] and 
                0 <= int(y) < mask.shape[0] and 
                mask[int(y), int(x)] > 0)
    
    # Generate noise for each coordinate point and frame
    randomized_coords = coords_array.copy()
    
    for point_idx in range(num_points):
        initial_x, initial_y = coords_array[point_idx, 0]
        
        if is_masked(initial_x, initial_y):
            for frame_idx in range(num_frames):
                # Current position
                curr_x, curr_y = coords_array[point_idx, frame_idx]
                
                # Time coordinate
                t = frame_idx / time_scale
                
                # Generate noise using current position for spatial coherence
                noise_x = multi_octave_noise(curr_x / spatial_scale, 
                                           curr_y / spatial_scale, 
                                           t, octaves) * intensity
                
                # Offset y-noise sampling to decorrelate from x-noise
                noise_y = multi_octave_noise((curr_x + 1000) / spatial_scale, 
                                           curr_y / spatial_scale, 
                                           t, octaves) * intensity
                
                # Apply noise
                new_x = curr_x + noise_x
                new_y = curr_y + noise_y
                
                # Convert back to integers
                randomized_coords[point_idx, frame_idx, 0] = round(new_x)
                randomized_coords[point_idx, frame_idx, 1] = round(new_y)
            
    
    # Convert back to original format with integer coordinates
    randomized_data = [
        [(int(randomized_coords[point, frame, 0]), int(randomized_coords[point, frame, 1])) 
         for frame in range(num_frames)]
        for point in range(num_points)
    ]
    
    return randomized_data




class PerlinCoordinateRandomizerNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tracking_results": ("STRING",),
            },
            "optional": {
                "images_for_marker": ("IMAGE", {"default": None}),
                "noise_mask": ("MASK", {"tooltip": "Mask for randomize"}),
                "spatial_scale": ("INT", {
                    "default": 1000,
                    "min": 1,
                    "max": 9999,
                    "step": 1,
                    "tooltip": "spatial_scale (pixels) / Larger → Smooth, coherent movement (nearby points move similarly) / Smaller → Chaotic, erratic movement (neighboring points move randomly)"
                }),
                "time_scale": ("INT", {
                    "default": 60,
                    "min": 1,
                    "max": 1000,
                    "step": 1,
                    "tooltip": "time_scale (frames) / Larger → Slow movement / Smaller → Fast movement"
                }),
                "intensity": ("INT", {
                    "default": 100,
                    "min": 1,
                    "max": 1000,
                    "step": 1,
                    "tooltip": "intensity (pixels) / Larger → Big displacement / Smaller → Small displacement"
                }),
                "octaves": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "tooltip": "octaves (layers) / Larger → Complex, detailed movement / Smaller → Simple, basic movement"
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
                "enabled": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("STRING","IMAGE")
    RETURN_NAMES = ("randomized_results","image_with_results")
    FUNCTION = "apply_perlin_noise"
    CATEGORY = "tracking/utility"

    def apply_perlin_noise(self, tracking_results, images_for_marker=None, noise_mask=None, spatial_scale=1000, time_scale=60, intensity=100, octaves=3, seed=42, enabled=True):
        
        if enabled == False:
            return (tracking_results, images_for_marker)
        
        if noise_mask is not None:
            noise_mask = noise_mask.cpu().numpy()
            if len(noise_mask.shape) == 3 and noise_mask.shape[0] == 1:
                noise_mask = noise_mask[0]
        
        
        raw_data = [[(d["x"], d["y"]) for d in json.loads(s)] for s in tracking_results]
        
        # Apply Perlin noise randomization
        randomized_data = randomize_coordinates_with_perlin(
            raw_data,
            spatial_scale=spatial_scale,    # spatial smoothness (larger = smoother)
            time_scale=time_scale,          # temporal smoothness (larger = slower changes)
            intensity=intensity,            # noise amplitude
            octaves=octaves,                # noise detail levels
            seed=seed,                      # for reproducibility
            mask=noise_mask
        )
        
        if images_for_marker is not None:
            images_with_markers = self.apply_marker(randomized_data, images_for_marker)
        else:
            images_with_markers = None
        
        result = [json.dumps([{"x": x, "y": y} for x, y in coords]) for coords in randomized_data]
        
        return (result, images_with_markers)
    
    def apply_marker(self, randomized_data, images):
        
        images_np = images.cpu().numpy()
        images_np = (images_np * 255).astype(np.uint8)
        
        marker_radius = 3
        marker_thickness = -1
        marker_color = (0, 0, 255)
        
        for coords in randomized_data:
            for i,(x,y) in enumerate(coords):
                if i < images_np.shape[0]:
                    cv2.circle(images_np[i], (int(x), int(y)), marker_radius, marker_color, marker_thickness)
        
        images_with_markers = torch.from_numpy(images_np)
        images_with_markers = images_with_markers.float() / 255.0
        
        return images_with_markers



NODE_CLASS_MAPPINGS = {
    "PerlinCoordinateRandomizerNode": PerlinCoordinateRandomizerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PerlinCoordinateRandomizerNode": "PerlinNoise Coordinate Randomizer"
}

