import torch
import cv2
import numpy as np
from transformers import pipeline
from PIL import Image

class DepthEngine:
    def __init__(self):
        # Using Depth-Anything-V2-Small for speed on Mac/Mobile
        # It provides incredible edge-accuracy for 4K frames
        self.pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device="mps")

    def generate_map(self, image):
        """
        Generates a normalized depth map (0-255) from a 4K image.
        """
        # Convert OpenCV BGR to PIL RGB for the transformer
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)
        
        # Run inference
        print("Analyzing scene depth...")
        depth = self.pipe(pil_img)["depth"]
        
        # Convert back to NumPy and resize to match original 4K dimensions
        depth_np = np.array(depth)
        depth_rescaled = cv2.resize(depth_np, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
        
        # Normalize to 0-255 for visualization and processing
        depth_normalized = cv2.normalize(depth_rescaled, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        return depth_normalized