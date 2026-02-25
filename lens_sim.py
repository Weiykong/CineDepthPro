import cv2
import numpy as np

def apply_variable_blur(image, depth_map, focus_depth=255, max_bokeh=20):
    """
    Simulates a shallow depth of field.
    focus_depth: The 'distance' that remains sharp (255 = closest).
    max_bokeh: The maximum size of the blur circles.
    """
    # Calculate the blur intensity based on distance from the focus point
    blur_map = np.abs(depth_map.astype(float) - focus_depth) / 255.0
    blur_map = (blur_map * max_bokeh).astype(np.uint8)
    
    # In a professional app, we use a 'Disk Kernel' to simulate lens iris
    # For the prototype, we use a sophisticated bilateral-style blend
    result = image.copy()
    for size in range(1, max_bokeh + 1):
        # Create a blurred version for this specific depth layer
        layer_blur = cv2.GaussianBlur(image, (size*2+1, size*2+1), 0)
        
        # Apply only to pixels that belong to this depth 'slice'
        mask = cv2.threshold(blur_map, size, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
        
        result = (result * (1 - mask) + layer_blur * mask).astype(np.uint8)
        
    return result