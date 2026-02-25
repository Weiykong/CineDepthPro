import cv2
import numpy as np

def apply_variable_blur(image, depth_map, focus_depth=255, max_bokeh=20):
    """
    Simulates a shallow depth of field using a soft-edge, layer-based algorithm.
    This method creates smoother transitions between blur levels by blending
    a few pre-blurred image layers, which is more efficient and produces
    higher quality results than iterative blurring.

    focus_depth: The grayscale value (0-255) in the depth map that should remain
                 in sharp focus. 255 is typically closest.
    max_bokeh:   The maximum blur radius, corresponding to the most out-of-focus
                 areas. A larger value creates a more intense depth of field effect.
    """
    if max_bokeh <= 0:
        return image

    # 1. Normalize the depth map to create a blur radius map.
    # This map indicates the desired blur radius (from 0 to max_bokeh) for each pixel.
    blur_radius_map = np.abs(depth_map.astype(np.float32) - focus_depth)
    blur_radius_map = (blur_radius_map / 255.0) * max_bokeh

    # 2. Create a discrete set of blurred image layers.
    # The number of layers is a trade-off between quality (more layers) and
    # performance (fewer layers). 4-8 layers is usually a good range.
    num_layers = int(max_bokeh / 3) + 2
    num_layers = max(2, min(num_layers, 10)) # Clamp between 2 and 10 layers

    blur_levels = np.linspace(0, max_bokeh, num_layers)
    
    layers = []
    for level in blur_levels:
        # Kernel size must be an odd integer.
        kernel_size = int(level) * 2 + 1
        if kernel_size > 1:
            # Use GaussianBlur for a realistic circular blur (bokeh).
            layers.append(cv2.GaussianBlur(image, (kernel_size, kernel_size), 0))
        else:
            # The sharpest layer is the original image.
            layers.append(image)

    # 3. Perform a vectorized blend between layers.
    # This is the core of the "soft-edge" algorithm. For each pixel, we find which
    # two blur layers it falls between and blend them with a weight.
    layers = [l.astype(np.float32) for l in layers]
    final_image = np.zeros_like(layers[0])

    # The first layer (sharpest) contributes to pixels with a blur radius
    # up to the second blur level.
    # Calculate the weight for the first layer (it fades out as we approach blur_levels[1])
    weight = 1.0 - np.clip((blur_radius_map - blur_levels[0]) / (blur_levels[1] - blur_levels[0]), 0, 1)
    final_image += layers[0] * np.expand_dims(weight, axis=2)

    # Blend the intermediate layers.
    for i in range(1, num_layers - 1):
        # Weight for this layer has two components:
        # 1. Fade-in from the previous level (i-1).
        w_in = np.clip((blur_radius_map - blur_levels[i-1]) / (blur_levels[i] - blur_levels[i-1]), 0, 1)
        # 2. Fade-out to the next level (i+1).
        w_out = 1.0 - np.clip((blur_radius_map - blur_levels[i]) / (blur_levels[i+1] - blur_levels[i]), 0, 1)
        # The final weight is the intersection of these two ramps (a triangle shape).
        weight = np.minimum(w_in, w_out)
        final_image += layers[i] * np.expand_dims(weight, axis=2)

    # The final layer (blurriest) contributes to pixels with a blur radius
    # starting from the second to last blur level.
    # It fades in as we approach blur_levels[-1].
    weight = np.clip((blur_radius_map - blur_levels[-2]) / (blur_levels[-1] - blur_levels[-2]), 0, 1)
    final_image += layers[-1] * np.expand_dims(weight, axis=2)

    return final_image.astype(np.uint8)
