import cv2
import numpy as np

def _smoothstep(x, edge0, edge1):
    denom = max(edge1 - edge0, 1e-6)
    t = np.clip((x - edge0) / denom, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _line_kernel(length, orientation):
    """Build a small normalized motion-blur kernel."""
    length = max(3, int(length))
    if length % 2 == 0:
        length += 1

    k = np.zeros((length, length), dtype=np.float32)
    c = length // 2

    if orientation == "h":
        k[c, :] = 1.0
    elif orientation == "v":
        k[:, c] = 1.0
    elif orientation == "d1":
        np.fill_diagonal(k, 1.0)
    else:  # d2
        np.fill_diagonal(np.fliplr(k), 1.0)

    # Slightly soften the line to avoid crunchy streak artifacts.
    k = cv2.GaussianBlur(k, (0, 0), sigmaX=max(0.6, length / 10.0), sigmaY=max(0.6, length / 10.0))
    s = float(k.sum())
    if s > 0:
        k /= s
    return k


def _add_premium_lens_character(base_image, source_image, blur_radius_map, max_bokeh,
                                cats_eye_strength=0.30, flare_strength=0.10):
    """
    Adds lens-character artifacts that mimic premium fast glass:
    - Cat's-eye bokeh bias on bright, out-of-focus highlights near frame edges
    - Subtle bloom / streak flare on very bright light sources
    """
    if max_bokeh <= 0 or (cats_eye_strength <= 0 and flare_strength <= 0):
        return base_image

    h, w = source_image.shape[:2]
    base = base_image.astype(np.float32) / 255.0
    src = source_image.astype(np.float32) / 255.0

    blur_norm = np.clip(blur_radius_map / max(float(max_bokeh), 1e-6), 0.0, 1.0)

    # Luminance on [0,1] (BGR input)
    luma = 0.114 * src[:, :, 0] + 0.587 * src[:, :, 1] + 0.299 * src[:, :, 2]

    # Bright speculars become bokeh highlights when out-of-focus.
    highlight_mask = _smoothstep(luma, 0.65, 0.98)
    defocus_mask = _smoothstep(blur_norm, 0.18, 0.95)

    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5
    dx = x - cx
    dy = y - cy
    rx = dx / max(cx, 1.0)
    ry = dy / max(cy, 1.0)
    r = np.sqrt(rx * rx + ry * ry)
    edge_mask = _smoothstep(r, 0.28, 1.0)

    # Seed for glow / bokeh highlights: use only bright regions, preserve their colors.
    bokeh_seed = src * (highlight_mask * defocus_mask)[:, :, None]

    if cats_eye_strength > 0:
        line_len = max(5, int(max_bokeh * 2.0) | 1)
        dir_h = cv2.filter2D(bokeh_seed, -1, _line_kernel(line_len, "h"), borderType=cv2.BORDER_REPLICATE)
        dir_v = cv2.filter2D(bokeh_seed, -1, _line_kernel(line_len, "v"), borderType=cv2.BORDER_REPLICATE)
        dir_d1 = cv2.filter2D(bokeh_seed, -1, _line_kernel(line_len, "d1"), borderType=cv2.BORDER_REPLICATE)
        dir_d2 = cv2.filter2D(bokeh_seed, -1, _line_kernel(line_len, "d2"), borderType=cv2.BORDER_REPLICATE)
        iso_sigma = max(1.0, max_bokeh * 0.45)
        dir_iso = cv2.GaussianBlur(bokeh_seed, (0, 0), sigmaX=iso_sigma, sigmaY=iso_sigma)

        # Tangential direction around the optical center drives cat's-eye orientation.
        eps = 1e-6
        inv_len = 1.0 / np.maximum(np.sqrt(dx * dx + dy * dy), eps)
        tx = -dy * inv_len
        ty = dx * inv_len

        # Orientation weights (axis-based, not arrow-based).
        w_h = np.abs(tx)
        w_v = np.abs(ty)
        inv_sqrt2 = 0.70710678
        w_d1 = np.abs(tx * inv_sqrt2 + ty * inv_sqrt2)
        w_d2 = np.abs(tx * inv_sqrt2 - ty * inv_sqrt2)
        w_sum = w_h + w_v + w_d1 + w_d2 + 1e-6
        w_h /= w_sum
        w_v /= w_sum
        w_d1 /= w_sum
        w_d2 /= w_sum

        oriented = (
            dir_h * w_h[:, :, None] +
            dir_v * w_v[:, :, None] +
            dir_d1 * w_d1[:, :, None] +
            dir_d2 * w_d2[:, :, None]
        )

        # Inward offset increases near the edge, mimicking clipped off-axis bokeh.
        shift_px = edge_mask * defocus_mask * np.clip(max_bokeh * 0.35, 0.0, 10.0)
        rr = np.maximum(np.sqrt(dx * dx + dy * dy), 1.0)
        nx = dx / rr
        ny = dy / rr
        map_x = (x + nx * shift_px).astype(np.float32)
        map_y = (y + ny * shift_px).astype(np.float32)
        oriented_inward = cv2.remap(oriented, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)

        cat_mix = dir_iso * (1.0 - edge_mask[:, :, None]) + oriented_inward * edge_mask[:, :, None]
        # Stronger on highlights + defocus, but keep it restrained.
        cat_gain = (0.6 + 0.8 * highlight_mask) * defocus_mask * (0.3 + 0.7 * edge_mask)
        base += cats_eye_strength * cat_mix * cat_gain[:, :, None]

    if flare_strength > 0:
        # Very bright lights create bloom and a faint streak. Keep it subtle to avoid gimmicky flare.
        flare_mask = _smoothstep(luma, 0.82, 1.0)
        flare_seed = src * flare_mask[:, :, None]

        bloom_sigma = max(1.5, max_bokeh * 0.65)
        bloom = cv2.GaussianBlur(flare_seed, (0, 0), sigmaX=bloom_sigma, sigmaY=bloom_sigma)

        streak_len = max(9, int(max_bokeh * 5) | 1)
        streak_kernel = _line_kernel(streak_len, "h")
        streak = cv2.filter2D(flare_seed, -1, streak_kernel, borderType=cv2.BORDER_REPLICATE)
        streak = cv2.GaussianBlur(streak, (0, 0), sigmaX=max_bokeh * 0.8 + 1.0, sigmaY=0.8)

        # Slight warm tint on bloom and cooler streak to feel more glass-like.
        bloom_tint = np.array([0.95, 1.00, 1.08], dtype=np.float32).reshape(1, 1, 3)  # BGR
        streak_tint = np.array([1.08, 1.00, 0.95], dtype=np.float32).reshape(1, 1, 3)  # BGR

        base += flare_strength * (0.90 * bloom * bloom_tint + 0.35 * streak * streak_tint)

    return np.clip(base * 255.0, 0, 255).astype(np.uint8)


def apply_variable_blur(image, depth_map, focus_depth=255, max_bokeh=20,
                        premium_look=True, cats_eye_strength=0.30, flare_strength=0.10):
    """
    Simulates a shallow depth of field using a soft-edge, layer-based algorithm.
    This method creates smoother transitions between blur levels by blending
    a few pre-blurred image layers, which is more efficient and produces
    higher quality results than iterative blurring.

    focus_depth: The grayscale value (0-255) in the depth map that should remain
                 in sharp focus. 255 is typically closest.
    max_bokeh:   The maximum blur radius, corresponding to the most out-of-focus
                 areas. A larger value creates a more intense depth of field effect.
    premium_look: If True, adds highlight-weighted cat's-eye bokeh bias and
                  subtle bloom / flare to mimic premium fast lenses.
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

    final_image = final_image.astype(np.uint8)

    if premium_look:
        final_image = _add_premium_lens_character(
            final_image,
            image,
            blur_radius_map,
            max_bokeh=max_bokeh,
            cats_eye_strength=cats_eye_strength,
            flare_strength=flare_strength,
        )

    return final_image
