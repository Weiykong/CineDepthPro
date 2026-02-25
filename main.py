import os
import cv2
import numpy as np
from depth_engine import DepthEngine
from lens_sim import apply_variable_blur
from evaluator import pick_hero_frame

def run_lsdr_pipeline(burst_folder):
    # 1. Load Images
    print("Loading 4K burst images...")
    files = sorted([os.path.join(burst_folder, f) for f in os.listdir(burst_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    images = [cv2.imread(f) for f in files]
    
    if not images:
        print("No images found.")
        return

    # 2. Selection (Find the Anchor)
    # We use the sharpest frame to ensure the subject is perfectly clear
    hero_idx = pick_hero_frame(images)
    hero_frame = images[hero_idx]
    print(f"Hero frame selected: {files[hero_idx]}")

    # 3. Depth Estimation (The Niche Core)
    # This identifies what is foreground and what is background
    engine = DepthEngine()
    depth_map = engine.generate_map(hero_frame)
    
    # Save depth map for debugging (optional)
    cv2.imwrite("depth_map_debug.jpg", depth_map)

    # 4. Physically Accurate Lens Simulation
    # focus_depth 255 = focus on the closest object (the person)
    # max_bokeh = 15-20 for a subtle, professional look
    print("Simulating professional large-sensor depth of field...")
    final_photo = apply_variable_blur(hero_frame, depth_map, focus_depth=255, max_bokeh=15)

    # 5. Saving the Result
    output_path = "zero_waste_lsdr_result.jpg"
    cv2.imwrite(output_path, final_photo, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    print(f"Success! Professional LSDR photo saved to {output_path}")

if __name__ == "__main__":
    sample_path = "/Users/weiyuankong/Projects/zero_waste_photo/sample"
    run_lsdr_pipeline(sample_path)