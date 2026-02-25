import os
import cv2
import numpy as np
from depth_engine import DepthEngine
from lens_sim import apply_variable_blur
from evaluator import pick_hero_frame
from refiner import DepthRefiner

# Global variable to track the focus depth from user clicks
current_focus_depth = 255

def mouse_callback(event, x, y, flags, param):
    """Handles the user clicking on the image to select a focus point."""
    global current_focus_depth
    if event == cv2.EVENT_LBUTTONDOWN:
        depth_map = param['depth_map']
        # Look up the depth value at the specific (x, y) coordinate clicked
        current_focus_depth = depth_map[y, x]
        print(f"Focus set to depth: {current_focus_depth} (Coordinate: {x}, {y})")

def run_lsdr_app(burst_folder):
    global current_focus_depth
    
    # 1. Load 4K Burst
    print("Loading 4K burst images...")
    files = sorted([os.path.join(burst_folder, f) for f in os.listdir(burst_folder) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    images = [cv2.imread(f) for f in files]
    
    if not images:
        print("No images found in the specified folder.")
        return

    # 2. Automatically Select the Sharpest 'Hero' Frame
    hero_idx = pick_hero_frame(images)
    hero_frame = images[hero_idx]
    print(f"Hero frame selected: {files[hero_idx]}")

    # 3. Generate the High-Res Depth Map
    # This uses the Transformer-based Depth-Anything-V2
    engine = DepthEngine()
    depth_map = engine.generate_map(hero_frame)

    # 4. Interactive UI Setup
    window_name = "LSDR Interactive Focus - Click to Focus"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720) # Scaled for display
    cv2.setMouseCallback(window_name, mouse_callback, {'depth_map': depth_map})

    print("\n--- APP CONTROLS ---")
    print("LEFT CLICK: Set focus at that depth")
    print("PRESS 'S' : Save current view at 100% 4K Quality")
    print("PRESS 'Q' : Quit application")

    while True:
        # 5. Apply Physically-Accurate Lens Blur
        # max_bokeh=20 simulates a fast f/1.4 or f/1.8 lens
        display_img = apply_variable_blur(hero_frame, depth_map, 
                                          focus_depth=current_focus_depth, 
                                          max_bokeh=20,
                                          premium_look=True,
                                          cats_eye_strength=0.28,
                                          flare_strength=0.08)

        cv2.imshow(window_name, display_img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            save_path = f"lsdr_focus_{current_focus_depth}.jpg"
            cv2.imwrite(save_path, display_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            print(f"Saved 4K high-quality result to: {save_path}")
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
    
    
    
import cv2
import numpy as np
from depth_engine import DepthEngine
from lens_sim import apply_variable_blur
from evaluator import pick_hero_frame

# Global focus state
current_focus_depth = 255
needs_update = True

def mouse_callback(event, x, y, flags, param):
    global current_focus_depth, needs_update
    if event == cv2.EVENT_LBUTTONDOWN:
        # Scale mouse coordinates from preview-size back to 4K-size
        scale_x = int(x * (param['width'] / param['view_w']))
        scale_y = int(y * (param['height'] / param['view_h']))
        
        depth_map = param['depth_map']
        current_focus_depth = depth_map[scale_y, scale_x]
        needs_update = True
        print(f"Focus set to depth: {current_focus_depth}")

def run_fast_lsdr(burst_folder):
    global current_focus_depth, needs_update
    
    # 1. Setup
    files = sorted([os.path.join(burst_folder, f) for f in os.listdir(burst_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    hero_frame = cv2.imread(files[0]) # Simplified for now
    h, w = hero_frame.shape[:2]

    # 2. Pre-calculate Depth (The only slow AI part)
    engine = DepthEngine()
    depth_map = engine.generate_map(hero_frame)

    # 3. Create a Fast Preview Frame (e.g., 1280px wide)
    preview_w = 1280
    preview_h = int(h * (preview_w / w))
    hero_preview = cv2.resize(hero_frame, (preview_w, preview_h))
    depth_preview = cv2.resize(depth_map, (preview_w, preview_h))

    # Save depth map for debugging (optional)
    cv2.imwrite("depth_map_debug.jpg", depth_map)
    
    # 4. UI Setup
    window_name = "Fast LSDR Preview (Click to Focus)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback, {
        'depth_map': depth_map, 'width': w, 'height': h, 
        'view_w': preview_w, 'view_h': preview_h
    })

    print("READY. Press 'S' for 4K Save, 'Q' to Quit.")

    display_img = hero_preview.copy()

    while True:
        if needs_update:
            # Process on the TINY preview frame for instant feedback
            display_img = apply_variable_blur(hero_preview, depth_preview, 
                                              focus_depth=current_focus_depth, 
                                              max_bokeh=15,
                                              premium_look=True,
                                              cats_eye_strength=0.22,
                                              flare_strength=0.06)
            needs_update = False

        cv2.imshow(window_name, display_img)
        key = cv2.waitKey(30) & 0xFF
        # Inside your main loop...
        if key == ord('r'):
            print("Running Multi-View Refinement... using secondary frame for edge truth.")

            refiner = DepthRefiner()
            
            # We use images[0] and images[1] from your burst folder
            second_frame = cv2.imread(files[1]) 
            
            # Update the global depth_map with the refined version
            depth_map = refiner.refine(hero_frame, second_frame, depth_map)
            
            # Update preview versions too
            depth_preview = cv2.resize(depth_map, (preview_w, preview_h))
            # Save depth map for debugging (optional)
            cv2.imwrite("depth_map_refined_debug.jpg", depth_map)
            needs_update = True
            print("Refinement Complete. Edges should now be mathematically sharper.")
            
        if key == ord('s'):
            print("Processing 4K Final Render... please wait...")
            # Only do the heavy 4K math when saving
            final_4k = apply_variable_blur(hero_frame, depth_map, 
                                           focus_depth=current_focus_depth, 
                                           max_bokeh=20,
                                           premium_look=True,
                                           cats_eye_strength=0.32,
                                           flare_strength=0.10)
            cv2.imwrite(f"final_4k_focus_{current_focus_depth}.jpg", final_4k)
            print("4K Save Complete!")
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Ensure this points to your specific folder
    MY_BURST_PATH = "/Users/weiyuankong/Projects/zero_waste_photo/sample"
    run_fast_lsdr(MY_BURST_PATH)
