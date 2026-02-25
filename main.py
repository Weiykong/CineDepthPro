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
                                              max_bokeh=15)
            needs_update = False

        cv2.imshow(window_name, display_img)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('s'):
            print("Processing 4K Final Render... please wait...")
            # Only do the heavy 4K math when saving
            final_4k = apply_variable_blur(hero_frame, depth_map, 
                                           focus_depth=current_focus_depth, 
                                           max_bokeh=20)
            cv2.imwrite(f"final_4k_focus_{current_focus_depth}.jpg", final_4k)
            print("4K Save Complete!")
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    # Ensure this points to your specific folder
    MY_BURST_PATH = "/Users/weiyuankong/Projects/zero_waste_photo/sample"
    run_fast_lsdr(MY_BURST_PATH)