import cv2
import numpy as np

class DepthRefiner:
    def refine(self, hero_img, second_img, ai_depth_map):
        """
        Uses a second frame to sharpen the edges of the AI-generated depth map.
        """
        # 1. Align the second image to the hero frame perfectly
        # We need this so the 'shift' we see is only from depth parallax
        gray_hero = cv2.cvtColor(hero_img, cv2.COLOR_BGR2GRAY)
        gray_second = cv2.cvtColor(second_img, cv2.COLOR_BGR2GRAY)
        
        # 2. Calculate Geometric Disparity (The 'Truth' Map)
        # numDisparities must be a multiple of 16. Higher = better for close objects.
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=32, 
            blockSize=5,
            P1=8 * 3 * 5**2,
            P2=32 * 3 * 5**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32
        )
        
        disparity = stereo.compute(gray_hero, gray_second).astype(np.float32) / 16.0
        
        # 3. Normalize Geometric Depth
        geo_depth = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # 4. Create an Edge Mask (Where AI usually struggles)
        edges = cv2.Canny(gray_hero, 100, 200)
        edge_mask = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1) / 255.0
        
        # 5. Instead of a simple blend, use the new 'refined_blend' logic
        # This uses the Hero image to 'guide' the depth map
        geo_smooth = cv2.ximgproc.guidedFilter(guide=hero_img, src=geo_depth, radius=10, eps=0.01)
        
        # 6. Confidence-based synthesis
        diff = cv2.absdiff(ai_depth_map, geo_smooth)
        confidence = np.where(diff < 30, 1.0, 0.0).astype(np.float32)
        
        # Blend: We keep the AI's smoothness but inject the Geo's edge truth
        refined_map = (ai_depth_map * (1 - 0.4 * confidence) + geo_smooth * (0.4 * confidence)).astype(np.uint8)
        
        return refined_map
    
    def refined_blend(ai_map, geo_map, hero_img):
        # 1. Use a Guided Filter to smooth the noisy geometric map
        # This uses the original photo to 'guide' the depth, ensuring 
        # the noise doesn't cross over sharp visual edges.
        geo_smooth = cv2.ximgproc.guidedFilter(guide=hero_img, src=geo_map, radius=10, eps=0.01)
        
        # 2. Create a 'Confidence Mask'
        # We only trust the geometric map where the AI and Geo maps 
        # roughly agree. If they differ by too much, it's probably noise.
        diff = cv2.absdiff(ai_map, geo_smooth)
        confidence = np.where(diff < 30, 1.0, 0.0).astype(np.float32)
        
        # 3. Final Blend: Trust AI mostly, use Geo to 'tighten' the edges
        final = (ai_map * (1 - 0.3 * confidence) + geo_smooth * (0.3 * confidence)).astype(np.uint8)
        return final