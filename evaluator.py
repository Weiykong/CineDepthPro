
import cv2
import numpy as np
def get_sharpness_score(image):
    """Calculates the Laplacian variance to estimate sharpness."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def pick_hero_frame(image_list):
    """Returns the index of the sharpest image in the burst."""
    scores = [get_sharpness_score(img) for img in image_list]
    return np.argmax(scores)