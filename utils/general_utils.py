import cv2
import numpy as np
from pathlib import Path

def load_image(path: str):
    if not Path(path).exists():
        return None
    
    img = cv2.imread(path)
    if img is None or img.size == 0:
        return None
        
    return img

def compute_mask_centroid(mask: np.ndarray):
    y, x = np.where(mask)
    if len(x) == 0 or len(y) == 0:
        return None
    return np.array([np.mean(x), np.mean(y)])

def compute_mask_centroid_3d(mask: np.ndarray, points_3d: np.ndarray, depth_map: np.ndarray):
    h, w = depth_map.shape
    y, x = np.where(mask)
    
    if len(x) == 0 or len(y) == 0:
        return None

    cy = min(max(int(np.mean(y)), 0), h-1)
    cx = min(max(int(np.mean(x)), 0), w-1)

    return points_3d[cy, cx]

def rotation_matrix(from_vec: np.ndarray, to_vec: np.ndarray):
    a = from_vec / np.linalg.norm(from_vec)
    b = to_vec / np.linalg.norm(to_vec)
    v = np.cross(a, b)
    c = np.dot(a, b)
    
    if c < -1 + 1e-6:
        return -np.eye(3)
    
    skew = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    return np.eye(3) + skew + skew @ skew * (1 / (1 + c))