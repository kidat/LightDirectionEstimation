import cv2
import torch
import numpy as np

def estimate_depth(img, moge_model, device, patch_size=16):
    """Generate 3D points using MoGe model"""
    input_tensor = torch.tensor(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1),
        dtype=torch.float32
    ).unsqueeze(0).to(device) / 255.0
    
    with torch.no_grad():
        output = moge_model(
            input_tensor,
            num_tokens=(img.shape[0]//patch_size)*(img.shape[1]//patch_size)
        )
    
    points_3d = output["points"].squeeze().cpu().numpy()
    points_3d[..., 1] *= -1  # Y-down → Y-up
    points_3d[..., 2] *= -1  # Z-forward → Z-backward
    return points_3d

def get_depth_map(points_3d):
    depth = points_3d[..., 2]
    depth_min = depth.min()
    depth_range = depth.max() - depth_min
    return (depth - depth_min) / (depth_range + 1e-6)