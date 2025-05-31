import numpy as np
from scipy.spatial import KDTree
from utils.general_utils import compute_mask_centroid, compute_mask_centroid_3d

def get_object_shadow_pairs(instances, points_3d, depth_map):
    objects = {"masks": [], "centroids_2d": [], "centroids_3d": []}
    shadows = {"masks": [], "centroids_2d": [], "centroids_3d": []}
    
    try:
        pred_classes = instances.pred_classes.cpu().numpy()
        pred_masks = instances.pred_masks.cpu().numpy()
    except AttributeError:
        pred_classes = np.array(instances.pred_classes)
        pred_masks = np.array(instances.pred_masks)

    for class_id, mask in zip(pred_classes, pred_masks):
        centroid_2d = compute_mask_centroid(mask)
        centroid_3d = compute_mask_centroid_3d(mask, points_3d, depth_map)

        if centroid_2d is None or centroid_3d is None:
            continue

        target = objects if class_id == 0 else shadows
        target["masks"].append(mask)
        target["centroids_2d"].append(centroid_2d)
        target["centroids_3d"].append(centroid_3d)

    objects["centroids_3d"] = np.array(objects["centroids_3d"])
    shadows["centroids_3d"] = np.array(shadows["centroids_3d"])

    pairs = []
    if len(objects["centroids_3d"]) > 0 and len(shadows["centroids_3d"]) > 0:
        object_tree = KDTree(objects["centroids_3d"])
        used_objects = set()
        used_shadows = set()

        for shd_idx, shd_centroid in enumerate(shadows["centroids_3d"]):
            if shd_idx in used_shadows:
                continue

            _, obj_idx = object_tree.query(shd_centroid)
            if obj_idx not in used_objects:
                pairs.append((obj_idx, shd_idx))
                used_objects.add(obj_idx)
                used_shadows.add(shd_idx)

    return pairs, objects, shadows