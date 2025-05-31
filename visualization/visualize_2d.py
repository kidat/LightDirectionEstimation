import cv2
from pathlib import Path

def save_2d_outputs(output_dir, vis_output, img, pairs, objects, shadows, base_name):
    # Save SSIS visualization
    cv2.imwrite(str(output_dir/f"{base_name}_ssis.png"), 
            vis_output.get_image()[:, :, ::-1])
    
    # Create pair visualization
    pair_img = img.copy()
    for obj_idx, shd_idx in pairs:
        try:
            obj_centroid = tuple(map(int, objects["centroids_2d"][obj_idx]))
            shd_centroid = tuple(map(int, shadows["centroids_2d"][shd_idx]))
            
            cv2.circle(pair_img, obj_centroid, 10, (0, 255, 0), -1)
            cv2.circle(pair_img, shd_centroid, 10, (0, 0, 255), -1)
            cv2.arrowedLine(pair_img, shd_centroid, obj_centroid, 
                        (255, 0, 0), 4, tipLength=0.3)
        except Exception:
            continue
    
    cv2.imwrite(str(output_dir/f"{base_name}_pairs.jpg"), pair_img)