import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from .depth_estimation import estimate_depth, get_depth_map
from .pair_matching import get_object_shadow_pairs
from .light_estimation import calculate_light_directions
from visualization.visualize_2d import save_2d_outputs
from visualization.visualize_3d import generate_3d_visualizations
from utils.general_utils import load_image

logger = logging.getLogger(__name__)

class LightEstimator:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.patch_size = 16
        
        # Initialize models and configuration
        from config.setup import setup_cfg
        from models.load_models import load_ssis, load_moge
        self.cfg = setup_cfg(args)
        self.demo = load_ssis(self.cfg)
        self.moge_model = load_moge(self.device)
        
        # Set up output directory
        self.output_dir = Path(args.output)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized LightEstimator on {self.device}")

    def process_image(self, image_path: str):
        try:
            img = load_image(image_path)
            if img is None:
                return

            base_name = Path(image_path).stem
            h, w = img.shape[:2]

            instances, vis_output = self.demo.run_on_image(img)
            points_3d = estimate_depth(img, self.moge_model, self.device, self.patch_size)
            depth_map = get_depth_map(points_3d)
            pairs, objects, shadows = get_object_shadow_pairs(instances, points_3d, depth_map)
            
            if not pairs:
                logger.warning(f"No valid pairs found in {image_path}. Skipping.")
                return

            light_data, original_lines, _ = calculate_light_directions(pairs, objects, shadows)

            if not light_data:
                logger.warning(f"No light vectors for {image_path}. Skipping.")
                return

            # Compute shared sun position
            avg_vec = np.mean([np.array(ld["vector"]) for ld in light_data], axis=0)
            avg_vec /= np.linalg.norm(avg_vec) + 1e-8
            scene_center = np.mean(points_3d.reshape(-1, 3), axis=0)
            sun_pos = scene_center + avg_vec * 5

            # Recompute extended lines to point to sun
            _, _, extended_lines = calculate_light_directions(
                pairs, objects, shadows, sun_pos=sun_pos
            )

            self._generate_outputs(
                img=img,
                vis_output=vis_output,
                points_3d=points_3d,
                depth_map=depth_map,
                pairs=pairs,
                objects=objects,
                shadows=shadows,
                light_data=light_data,
                original_lines=original_lines,
                extended_lines=extended_lines,
                sun_pos=sun_pos,
                base_name=base_name
            )

        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            raise
        finally:
            torch.cuda.empty_cache()

    def _generate_outputs(self, img, vis_output, points_3d, depth_map, 
                        pairs, objects, shadows, light_data, 
                        original_lines, extended_lines, sun_pos, base_name):
        # Save 2D outputs
        save_2d_outputs(
            self.output_dir,
            vis_output,
            img,
            pairs,
            objects,
            shadows,
            base_name
        )

        # Save depth map
        plt.imsave(self.output_dir / f"{base_name}_depth.png", -points_3d[..., 2], cmap="viridis")

        # Generate 3D visualizations
        if light_data:
            generate_3d_visualizations(
                output_dir=self.output_dir,
                depth_map=depth_map,
                img=img,
                points_3d=points_3d,
                light_data=light_data,
                original_lines=original_lines,
                extended_lines=extended_lines,
                sun_pos=sun_pos,
                base_name=base_name
            )