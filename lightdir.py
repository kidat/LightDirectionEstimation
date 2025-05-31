""" Light Direction Estimation using MoGe and SSIS with Open3D visualization """
import sys
import os
import logging
import argparse
import glob
import multiprocessing as mp
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import cv2
import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
import open3d as o3d
from scipy.spatial import KDTree

# Suppress unnecessary warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Configure environment paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../MoGe")))
from predictor import VisualizationDemo
from adet.config import get_cfg
from moge.model.v1 import MoGeModel

# Configure logging
logger = logging.getLogger(__name__)

class LightEstimator:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.patch_size = 16
        
        # Initialize models and configuration
        self.cfg = self._setup_cfg()
        self.demo = VisualizationDemo(self.cfg)
        self.moge_model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(self.device).eval()
        
        # Set up output directory
        self.output_dir = Path(args.output)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized LightEstimator on {self.device}")

    def _setup_cfg(self):
        """Configure Detectron2 settings"""
        cfg = get_cfg()
        cfg.VERSION = 2  # Add default version
        
        if not Path(self.args.config_file).exists():
            raise FileNotFoundError(f"Config file {self.args.config_file} not found!")
        
        cfg.merge_from_file(self.args.config_file)
        cfg.merge_from_list(self.args.opts)
 
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        return cfg.clone()
    
    def process_image(self, image_path: str):
        """Full processing pipeline for a single image"""
        try:
            img = self._load_image(image_path)
            if img is None:
                return

            base_name = Path(image_path).stem
            h, w = img.shape[:2]

            instances, vis_output = self.demo.run_on_image(img)
            self.points_3d = self._estimate_depth(img, h, w)
            pairs, self.objects, self.shadows = self._get_object_shadow_pairs(instances)
            if not pairs:
                logger.warning(f"No valid pairs found in {image_path}. Skipping.")
                return

            light_data, original_lines, _ = self._calculate_light_directions(pairs, self.objects, self.shadows)

            if not light_data:
                logger.warning(f"No light vectors for {image_path}. Skipping.")
                return

            # Compute shared sun position
            avg_vec = np.mean([np.array(ld["vector"]) for ld in light_data], axis=0)
            avg_vec /= np.linalg.norm(avg_vec) + 1e-8
            scene_center = np.mean(self.points_3d.reshape(-1, 3), axis=0)
            sun_pos = scene_center + avg_vec * 5

            # Recompute extended lines to point to sun
            _, _, extended_lines = self._calculate_light_directions(
                pairs, self.objects, self.shadows, sun_pos=sun_pos
            )

            self._generate_outputs(
                img=img,
                vis_output=vis_output,
                pairs=pairs,
                points_3d=self.points_3d,
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
            if hasattr(self, 'points_3d'):
                del self.points_3d
            torch.cuda.empty_cache()

    def _get_object_shadow_pairs(self, instances) -> Tuple[List, Dict, Dict]:
        """Identify object-shadow pairs with 3D centroids (one-to-one pairing)"""
        objects = {"masks": [], "centroids_2d": [], "centroids_3d": []}
        shadows = {"masks": [], "centroids_2d": [], "centroids_3d": []}
        depth_map = self._get_depth_map()

        try:
            pred_classes = instances.pred_classes.cpu().numpy()
            pred_masks = instances.pred_masks.cpu().numpy()
        except AttributeError:
            pred_classes = np.array(instances.pred_classes)
            pred_masks = np.array(instances.pred_masks)

        # Separate objects and shadows
        for class_id, mask in zip(pred_classes, pred_masks):
            centroid_2d = self._compute_mask_centroid(mask)
            centroid_3d = self._compute_mask_centroid_3d(mask, depth_map)

            if centroid_2d is None or centroid_3d is None:
                continue

            target = objects if class_id == 0 else shadows
            target["masks"].append(mask)
            target["centroids_2d"].append(centroid_2d)
            target["centroids_3d"].append(centroid_3d)

        # Convert centroids to arrays
        objects["centroids_3d"] = np.array(objects["centroids_3d"])
        shadows["centroids_3d"] = np.array(shadows["centroids_3d"])

        pairs = []
        used_objects = set()
        used_shadows = set()

        if len(objects["centroids_3d"]) > 0 and len(shadows["centroids_3d"]) > 0:
            object_tree = KDTree(objects["centroids_3d"])

            for shd_idx, shd_centroid in enumerate(shadows["centroids_3d"]):
                if shd_idx in used_shadows:
                    continue

                # Find nearest object to this shadow
                _, obj_idx = object_tree.query(shd_centroid)

                # Ensure object is not already used
                if obj_idx not in used_objects:
                    pairs.append((obj_idx, shd_idx))
                    used_objects.add(obj_idx)
                    used_shadows.add(shd_idx)

        return pairs, objects, shadows


    def _generate_3d_visualizations(self, img: np.ndarray, depth_map: np.ndarray,
                                    points_3d: np.ndarray, light_data: List,
                                    original_lines: List, extended_lines: List,
                                    sun_pos: np.ndarray, base_name: str):
        """Save 3D scene with light directions and sun symbol as PLY"""
        try:
            pcd = self._create_o3d_pointcloud(depth_map, img)
            pcd.estimate_normals()

            combined_mesh = o3d.geometry.TriangleMesh()

            # Point cloud mesh
            radii = [0.005, 0.01, 0.02, 0.04]
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector(radii))
            combined_mesh += mesh

            # Add original (shadow → object) red arrows
            for start, end in original_lines:
                arrow = self._create_3d_arrow(start, end, color=[1, 0, 0])
                combined_mesh += arrow

            # Add extended (shadow → sun) yellow arrows
            for start, end in extended_lines:
                arrow = self._create_3d_arrow(start, end, color=[1, 1, 0])
                combined_mesh += arrow

            # Add sun marker
            sun_core = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
            sun_core.paint_uniform_color([1.0, 1.0, 1.0])
            sun_core.translate(sun_pos)

            sun_glow = o3d.geometry.TriangleMesh.create_sphere(radius=0.10)
            sun_glow.paint_uniform_color([1.0, 0.85, 0.3])
            sun_glow.translate(sun_pos)

            combined_mesh += sun_core + sun_glow

            o3d.io.write_triangle_mesh(
                str(self.output_dir / f"{base_name}_full_scene.ply"),
                combined_mesh,
                write_vertex_colors=True
            )
        except Exception as e:
            logger.error(f"3D visualization failed: {str(e)}")

    def _create_3d_arrow(self, start: np.ndarray, end: np.ndarray, color=[1, 0, 0]) -> o3d.geometry.TriangleMesh:
        """Creates a 3D arrow from start to end"""
        start = np.asarray(start)
        end = np.asarray(end)
        vec = end - start
        length = np.linalg.norm(vec)

        if length < 1e-6:
            return o3d.geometry.TriangleMesh()

        direction = vec / length

        # Arrow proportions
        cone_height = 0.2 * length
        cylinder_height = 0.8 * length

        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.01, height=cylinder_height)
        cone = o3d.geometry.TriangleMesh.create_cone(radius=0.02, height=cone_height)

        cylinder.paint_uniform_color(color)
        cone.paint_uniform_color(color)

        # Align arrow to direction
        rot = self._rotation_matrix(np.array([0, 0, 1]), direction)
        cylinder.rotate(rot)
        cone.rotate(rot)

        cylinder.translate(start + direction * (cylinder_height / 2))
        cone.translate(start + direction * (cylinder_height + cone_height / 2))

        return cylinder + cone
            
    def _create_basic_lines(self, line_segments):
        """Create line set with customizable properties"""
        line_set = o3d.geometry.LineSet()
        points = []
        lines = []
        
        # Increase line thickness using cylinder meshes
        for start, end in line_segments:
            cylinder = o3d.geometry.TriangleMesh.create_cylinder(
                radius=0.005, 
                height=np.linalg.norm(end - start)
            )
            cylinder.translate((start + end)/2)
            cylinder.rotate(self._rotation_matrix(
                [0,1,0],  # Default cylinder orientation
                (end - start)/np.linalg.norm(end - start)
            ))
            line_set += cylinder
            
        return line_set


    def _estimate_depth(self, img: np.ndarray, h: int, w: int) -> np.ndarray:
        """Generate 3D points using MoGe model"""
        input_tensor = torch.tensor(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1),
            dtype=torch.float32
        ).unsqueeze(0).to(self.device) / 255.0
        
        with torch.no_grad():
            output = self.moge_model(
                input_tensor,
                num_tokens=(h//self.patch_size)*(w//self.patch_size)
            )

        
        # Store the 3D points for later use (this is to convert from openCV to open3D)
        self.points_3d = output["points"].squeeze().cpu().numpy()
        self.points_3d[..., 1] *= -1  # Y-down → Y-up
        self.points_3d[..., 2] *= -1  # Z-forward → Z-backward
        return self.points_3d
    
    def _get_depth_map(self):
        """Safer depth normalization with epsilon"""
        depth = self.points_3d[..., 2]
        depth_min = depth.min()
        depth_range = depth.max() - depth_min
        return (depth - depth_min) / (depth_range + 1e-6)

    def _load_image(self, path: str) -> Optional[np.ndarray]:
        """Load and validate input image"""
        if not Path(path).exists():
            logger.error(f"Image not found: {path}")
            return None
        
        img = cv2.imread(path)
        if img is None:
            logger.error(f"Failed to read image: {path}")
            return None
        
        if img.size == 0:
            logger.error(f"Empty image: {path}")
            return None
            
        return img

    def _compute_mask_centroid(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """Calculate 2D centroid of a binary mask"""
        y, x = np.where(mask)
        if len(x) == 0 or len(y) == 0:
            return None
        return np.array([np.mean(x), np.mean(y)])

    def _compute_mask_centroid_3d(self, mask: np.ndarray, depth_map: np.ndarray) -> Optional[np.ndarray]:
        """Calculate 3D centroid with safety checks"""
        h, w = depth_map.shape
        y, x = np.where(mask)
        
        if len(x) == 0 or len(y) == 0:
            return None

        # Add coordinate clamping
        cy = min(max(int(np.mean(y)), 0), h-1)
        cx = min(max(int(np.mean(x)), 0), w-1)

        # Get depth value
        z = depth_map[cy, cx]  

        return self.points_3d[cy, cx]

    def _calculate_light_directions(self, pairs, objects, shadows, sun_pos=None):
        light_data = []
        original_lines = []
        extended_lines = []

        EXTENSION_FACTOR = 2.0  # Only used if sun_pos is None

        for obj_idx, shd_idx in pairs:
            try:
                obj_3d = objects["centroids_3d"][obj_idx]
                shd_3d = shadows["centroids_3d"][shd_idx]

                vector = obj_3d - shd_3d
                direction = vector / np.linalg.norm(vector)

                original_lines.append((shd_3d, obj_3d))

                # If sun_pos is provided, use it
                if sun_pos is not None:
                    extended_end = sun_pos
                else:
                    extended_end = obj_3d + direction * EXTENSION_FACTOR

                extended_lines.append((shd_3d, extended_end))

                light_data.append({
                    "vector": direction.tolist(),
                    "object": obj_3d.tolist(),
                    "shadow": shd_3d.tolist(),
                    "extended": extended_end.tolist()
                })

            except Exception as e:
                logger.warning(f"Skipping pair {obj_idx}-{shd_idx}: {str(e)}")

        return light_data, original_lines, extended_lines

    def _save_2d_outputs(self, vis_output, img: np.ndarray, pairs: List, base_name: str):
        """Complete 2D visualization handling"""
        # Save SSIS visualization
        cv2.imwrite(str(self.output_dir/f"{base_name}_ssis.png"), 
                vis_output.get_image()[:, :, ::-1])  # Convert RGB to BGR
        
        # Create pair visualization
        pair_img = img.copy()
        for obj_idx, shd_idx in pairs:
            try:
                # Use precomputed centroids from object/shadow data
                obj_centroid = tuple(map(int, self.objects["centroids_2d"][obj_idx]))
                shd_centroid = tuple(map(int, self.shadows["centroids_2d"][shd_idx]))
                
                # Draw elements
                cv2.circle(pair_img, obj_centroid, 10, (0, 255, 0), -1)  # Green
                cv2.circle(pair_img, shd_centroid, 10, (0, 0, 255), -1)  # Red
                cv2.arrowedLine(pair_img, shd_centroid, obj_centroid, 
                            (255, 0, 0), 4, tipLength=0.3)  # Blue arrow
            except Exception as e:
                logger.warning(f"Skipping pair drawing: {str(e)}")
        
        # Save pair visualization
        cv2.imwrite(str(self.output_dir/f"{base_name}_pairs.jpg"), pair_img)

    def _create_o3d_pointcloud(self, depth_map: np.ndarray, image: np.ndarray) -> o3d.geometry.PointCloud:
        """Create point cloud using MoGe's actual 3D point predictions"""
        h, w = depth_map.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Use MoGe output points directly
        points = self.points_3d.reshape(-1, 3).copy()
        colors = image_rgb.reshape(-1, 3) / 255.0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd.estimate_normals()
        return pcd

    @staticmethod
    def _rotation_matrix(from_vec: np.ndarray, to_vec: np.ndarray) -> np.ndarray:
        """Calculate rotation matrix between two vectors"""
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
        
    def _generate_outputs(self, img: np.ndarray, vis_output, 
                        pairs: List, points_3d: np.ndarray, light_data: List,
                        original_lines: List, extended_lines: List,
                        sun_pos: np.ndarray, base_name: str):
        """Complete outputs generation"""
        self._save_2d_outputs(vis_output, img, pairs, base_name)

        depth_map = -points_3d[..., 2]
        plt.imsave(self.output_dir / f"{base_name}_depth.png", depth_map, cmap="viridis")

        if light_data:
            self._generate_3d_visualizations(
                img=img,
                depth_map=depth_map,
                points_3d=points_3d,
                light_data=light_data,
                original_lines=original_lines,
                extended_lines=extended_lines,
                sun_pos=sun_pos,
                base_name=base_name
            )

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 + MoGe: Object-Shadow 3D Estimation")
    parser.add_argument(
        "--config-file", 
        default="../configs/SSIS/MS_R_101_BiFPN_SSISv2_demo.yaml", 
        metavar="FILE",
        help="Path to config file (default: %(default)s)"
    )
    parser.add_argument(
        "--input", 
        default="./", 
        help="Path to input image or directory (default: %(default)s)"
    )
    parser.add_argument(
        "--output", 
        default="./res/", 
        help="Output directory (default: %(default)s)"
    )
    parser.add_argument(
        "--confidence-threshold", 
        type=float, 
        default=0.1,
        help="Detection confidence threshold (default: %(default)s)"
    )
    parser.add_argument(
        "--opts", 
        default=[], 
        nargs=argparse.REMAINDER,
        help="Modify config options using command-line"
    )
    return parser

def main():
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    
    # Validate paths
    if not Path(args.config_file).exists():
        raise FileNotFoundError(f"Config file {args.config_file} not found!")
    if not Path(args.input).exists():
        raise FileNotFoundError(f"Input path {args.input} not found!")
    
    # Initialize and process
    estimator = LightEstimator(args)
    input_paths = []
    
    if os.path.isdir(args.input):
        for ext in ('*.png', '*.jpg', '*.jpeg'):
            input_paths.extend(glob.glob(os.path.join(args.input, ext)))
    else:
        input_paths = [args.input]
    
    for path in tqdm.tqdm(input_paths, desc="Processing Images"):
        estimator.process_image(path)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main()

    