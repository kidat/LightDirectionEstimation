import open3d as o3d
import numpy as np
import logging
from utils.general_utils import rotation_matrix

logger = logging.getLogger(__name__)

def generate_3d_visualizations(output_dir, depth_map, img, points_3d, light_data, 
                               original_lines, extended_lines, sun_pos, base_name):
    try:
        pcd = create_o3d_pointcloud(points_3d, img)
        pcd.estimate_normals()

        combined_mesh = o3d.geometry.TriangleMesh()

        # Point cloud mesh
        radii = [0.005, 0.01, 0.02, 0.04]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii))
        combined_mesh += mesh

        # Add original (shadow → object) red arrows
        for start, end in original_lines:
            arrow = create_3d_arrow(start, end, color=[1, 0, 0])
            combined_mesh += arrow

        # Add extended (shadow → sun) yellow arrows
        for start, end in extended_lines:
            arrow = create_3d_arrow(start, end, color=[1, 1, 0])
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
            str(output_dir / f"{base_name}_full_scene.ply"),
            combined_mesh,
            write_vertex_colors=True
        )
    except Exception as e:
        logger.error(f"3D visualization failed: {str(e)}")

def create_3d_arrow(start, end, color=[1, 0, 0]):
    start = np.asarray(start)
    end = np.asarray(end)
    vec = end - start
    length = np.linalg.norm(vec)

    if length < 1e-6:
        return o3d.geometry.TriangleMesh()

    direction = vec / length
    cone_height = 0.2 * length
    cylinder_height = 0.8 * length

    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.01, height=cylinder_height)
    cone = o3d.geometry.TriangleMesh.create_cone(radius=0.02, height=cone_height)

    cylinder.paint_uniform_color(color)
    cone.paint_uniform_color(color)

    rot = rotation_matrix(np.array([0, 0, 1]), direction)
    cylinder.rotate(rot)
    cone.rotate(rot)

    cylinder.translate(start + direction * (cylinder_height / 2))
    cone.translate(start + direction * (cylinder_height + cone_height / 2))

    return cylinder + cone

def create_o3d_pointcloud(points_3d, image):
    h, w = points_3d.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    points = points_3d.reshape(-1, 3).copy()
    colors = image_rgb.reshape(-1, 3) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.estimate_normals()
    return pcd