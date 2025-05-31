import numpy as np

def calculate_light_directions(pairs, objects, shadows, sun_pos=None, extension_factor=2.0):
    light_data = []
    original_lines = []
    extended_lines = []

    for obj_idx, shd_idx in pairs:
        try:
            obj_3d = objects["centroids_3d"][obj_idx]
            shd_3d = shadows["centroids_3d"][shd_idx]

            vector = obj_3d - shd_3d
            direction = vector / np.linalg.norm(vector)

            original_lines.append((shd_3d, obj_3d))

            if sun_pos is not None:
                extended_end = sun_pos
            else:
                extended_end = obj_3d + direction * extension_factor

            extended_lines.append((shd_3d, extended_end))

            light_data.append({
                "vector": direction.tolist(),
                "object": obj_3d.tolist(),
                "shadow": shd_3d.tolist(),
                "extended": extended_end.tolist()
            })

        except Exception:
            continue

    return light_data, original_lines, extended_lines