import os
import cv2
import json
from shapely.geometry import Polygon, box

def clip_polygon_to_patch(polygon_points, patch_x, patch_y, patch_size):
    patch_rect = box(patch_x, patch_y, patch_x + patch_size, patch_y + patch_size)
    poly = Polygon(polygon_points)
    if not poly.is_valid or poly.is_empty:
        return []
    inter = poly.intersection(patch_rect)
    if inter.is_empty:
        return []
    
    if inter.geom_type == 'Polygon':
        coords = list(inter.exterior.coords)
        if len(coords) > 1 and coords[0] == coords[-1]:
            coords = coords[:-1]
        coords = [[x - patch_x, y - patch_y] for x, y in coords]
        return [coords]

def update_bbox(points):
    xs = [pt[0] for pt in points]
    ys = [pt[1] for pt in points]
    min_x = min(xs)
    min_y = min(ys)
    max_x = max(xs)
    max_y = max(ys)
    return {"x": min_x, "y": min_y, "width": max_x - min_x, "height": max_y - min_y}

def process_image_and_json(image_path, json_path, output_dir, patch_size, stride):
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    image = cv2.imread(image_path)
    with open(json_path, "r") as f:
        annotation = json.load(f)
    
    img_height, img_width = image.shape[:2]
    patch_idx = 0
    for y in range(0, img_height - patch_size + 1, stride):
        for x in range(0, img_width - patch_size + 1, stride):
            patch = image[y:y+patch_size, x:x+patch_size]
            patch_filename = f"{base_name}_patch_{patch_idx}.bmp"
            patch_json_filename = f"{base_name}_patch_{patch_idx}.json"
            patch_image_path = os.path.join(output_dir, patch_filename)
            patch_json_path = os.path.join(output_dir, patch_json_filename)
            
            cv2.imwrite(patch_image_path, patch)
            
            new_annotation = {
                "version": annotation.get("version", ""),
                "flags": annotation.get("flags", {}),
                "shapes": [],
                "imagePath": patch_filename,
                "imageHeight": patch_size,
                "imageWidth": patch_size
            }
            
            for shape in annotation.get("shapes", []):
                points = shape.get("points", [])
                clipped_polygons = clip_polygon_to_patch(points, x, y, patch_size)
                for polygon in clipped_polygons:
                    if len(polygon) >= 3:
                        new_shape = {
                            "label": shape.get("label", ""),
                            "points": polygon,
                            "shape_type": shape.get("shape_type", "polygon"),
                            "flags": shape.get("flags", {}),
                            "bbox": update_bbox(polygon)
                        }
                        new_annotation["shapes"].append(new_shape)
            
            with open(patch_json_path, "w") as f:
                json.dump(new_annotation, f, indent=4)
            
            patch_idx += 1

def process_folder(folder_path, output_dir, patch_size, stride):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".bmp"):
            image_path = os.path.join(folder_path, filename)
            json_filename = os.path.splitext(filename)[0] + ".json"
            json_path = os.path.join(folder_path, json_filename)
            if os.path.exists(json_path):
                process_image_and_json(image_path, json_path, output_dir, patch_size, stride)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Generate patches")
    parser.add_argument("--folder", type=str, default="./total_dataset/train/")
    parser.add_argument("--output", type=str, default="./total_dataset/train_dataset/")
    parser.add_argument("--patch_size", type=int, default=832)
    parser.add_argument("--stride", type=int, default=277)
    args = parser.parse_args()

    process_folder(args.folder, args.output, args.patch_size, args.stride)
