import json
import cv2
import numpy as np
import albumentations as A
import random
import os
import glob
import argparse

# 증강 파이프라인 구성 (각 transform은 p=0.5로 적용)
transform = A.Compose(
    [
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.Resize(100, 100, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        A.Blur(blur_limit=3, p=0.5),
        A.ToGray(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
    ],
    keypoint_params=A.KeypointParams(format='xy')
)

def load_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, json_path):
    # NumPy 타입을 기본 자료형으로 변환
    data = convert_numpy_types(data)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    else:
        return obj

def boxes_overlap(box1, box2):
    """
    두 사각형(box: (x, y, width, height))이 겹치는지 확인합니다.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    if x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1:
        return False
    return True

def find_non_overlapping_position(img_width, img_height, region_width, region_height, existing_boxes, max_attempts=100):
    """
    이미지 내에서 기존 영역(existing_boxes)과 겹치지 않는 좌측 상단 좌표를 찾습니다.
    """
    for _ in range(max_attempts):
        x = random.randint(0, img_width - region_width)
        y = random.randint(0, img_height - region_height)
        new_box = (x, y, region_width, region_height)
        if all(not boxes_overlap(new_box, box) for box in existing_boxes):
            return x, y
    return None

def process_random_stabbed_region(source_image, source_shapes, dest_image, placed_boxes, label_suffix=""):
    """
    후보 이미지에서 STABBED 영역을 무작위로 선택하여 증강한 후,
    dest_image(negative 이미지)의 non-overlapping 영역에 붙입니다.
    """
    if not source_shapes:
        return None
    shape = random.choice(source_shapes)
    polygon = shape['points']
    xs = [pt[0] for pt in polygon]
    ys = [pt[1] for pt in polygon]
    x_min, y_min, x_max, y_max = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
    
    region = source_image[y_min:y_max, x_min:x_max]
    if region.size == 0:
        return None

    # ROI 기준 좌표로 변환
    local_polygon = [
        [min(pt[0] - x_min, region.shape[1]-1), min(pt[1] - y_min, region.shape[0]-1)]
        for pt in polygon
    ]
    
    transformed = transform(image=region, keypoints=local_polygon)
    aug_region = transformed['image']
    aug_keypoints = transformed['keypoints']
    
    h, w = aug_region.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(aug_keypoints, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)
    
    img_height, img_width = dest_image.shape[:2]
    pos = find_non_overlapping_position(img_width, img_height, w, h, placed_boxes)
    if pos is None:
        return None
    target_x, target_y = pos

    # negative 이미지에 증강된 영역 붙이기
    target_region = dest_image[target_y:target_y+h, target_x:target_x+w].copy()
    target_region[mask == 255] = aug_region[mask == 255]
    dest_image[target_y:target_y+h, target_x:target_x+w] = target_region
    
    new_polygon = [[kp[0] + target_x, kp[1] + target_y] for kp in aug_keypoints]
    new_shape = {
        "label": shape["label"] + label_suffix,
        "points": new_polygon,
        "bbox": {"x": target_x, "y": target_y, "width": w, "height": h},
        "shape_type": "polygon",
        "flags": {}
    }
    return new_shape, (target_x, target_y, w, h)

def process_negative_image_with_fixed_stabbed_count(negative_image_path, negative_json_path, candidate_files, target_count, output_dir):
    """
    한 negative 이미지에 대해 후보 이미지에서 무작위로 STABBED 영역을 추출하여
    정확히 target_count 개의 영역을 부착한 후, 결과 이미지와 JSON 파일을 저장합니다.
    """
    image = cv2.imread(negative_image_path)
    data = load_json(negative_json_path)
    augmented_shapes = data['shapes'].copy()
    placed_boxes = []
    
    candidate_used = []  # 사용한 후보 이미지의 basename 기록
    attempts = 0
    max_attempts = target_count * 5  # 무한루프 방지
    
    while len(candidate_used) < target_count and attempts < max_attempts:
        attempts += 1
        candidate_image_path = random.choice(candidate_files)
        candidate_base = os.path.splitext(os.path.basename(candidate_image_path))[0]
        # 후보 이미지의 중복 사용 방지 (후보가 충분하다면)
        if candidate_base in candidate_used and len(candidate_used) < len(candidate_files):
            continue
        
        candidate_json_path = os.path.splitext(candidate_image_path)[0] + ".json"
        candidate_image = cv2.imread(candidate_image_path)
        candidate_data = load_json(candidate_json_path)
        candidate_stabbed = [s for s in candidate_data['shapes'] if s['label'] == 'STABBED']
        if not candidate_stabbed:
            continue
        
        result = process_random_stabbed_region(candidate_image, candidate_stabbed, image, placed_boxes)
        if result is not None:
            new_shape, box = result
            augmented_shapes.append(new_shape)
            placed_boxes.append(box)
            candidate_used.append(candidate_base)
    
    data['shapes'] = augmented_shapes
    
    # 출력 파일명 구성 (negative 이미지 이름, 사용 후보 이미지, target_count 포함)
    negative_base = os.path.splitext(os.path.basename(negative_image_path))[0]
    candidate_str = "_".join(candidate_used) if candidate_used else "none"
    output_filename = f"{negative_base}_{target_count}.bmp"
    output_image_path = os.path.join(output_dir, output_filename)
    output_json_path = os.path.splitext(output_image_path)[0] + ".json"
    data['imagePath'] = output_filename
    save_json(data, output_json_path)
    cv2.imwrite(output_image_path, image)
    print(f"Saved: {output_image_path} and {output_json_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="STABBED 영역을 이용한 negative 이미지 증강 스크립트"
    )
    parser.add_argument(
        "--negative_input_folder", type=str, required=True,
        help="Negative 이미지들이 있는 폴더 경로 (BMP와 JSON 파일)"
    )
    parser.add_argument(
        "--candidate_input_folder", type=str, required=True,
        help="Candidate 이미지들이 있는 폴더 경로 (BMP와 JSON 파일)"
    )
    parser.add_argument(
        "--output_folder", type=str, required=True,
        help="결과 파일을 저장할 출력 폴더 경로"
    )
    parser.add_argument(
        "--target_counts", type=int, nargs="+", default=[2, 3, 4, 5, 6],
        help="부착할 STABBED 영역 개수 리스트 (예: 2 3 4 5 6)"
    )
    args = parser.parse_args()
    
    negative_input_folder = args.negative_input_folder
    candidate_input_folder = args.candidate_input_folder
    output_folder = args.output_folder
    target_counts = args.target_counts
    
    os.makedirs(output_folder, exist_ok=True)
    
    # negative 이미지와 후보 이미지 파일 목록 가져오기
    negative_files = glob.glob(os.path.join(negative_input_folder, "*.bmp"))
    negative_files.sort()
    candidate_files = glob.glob(os.path.join(candidate_input_folder, "*.bmp"))
    candidate_files.sort()
    
    # 각 negative 이미지마다, 지정된 target_count 개의 STABBED 영역을 부착하여 증강
    for negative_image_path in negative_files:
        negative_json_path = os.path.splitext(negative_image_path)[0] + ".json"
        for target_count in target_counts:
            process_negative_image_with_fixed_stabbed_count(
                negative_image_path,
                negative_json_path,
                candidate_files,
                target_count,
                output_folder
            )
