import json
import cv2
import numpy as np
import random
import os
import glob
import argparse

# JSON 입출력 및 타입 변환 함수
def load_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, json_path):
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

# 박스 겹침 체크 (box: (x, y, width, height))
def boxes_overlap(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    if x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1:
        return False
    return True

def find_non_stabbed_region(source_image, source_shapes, region_width, region_height, max_attempts=100):
    """
    secondary 이미지에서, 주어진 region_width x region_height 크기의 영역이  
    source_shapes에 기록된 STABBED 영역(바운딩 박스)과 겹치지 않는 위치를 랜덤으로 찾습니다.
    """
    img_height, img_width = source_image.shape[:2]
    # secondary 이미지의 모든 STABBED 영역 바운딩 박스 계산
    stabbed_boxes = []
    for shape in source_shapes:
        pts = shape['points']
        xs = [pt[0] for pt in pts]
        ys = [pt[1] for pt in pts]
        x_min, y_min, x_max, y_max = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        stabbed_boxes.append((x_min, y_min, x_max - x_min, y_max - y_min))
    for _ in range(max_attempts):
        x = random.randint(0, img_width - region_width)
        y = random.randint(0, img_height - region_height)
        new_box = (x, y, region_width, region_height)
        if all(not boxes_overlap(new_box, box) for box in stabbed_boxes):
            return x, y
    return None

def process_image_without_stabbed(primary_image_path, json_path, 
                                  other_image_path, other_json_path, output_dir):
    """
    primary 이미지에서 STABBED 영역을 선택하여,  
    secondary 이미지의 STABBED 없는 영역(patch)을 복사해 덮어씌웁니다.
    최종적으로 STABBED 영역이 제거된 이미지를 생성한 후,
    JSON 내의 영역 데이터는 모두 삭제하여 저장합니다.
    """
    # primary 이미지와 JSON 로드
    primary_image = cv2.imread(primary_image_path)
    primary_data = load_json(json_path)
    primary_stabbed = [s for s in primary_data.get('shapes', []) if s['label'] == 'STABBED']
    
    if not primary_stabbed:
        print(f"{primary_image_path}에는 STABBED 영역이 없습니다.")
        return

    # secondary 이미지와 JSON 로드 (clean patch 추출 대상)
    other_image = cv2.imread(other_image_path)
    other_data = load_json(other_json_path)
    
    # primary 이미지의 각 STABBED 영역을 순회하며 처리
    for shape in primary_stabbed:
        pts = shape['points']
        xs = [pt[0] for pt in pts]
        ys = [pt[1] for pt in pts]
        x_min, y_min, x_max, y_max = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        w, h = x_max - x_min, y_max - y_min
        
        # secondary 이미지에서, w x h 크기로 STABBED 영역과 겹치지 않는 clean patch 찾기
        pos = find_non_stabbed_region(other_image, other_data.get('shapes', []), w, h)
        if pos is None:
            print(f"Non-stabbed patch를 찾지 못했습니다. ({x_min}, {y_min}, {w}, {h})")
            continue
        src_x, src_y = pos
        patch = other_image[src_y:src_y+h, src_x:src_x+w].copy()
        
        # primary 이미지의 STABBED 영역 위치에 patch를 복사하여 덮어씌움
        primary_image[y_min:y_min+h, x_min:x_min+w] = patch
    
    # 최종적으로 JSON 내의 영역 데이터는 모두 제거 (필요한 경우 다른 메타데이터는 유지)
    primary_data['shapes'] = []
    if "rois" in primary_data:
        del primary_data["rois"]
    
    # 출력 파일명 생성 (예: primary_non_stabbed.bmp, primary_non_stabbed.json)
    base = os.path.splitext(os.path.basename(primary_image_path))[0]
    output_image_path = os.path.join(output_dir, f"{base}_non_stabbed.bmp")
    output_json_path = os.path.join(output_dir, f"{base}_non_stabbed.json")
    primary_data["imagePath"] = f"{base}_non_stabbed.bmp"
    save_json(primary_data, output_json_path)
    cv2.imwrite(output_image_path, primary_image)
    print(f"Saved non-stabbed image: {output_image_path} and {output_json_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="STABBED 영역 제거를 위한 negative data generation 스크립트"
    )
    parser.add_argument(
        "--input_folder", type=str, required=True,
        help="입력 폴더 경로 (BMP와 JSON 파일은 동일 basename을 갖음)"
    )
    parser.add_argument(
        "--output_folder", type=str, required=True,
        help="출력 폴더 경로"
    )
    args = parser.parse_args()
    
    input_folder = args.input_folder
    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)
    
    bmp_files = glob.glob(os.path.join(input_folder, "*.bmp"))
    bmp_files.sort()
    
    if len(bmp_files) < 2:
        print("최소 2개의 BMP 파일이 필요합니다.")
        exit(1)
    
    # 각 primary 이미지마다, secondary 이미지(다음 파일 선택)를 사용하여 STABBED 영역 제거 처리
    for i, primary_image_path in enumerate(bmp_files):
        primary_json_path = os.path.splitext(primary_image_path)[0] + ".json"
        # secondary 이미지 선택 (자기 자신 제외; 없으면 첫 번째 사용)
        other_index = (i + 1) % len(bmp_files)
        other_image_path = bmp_files[other_index]
        other_json_path = os.path.splitext(other_image_path)[0] + ".json"
        
        process_image_without_stabbed(primary_image_path, primary_json_path, 
                                      other_image_path, other_json_path, output_folder)
