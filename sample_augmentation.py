from data.bbox_utils import draw_rect
from data.CutMix import CutMix
import pickle as pkl
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image
import json
import albumentations as A



def json_to_bboxes(json_ann):
    shapes = json_ann.get("shapes", [])
    bboxes = []
    for shape in shapes:
        bbox = shape.get("bbox")
        if bbox is not None:
            x = bbox["x"]
            y = bbox["y"]
            w = bbox["width"]
            h = bbox["height"]
            # 변환: [x, y, x+width, y+height]
            bboxes.append([x, y, x + w, y + h])
    return np.array(bboxes)

if __name__ == "__main__":


    # 파일 경로 (사용 환경에 맞게 수정)
    png_path1 = "./train_augmented/1241219023343044_1_Outer_non_stabbed_2.bmp"
    json_path1 = "./train_augmented/1241219023343044_1_Outer_non_stabbed_2.json"
    
    png_path2 = "./train_augmented/12412191614528614_1_Outer_non_stabbed_8.bmp"
    json_path2 = "./train_augmented/12412191614528614_1_Outer_non_stabbed_8.json"
    # 만약 다른 이미지가 있다면 경로를 수정하세요.

    # PIL로 이미지 읽기 후 numpy array로 변환
    img1 = Image.open(png_path1)
    img1 = np.array(img1)
    img2 = Image.open(png_path2)
    img2 = np.array(img2)
    
    # JSON 파일 읽기 및 파싱
    with open(json_path1, "r") as f:
        json_ann1 = json.load(f)
    with open(json_path2, "r") as f:
        json_ann2 = json.load(f)
    
    # JSON 어노테이션에서 bbox 정보 변환
    bboxes1 = json_to_bboxes(json_ann1)
    bboxes2 = json_to_bboxes(json_ann2)
    
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussianBlur(p=0.3),
        A.GaussNoise(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=30, p=0.5),
        # A.ElasticTransform(p=0.3)
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))
    
    # 각 이미지에 augmentation 파이프라인 적용
    aug_img1 = transform(image=img1, bboxes=bboxes1, category_ids=[0]*len(bboxes1))
    aug_img2= transform(image=img2, bboxes=bboxes2, category_ids=[0]*len(bboxes2))
    
    img1_trans = aug_img1["image"]
    bboxes1_trans = aug_img1["bboxes"]
    
    img2_trans = aug_img2["image"]
    bboxes2_trans = aug_img2["bboxes"]
    
    # CutMix augmentation 적용 (두 이미지와 bbox들을 결합)
    cutmix = CutMix(beta=1.0, min_area_ratio=0.1)
    img_cutmix, bboxes_cutmix = cutmix(img1_trans, bboxes1_trans, img2_trans, bboxes2_trans)
    
    # 결과 이미지에 bounding box overlay (data_aug의 draw_rect 함수 사용)
    result_img = draw_rect(img_cutmix, bboxes_cutmix)
    # 결과 이미지 저장 (PIL 이미지로 변환)
    result = Image.fromarray(result_img)
    result.save("augmented_cutmix_image_with_bbox.png")