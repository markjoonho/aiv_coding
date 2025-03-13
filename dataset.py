import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class ImageTextBBoxDataset(Dataset):
    def __init__(self, image_dir, processor, transform=None, annotation_ext='.json'):
        """
        image_dir: 이미지가 저장된 디렉토리 경로
        processor: 텍스트와 이미지를 전처리할 processor (예: HuggingFace processor)
        transform: 이미지에 적용할 transform (선택 사항)
        annotation_ext: JSON 파일 확장자 (기본값: '.json')
        """
        self.image_dir = image_dir
        self.processor = processor
        self.transform = transform
        self.annotation_ext = annotation_ext
        
        self.image_paths = [
            os.path.join(root, f)
            for root, _, files in os.walk(self.image_dir)
            for f in files if f.lower().endswith(".bmp")
        ]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 이미지 로드
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        
        
        # 같은 경로에서 JSON annotation 파일 찾기 (예: "xxx.bmp" -> "xxx.json")
        base, _ = os.path.splitext(image_path)
        json_path = base + self.annotation_ext
        
        bboxes = []
        category_ids = []  # Albumentations transform에 사용 (bbox에 대한 label 정보)

        label = None
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                annotation = json.load(f)
            shapes = annotation.get("shapes", [])
            if shapes:
                label = "stabbed exist"
                # 각 shape에서 "bbox" 필드 추출 (예: {"x": ..., "y": ..., "width": ..., "height": ...})
                for shape in shapes:
                    if "bbox" in shape:
                        bbox_dict = shape["bbox"]
                        x = bbox_dict.get("x", 0)
                        y = bbox_dict.get("y", 0)
                        width = bbox_dict.get("width", 0)
                        height = bbox_dict.get("height", 0)
                        bbox_pascal = [x, y, x + width, y + height]
                        bboxes.append(bbox_pascal)
                        category_ids.append(1)
            else:
                label = "stabbed not exist"
      
        # transform이 제공된다면 이미지와 bbox에 적용
        if self.transform is not None:
            # PIL 이미지를 numpy array로 변환
            image_np = np.array(image)
            # transform에 넣을 때, bbox가 없으면 빈 리스트, label 정보는 category_ids
            transformed = self.transform(image=image_np, bboxes=bboxes, category_ids=category_ids)
            image_np = transformed["image"]
            bboxes = transformed["bboxes"]  # 여전히 pascal voc 형식
            # numpy array를 다시 PIL 이미지로 변환
            image = Image.fromarray(image_np)
            # pascal voc 형식을 다시 [x, y, width, height]로 변환
            bboxes = [[bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]] for bbox in bboxes]
        
        # bboxes가 있을 경우 tensor로, 없으면 (0,4) 텐서 반환
        if bboxes:
            bboxes = torch.tensor(bboxes, dtype=torch.float)
        else:
            bboxes = torch.empty((0, 4), dtype=torch.float)
        
        # processor를 통해 텍스트와 이미지 전처리 (토크나이저, 정규화 등)
        inputs = self.processor(text=label, images=image, return_tensors="pt")
        
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),  # (3, H, W)
            "input_ids": inputs["input_ids"].squeeze(0),          # (sequence_length,)
            "attention_mask": inputs["attention_mask"].squeeze(0),  # (sequence_length,)
            "bboxes": bboxes                                      # (num_boxes, 4) in [x, y, width, height] format
        }


# ✅ Collate Function 정의 (배치 데이터 변환)
def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    # bboxes의 경우 이미지마다 box 개수가 다를 수 있으므로 리스트 형태로 반환합니다.
    bboxes = [item["bboxes"] for item in batch]
    
    return {
        "pixel_values": pixel_values,      # (batch, 3, H, W)
        "input_ids": input_ids,            # (batch, sequence_length)
        "attention_mask": attention_mask,  # (batch, sequence_length)
        "bboxes": bboxes                   # List of tensors, 각 tensor shape: (num_boxes, 4)
    }
if __name__ == "__main__":
    import albumentations as A
    from transformers import OwlViTProcessor, OwlViTForObjectDetection

    transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.GaussianBlur(p=0.3),
            A.GaussNoise(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=30, p=0.5),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")


    train_dataset = ImageTextBBoxDataset("./total_dataset/train_dataset/", processor, transform=transform)
    import ipdb; ipdb.set_trace()
    