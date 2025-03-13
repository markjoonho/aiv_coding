import numpy as np
import random

from .CutMix import CutMix
from .dataset import OWLVITJSONDataset

def update_boxes_for_resized_image(boxes, orig_size, new_size):
    """
    이미지 크기가 변경되었을 때, bounding box를 새로운 크기에 맞게 조정하는 함수
    - boxes: 원본 bounding box 리스트 (x1, y1, x2, y2)
    - orig_size: 원본 이미지 크기 (H, W)
    - new_size: 변환된 이미지 크기 (H, W)
    """
    orig_h, orig_w = orig_size
    new_h, new_w = new_size

    scale_w = new_w / orig_w  # 가로 비율
    scale_h = new_h / orig_h  # 세로 비율

    # Bounding box 크기 변환
    new_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        new_x1 = x1 * scale_w
        new_y1 = y1 * scale_h
        new_x2 = x2 * scale_w
        new_y2 = y2 * scale_h
        new_boxes.append([new_x1, new_y1, new_x2, new_y2])

    return np.array(new_boxes)


def collate_fn(batch, text_queries, cutmix_augmentor=None, processor=None):
    if cutmix_augmentor is not None and len(batch) >= 2:
        # 모든 샘플에 대해 CutMix 수행 (자신을 제외한 랜덤한 샘플과)
        new_batch = []
        for i in range(len(batch)):
            if random.random() < 0.0:
                # 자신을 제외한 랜덤 샘플 선택
                available_indices = list(range(len(batch)))
                available_indices.remove(i)
                rand_idx = random.choice(available_indices)
                mixed_sample = cutmix_augmentor(batch[i], batch[rand_idx])
                new_batch.append(mixed_sample)
            else:
                new_batch.append(batch[i])
        batch = new_batch 
    # CutMix 적용된 batch에서 이미지, bbox, labels 추출
    images = [sample["image"] for sample in batch]
    boxes = [sample["boxes"] for sample in batch]
    labels = [sample["labels"] for sample in batch]
    orig_sizes = [img.shape[1:3] for img in images]  # (H, W) 저장

    # 🟢 Processor 적용: 여러 개의 이미지(batch)를 한번에 변환
    if processor is not None:
        encoding = processor(text=text_queries * len(batch), images=images, return_tensors="pt")
        input_ids = encoding["input_ids"]
        images = encoding["pixel_values"]  # 모델 입력값
        new_size = images.shape[2:]  # (H, W) 변환된 이미지 크기

        # 🟢 Bounding box 크기 변환 적용
        new_boxes = [
            update_boxes_for_resized_image(box, orig_size, new_size) 
            for box, orig_size in zip(boxes, orig_sizes)
        ]
    else:
        input_ids = None
        new_boxes = boxes

    return {"image": images, "boxes": new_boxes, "labels": labels, "input_ids": input_ids}
