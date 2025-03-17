import os
import json
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # 좌우 반전
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomEqualize(),
        transforms.RandomRotation(degrees=(0, 180)),  # 회전
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 이동 변환
    ])

image_path = "./total_dataset/train_dataset/1241218144156467_1_Outer_patch_0.bmp"
# 이미지 로드
image = Image.open(image_path).convert("RGB")


# 랜덤 변환 이미지 표시
num_augmentations = 5
for i in range(num_augmentations):
    augmented_image = transform(image)
    save_path = f"./augmented_{i+1}.jpg"
    augmented_image.save(save_path)  # 이미지 저장
    print(f"Saved: {save_path}")

