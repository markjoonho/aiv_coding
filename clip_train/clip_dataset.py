import os
import json
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from PIL import Image

class ImageTextDataset(Dataset):
    def __init__(self, image_dir, processor, transform=None):
        self.image_dir = image_dir
        
        self.image_paths = [
            os.path.join(root, f)
            for root, _, files in os.walk(self.image_dir)
            for f in files if f.endswith(".bmp")
        ] * 5  # Oversampling 5배 증가
        
        self.processor = processor
        self.transform = transform
        self.first_epoch_data_saved = False  # 첫 번째 에포크 데이터 저장 여부

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        # JSON 파일에서 라벨 추출
        json_path = image_path.replace(".bmp", ".json")
        stabbed_exist = False
        
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for shape in data.get("shapes", []):
                    if shape.get("label") == "STABBED":
                        stabbed_exist = True
                        break
        
        y = "stabbed exist" if stabbed_exist else "stabbed not exist"
        
        # processor 적용
        inputs = self.processor(text=y, images=image, return_tensors="pt")
        
        # 첫 번째 배치 데이터 저장 (한 번만 실행)
        if not self.first_epoch_data_saved:
            self.save_first_batch_data(image, y, idx)
        
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),  # (3, 768, 768)
            "input_ids": inputs["input_ids"].squeeze(0),  # (sequence_length,)
            "attention_mask": inputs["attention_mask"].squeeze(0)  # (sequence_length,)
        }
    
    def save_first_batch_data(self, image, label, idx):
        """첫 번째 배치 데이터를 저장"""
        save_dir = "first_epoch_data"
        os.makedirs(save_dir, exist_ok=True)
        
        # 이미지 저장
        image_path = os.path.join(save_dir, f"image_{idx}.png")
        image.save(image_path)
        
        # 텍스트 라벨 저장
        label_path = os.path.join(save_dir, f"label_{idx}.txt")
        with open(label_path, "w") as f:
            f.write(label)
        
        # 이미지와 라벨 시각화 저장
        plt.figure(figsize=(6, 6))
        plt.imshow(image)
        plt.axis("off")
        plt.title(label, fontsize=12, color="red")
        plt.savefig(os.path.join(save_dir, f"visualized_{idx}.png"))
        plt.close()
        
        self.first_epoch_data_saved = True

# ✅ Collate Function 정의 (배치 데이터 변환)
def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])  # 이미지 배치화
    input_ids = torch.stack([item["input_ids"] for item in batch])  # 텍스트 배치화
    attention_mask = torch.stack([item["attention_mask"] for item in batch])  # 마스크 배치화

    return {
        "pixel_values": pixel_values,  # (batch, 3, 768, 768)
        "input_ids": input_ids,  # (batch, sequence_length)
        "attention_mask": attention_mask  # (batch, sequence_length)
    }
