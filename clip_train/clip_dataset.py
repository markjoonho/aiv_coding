import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class ImageTextDataset(Dataset):
    def __init__(self, image_dir, processor):
        self.image_dir = image_dir
        
        self.image_paths = [
            os.path.join(root, f)
            for root, _, files in os.walk(self.image_dir)
            for f in files if f.endswith(".bmp")
        ]

        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        # 라벨 추출
        if "negative" in image_path:
            y = "no stabbed"
        elif "positive" in image_path:
            y = "stabbed exist"

            # stabbed 숫자 체크
            try:
                n_stabbed = int(image_path.split("_")[-1].replace(".bmp", ""))
                y = f"{n_stabbed} stabbed exists"
            except ValueError:
                y = "1 stabbed exists"
        else:
            y = "stabbed exist"
        # processor
        inputs = self.processor(text=y, images=image, return_tensors="pt")

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),  # (3, 768, 768)
            "input_ids": inputs["input_ids"].squeeze(0),  # (sequence_length,)
            "attention_mask": inputs["attention_mask"].squeeze(0)  # (sequence_length,)
        }

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

