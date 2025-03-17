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
        ] * 5 
        
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
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
        
        inputs = self.processor(text=y, images=image, return_tensors="pt")
        
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0)
        }
    
    

def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }
