import os
import json
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class OWLVITJSONDataset(Dataset):
    def __init__(self, folders, label2id, transform=None):
        self.json_paths = []
        for folder in folders:
            for root, _, files in os.walk(folder):
                for file in files:
                    if file.endswith('.json'):
                        self.json_paths.append(os.path.join(root, file))
        self.label2id = label2id
        self.transform = transform

    def __len__(self):
        return len(self.json_paths)

    def __getitem__(self, idx):
        json_path = self.json_paths[idx]
        with open(json_path, 'r') as f:
            data = json.load(f)
        # 이미지 파일 경로 생성 및 로드
        image_path = os.path.join(os.path.dirname(json_path), data["imagePath"])
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        
        boxes = []
        labels = []
        for shape in data.get("shapes", []):
            if "bbox" in shape:
                bbox = shape["bbox"]
                x1 = bbox["x"]
                y1 = bbox["y"]
                x2 = x1 + bbox["width"]
                y2 = y1 + bbox["height"]
            else:
                points = shape.get("points", [])
                if points:
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                else:
                    continue
            boxes.append([x1, y1, x2, y2])
            label_str = shape.get("label", "")
            labels.append(self.label2id.get(label_str, 0))
        
        if self.transform:
            transformed = self.transform(image=image_np, bboxes=boxes, category_ids=labels)
            image_np = transformed['image']
            boxes = transformed.get('bboxes', [0,0,0,0])
            labels = transformed.get('category_ids', [0])
        
        return {"image": image_np, "boxes": np.array(boxes), "labels": np.array(labels)}
