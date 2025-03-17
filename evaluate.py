import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from sklearn.metrics import average_precision_score
from train import OWLVITCLIPModel
from dataset import ImageTextBBoxDataset, collate_fn
from transformers import OwlViTProcessor
from torch.utils.data import DataLoader
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_model_and_dataset(val_dataset_dir, checkpoint_path, use_lora=True):
    val_dataset = ImageTextBBoxDataset(val_dataset_dir, transform=None, oversample=1)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    model_wrapper = OWLVITCLIPModel(use_lora=use_lora)
    model_wrapper.load_checkpoint(checkpoint_path)
    
    model_wrapper.model.to(device)
    model_wrapper.model.eval()
    
    return model_wrapper, val_loader

def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    inter_x_min = max(x1 - w1 / 2, x2 - w2 / 2)
    inter_y_min = max(y1 - h1 / 2, y2 - h2 / 2)
    inter_x_max = min(x1 + w1 / 2, x2 + w2 / 2)
    inter_y_max = min(y1 + h1 / 2, y2 + h2 / 2)
    
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

def apply_sliding_window(image, processor, model_wrapper, window_size=832, stride=400):
    w, h = image.size
    detected_boxes, detected_scores = [], []

    if w <= window_size and h <= window_size:
        inputs = processor(text='stabbed exist', images=image, return_tensors='pt')
        outputs = model_wrapper.model(pixel_values=inputs["pixel_values"].to(device),
                                      input_ids=inputs["input_ids"].to(device),
                                      attention_mask=inputs["attention_mask"].to(device))
        scores = torch.sigmoid(torch.max(outputs["logits"][0], dim=-1).values).detach().cpu().numpy()
        pred_boxes = outputs["pred_boxes"][0].detach().cpu().numpy()
        return pred_boxes, scores

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            x_end = min(x + window_size, w)
            y_end = min(y + window_size, h)

            cropped_image = image.crop((x, y, x_end, y_end))
            inputs = processor(text='stabbed exist', images=cropped_image, return_tensors='pt')

            outputs = model_wrapper.model(pixel_values=inputs["pixel_values"].to(device),
                                          input_ids=inputs["input_ids"].to(device),
                                          attention_mask=inputs["attention_mask"].to(device))
            
            scores = torch.sigmoid(torch.max(outputs["logits"][0], dim=-1).values).detach().cpu().numpy()
            pred_boxes = outputs["pred_boxes"][0].detach().cpu().numpy()

            for score, box in zip(scores, pred_boxes):
                if score > 0.5:
                    box[0] += x  # 원본 좌표계로 변환
                    box[1] += y
                    detected_boxes.append(box)
                    detected_scores.append(score)

    return detected_boxes, detected_scores

def draw_boxes_on_image(image, boxes, scores, i):
    """ 이미지에 바운딩 박스 & 스코어 그리기 """
    draw = ImageDraw.Draw(image)
    
    for (box, score) in zip(boxes, scores):
        x, y, w, h = map(int, box)
        x_min, y_min, x_max, y_max = x - w // 2, y - h // 2, x + w // 2, y + h // 2
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
        draw.text((x_min, y_min - 10), f"{score:.2f}", fill="red")
    
    image.save(f"detection_result_{i}.jpg")
    return image

def plot_metrics(results):
    """ Precision, Recall, F1-score 그래프 저장 """
    plt.figure(figsize=(10, 6))
    plt.plot(results["score_threshold"], results["precision"], label="Precision", marker='o')
    plt.plot(results["score_threshold"], results["recall"], label="Recall", marker='s')
    plt.plot(results["score_threshold"], results["f1_score"], label="F1 Score", marker='^')
    plt.plot(results["score_threshold"], results["average_precision"], label="Average Precision", marker='x')
    plt.xlabel("Score Threshold")
    plt.ylabel("Metric Value")
    plt.title("Object Detection Metrics vs. Score Threshold")
    plt.legend()
    plt.grid(True)
    plt.savefig('./metrics_plot_test.jpeg')
    plt.show()

def evaluate_model(model_wrapper, val_loader, processor, iou_threshold=0.5):
    results = {"score_threshold": [], "precision": [], "recall": [], "f1_score": [], "average_precision": []}
    score_thresholds = np.linspace(0.1, 0.9, 9)  # 다양한 score_threshold 실험

    for score_threshold in score_thresholds:
    
        all_tp, all_fp, all_fn, all_pred_scores = 0, 0, 0, []

        for i, batch in enumerate(val_loader):
            image = batch['images'][0]  # PIL 이미지
            gt_boxes = [bbox.cpu().numpy() for bbox_tensor in batch['bboxes'] for bbox in bbox_tensor]

            pred_boxes, pred_scores = apply_sliding_window(image, processor, model_wrapper)
            
            # 탐지 결과 시각화
            # drawn_image = draw_boxes_on_image(image.copy(), pred_boxes, pred_scores, i)

            # Score Threshold 적용
            filtered_pred_boxes = [box for i, box in enumerate(pred_boxes) if pred_scores[i] >= score_threshold]
            filtered_pred_scores = [score for score in pred_scores if score >= score_threshold]

            # TP, FP, FN 계산
            tp = sum(1 for gt in gt_boxes if any(compute_iou(gt, pred) >= iou_threshold for pred in filtered_pred_boxes))
            fp = len(filtered_pred_boxes) - tp
            fn = len(gt_boxes) - tp

            all_tp += tp
            all_fp += fp
            all_fn += fn
            all_pred_scores.extend(filtered_pred_scores)

        # 🔥 Score Threshold 별 Precision, Recall, F1-score, AP 계산
        precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
        recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        average_precision = average_precision_score([1] * all_tp + [0] * all_fp, all_pred_scores) if len(all_pred_scores) > 0 else 0

        results["score_threshold"].append(score_threshold)
        results["precision"].append(precision)
        results["recall"].append(recall)
        results["f1_score"].append(f1)
        results["average_precision"].append(average_precision)

    plot_metrics(results)

if __name__ == "__main__":
    val_dataset_dir = "./total_dataset/val_dataset/"
    # checkpoint_path = "./ckpt_final/20250318_031452/epoch_20.pth"
    checkpoint_path = "./ckpt_final/20250318_050935/epoch_20.pth"
        
    model_wrapper, val_loader = load_model_and_dataset(val_dataset_dir, checkpoint_path, use_lora=True)
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    
    evaluate_model(model_wrapper, val_loader, processor)
