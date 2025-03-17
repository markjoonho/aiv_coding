import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, average_precision_score
from torch.utils.data import DataLoader
from dataset import ImageTextBBoxDataset, collate_fn  # 사용자 정의 데이터셋 모듈
from transformers import OwlViTProcessor
from OWLVITCLIPModel import OWLVITCLIPModel  # 위에 정의한 모델 클래스가 저장된 모듈

# bbox 좌표 변환: (cx, cy, w, h) -> (x_min, y_min, x_max, y_max)
def box_cxcywh_to_xyxy(x):
    cx, cy, w, h = x.unbind(-1)
    x_min = cx - 0.5 * w
    y_min = cy - 0.5 * h
    x_max = cx + 0.5 * w
    y_max = cy + 0.5 * h
    return torch.stack([x_min, y_min, x_max, y_max], dim=-1)

# 평가 함수: 지정한 threshold에서 전체 데이터셋에 대해 GT와 예측을 모아 metric 계산
def evaluate_model(model_wrapper, dataloader, threshold=0.5):
    model_wrapper.model.eval()
    all_gt = []
    all_preds = []
    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(model_wrapper.device)
            input_ids = batch["input_ids"].to(model_wrapper.device)
            attention_mask = batch["attention_mask"].to(model_wrapper.device)
            # ground truth 생성 (각 이미지 내 bbox가 모두 0이면 0, 아니면 1)
            for bbox_tensor in batch['bboxes']:
                for bbox in bbox_tensor:
                    label = 0 if bbox.sum().item() == 0 else 1
                    all_gt.append(label)
            
            outputs = model_wrapper.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            # logits shape: (batch, 577, 1) -> apply sigmoid and squeeze 마지막 차원
            pred_probs = torch.sigmoid(outputs.logits).squeeze(-1)  # shape: (batch, 577)
            # threshold 적용하여 예측 (flatten)
            preds = (pred_probs > threshold).int().view(-1).cpu().numpy().tolist()
            all_preds.extend(preds)
    precision = precision_score(all_gt, all_preds, zero_division=0)
    recall = recall_score(all_gt, all_preds, zero_division=0)
    f1 = f1_score(all_gt, all_preds, zero_division=0)
    return precision, recall, f1, all_gt, all_preds

# Threshold 최적화: validation set에서 여러 threshold에 대해 f1 score 평가 후 최적 threshold 선택
def find_best_threshold(model_wrapper, dataloader, thresholds=np.linspace(0, 1, 101)):
    best_threshold = None
    best_f1 = 0
    metrics_list = []
    for t in thresholds:
        precision, recall, f1, _, _ = evaluate_model(model_wrapper, dataloader, threshold=t)
        metrics_list.append((t, precision, recall, f1))
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
    return best_threshold, best_f1, metrics_list

# Precision-Recall Curve 시각화
def plot_precision_recall_curve(all_gt, all_pred_probs):
    precision_vals, recall_vals, thresholds = precision_recall_curve(all_gt, all_pred_probs)
    ap = average_precision_score(all_gt, all_pred_probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall_vals, precision_vals, label=f'AP = {ap:.2f}')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

# 모든 예측 확률을 모으기 위한 함수 (평가 시 사용)
def get_all_pred_probs(model_wrapper, dataloader):
    model_wrapper.model.eval()
    all_pred_probs = []
    all_gt = []
    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(model_wrapper.device)
            input_ids = batch["input_ids"].to(model_wrapper.device)
            attention_mask = batch["attention_mask"].to(model_wrapper.device)
            # ground truth
            for bbox_tensor in batch['bboxes']:
                for bbox in bbox_tensor:
                    label = 0 if bbox.sum().item() == 0 else 1
                    all_gt.append(label)
            outputs = model_wrapper.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            # logits: (batch, 577, 1) -> apply sigmoid
            pred_probs = torch.sigmoid(outputs.logits).squeeze(-1)  # (batch, 577)
            all_pred_probs.extend(pred_probs.view(-1).cpu().numpy().tolist())
    return all_gt, all_pred_probs

# 예측 결과를 이미지에 overlay하여 시각화하는 함수
def visualize_predictions(model_wrapper, dataloader, threshold=0.5, num_images=5):
    model_wrapper.model.eval()
    shown = 0
    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(model_wrapper.device)
            input_ids = batch["input_ids"].to(model_wrapper.device)
            attention_mask = batch["attention_mask"].to(model_wrapper.device)
            outputs = model_wrapper.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            # 예측 확률: (batch, 577, 1) -> sigmoid 후 squeeze
            pred_probs = torch.sigmoid(outputs.logits).squeeze(-1)  # (batch, 577)
            # bbox: (batch, 577, 4) in cx,cy,w,h format
            pred_boxes = outputs.pred_boxes  
            batch_size = pixel_values.size(0)
            for i in range(batch_size):
                # 이미지 tensor -> numpy (채널 순서 변환)
                image = pixel_values[i].cpu().permute(1, 2, 0).numpy()
                # bbox 변환: cxcywh -> xyxy
                boxes = pred_boxes[i]  # (577, 4)
                boxes_xyxy = box_cxcywh_to_xyxy(boxes)  # (577, 4)
                probs = pred_probs[i]  # (577,)
                keep = probs > threshold
                filtered_boxes = boxes_xyxy[keep].cpu().numpy()
                filtered_probs = probs[keep].cpu().numpy()
                
                fig, ax = plt.subplots(1, figsize=(8, 8))
                ax.imshow(image)
                height, width, _ = image.shape
                for box, prob in zip(filtered_boxes, filtered_probs):
                    # bbox가 normalized 되어 있다고 가정하고, pixel 좌표로 변환
                    x_min = box[0] * width
                    y_min = box[1] * height
                    x_max = box[2] * width
                    y_max = box[3] * height
                    rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                         fill=False, edgecolor='red', linewidth=2)
                    ax.add_patch(rect)
                    ax.text(x_min, y_min, f"{prob:.2f}", color='red', fontsize=12)
                ax.set_title("Predicted Bounding Boxes")
                plt.axis("off")
                plt.show()
                shown += 1
                if shown >= num_images:
                    return

if __name__ == "__main__":
    # 데이터셋 경로 (사용자 환경에 맞게 수정)
    val_dataset_dir = "./total_dataset/val/"
    test_dataset_dir = "./test/"  # test set 경로 (존재한다고 가정)

    # 데이터 로더 생성
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    val_dataset = ImageTextBBoxDataset(val_dataset_dir, processor, transform=None)
    test_dataset = ImageTextBBoxDataset(test_dataset_dir, processor, transform=None)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # 모델 로드 (ckpt 경로 수정)
    model_wrapper = OWLVITCLIPModel(use_lora=True)
    checkpoint_path = "./ckpt/20250313_172710/best_model.pth"  # 사용중인 ckpt 경로
    model_wrapper.load_checkpoint(checkpoint_path)

    # validation set에서 최적 threshold 찾기
    thresholds = np.linspace(0, 1, 101)
    best_threshold, best_f1, metrics_list = find_best_threshold(model_wrapper, val_loader, thresholds)
    print(f"최적의 threshold: {best_threshold:.2f} / F1: {best_f1:.4f}")

    # 모든 예측 확률을 모아 Precision-Recall curve 시각화 (validation set)
    gt, all_pred_probs = get_all_pred_probs(model_wrapper, val_loader)
    plot_precision_recall_curve(gt, all_pred_probs)

    # 최적 threshold로 validation set 평가
    val_precision, val_recall, val_f1, _, _ = evaluate_model(model_wrapper, val_loader, threshold=best_threshold)
    print(f"[Validation] Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")

    # test set 평가
    test_precision, test_recall, test_f1, _, _ = evaluate_model(model_wrapper, test_loader, threshold=best_threshold)
    print(f"[Test] Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")

    # 예측 결과 시각화 (test set)
    visualize_predictions(model_wrapper, test_loader, threshold=best_threshold, num_images=5)
