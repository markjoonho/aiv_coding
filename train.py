import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from data import CutMix, collate_fn, OWLVITJSONDataset
from loss2 import OWLVITLoss  # 손실 함수 불러오기
import torch.nn as nn

#############################
#  Custom Head 및 Utility 함수
#############################
class CustomOwlViTClassHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)
    
    def forward(self, image_feats, query_embeds, query_mask):
        # image_feats에 대해 선형 변환 적용
        pred_logits = self.linear(image_feats)  # [B, num_queries, num_classes]
        # 원래 head는 (pred_logits, image_class_embeds)를 반환하는데, 여기서는 image_feats를 그대로 반환
        return pred_logits, image_feats

def normalize_boxes(boxes, image):
    """
    boxes: [N,4] 텐서, 절대 좌표 [x1, y1, x2, y2] (pascal_voc 형식)
    image: [C, H, W] 텐서
    반환: [N,4] 텐서, 정규화된 [cx, cy, w, h]
    """
    # 박스가 없는 경우, 빈 텐서를 반환 (shape: [0, 4])
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 4))
    
    # 단일 박스의 경우, shape가 [4]라면 [1, 4]로 변환
    if boxes.dim() == 1:
        boxes = boxes.unsqueeze(0)
        
    H, W = image.shape[-2], image.shape[-1]
    boxes = boxes.float()
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    cx = (x1 + x2) / 2.0 / W
    cy = (y1 + y2) / 2.0 / H
    w = (x2 - x1) / W
    h = (y2 - y1) / H
    return torch.stack([cx, cy, w, h], dim=1)


def denormalize_image(image, mean, std):
    """
    image: [C, H, W] 텐서, 정규화된 이미지 (float)
    mean, std: 리스트 혹은 텐서 (예: [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
    반환: denormalize된 이미지 (0~1 범위)
    """
    mean = torch.tensor(mean, device=image.device).view(-1, 1, 1)
    std = torch.tensor(std, device=image.device).view(-1, 1, 1)
    return image * std + mean

def visualize_batch(batch, save_path=None, num_cols=4):
    """
    배치 내 이미지와 박스를 시각화합니다.
    - image: [C, H, W] 텐서 (절대 좌표 이미지)
    - boxes: [x1, y1, x2, y2] 형태의 박스 리스트
    save_path: 저장할 파일 경로. 지정되면 해당 경로에 figure를 저장합니다.
    """
    num_images = len(batch["image"])
    num_rows = (num_images + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4))
    axes = axes.flatten()

    for i in range(num_images):
        image = batch["image"][i]
        boxes = batch["boxes"][i]
        if torch.is_tensor(image):
            # tensor를 numpy array로 변환 (채널 마지막)
            image = image.permute(1, 2, 0).cpu().numpy()
        ax = axes[i]
        ax.imshow(image)
        ax.axis("off")
        for box in boxes:
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     edgecolor="red", facecolor="none", lw=2)
            ax.add_patch(rect)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)

#####################################
# Training / Validation / Inference
#####################################
def train(model, dataloader, optimizer, lr_scheduler, loss_fn, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        images = batch["image"].to(device)            # [B, C, H, W]
        input_ids = batch["input_ids"].to(device)

        # GT 박스를 정규화: 각 이미지별로 절대 좌표 -> normalized [cx, cy, w, h]
        normalized_boxes = []
        for i, b in enumerate(batch["boxes"]):
            b_tensor = b if torch.is_tensor(b) else torch.tensor(b)
            b_tensor = b_tensor.to(device)
            norm_b = normalize_boxes(b_tensor, images[i])
            normalized_boxes.append(norm_b)
        
        labels = [torch.tensor(l).to(device) for l in batch["labels"]]

        outputs = model(pixel_values=images, input_ids=input_ids)
        pred_logits = outputs.logits     # [B, num_queries, num_classes]
        pred_boxes = outputs.pred_boxes    # [B, num_queries, 4] (normalized [cx,cy,w,h])
        
        targets = [{'labels': lbl, 'boxes': bx} for lbl, bx in zip(labels, normalized_boxes)]
        loss = loss_fn({'pred_logits': pred_logits, 'pred_boxes': pred_boxes}, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    lr_scheduler.step()
    avg_loss = total_loss / len(dataloader)
    print(f"Train Loss: {avg_loss:.4f}")
    return avg_loss

def validate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            normalized_boxes = []
            for i, b in enumerate(batch["boxes"]):
                b_tensor = b if torch.is_tensor(b) else torch.tensor(b)
                b_tensor = b_tensor.to(device)
                norm_b = normalize_boxes(b_tensor, images[i])
                normalized_boxes.append(norm_b)
            labels = [torch.tensor(l).to(device) for l in batch["labels"]]
            outputs = model(pixel_values=images, input_ids=input_ids)
            pred_logits = outputs.logits
            pred_boxes = outputs.pred_boxes
            targets = [{'labels': lbl, 'boxes': bx} for lbl, bx in zip(labels, normalized_boxes)]
            loss = loss_fn({'pred_logits': pred_logits, 'pred_boxes': pred_boxes}, targets)
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss

def inference_and_visualize(model, dataloader, device, threshold=0.1, save_path="inference_result.png"):
    """
    한 배치에 대해 모델 추론을 수행하고, object 클래스 확률이 threshold 이상인 예측 박스를 시각화한 후 저장합니다.
    이미지가 processor로 정규화되어 있다면, denormalize하여 시각화합니다.
    """
    model.eval()
    with torch.no_grad():
        batch = next(iter(dataloader))
        images = batch["image"].to(device)        # [B, C, H, W]
        input_ids = batch["input_ids"].to(device)
        outputs = model(pixel_values=images, input_ids=input_ids)
        pred_logits = outputs.logits               # [B, num_queries, num_classes]
        pred_boxes = outputs.pred_boxes            # [B, num_queries, 4] normalized
        
        # 소프트맥스 확률 계산: [B, num_queries, num_classes]
        probs = pred_logits.softmax(-1)
        # 객체 클래스의 확률 (클래스 1: 객체, 클래스 0: background)
        object_probs = probs[..., 1]

        # processor 정규화 값 (OwlViTProcessor 기본값)
        mean = [0.48145466, 0.4578275, 0.40821073]
        std  = [0.26862954, 0.26130258, 0.27577711]
        
        denorm_boxes_batch = []
        denorm_images = []
        # 각 이미지별로 threshold 이상의 박스만 선택하고, normalized -> absolute 좌표 변환
        for i in range(images.shape[0]):
            image_tensor = images[i]             # [C, H, W]
            # denormalize 이미지
            img_denorm = denormalize_image(image_tensor, mean, std)
            denorm_images.append(img_denorm.cpu())
            _, H, W = img_denorm.shape
            boxes = pred_boxes[i]                # [num_queries, 4]
            probs_i = object_probs[i]            # [num_queries]
            keep = probs_i > threshold
            boxes = boxes[keep]
            if boxes.numel() > 0:
                cx, cy, w, h = boxes.unbind(-1)
                x1 = (cx - 0.5 * w) * W
                y1 = (cy - 0.5 * h) * H
                x2 = (cx + 0.5 * w) * W
                y2 = (cy + 0.5 * h) * H
                boxes_abs = torch.stack([x1, y1, x2, y2], dim=-1)
            else:
                boxes_abs = torch.empty((0, 4))
            denorm_boxes_batch.append(boxes_abs)
        
        vis_batch = {
            "image": denorm_images,
            "boxes": denorm_boxes_batch
        }
        visualize_batch(vis_batch, save_path=save_path)

#####################
#   Main Execution
#####################
if __name__ == "__main__":
    train_folders = ["./train"]
    vali_folders = ["./validation"]
    label2id = {"STABBED": 1}

    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussianBlur(p=0.3),
        A.GaussNoise(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=30, p=0.5),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))

    # 검증 시에는 augmentation 없이 ToTensorV2만 적용
    val_transform = A.Compose([
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = OWLVITJSONDataset(train_folders, label2id, transform)
    vali_dataset = OWLVITJSONDataset(vali_folders, label2id, val_transform)

    cutmix = CutMix(beta=1.0, min_area_ratio=0.0)
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
    in_features = list(model.class_head.parameters())[0].shape[1]
    # 2개 클래스 (배경: 0, 객체: 1)를 예측하도록 classification head 재정의
    model.class_head = CustomOwlViTClassHead(in_features, 2).to(device)
    model = model.to(device)

    # Backbone Freeze: heads만 학습
    for param in model.parameters():
        param.requires_grad = False
    for param in model.class_head.parameters():
        param.requires_grad = True
    for param in model.box_head.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=5e-4, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    loss_fn = OWLVITLoss()  # 손실 함수 인스턴스화

    text_queries = [['stabbed']]
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, 
                                  collate_fn=lambda batch: collate_fn(batch, text_queries, cutmix_augmentor=None, processor=processor))
    vali_dataloader = DataLoader(vali_dataset, batch_size=16, shuffle=False,
                                 collate_fn=lambda batch: collate_fn(batch, text_queries, cutmix_augmentor=None, processor=processor))
    
    num_epochs = 50
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss = train(model, train_dataloader, optimizer, lr_scheduler, loss_fn, device)
        val_loss = validate(model, vali_dataloader, loss_fn, device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_owlvit_model.pth")
            print("✅ Best model saved!")
        torch.save(model.state_dict(), f"exp_0_epoch_{epoch}.pth")
        
        # 5 epoch마다 inference 결과 시각화 (이미지를 저장)
        if (epoch + 1) % 5 == 0:
            save_path = f"inference_epoch_{epoch+1}.png"
            print(f"Inference & Visualization, saving to {save_path}")
            inference_and_visualize(model, vali_dataloader, device, threshold=0.1, save_path=save_path)
