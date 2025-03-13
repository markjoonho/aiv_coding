import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from data import CutMix, collate_fn, OWLVITJSONDataset
from loss2 import OWLVITLoss  # 손실 함수 불러오기

def visualize_batch(batch, num_cols=4):
    """
    배치 내 모든 이미지를 하나의 figure에 시각화하는 함수.
    """
    num_images = len(batch["image"])
    num_rows = (num_images + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4))
    axes = axes.flatten()

    for i in range(num_images):
        image = batch["image"][i]
        boxes = batch["boxes"][i]

        if torch.is_tensor(image):
            image = image.permute(1, 2, 0).cpu().numpy()

        ax = axes[i]
        ax.imshow(image)
        ax.axis("off")

        for box in boxes:
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                edgecolor="red", facecolor="none", lw=2
            )
            ax.add_patch(rect)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

def train(model, dataloader, optimizer, lr_scheduler, loss_fn, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        optimizer.zero_grad()

        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        boxes = [torch.tensor(b).to(device) for b in batch["boxes"]]
        labels = [torch.tensor(l).to(device) for l in batch["labels"]]

        outputs = model(pixel_values=images, input_ids=input_ids)
        
        # 예측값 추출
        pred_logits = outputs.logits
        pred_boxes = outputs.pred_boxes

        # 손실 계산
        loss = loss_fn(pred_logits, pred_boxes, labels, boxes)
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
            boxes = [torch.tensor(b).to(device) for b in batch["boxes"]]
            labels = [torch.tensor(l).to(device) for l in batch["labels"]]

            outputs = model(pixel_values=images, input_ids=input_ids)
            pred_logits = outputs.logits
            pred_boxes = outputs.pred_boxes

            loss = loss_fn(pred_logits, pred_boxes, labels, boxes)

            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss

if __name__ == "__main__":
    train_folders = [
        "./train",
        # "./train_augmented",
        # "./augmented_non_stabbed"
    ]
    vali_folders = [
        "./validation"
    ]
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = OWLVITJSONDataset(train_folders, label2id, transform)
    vali_dataset = OWLVITJSONDataset(vali_folders, label2id, None)

    cutmix = CutMix(beta=1.0, min_area_ratio=0.0)
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)

    for param in model.parameters():
        param.requires_grad = False  # Backbone Freeze
    for param in model.class_head.parameters():
        param.requires_grad = True
    for param in model.box_head.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
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
