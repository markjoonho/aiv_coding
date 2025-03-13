import torch
import torch.nn as nn
import torch.optim as optim
import logging
import os
import datetime
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from clip_dataset import ImageTextDataset, collate_fn
from loss import CLIPContrastiveLoss

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_ckpt_dir():
    """ 체크포인트 저장 폴더 생성 """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = f"ckpt/{timestamp}"
    os.makedirs(ckpt_dir, exist_ok=True)
    return ckpt_dir

def load_model():
    """ OWL-ViT 모델 로드 및 학습 가능한 레이어 설정 """
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)
    model.train()
    
    # 모든 파라미터 Freeze
    for param in model.parameters():
        param.requires_grad = False

    # 특정 레이어 Unfreeze
    trainable_layers = [
        model.owlvit.text_projection,  # 텍스트 임베딩 투영
        model.owlvit.visual_projection,  # 비전 임베딩 투영
        model.owlvit.text_model.encoder.layers[-2:],  # 텍스트 모델 마지막 두 개 레이어
        model.owlvit.vision_model.encoder.layers[-2:],
        model.owlvit.text_model.final_layer_norm,
        model.owlvit.vision_model.post_layernorm
          # 비전 모델 마지막 두 개 레이어
    ]
    
    for layer in trainable_layers:
        for param in layer.parameters():
            param.requires_grad = True
    
    model.owlvit.logit_scale.requires_grad = True  # Logit Scale 학습
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"🚀 Trainable Parameters: {trainable_params / 1e6:.2f}M")
    
    return model, processor

def get_optimizer(model, lr=1e-4):
    """ 옵티마이저 반환 """
    return optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

def get_dataloaders(processor, train_dir, val_dir, batch_size=5):
    """ 데이터 로더 생성 """
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # 좌우 반전
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 색상 변화
        transforms.RandomRotation(degrees=15),  # 회전
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 이동 변환
    ])

    train_dataset = ImageTextDataset(train_dir, processor, transform=transform)
    val_dataset = ImageTextDataset(val_dir, processor)
    
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    
    return train_dataloader, val_dataloader

def train_model(model, processor, train_dir, val_dir, epochs=10, batch_size=16, lr=1e-4):
    """ 모델 학습 """
    train_dataloader, val_dataloader = get_dataloaders(processor, train_dir, val_dir, batch_size)
    optimizer = get_optimizer(model, lr)
    contrastive_loss = CLIPContrastiveLoss().to(device)
    
    ckpt_dir = create_ckpt_dir()
    best_val_loss = float("inf")
    
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        
        for batch in train_dataloader:
            optimizer.zero_grad()
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            vision_embeds = outputs.image_embeds.mean(dim=(1, 2))
            text_embeds = outputs.text_embeds.squeeze(1)
            
            vision_embeds = model.owlvit.visual_projection(vision_embeds)
            text_embeds = model.owlvit.text_projection(text_embeds)
            
            loss = contrastive_loss(vision_embeds, text_embeds)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        val_loss = validate_model(model, val_dataloader, contrastive_loss)
        logging.info(f"Epoch {epoch+1} | Train Loss: {total_loss / len(train_dataloader):.4f} | Val Loss: {val_loss:.4f}")
        
        save_checkpoint(model, optimizer, epoch, total_loss, val_loss, ckpt_dir, best_val_loss)

def validate_model(model, dataloader, contrastive_loss):
    """ 검증 루프 """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            vision_embeds = outputs.image_embeds.mean(dim=(1, 2))
            text_embeds = outputs.text_embeds.squeeze(1)
            
            vision_embeds = model.owlvit.visual_projection(vision_embeds)
            text_embeds = model.owlvit.text_projection(text_embeds)
            
            loss = contrastive_loss(vision_embeds, text_embeds)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, ckpt_dir, best_val_loss):
    """ 체크포인트 저장 """
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss
    }
    torch.save(checkpoint, f"{ckpt_dir}/epoch_{epoch+1}.pth")
    
    if val_loss < best_val_loss:
        torch.save(checkpoint, f"{ckpt_dir}/best_model.pth")
        logging.info(f"🔹 Best model updated at {ckpt_dir}/best_model.pth")

if __name__ == "__main__":
    model, processor = load_model()
    train_model(model, processor, "./total_dataset/train_dataset/", "./total_dataset/validation/")
