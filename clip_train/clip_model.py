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
from peft import LoraConfig, get_peft_model

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_ckpt_dir():
    """체크포인트 저장 폴더 생성"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = f"ckpt/{timestamp}"
    os.makedirs(ckpt_dir, exist_ok=True)
    return ckpt_dir

def get_dataloaders(processor, train_dir, val_dir, batch_size=5):
    """데이터 로더 생성"""
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

class OWLVITCLIPModel:
    """
    OwlViT 모델을 로드하고, LoRA를 적용한 후 학습/검증 및 체크포인트 저장 기능을 포함하는 클래스입니다.
    """
    def __init__(self, model_name="google/owlvit-base-patch32", use_lora=True, lora_config_params=None):
        # 프로세서 및 기본 모델 로드
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.model = OwlViTForObjectDetection.from_pretrained(model_name).to(device)
        self.model.train()

        # 전체 파라미터 Freeze
        for param in self.model.parameters():
            param.requires_grad = False

        if use_lora:
            # 기본 LoRA 하이퍼파라미터 값 (필요시 조정)
            if lora_config_params is None:
                lora_config_params = {"r": 4, "lora_alpha": 32, "lora_dropout": 0.1}
            lora_config = LoraConfig(
                task_type="OTHER",  # 태스크에 따라 적절한 task_type으로 변경 가능
                r=lora_config_params["r"],
                lora_alpha=lora_config_params["lora_alpha"],
                lora_dropout=lora_config_params["lora_dropout"],
                target_modules=["text_projection", "visual_projection"]
            )
            # PEFT 라이브러리를 이용하여 LoRA 어댑터 추가
            self.model = get_peft_model(self.model, lora_config)
        else:
            # LoRA를 사용하지 않는 경우, 특정 레이어만 Unfreeze
            trainable_layers = [
                self.model.owlvit.text_projection,
                self.model.owlvit.visual_projection
            ]
            for layer in trainable_layers:
                for param in layer.parameters():
                    param.requires_grad = True
            self.model.owlvit.logit_scale.requires_grad = True

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logging.info(f"🚀 Trainable Parameters: {trainable_params / 1e6:.2f}M")

    def get_optimizer(self, lr=1e-4):
        """옵티마이저 반환 (LoRA 어댑터 파라미터만 업데이트)"""
        return optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)

    def train(self, train_dir, val_dir, epochs=100, batch_size=16, lr=1e-4):
        """모델 학습"""
        train_dataloader, val_dataloader = get_dataloaders(self.processor, train_dir, val_dir, batch_size)
        optimizer = self.get_optimizer(lr)
        contrastive_loss = CLIPContrastiveLoss().to(device)
        ckpt_dir = create_ckpt_dir()
        best_val_loss = float("inf")

        for epoch in range(epochs):
            total_loss = 0
            self.model.train()

            for batch in train_dataloader:
                optimizer.zero_grad()
                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                outputs = self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                # 비전 및 텍스트 임베딩 처리
                vision_embeds = outputs.image_embeds.mean(dim=(1, 2))
                text_embeds = outputs.text_embeds.squeeze(1)

                # 프로젝션 레이어 적용
                vision_embeds = self.model.owlvit.visual_projection(vision_embeds)
                text_embeds = self.model.owlvit.text_projection(text_embeds)

                loss = contrastive_loss(vision_embeds, text_embeds)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                break
            val_loss = self.validate(val_dataloader, contrastive_loss)
            logging.info(f"Epoch {epoch+1} | Train Loss: {total_loss / len(train_dataloader):.4f} | Val Loss: {val_loss:.4f}")

            best_val_loss = self.save_checkpoint(optimizer, epoch, total_loss, val_loss, ckpt_dir, best_val_loss)

    def validate(self, dataloader, contrastive_loss):
        """검증 루프"""
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                outputs = self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                vision_embeds = outputs.image_embeds.mean(dim=(1, 2))
                text_embeds = outputs.text_embeds.squeeze(1)

                vision_embeds = self.model.owlvit.visual_projection(vision_embeds)
                text_embeds = self.model.owlvit.text_projection(text_embeds)

                loss = contrastive_loss(vision_embeds, text_embeds)
                total_loss += loss.item()
        return total_loss / len(dataloader)

    def save_checkpoint(self, optimizer, epoch, train_loss, val_loss, ckpt_dir, best_val_loss):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
        }
        torch.save(checkpoint, f"{ckpt_dir}/epoch_{epoch+1}.pth")
        if val_loss < best_val_loss:
            torch.save(checkpoint, f"{ckpt_dir}/best_model.pth")
            logging.info(f"🔹 Best model updated at {ckpt_dir}/best_model.pth")
            best_val_loss = val_loss
        return best_val_loss