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
    log_file = os.path.join(ckpt_dir, f"train_{timestamp}.log")

    # 기존 로거 가져오기
    logger = logging.getLogger()
    
    # 기존 핸들러 제거 (중복 방지)
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # 콘솔 및 파일 핸들러 추가
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    
    # 파일 로그 저장
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 콘솔 출력 핸들러 추가
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 로깅 기본 레벨 설정
    logger.setLevel(logging.INFO)

    return ckpt_dir


def get_dataloaders(processor, train_dir, val_dir, batch_size=5):
    """데이터 로더 생성"""
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # 좌우 반전
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomEqualize(),
        transforms.RandomRotation(degrees=(0, 180)),  # 회전
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
    def __init__(self, model_name="google/owlvit-base-patch32", use_lora=True, lora_config_params=None):
        # 프로세서 및 기본 모델 로드
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.model = OwlViTForObjectDetection.from_pretrained(model_name).to(device)
        self.model.train()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # 전체 파라미터 Freeze
        for param in self.model.parameters():
            param.requires_grad = False

        if use_lora:
            # 기본 LoRA 하이퍼파라미터 값 (필요시 조정)
            if lora_config_params is None:
                lora_config_params = {"r": 4, "lora_alpha": 32, "lora_dropout": 0.1}
            lora_config = LoraConfig(
                task_type="OTHER",
                r=lora_config_params["r"],
                lora_alpha=lora_config_params["lora_alpha"],
                lora_dropout=lora_config_params["lora_dropout"],
                target_modules=["text_projection", "visual_projection"]
            )
            self.model = get_peft_model(self.model, lora_config)
        else:
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
        return optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)

    def train(self, train_dir, val_dir, epochs=100, batch_size=16, lr=1e-4):
        train_dataloader, val_dataloader = get_dataloaders(self.processor, train_dir, val_dir, batch_size)
        optimizer = self.get_optimizer(lr)
    
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
        )
        
        contrastive_loss = CLIPContrastiveLoss().to(device)
        ckpt_dir = create_ckpt_dir()
        logging.info("🚀 Training Configuration:")
        logging.info(f"🔹 Batch Size: {batch_size}")
        logging.info(f"🔹 Learning Rate: {lr}")
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
                vision_embeds = outputs.image_embeds
                text_embeds = outputs.text_embeds

                # 프로젝션 레이어 적용
                vision_embeds = self.model.owlvit.visual_projection(vision_embeds)
                vision_embeds = vision_embeds.permute(0, 3, 1, 2)  
                vision_embeds = self.pool(vision_embeds)  
                vision_embeds = vision_embeds.squeeze(-1).squeeze(-1)
                
                text_embeds = self.model.owlvit.text_projection(text_embeds).squeeze(1)

                loss = contrastive_loss(vision_embeds, text_embeds)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            val_loss = self.validate(val_dataloader, contrastive_loss)
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']

            logging.info(f"Epoch {epoch+1} | Train Loss: {total_loss / len(train_dataloader):.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.6f}")


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

                # 비전 및 텍스트 임베딩 처리
                vision_embeds = outputs.image_embeds
                text_embeds = outputs.text_embeds

                # 프로젝션 레이어 적용
                vision_embeds = self.model.owlvit.visual_projection(vision_embeds)
                vision_embeds = vision_embeds.permute(0, 3, 1, 2)  
                vision_embeds = self.pool(vision_embeds)  
                vision_embeds = vision_embeds.squeeze(-1).squeeze(-1)
                
                text_embeds = self.model.owlvit.text_projection(text_embeds).squeeze(1)

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
            logging.info(f"Best model updated at {ckpt_dir}/best_model.pth")
            best_val_loss = val_loss
        return best_val_loss