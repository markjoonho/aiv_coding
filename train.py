import os
import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
import datetime
import logging
from torch.utils.data import DataLoader
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from peft import LoraConfig, get_peft_model
from dataset import ImageTextBBoxDataset, collate_fn  # 사용자 정의 데이터셋 모듈
from loss import HungarianMatcher, OWLVITLoss              # 사용자 정의 손실함수

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class OWLVITCLIPModel:
    """
    OwlViT 모델을 로드하고, LoRA를 적용한 후 head만 학습할 수 있도록 하는 클래스입니다.
    여기서는 bbox 예측 head(box_head)와 클래스 예측 head(class_head)만 학습합니다.
    """
    def __init__(self, model_name="google/owlvit-base-patch32", device='cuda', use_lora=True, lora_config_params=None):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        # 프로세서 및 기본 모델 로드
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.model = OwlViTForObjectDetection.from_pretrained(model_name).to(self.device)
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
            # LoRA를 사용하지 않는 경우, 예시로 text_projection, visual_projection만 unfreeze
            trainable_layers = [
                self.model.owlvit.text_projection,
                self.model.owlvit.visual_projection
            ]
            for layer in trainable_layers:
                for param in layer.parameters():
                    param.requires_grad = True
            self.model.owlvit.logit_scale.requires_grad = True

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logging.info(f"🚀 초기 trainable 파라미터: {trainable_params / 1e6:.2f}M")

    def load_checkpoint(self, checkpoint_path):
        """
        checkpoint에서 모델 state_dict를 로드합니다.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logging.info(f"Checkpoint loaded from {checkpoint_path}")

    def freeze_except_heads(self):
        """
        모델의 모든 파라미터를 freeze하고, 'box_head'와 'class_head'에 해당하는 파라미터만 학습 가능하도록 설정합니다.
        """
        for name, param in self.model.named_parameters():
            if "box_head" in name or "class_head" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logging.info(f"🚀 Head만 학습 가능하도록 설정됨. Trainable 파라미터: {trainable_params / 1e6:.2f}M")

    def reinitialize_heads(self):
        """
        box_head와 class_head에 해당하는 모듈들의 파라미터를 재초기화합니다.
        """
        def _reinit_module(module, module_name):
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
                logging.info(f"{module_name} 재초기화됨.")
        for name, module in self.model.named_modules():
            if "box_head" in name or "class_head" in name:
                _reinit_module(module, name)

    def get_optimizer(self, lr=1e-4):
        """학습 가능한 파라미터(여기서는 head만)를 업데이트하는 옵티마이저 반환"""
        return optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)

    def get_dataloaders(self, train_dir, val_dir, batch_size=16):
        """데이터 로더 생성"""
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.GaussianBlur(p=0.3),
            A.GaussNoise(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=30, p=0.5),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))


        train_dataset = ImageTextBBoxDataset(train_dir, self.processor, transform=transform)
        val_dataset = ImageTextBBoxDataset(val_dir, self.processor)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        return train_loader, val_loader

    def train(self, train_dir, val_dir, epochs=10, batch_size=16, lr=1e-4, ckpt_base_dir="ckpt"):
        """
        학습 및 검증 루프.
        학습 전에 freeze_except_heads()를 호출하여 head만 학습하도록 합니다.
        """
        # head만 학습할 수 있도록 설정
        self.freeze_except_heads()

        train_loader, val_loader = self.get_dataloaders(train_dir, val_dir, batch_size)
        optimizer = self.get_optimizer(lr)
        matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)
        weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}
        criterion = OWLVITLoss(num_classes=2, matcher=matcher, weight_dict=weight_dict, eos_coef=0.1, losses=['labels', 'boxes'])
    

        # 체크포인트 저장 폴더 생성
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_dir = os.path.join(ckpt_base_dir, timestamp)
        os.makedirs(ckpt_dir, exist_ok=True)

        best_val_loss = float("inf")

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                pixel_values = batch["pixel_values"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                bboxes = batch['bboxes']
                all_labels = []  # 이미지별로 label 리스트를 저장 (각 이미지: tensor of shape (num_boxes,))
                for bbox_tensor in bboxes:
                    image_labels = []
                    for bbox in bbox_tensor:  # bbox는 (4,) 텐서
                        # bbox의 합이 0이면 label 0, 아니면 1로 지정
                        label = 0 if bbox.sum().item() == 0 else 1
                        image_labels.append(label)
                    all_labels.append(torch.tensor(image_labels, dtype=torch.int64))
                all_labels = [torch.tensor(image_labels, dtype=torch.int64, device=self.device) for image_labels in all_labels]
                bboxes = [bbox.to(self.device) for bbox in bboxes]

                # import ipdb; ipdb.set_trace()
                
                
                outputs = self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                outputs = {
                    "pred_logits": outputs.logits,
                    "pred_boxes": outputs.pred_boxes
                }
                targets = [{"labels": lbl, "boxes": box} for lbl, box in zip(all_labels, bboxes)]

                loss = criterion(outputs, targets)
                # 'loss_ce': weight, 'loss_bbox': weight, 'loss_giou': weight
                loss = loss['total_loss']
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                break
            avg_train_loss = total_loss / len(train_loader)
            avg_val_loss = self.validate(val_loader, criterion)
            logging.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            # 체크포인트 저장
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss
            }
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch+1}.pth")
            torch.save(checkpoint, ckpt_path)
            logging.info(f"Checkpoint saved: {ckpt_path}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_ckpt_path = os.path.join(ckpt_dir, "best_model.pth")
                torch.save(checkpoint, best_ckpt_path)
                logging.info(f"Best model updated: {best_ckpt_path}")

    def validate(self, val_loader, criterion):
        """검증 루프 - train과 동일한 타겟 구성 방식을 사용"""
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                bboxes = batch['bboxes']
                
                # 각 이미지에 대한 라벨 생성 (bbox 합이 0이면 0, 아니면 1)
                all_labels = []
                for bbox_tensor in bboxes:
                    image_labels = []
                    for bbox in bbox_tensor:  # bbox는 (4,) 텐서
                        label = 0 if bbox.sum().item() == 0 else 1
                        image_labels.append(label)
                    all_labels.append(torch.tensor(image_labels, dtype=torch.int64, device=self.device))
                
                # bboxes도 device 이동 (만약 이미 tensor라면)
                bboxes = [bbox.to(self.device) for bbox in bboxes]
                
                # 모델 추론 및 결과 구성
                outputs = self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                outputs = {
                    "pred_logits": outputs.logits,
                    "pred_boxes": outputs.pred_boxes
                }
                
                # 각 이미지에 대한 타겟 딕셔너리 생성
                targets = [{"labels": lbl, "boxes": box} for lbl, box in zip(all_labels, bboxes)]
                
                loss = criterion(outputs, targets)
                loss = loss['total_loss']
                total_loss += loss.item()
        return total_loss / len(val_loader)


if __name__ == "__main__":
    # 데이터셋 경로 (프로젝트에 맞게 수정)
    train_dataset_dir = "./total_dataset/train_dataset/"
    val_dataset_dir = "./total_dataset/val/"

    # 모델 인스턴스 생성 (LoRA 적용)
    model_wrapper = OWLVITCLIPModel(use_lora=True)

    # 기존 checkpoint에서 모델 로드 (원한다면 head 재초기화도 수행)
    # checkpoint_path = "./ckpt/20250313_172710/best_model.pth"
    checkpoint_path = './ckpt/20250313_184959/best_model.pth'
    model_wrapper.load_checkpoint(checkpoint_path)
    # (원하는 경우) head 재초기화
    model_wrapper.reinitialize_heads()

    # head만 학습하도록 설정한 후 학습 시작
    model_wrapper.train(
        train_dir=train_dataset_dir,
        val_dir=val_dataset_dir,
        epochs=10,
        batch_size=16,
        lr=1e-4,
        ckpt_base_dir="ckpt"
    )
