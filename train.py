import os
import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
import datetime
import logging
import argparse
from torch.utils.data import DataLoader
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from peft import LoraConfig, get_peft_model
from dataset import ImageTextBBoxDataset, collate_fn  # 사용자 정의 데이터셋 모듈
from loss import HungarianMatcher, OWLVITLoss              # 사용자 정의 손실함수
import warnings
warnings.filterwarnings('ignore')

# 로깅 기본 포맷 설정 (나중에 FileHandler를 추가할 예정)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")


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

        # 전체 파라미터 Freeze 후, head와 layer_norm만 학습 가능하도록 변경
        for param in self.model.parameters():
            param.requires_grad = False
        for name, param in self.model.named_parameters():
            if "box_head" in name or "class_head" in name:
                param.requires_grad = True
            if "model.layer_norm." in name:
                param.requires_grad = True
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logging.info(f"trainable 파라미터: {trainable_params / 1e6:.2f}M")
    
    def load_checkpoint(self, checkpoint_path):
        """
        checkpoint에서 모델 state_dict를 로드합니다.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logging.info(f"Checkpoint loaded from {checkpoint_path}")

    def get_optimizer(self, lr=1e-4):
        """학습 가능한 파라미터(여기서는 head만)를 업데이트하는 옵티마이저 반환"""
        return optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)

    def get_dataloaders(self, train_dir, val_dir, batch_size=16):
        """데이터 로더 생성"""
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=30, p=0.5),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))

        train_dataset = ImageTextBBoxDataset(train_dir, self.processor, transform=transform)
        val_dataset = ImageTextBBoxDataset(val_dir, self.processor)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        return train_loader, val_loader

    def train(self, train_dir, val_dir, epochs=10, batch_size=16, lr=1e-4, ckpt_dir=None, loss_weights=None):
        """
        학습 및 검증 루프.
        loss_weights는 {'loss_ce': value, 'loss_bbox': value, 'loss_giou': value} 형식으로 외부에서 주입할 수 있습니다.
        ckpt_dir가 제공되면 해당 디렉토리에 checkpoint를 저장합니다.
        """
        train_loader, val_loader = self.get_dataloaders(train_dir, val_dir, batch_size)
        optimizer = self.get_optimizer(lr)
        matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)
        if loss_weights is None:
            weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}
        else:
            weight_dict = loss_weights
        criterion = OWLVITLoss(num_classes=1, matcher=matcher, weight_dict=weight_dict)
    
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=0.1,
            patience=3,
            verbose=True
        )
        # ckpt_dir가 제공되지 않으면 새로 생성
        if ckpt_dir is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            ckpt_dir = os.path.join("ckpt_final", timestamp)
            os.makedirs(ckpt_dir, exist_ok=True)

        best_val_loss = float("inf")
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            total_loss_ce = 0.0
            total_loss_bbox = 0.0
            total_loss_giou = 0.0
            
            for batch in train_loader:
                optimizer.zero_grad()
                pixel_values = batch["pixel_values"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                bboxes = batch['bboxes']
                all_labels = []
                for bbox_tensor in bboxes:
                    image_labels = []
                    for bbox in bbox_tensor:
                        label = 0 if bbox.sum().item() == 0 else 1
                        image_labels.append(label)
                    all_labels.append(torch.tensor(image_labels, dtype=torch.int64, device=self.device))
                bboxes = [bbox.to(self.device) for bbox in bboxes]
                
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

                loss_dict = criterion(outputs, targets)
                total_loss_ce += loss_dict['loss_ce'].item()
                total_loss_bbox += loss_dict['loss_bbox'].item()
                total_loss_giou += loss_dict['loss_giou'].item()
                loss = loss_dict['total_loss']
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            avg_train_loss = total_loss / len(train_loader)
            avg_loss_ce = total_loss_ce / len(train_loader)
            avg_loss_bbox = total_loss_bbox / len(train_loader)
            avg_loss_giou = total_loss_giou / len(train_loader)
            
            # validation 단계 실행 (로그는 내부에서 남기지 않고 결과만 반환)
            val_metrics = self.validate(val_loader, criterion)
            avg_val_loss = val_metrics["avg_val_loss"]
            scheduler.step(avg_val_loss)
            
            # train 관련 로그 출력 후 validation 로그 출력
            logging.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}")
            logging.info(f"Epoch {epoch+1}/{epochs} - Train - ce: {avg_loss_ce:.4f}, bbox: {avg_loss_bbox:.4f}, giou: {avg_loss_giou:.4f}")
            logging.info(f"Epoch {epoch+1}/{epochs} - Validation Loss: {avg_val_loss:.4f}")
            logging.info(f"Epoch {epoch+1}/{epochs} - Validation - ce: {val_metrics['avg_loss_ce']:.4f}, bbox: {val_metrics['avg_loss_bbox']:.4f}, giou: {val_metrics['avg_loss_giou']:.4f}")
            
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss
            }
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch+1}.pth")
            torch.save(checkpoint, ckpt_path)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_ckpt_path = os.path.join(ckpt_dir, "best_model.pth")
                torch.save(checkpoint, best_ckpt_path)
                logging.info(f"Best model updated: {best_ckpt_path}")

    def validate(self, val_loader, criterion):
        """검증 루프 - 로그는 남기지 않고 각 손실값을 반환합니다."""
        self.model.eval()
        total_loss = 0.0
        total_loss_ce = 0.0
        total_loss_bbox = 0.0
        total_loss_giou = 0.0
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                bboxes = batch['bboxes']
                all_labels = []
                for bbox_tensor in bboxes:
                    image_labels = []
                    for bbox in bbox_tensor:
                        label = 0 if bbox.sum().item() == 0 else 1
                        image_labels.append(label)
                    all_labels.append(torch.tensor(image_labels, dtype=torch.int64, device=self.device))
                bboxes = [bbox.to(self.device) for bbox in bboxes]

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
                
                loss_dict = criterion(outputs, targets)
                total_loss += loss_dict['total_loss'].item()
                total_loss_ce += loss_dict['loss_ce'].item()
                total_loss_bbox += loss_dict['loss_bbox'].item()
                total_loss_giou += loss_dict['loss_giou'].item()
        avg_val_loss = total_loss / len(val_loader)
        avg_loss_ce = total_loss_ce / len(val_loader)
        avg_loss_bbox = total_loss_bbox / len(val_loader)
        avg_loss_giou = total_loss_giou / len(val_loader)
        
        return {
            "avg_val_loss": avg_val_loss,
            "avg_loss_ce": avg_loss_ce,
            "avg_loss_bbox": avg_loss_bbox,
            "avg_loss_giou": avg_loss_giou
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OwlViT CLIP Training")
    parser.add_argument("--train_dir", type=str, default="./total_dataset/train_dataset/", help="Training dataset directory")
    parser.add_argument("--val_dir", type=str, default="./total_dataset/val/", help="Validation dataset directory")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--ckpt_base_dir", type=str, default="ckpt_final", help="Base directory for saving checkpoints")
    parser.add_argument("--checkpoint_path", type=str, default="./ckpt/20250317_164612/train_20250317_164612/best_model.pth", help="Path to checkpoint for model initialization")
    parser.add_argument("--model_name", type=str, default="google/owlvit-base-patch32", help="Pretrained model name")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training")
    parser.add_argument("--use_lora", type=lambda x: (str(x).lower() == 'true'), default=True, help="Whether to use LoRA (True/False)")
    parser.add_argument("--lora_r", type=int, default=4, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout rate")
    parser.add_argument("--loss_weights", type=str, default="1:5:2", help="Loss weights for ce, bbox, giou in the format '1:5:2'")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level")
    args = parser.parse_args()

    # 로깅 레벨 설정
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # ckpt 디렉토리 미리 생성 (여기서 생성하면 args 로그 포함 모든 로그가 파일에 기록됨)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = os.path.join(args.ckpt_base_dir, timestamp)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # FileHandler 추가 (이 시점부터 발생하는 로그는 ckpt_dir/train.log에 기록)
    log_file = os.path.join(ckpt_dir, "train.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, args.log_level))
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)
    
    # 파싱된 인자 값 로그에 기록
    logging.info(f"Parsed arguments: {args}")

    # loss_weights 파싱 (예: "1:5:2")
    try:
        ce_weight, bbox_weight, giou_weight = [float(w) for w in args.loss_weights.split(":")]
        loss_weights = {"loss_ce": ce_weight, "loss_bbox": bbox_weight, "loss_giou": giou_weight}
    except Exception as e:
        logging.error("loss_weights 파싱 오류. '1:5:2' 형식으로 입력해 주세요.")
        raise e

    # LoRA 설정 파라미터
    lora_config_params = {"r": args.lora_r, "lora_alpha": args.lora_alpha, "lora_dropout": args.lora_dropout}

    # 모델 인스턴스 생성
    model_wrapper = OWLVITCLIPModel(model_name=args.model_name, device=args.device, use_lora=args.use_lora, lora_config_params=lora_config_params)

    # checkpoint가 존재하면 로드
    if args.checkpoint_path and os.path.isfile(args.checkpoint_path):
        model_wrapper.load_checkpoint(args.checkpoint_path)

    # 학습 시작 (미리 생성한 ckpt_dir를 전달)
    model_wrapper.train(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        ckpt_dir=ckpt_dir,
        loss_weights=loss_weights
    )
