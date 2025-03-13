from clip_model import OWLVITCLIPModel



if __name__ == "__main__":
    # 모델 인스턴스 생성 (LoRA 사용)
    model_wrapper = OWLVITCLIPModel(use_lora=True)
    # 학습 진행 (데이터셋 경로 및 하이퍼파라미터 필요에 따라 조정)
    model_wrapper.train(
        train_dir="./total_dataset/train_dataset/",
        val_dir="./total_dataset/val/",
        epochs=100,
        batch_size=16,
        lr=1e-2
    )
    # "./ckpt/20250313_172710/best_model.pth"