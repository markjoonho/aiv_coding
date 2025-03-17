CUDA_VISIBLE_DEVICES=4 python train.py \
--checkpoint_path ./ckpt/20250317_205659/best_model.pth \
--train_dir "./total_dataset/train_dataset/" \
--val_dir "./total_dataset/val_dataset/" \
--epochs 100 \
--batch_size 64 \
--lr 5e-4 \
--loss_weights "1:5:2"

CUDA_VISIBLE_DEVICES=4 python train.py --use_lora \
--checkpoint_path ./ckpt/20250317_205757/best_model.pth \
--train_dir "./total_dataset/train_dataset/" \
--val_dir "./total_dataset/val_dataset/" \
--epochs 50 \
--batch_size 64 \
--lr 5e-4 \
--loss_weights "1:5:2"


