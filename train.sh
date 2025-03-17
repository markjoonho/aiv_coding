CUDA_VISIBLE_DEVICES=3 python train.py \
--train_dir "./total_dataset/train_dataset/" \
--val_dir "./total_dataset/val_dataset/" \
--epochs 100 \
--batch_size 64 \
--lr 1e-5 \
--loss_weights "1:10:1" \
--log_level "INFO"

CUDA_VISIBLE_DEVICES=3 python train.py \
--train_dir "./total_dataset/train_dataset/" \
--val_dir "./total_dataset/val_dataset/" \
--epochs 100 \
--batch_size 64 \
--lr 5e-5 \
--loss_weights "1:10:1" \
--log_level "INFO"
