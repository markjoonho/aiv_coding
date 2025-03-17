CUDA_VISIBLE_DEVICES=5 python train.py \
--train_dir "./total_dataset/train_dataset/" \
--val_dir "./total_dataset/val/" \
--epochs 100 \
--batch_size 64 \
--lr 2e-3 \
--loss_weights "1:3:2" \
--log_level "INFO"

CUDA_VISIBLE_DEVICES=5 python train.py \
--train_dir "./total_dataset/train_dataset/" \
--val_dir "./total_dataset/val/" \
--epochs 100 \
--batch_size 64 \
--lr 9e-4 \
--loss_weights "1:3:2" \
--log_level "INFO"
