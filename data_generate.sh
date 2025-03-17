python ./data_generation/split_data.py --data_dir ./train \
--output_dir ./total_dataset \
--train_ratio 0.8


python ./data_generation/patch_generate.py \
--folder ./total_dataset/train/ \
--output ./total_dataset/train_dataset/ \
--patch_size 832 \
--stride 208

python ./data_generation/patch_generate.py \
--folder ./total_dataset/val/ \
--output ./total_dataset/val_dataset/ \
--patch_size 832 \
--stride 832