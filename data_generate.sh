python ./data_generation/split_data.py --data_dir ./train \
--output_dir ./total_dataset \
--train_ratio 0.8

python ./data_generation/negative_data_generate.py \
--input_folder ./total_dataset/train \
--output_folder ./total_dataset/train_dataset/negative

python ./data_generation/positive_data_generate.py \
--negative_input_folder ./total_dataset/train_dataset/negative \
--candidate_input_folder ./total_dataset/train \
--output_folder ./total_dataset/train_dataset/positive \
--target_counts 2 3 4 5 6 7 8 9 10

cp -r ./total_dataset/train/* ./total_dataset/train_dataset/positive/