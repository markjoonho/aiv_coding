python ./data_generation/negative_data_generate.py \
--input_folder ./train \
--output_folder ./train_dataset/negative

python ./data_generation/positive_data_generate.py \
--negative_input_folder ./train_dataset/negative \
--candidate_input_folder ./train \
--output_folder ./train_dataset/positive \
--target_counts 2 3 4 5 6 7 8 9 10

cp -r ./train/* ./train_dataset/positive/