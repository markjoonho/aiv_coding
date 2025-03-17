import os
import random
import shutil
import argparse

def split_data(data_dir, output_dir, train_ratio=0.8, random_seed=42):
    
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")

    if os.path.exists(val_dir):
        return

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    bmp_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".bmp")])
    json_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".json")])

    pairs = [(bmp, bmp.replace(".bmp", ".json")) for bmp in bmp_files if bmp.replace(".bmp", ".json") in json_files]

    random.seed(random_seed)
    random.shuffle(pairs)

    # Train/Validation 분할
    train_size = int(len(pairs) * train_ratio)
    train_pairs = pairs[:train_size]
    val_pairs = pairs[train_size:]
    print(len(train_pairs), len(val_pairs))

    def move_files(pairs, target_dir):
        for bmp, json in pairs:
            shutil.copy(os.path.join(data_dir, bmp), os.path.join(target_dir, bmp))
            shutil.copy(os.path.join(data_dir, json), os.path.join(target_dir, json))

    move_files(train_pairs, train_dir)
    move_files(val_pairs, val_dir)

    print(f"Train: {len(train_pairs)} pairs → {train_dir}")
    print(f"Validation: {len(val_pairs)} pairs → {val_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/Validation data split")
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    
    args = parser.parse_args()
    split_data(args.data_dir, args.output_dir, args.train_ratio)
