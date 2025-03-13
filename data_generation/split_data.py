import os
import random
import shutil
import argparse

def split_data(data_dir, output_dir, train_ratio=0.8, random_seed=42):
    """
    .bmp와 .json 파일을 train/validation으로 나눠서 저장하는 함수.

    Args:
        data_dir (str): 원본 데이터 폴더 (bmp와 json이 포함된 폴더)
        output_dir (str): 분할된 데이터 저장 폴더
        train_ratio (float): Train 데이터의 비율 (default: 0.8)
        random_seed (int): 랜덤 시드 (default: 42)
    """
    # Train / Validation 폴더 경로
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")

    # Validation 폴더가 이미 있으면 종료
    if os.path.exists(val_dir):
        print(f"✅ Validation 폴더({val_dir})가 이미 존재합니다. 데이터 분할을 건너뜁니다.")
        return

    # 폴더 생성
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # 파일 목록 불러오기
    bmp_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".bmp")])
    json_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".json")])

    # .bmp와 .json Pair 매칭
    pairs = [(bmp, bmp.replace(".bmp", ".json")) for bmp in bmp_files if bmp.replace(".bmp", ".json") in json_files]

    # 랜덤 섞기
    random.seed(random_seed)
    random.shuffle(pairs)

    # Train/Validation 분할
    train_size = int(len(pairs) * train_ratio)
    train_pairs = pairs[:train_size]
    val_pairs = pairs[train_size:]

    # 파일 복사 함수
    def move_files(pairs, target_dir):
        for bmp, json in pairs:
            shutil.copy(os.path.join(data_dir, bmp), os.path.join(target_dir, bmp))
            shutil.copy(os.path.join(data_dir, json), os.path.join(target_dir, json))

    # Train/Validation 데이터 저장
    move_files(train_pairs, train_dir)
    move_files(val_pairs, val_dir)

    print(f"✅ 데이터 분할 완료!")
    print(f"Train: {len(train_pairs)} pairs → {train_dir}")
    print(f"Validation: {len(val_pairs)} pairs → {val_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/Validation 데이터 분할 스크립트")
    parser.add_argument("--data_dir", type=str, help="원본 데이터 폴더 경로")
    parser.add_argument("--output_dir", type=str, help="결과 데이터가 저장될 폴더")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train 데이터 비율 (default: 0.8)")
    
    args = parser.parse_args()
    split_data(args.data_dir, args.output_dir, args.train_ratio)
