import argparse
from clip_model import OWLVITCLIPModel

def main():
    parser = argparse.ArgumentParser(description="Train OWLVITCLIPModel with custom parameters")
    parser.add_argument('--train_dir', type=str, default="./total_dataset/train_dataset/")
    parser.add_argument('--val_dir', type=str, default="./total_dataset/val_dataset/")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--use_lora', action='store_true')
    
    args = parser.parse_args()
    
    model_wrapper = OWLVITCLIPModel(use_lora=args.use_lora)
    model_wrapper.train(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )

if __name__ == "__main__":
    main()
