import argparse
from clip_model import OWLVITCLIPModel

def main():
    parser = argparse.ArgumentParser(description="Train OWLVITCLIPModel with custom parameters")
    parser.add_argument('--train_dir', type=str, default="./total_dataset/train_dataset/", help='Path to training dataset')
    parser.add_argument('--val_dir', type=str, default="./total_dataset/val_dataset/", help='Path to validation dataset')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate for training')
    
    args = parser.parse_args()
    
    model_wrapper = OWLVITCLIPModel(use_lora=True) #True, False
    model_wrapper.train(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )

if __name__ == "__main__":
    main()