import torch
from transformers import OwlViTProcessor
from train import OWLVITCLIPModel
from dataset import ImageTextBBoxDataset, collate_fn
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_dataset(val_dataset_dir, checkpoint_path, use_lora=True):

    val_dataset = ImageTextBBoxDataset(val_dataset_dir, transform=None, oversample=1)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model_wrapper = OWLVITCLIPModel(use_lora=use_lora)
    model_wrapper.load_checkpoint(checkpoint_path)
    
    return model_wrapper, val_loader

def plot_predictions(input_image, text_queries, scores, boxes, labels, image_path, output_dir="val_vis", score_threshold=0.5):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(input_image, extent=(0, 1, 1, 0))
    ax.set_axis_off()
    
    for score, box, label in zip(scores, boxes, labels):
        if score < score_threshold:
            continue
        
        cx, cy, w, h = box
        ax.plot([cx - w/2, cx + w/2, cx + w/2, cx - w/2, cx - w/2],
                [cy - h/2, cy - h/2, cy + h/2, cy + h/2, cy - h/2], "r")
        ax.text(cx - w / 2, cy + h / 2 + 0.015,
                f"{text_queries[label]}: {score:.2f}",
                ha="left", va="top", color="red",
                bbox={"facecolor": "white", "edgecolor": "red", "boxstyle": "square,pad=.3"})
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(image_path).replace(".bmp", "_vis.png"))
    plt.savefig(output_path)
    plt.close()
    

def inference_and_visualization(model_wrapper, val_loader, text_queries, output_dir="val_vis", score_threshold=0.5):
    all_gt, all_preds, all_pred_probs = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            image = batch['images'][0]
            image_path = batch['image_path'][0]
            inputs = processor(text='stabbed exist', images=image, return_tensors='pt')
            outputs = model_wrapper.model(pixel_values=inputs["pixel_values"].to(device),
                                        input_ids=inputs["input_ids"].to(device),
                                        attention_mask=inputs["attention_mask"].to(device))
        
            # Get prediction logits
            logits = torch.max(outputs["logits"][0], dim=-1)
            scores = torch.sigmoid(logits.values).cpu().detach().numpy()

            labels = logits.indices.cpu().detach().numpy()
            boxes = outputs["pred_boxes"][0].cpu().detach().numpy()
            plot_predictions(image, text_queries, scores, boxes, labels, image_path)
        

if __name__ == "__main__":
    val_dataset_dir = "./total_dataset/val_dataset/"
    checkpoint_path = "./ckpt_final/20250318_050935/epoch_20.pth"
    
    text_queries = ['stabbed exist']
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    
    model_wrapper, val_loader = load_model_and_dataset(val_dataset_dir,checkpoint_path, True)
    inference_and_visualization(model_wrapper, val_loader, text_queries)