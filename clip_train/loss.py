import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIPContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(CLIPContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, image_features, text_features):
        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)
        logits = torch.matmul(image_features, text_features.T) / self.temperature
        batch_size = logits.shape[0]
        labels = torch.arange(batch_size).to(logits.device)
        loss_img = F.cross_entropy(logits, labels)
        loss_txt = F.cross_entropy(logits.T, labels)
        return (loss_img + loss_txt) / 2