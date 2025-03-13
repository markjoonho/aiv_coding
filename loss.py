import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou

def sigmoid_cost(
    logit: torch.Tensor,
    focal_loss: bool = False,
    focal_alpha: float = None,
    focal_gamma: float = None
) -> torch.Tensor:
    """Computes the classification cost."""
    neg_cost_class = -F.logsigmoid(-logit)
    pos_cost_class = -F.logsigmoid(logit)
    
    if focal_loss:
        neg_cost_class *= (1 - focal_alpha) * torch.sigmoid(logit) ** focal_gamma
        pos_cost_class *= focal_alpha * torch.sigmoid(-logit) ** focal_gamma
    
    return pos_cost_class - neg_cost_class  # [B, N, C]

def compute_cost(
    tgt_labels: torch.Tensor,
    out_logits: torch.Tensor,
    tgt_bbox: torch.Tensor,
    out_bbox: torch.Tensor,
    class_loss_coef: float,
    bbox_loss_coef: float,
    giou_loss_coef: float,
    focal_loss: bool = False,
    focal_alpha: float = None,
    focal_gamma: float = None,
) -> torch.Tensor:
    """Computes cost matrices for a batch of predictions."""
    if focal_loss and (focal_alpha is None or focal_gamma is None):
        raise ValueError('For focal loss, focal_alpha and focal_gamma must be set.')
    
    n_labels_per_instance = torch.sum(tgt_labels[..., 1:], dim=-1)
    mask = n_labels_per_instance > 0  # [B, M]
    
    tgt_labels = torch.cat([
        (~mask).unsqueeze(-1), tgt_labels[..., 1:]
    ], dim=-1)
    
    cost_class = sigmoid_cost(out_logits, focal_loss, focal_alpha, focal_gamma)  # [B, N, C]
    cost_class = torch.einsum('bnl,bml->bnm', cost_class, tgt_labels)
    
    cost = class_loss_coef * cost_class
    
    diff = torch.abs(out_bbox[:, :, None] - tgt_bbox[:, None, :])  # [B, N, M, 4]
    cost_bbox = torch.sum(diff, dim=-1)  # [B, N, M]
    cost = cost + bbox_loss_coef * cost_bbox
    
    cost_giou = -generalized_box_iou(
        out_bbox, tgt_bbox
    )
    cost = cost + giou_loss_coef * cost_giou
    
    mask = mask.unsqueeze(1)
    
    cost_mask_value = torch.max(torch.where(mask, cost, torch.tensor(-1e10, device=cost.device)), dim=(1, 2))[0]
    all_masked = torch.all(~mask, dim=(1, 2))
    cost_mask_value = torch.where(~all_masked, cost_mask_value, torch.tensor(1.0, device=cost.device))
    cost_mask_value = cost_mask_value[:, None, None] * 1.1 + 10.0
    
    cost = cost * mask + (1.0 - mask) * cost_mask_value
    cost = torch.nan_to_num(cost, nan=cost_mask_value, posinf=cost_mask_value, neginf=cost_mask_value)
    
    max_num_boxes = tgt_labels.shape[1]
    n_cols = torch.where(
        torch.max(mask, dim=1)[0],
        torch.arange(1, max_num_boxes + 1, device=cost.device).unsqueeze(0),
        torch.tensor(0, device=cost.device)
    )
    n_cols = torch.max(n_cols, dim=1)[0]
    
    return cost, n_cols


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