import torch
import torch.nn.functional as F
from torch import nn
from scipy.optimize import linear_sum_assignment

def box_xywh_to_cxcywh(x):
    x, y, w, h = x.unbind(-1)
    cx = x + 0.5 * w
    cy = y + 0.5 * h
    return torch.stack([cx, cy, w, h], dim=-1)

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h]
    return torch.stack(b, dim=-1)

def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]).clamp(0) * (boxes[:, 3] - boxes[:, 1]).clamp(0)

def generalized_box_iou(boxes1, boxes2):
    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    union = area1[:, None] + area2 - inter
    iou = inter / union

    convex_x1 = torch.min(boxes1[:, None, 0], boxes2[:, 0])
    convex_y1 = torch.min(boxes1[:, None, 1], boxes2[:, 1])
    convex_x2 = torch.max(boxes1[:, None, 2], boxes2[:, 2])
    convex_y2 = torch.max(boxes1[:, None, 3], boxes2[:, 3])
    convex_area = (convex_x2 - convex_x1).clamp(min=0) * (convex_y2 - convex_y1).clamp(min=0)
    
    giou = iou - (convex_area - union) / convex_area
    return giou

class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "모든 비용 값이 0이면 안됩니다."

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_prob = outputs["pred_logits"].sigmoid()  # Binary classification이므로 sigmoid 적용
        out_bbox = outputs["pred_boxes"]
        indices = []

        for b in range(bs):
            tgt_ids = targets[b]["labels"]  # [num_target]
            tgt_bbox = targets[b]["boxes"]  # [num_target, 4]
            
            num_target = tgt_ids.shape[0]

            if num_target == 0:
                indices.append((torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)))
                continue

            # 분류 비용: binary cross-entropy cost 계산 (예측 확률 p와 타겟 y)
            # log(0)를 방지하기 위해 clamp 사용
            p = out_prob[b].clamp(min=1e-8, max=1-1e-8)  # [num_queries, 1]
            tgt_ids_float = tgt_ids.float()  # 타겟은 0 또는 1로 가정
            cost_class = - (tgt_ids_float.unsqueeze(0) * torch.log(p) + (1 - tgt_ids_float).unsqueeze(0) * torch.log(1 - p))
            # cost_class shape: [num_queries, num_target]

            cost_bbox = torch.cdist(out_bbox[b], tgt_bbox, p=1)
            
            out_bbox_xyxy = box_cxcywh_to_xyxy(out_bbox[b])
            tgt_bbox_xyxy = box_cxcywh_to_xyxy(tgt_bbox)
            cost_giou = -generalized_box_iou(out_bbox_xyxy, tgt_bbox_xyxy)

            # 최종 비용 행렬 계산
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = C.cpu()
            indices_b = linear_sum_assignment(C)
            indices.append((torch.as_tensor(indices_b[0], dtype=torch.int64),
                            torch.as_tensor(indices_b[1], dtype=torch.int64)))

        return indices


class OWLVITLoss(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        
    def loss_labels(self, outputs, targets, indices, num_boxes):
        src_logits = outputs['pred_logits']  # [B, num_queries, 1]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        
        target_classes = torch.zeros_like(src_logits, dtype=torch.float)
        target_classes[idx] = 1  # 매칭된 객체에 대해 1

        loss_ce = F.binary_cross_entropy_with_logits(src_logits, target_classes)
        return {'loss_ce': loss_ce}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_bbox = loss_bbox.sum() / num_boxes

        src_boxes_xyxy = box_cxcywh_to_xyxy(src_boxes)
        target_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes)
        giou = generalized_box_iou(src_boxes_xyxy, target_boxes_xyxy)
        loss_giou = (1 - torch.diag(giou)).sum() / num_boxes
        return {'loss_bbox': loss_bbox, 'loss_giou': loss_giou}

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return (batch_idx, src_idx)

    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        
        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices, num_boxes))
        losses.update(self.loss_boxes(outputs, targets, indices, num_boxes))
        
        total_loss = sum(losses[k] * self.weight_dict.get(k, 1.0) for k in losses.keys())
        losses['total_loss'] = total_loss
        return losses
