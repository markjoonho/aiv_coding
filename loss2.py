import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

def generalized_box_iou(boxes1, boxes2):
    """
    boxes1, boxes2: [N, 4] 텐서, [x1, y1, x2, y2] 형식.
    각 쌍에 대해 IoU를 계산하고, GIoU를 반환한 후 평균을 구합니다.
    """
    inter_x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    inter_y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    inter_x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    inter_y2 = torch.min(boxes1[:, 3], boxes2[:, 3])
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union_area = area1 + area2 - inter_area

    iou = inter_area / (union_area + 1e-6)

    enc_x1 = torch.min(boxes1[:, 0], boxes2[:, 0])
    enc_y1 = torch.min(boxes1[:, 1], boxes2[:, 1])
    enc_x2 = torch.max(boxes1[:, 2], boxes2[:, 2])
    enc_y2 = torch.max(boxes1[:, 3], boxes2[:, 3])
    enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1) + 1e-6

    giou = iou - (enc_area - union_area) / enc_area
    return giou.mean()

class HungarianMatcher(nn.Module):
    def __init__(self, cls_cost_weight=1.0, bbox_cost_weight=5.0, giou_cost_weight=2.0):
        super().__init__()
        self.cls_cost_weight = cls_cost_weight
        self.bbox_cost_weight = bbox_cost_weight
        self.giou_cost_weight = giou_cost_weight

    @torch.no_grad()
    def forward(self, pred_logits, pred_boxes, targets):
        bs, num_queries, num_classes = pred_logits.shape
        indices = []
        for b in range(bs):
            out_prob = pred_logits[b].softmax(-1)  # [num_queries, num_classes]
            out_bbox = pred_boxes[b]               # [num_queries, 4]

            # 정수형으로 변환
            tgt_ids = targets[b]["labels"].long()         # [num_targets]
            tgt_bbox = targets[b]["boxes"].float()         

            if num_classes == 1:
                adjusted_tgt_ids = tgt_ids - 1
            else:
                adjusted_tgt_ids = tgt_ids

            # adjusted_tgt_ids가 long 타입임을 보장합니다.
            cost_cls = -out_prob[:, adjusted_tgt_ids]  # [num_queries, num_targets]

            cost_bbox = torch.cdist(out_bbox.float(), tgt_bbox, p=1)  # [num_queries, num_targets]
            out_bbox_xyxy = self.box_cxcywh_to_xyxy(out_bbox.float())  
            tgt_bbox_xyxy = self.box_cxcywh_to_xyxy(tgt_bbox)            
            cost_giou = -self.pairwise_giou(out_bbox_xyxy, tgt_bbox_xyxy)  

            C = self.cls_cost_weight * cost_cls + self.bbox_cost_weight * cost_bbox + self.giou_cost_weight * cost_giou
            C = C.cpu().detach().numpy()

            row_ind, col_ind = linear_sum_assignment(C)
            indices.append((torch.as_tensor(row_ind, dtype=torch.int64),
                            torch.as_tensor(col_ind, dtype=torch.int64)))
        return indices


    def box_cxcywh_to_xyxy(self, boxes):
        cx, cy, w, h = boxes.unbind(-1)
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        return torch.stack((x1, y1, x2, y2), dim=-1)

    def pairwise_giou(self, boxes1, boxes2):
        """
        boxes1: [N,4], boxes2: [M,4] (xyxy 형식)
        반환: [N, M] 크기의 GIoU 텐서
        """
        N = boxes1.shape[0]
        M = boxes2.shape[0]
        
        boxes1_exp = boxes1[:, None, :].expand(N, M, 4)
        boxes2_exp = boxes2[None, :, :].expand(N, M, 4)
        
        inter_x1 = torch.max(boxes1_exp[..., 0], boxes2_exp[..., 0])
        inter_y1 = torch.max(boxes1_exp[..., 1], boxes2_exp[..., 1])
        inter_x2 = torch.min(boxes1_exp[..., 2], boxes2_exp[..., 2])
        inter_y2 = torch.min(boxes1_exp[..., 3], boxes2_exp[..., 3])
        inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
        
        area1 = (boxes1_exp[..., 2] - boxes1_exp[..., 0]) * (boxes1_exp[..., 3] - boxes1_exp[..., 1])
        area2 = (boxes2_exp[..., 2] - boxes2_exp[..., 0]) * (boxes2_exp[..., 3] - boxes2_exp[..., 1])
        union_area = area1 + area2 - inter_area
        iou = inter_area / (union_area + 1e-6)
        
        enc_x1 = torch.min(boxes1_exp[..., 0], boxes2_exp[..., 0])
        enc_y1 = torch.min(boxes1_exp[..., 1], boxes2_exp[..., 1])
        enc_x2 = torch.max(boxes1_exp[..., 2], boxes2_exp[..., 2])
        enc_y2 = torch.max(boxes1_exp[..., 3], boxes2_exp[..., 3])
        enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1) + 1e-6
        
        giou = iou - (enc_area - union_area) / enc_area
        return giou

class OWLVITLoss(nn.Module):
    def __init__(self, cls_loss_weight=1.0, bbox_loss_weight=5.0, giou_loss_weight=2.0, no_object_label=0):
        """
        no_object_label: 객체가 없는 (배경) 클래스 인덱스 (일반적으로 0번)
        """
        super(OWLVITLoss, self).__init__()
        self.cls_loss_fn = nn.CrossEntropyLoss()  # 분류 손실 함수
        self.l1_loss_fn = nn.L1Loss()             # 박스 회귀 손실 함수
        self.cls_loss_weight = cls_loss_weight
        self.bbox_loss_weight = bbox_loss_weight
        self.giou_loss_weight = giou_loss_weight
        self.no_object_label = no_object_label
        # Hungarian matching 모듈 생성
        self.matcher = HungarianMatcher(cls_cost_weight=cls_loss_weight,
                                        bbox_cost_weight=bbox_loss_weight,
                                        giou_cost_weight=giou_loss_weight)

    def forward(self, outputs, targets):
        """
        outputs: 딕셔너리,  
            - 'pred_logits': [B, num_queries, num_classes] 텐서  
            - 'pred_boxes': [B, num_queries, 4] 텐서 (cx, cy, w, h 형식)
        targets: 길이 B의 리스트, 각 원소는 딕셔너리  
            - 'labels': [num_targets] 텐서  
            - 'boxes': [num_targets, 4] 텐서 (cx, cy, w, h 형식)
        """
        pred_logits = outputs['pred_logits']  # [B, num_queries, num_classes]
        pred_boxes = outputs['pred_boxes']      # [B, num_queries, 4]
        B, num_queries, num_classes = pred_logits.shape
        device = pred_logits.device

        # Hungarian matching을 통해 각 이미지별 매칭 결과 획득
        indices = self.matcher(pred_logits, pred_boxes, targets)
        
        # 전체 쿼리에 대해 기본적으로 no_object_label로 채운 타겟 텐서 생성
        target_labels_full = torch.full((B, num_queries), self.no_object_label, dtype=torch.long, device=device)
        target_boxes_full = torch.zeros((B, num_queries, 4), dtype=torch.float, device=device)

        for i, (pred_idx, tgt_idx) in enumerate(indices):
            if pred_idx.numel() > 0:
                # 매칭된 쿼리에 한해서 GT 값 대입
                target_labels_full[i, pred_idx] = targets[i]['labels'][tgt_idx]
                target_boxes_full[i, pred_idx] = targets[i]['boxes'][tgt_idx].float()

        # 모든 쿼리에 대해 분류 손실 계산
        pred_logits_flat = pred_logits.view(-1, num_classes)
        target_labels_flat = target_labels_full.view(-1)
        loss_cls = self.cls_loss_fn(pred_logits_flat, target_labels_flat)

        # 매칭된 (positive) 쿼리에 대해서만 bbox 및 GIoU 손실 계산
        matched_mask = target_labels_flat != self.no_object_label
        if matched_mask.sum() > 0:
            pred_boxes_flat = pred_boxes.view(-1, 4)[matched_mask]
            target_boxes_flat = target_boxes_full.view(-1, 4)[matched_mask]
            loss_bbox = self.l1_loss_fn(pred_boxes_flat, target_boxes_flat)
            loss_giou = 1 - generalized_box_iou(self.box_cxcywh_to_xyxy(pred_boxes_flat),
                                                self.box_cxcywh_to_xyxy(target_boxes_flat))
        else:
            loss_bbox = 0 * loss_cls
            loss_giou = 0 * loss_cls

        total_loss = (self.cls_loss_weight * loss_cls +
                      self.bbox_loss_weight * loss_bbox +
                      self.giou_loss_weight * loss_giou)
        return total_loss

    def box_cxcywh_to_xyxy(self, boxes):
        cx, cy, w, h = boxes.unbind(-1)
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        return torch.stack((x1, y1, x2, y2), dim=-1)
