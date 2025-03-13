import torch
import torch.nn.functional as F
from torch import nn
from scipy.optimize import linear_sum_assignment

def box_cxcywh_to_xyxy(x):
    """
    [center_x, center_y, width, height] 형식을 [x1, y1, x2, y2] 형식으로 변환합니다.
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h]
    return torch.stack(b, dim=-1)

def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]).clamp(0) * (boxes[:, 3] - boxes[:, 1]).clamp(0)

def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU (GIoU) 계산 (참고: https://giou.stanford.edu/)
    boxes1: [N, 4], boxes2: [M, 4] (두 박스 모두 [x1, y1, x2, y2] 형식)
    리턴값: [N, M] 텐서 (각각의 박스 쌍에 대한 GIoU)
    """
    # 교집합 영역 계산
    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    
    # 각 박스의 넓이와 합집합 계산
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    union = area1[:, None] + area2 - inter
    
    iou = inter / union

    # 두 박스를 포함하는 최소한의 convex box 계산
    convex_x1 = torch.min(boxes1[:, None, 0], boxes2[:, 0])
    convex_y1 = torch.min(boxes1[:, None, 1], boxes2[:, 1])
    convex_x2 = torch.max(boxes1[:, None, 2], boxes2[:, 2])
    convex_y2 = torch.max(boxes1[:, None, 3], boxes2[:, 3])
    convex_area = (convex_x2 - convex_x1).clamp(min=0) * (convex_y2 - convex_y1).clamp(min=0)
    
    giou = iou - (convex_area - union) / convex_area
    return giou

class HungarianMatcher(nn.Module):
    """
    Hungarian matching 모듈:
      - outputs: {"pred_logits": [B, num_queries, num_classes], "pred_boxes": [B, num_queries, 4]}
        (pred_boxes는 normalized된 [cx, cy, w, h] 형식)
      - targets: list (길이 B)이며, 각 원소는 {"labels": [num_target], "boxes": [num_target, 4]} 형식입니다.
    비용은 classification, L1(bbox) 및 GIoU 비용을 가중합하여 구성합니다.
    """
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "모든 비용 값이 0이면 안됩니다."

    @torch.no_grad()
    def forward(self, outputs, targets):
        # outputs: {"pred_logits": [B, num_queries, num_classes], "pred_boxes": [B, num_queries, 4]}
        # targets: 리스트, 각 원소 {"labels": [num_target], "boxes": [num_target, 4]}
        bs, num_queries = outputs["pred_logits"].shape[:2]
        # softmax를 통해 분류 확률을 계산합니다.
        if outputs["pred_logits"].shape[-1] == 1:
            zeros = torch.zeros_like(outputs["pred_logits"])
            outputs["pred_logits"] = torch.cat([zeros, outputs["pred_logits"]], dim=-1)  # 결과: [B, num_queries, 2]
        out_prob = outputs["pred_logits"].softmax(-1)  # [B, num_queries, num_classes]
        out_bbox = outputs["pred_boxes"]
        indices = []
        for b in range(bs):
            tgt_ids = targets[b]["labels"]  # [num_target]
            tgt_bbox = targets[b]["boxes"]  # [num_target, 4]
            
            # 비용 계산
            cost_class = -out_prob[b][:, tgt_ids]  # 음의 확률 (최대화 문제 -> 최소화 문제 변환)
            cost_bbox = torch.cdist(out_bbox[b], tgt_bbox, p=1)
            # GIoU 비용: 박스 형식을 [x1, y1, x2, y2]로 변환 후 계산
            out_bbox_xyxy = box_cxcywh_to_xyxy(out_bbox[b])
            tgt_bbox_xyxy = box_cxcywh_to_xyxy(tgt_bbox)
            cost_giou = -generalized_box_iou(out_bbox_xyxy, tgt_bbox_xyxy)
            
            # 총 비용 행렬 구성
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = C.cpu()
            indices_b = linear_sum_assignment(C)
            # indices_b[0]: 예측 인덱스, indices_b[1]: 타겟 인덱스
            indices.append((torch.as_tensor(indices_b[0], dtype=torch.int64),
                            torch.as_tensor(indices_b[1], dtype=torch.int64)))
        return indices

class OWLVITLoss(nn.Module):
    """
    OWL-ViT용 loss 모듈.
    내부적으로 HungarianMatcher를 통해 예측-타겟 매칭을 수행한 뒤,
    classification loss, L1 loss, 그리고 GIoU loss를 계산합니다.
    
    weight_dict: {'loss_ce': weight, 'loss_bbox': weight, 'loss_giou': weight}
    eos_coef: 백그라운드 클래스(혹은 'no-object' 클래스)에 대한 가중치 계수
    losses: 사용할 loss 종류 리스트 (예: ['labels', 'boxes'])
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        
        # 백그라운드 클래스의 가중치 설정 (클래스 0을 백그라운드로 가정)
        empty_weight = torch.ones(num_classes)
        empty_weight[0] = eos_coef
        self.register_buffer('empty_weight', empty_weight)
    
    def loss_labels(self, outputs, targets, indices, num_boxes):
        """
        분류에 대한 cross entropy loss 계산.
        """
        src_logits = outputs['pred_logits']  # [B, num_queries, num_classes]
        # 매칭된 인덱스에 대한 permutation index 생성
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        # 모든 예측에 대해 기본값(백그라운드, 0)으로 채우고, 매칭된 부분에 실제 GT 클래스를 할당
        target_classes = torch.full(src_logits.shape[:2], 0, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, weight=self.empty_weight)
        return {'loss_ce': loss_ce}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
        박스 회귀에 대한 L1 loss와 GIoU loss 계산.
        """
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_bbox = loss_bbox.sum() / num_boxes

        # GIoU loss 계산: 먼저 [x1,y1,x2,y2] 형식으로 변환
        src_boxes_xyxy = box_cxcywh_to_xyxy(src_boxes)
        target_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes)
        giou = generalized_box_iou(src_boxes_xyxy, target_boxes_xyxy)
        loss_giou = (1 - torch.diag(giou)).sum() / num_boxes
        return {'loss_bbox': loss_bbox, 'loss_giou': loss_giou}

    def _get_src_permutation_idx(self, indices):
        """
        batch 내 예측 결과의 permutation index를 생성합니다.
        """
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return (batch_idx, src_idx)

    def forward(self, outputs, targets):
        """
        전체 loss 계산:
          1. matcher를 통해 예측과 GT를 매칭
          2. 각 loss (labels, boxes 등) 계산 후 합산
        """

        indices = self.matcher(outputs, targets)
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        
        losses = {}
        loss_dict = self.loss_labels(outputs, targets, indices, num_boxes)
        losses.update(loss_dict)
        loss_dict = self.loss_boxes(outputs, targets, indices, num_boxes)
        losses.update(loss_dict)
        
        # 지정한 weight에 따라 loss 가중합
        total_loss = sum(losses[k] * self.weight_dict.get(k, 1.0) for k in losses.keys())
        losses['total_loss'] = total_loss
        return losses

# -------------------------------
# 사용 예시:
# -------------------------------
if __name__ == '__main__':
    # 예시 배치 크기와 num_queries, num_classes
    batch_size = 2
    num_queries = 100
    num_classes = 91  # 예를 들어 COCO의 클래스 수 (백그라운드 포함)

    # 임의의 예측값 생성 (pred_logits: [B, num_queries, num_classes], pred_boxes: [B, num_queries, 4])
    outputs = {
        "pred_logits": torch.randn(batch_size, num_queries, num_classes),
        "pred_boxes": torch.rand(batch_size, num_queries, 4)
    }

    # 타겟 생성 (각 배치마다 서로 다른 GT 수)
    targets = []
    for i in range(batch_size):
        num_targets = torch.randint(1, 10, (1,)).item()
        targets.append({
            "labels": torch.randint(1, num_classes, (num_targets,)),  # 0을 백그라운드로 가정
            "boxes": torch.rand(num_targets, 4)
        })

    # Matcher와 loss 모듈 생성
    matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)
    weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}
    criterion = OWLVITLoss(num_classes=num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=0.1, losses=['labels', 'boxes'])
    
    # loss 계산
    loss_dict = criterion(outputs, targets)
    print("Losses:", loss_dict)
