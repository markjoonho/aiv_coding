import numpy as np

class CutMix:
    def __init__(self, beta=1.0, min_area_ratio=0.1):
        self.beta = beta
        self.min_area_ratio = min_area_ratio

    def __call__(self, sample1, sample2):
        # sample1, sample2는 {"image": tensor, "boxes": np.array, "labels": np.array} 형태라고 가정
        img1 = sample1["image"]  # shape: (C, H, W)
        img2 = sample2["image"]

        # 이미지 tensor의 H, W는 1,2번 차원에 위치
        C, H, W = img1.shape
        lam = np.random.beta(self.beta, self.beta)
        cut_rat = np.sqrt(1 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        x1 = np.clip(cx - cut_w // 2, 0, W)
        y1 = np.clip(cy - cut_h // 2, 0, H)
        x2 = np.clip(cx + cut_w // 2, 0, W)
        y2 = np.clip(cy + cut_h // 2, 0, H)
        
        # torch tensor에서는 .clone()을 사용하여 복사합니다.
        new_img = img1.clone()
        # 텐서의 경우 (C, H, W) 형식이므로 H, W 영역에 대해 슬라이싱합니다.
        new_img[:, y1:y2, x1:x2] = img2[:, y1:y2, x1:x2]
        
        new_boxes = []
        new_labels = []
        
        boxes1 = sample1["boxes"]
        labels1 = sample1["labels"]
        boxes2 = sample2["boxes"]
        labels2 = sample2["labels"]
        
        # 이미지1의 bbox 처리: bbox 중심이 cut 영역 내부이면 제거
        for box, lab in zip(boxes1, labels1):
            bx1, by1, bx2, by2 = box
            cx_box = (bx1 + bx2) / 2
            cy_box = (by1 + by2) / 2
            if not (x1 <= cx_box <= x2 and y1 <= cy_box <= y2):
                new_boxes.append(box)
                new_labels.append(lab)
                
        # 이미지2의 bbox 처리: cut 영역과의 교집합 bbox 추가 (일정 비율 이상일 경우)
        for box, lab in zip(boxes2, labels2):
            bx1, by1, bx2, by2 = box
            inter_x1 = max(bx1, x1)
            inter_y1 = max(by1, y1)
            inter_x2 = min(bx2, x2)
            inter_y2 = min(by2, y2)
            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                orig_area = (bx2 - bx1) * (by2 - by1)
                if orig_area > 0 and (inter_area / orig_area) >= self.min_area_ratio:
                    new_boxes.append([inter_x1, inter_y1, inter_x2, inter_y2])
                    new_labels.append(lab)
                    
        # bbox 클리핑: 이미지 경계를 넘지 않도록 조정
        final_boxes = []
        final_labels = []
        for box, lab in zip(new_boxes, new_labels):
            bx1, by1, bx2, by2 = box
            bx1 = np.clip(bx1, 0, W)
            by1 = np.clip(by1, 0, H)
            bx2 = np.clip(bx2, 0, W)
            by2 = np.clip(by2, 0, H)
            if bx2 > bx1 and by2 > by1:
                final_boxes.append([bx1, by1, bx2, by2])
                final_labels.append(lab)
                
        return {"image": new_img, "boxes": np.array(final_boxes), "labels": np.array(final_labels)}
