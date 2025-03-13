import cv2 
import numpy as np


def draw_rect(im, cords, color = None):
    """Draw the rectangle on the image
    
    Parameters
    ----------
    
    im : numpy.ndarray
        numpy image 
    
    cords: numpy.ndarray
        Numpy array containing bounding boxes of shape `N X 4` where N is the 
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`
        
    Returns
    -------
    
    numpy.ndarray
        numpy image with bounding boxes drawn on it
        
    """
    
    im = im.copy()
    
    cords = cords[:,:4]
    cords = cords.reshape(-1,4)
    if not color:
        color = [0,255,0]
    for cord in cords:
        
        pt1, pt2 = (cord[0], cord[1]) , (cord[2], cord[3])
                
        pt1 = int(pt1[0]), int(pt1[1])
        pt2 = int(pt2[0]), int(pt2[1])
    
        im = cv2.rectangle(im.copy(), pt1, pt2, color, int(max(im.shape[:2])/200))
    return im

def bbox_area(bbox):
    return (bbox[:,2] - bbox[:,0])*(bbox[:,3] - bbox[:,1])
        
def clip_box(bbox, clip_box, alpha):
    """Clip the bounding boxes to the borders of an image
    
    Parameters
    ----------
    
    bbox: numpy.ndarray
        Numpy array containing bounding boxes of shape `N X 4` where N is the 
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`
    
    clip_box: numpy.ndarray
        An array of shape (4,) specifying the diagonal co-ordinates of the image
        The coordinates are represented in the format `x1 y1 x2 y2`
        
    alpha: float
        If the fraction of a bounding box left in the image after being clipped is 
        less than `alpha` the bounding box is dropped. 
    
    Returns
    -------
    
    numpy.ndarray
        Numpy array containing **clipped** bounding boxes of shape `N X 4` where N is the 
        number of bounding boxes left are being clipped and the bounding boxes are represented in the
        format `x1 y1 x2 y2` 
    
    """
    ar_ = (bbox_area(bbox))
    x_min = np.maximum(bbox[:,0], clip_box[0]).reshape(-1,1)
    y_min = np.maximum(bbox[:,1], clip_box[1]).reshape(-1,1)
    x_max = np.minimum(bbox[:,2], clip_box[2]).reshape(-1,1)
    y_max = np.minimum(bbox[:,3], clip_box[3]).reshape(-1,1)
    
    bbox = np.hstack((x_min, y_min, x_max, y_max, bbox[:,4:]))
    
    delta_area = ((ar_ - bbox_area(bbox))/ar_)
    
    mask = (delta_area < (1 - alpha)).astype(int)
    
    bbox = bbox[mask == 1,:]


    return bbox

def subtract_cut_from_bbox(box, cut, min_area_ratio=0.25):
    """
    주어진 bbox와 cut 영역([x1, y1, x2, y2] 형태)에 대해,
    bbox와 cut 영역의 교집합을 제거한 후 남은 영역(여러 후보 중 가장 넓은 영역)을 구합니다.
    
    Parameters
    ----------
    box: list or array-like, [x1, y1, x2, y2]
    cut: list or array-like, [cx1, cy1, cx2, cy2]
    min_area_ratio: float
        남은 후보 영역의 면적이 원래 bbox 면적의 이 비율 이상이어야 함.
    
    Returns
    -------
    list or None: 조정된 bbox [x1, y1, x2, y2] 또는 만족하는 후보가 없으면 None.
    """
    x1, y1, x2, y2 = box
    cx1, cy1, cx2, cy2 = cut
    
    # 교집합 계산
    inter_x1 = max(x1, cx1)
    inter_y1 = max(y1, cy1)
    inter_x2 = min(x2, cx2)
    inter_y2 = min(y2, cy2)
    
    # 교집합이 없으면 원래 bbox 그대로 반환
    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return box
    
    original_area = (x2 - x1) * (y2 - y1)
    candidates = []
    
    # 후보 1: 좌측 영역 (bbox의 왼쪽 부분)
    if x1 < cx1:
        candidates.append([x1, y1, min(x2, cx1), y2])
    # 후보 2: 우측 영역
    if x2 > cx2:
        candidates.append([max(x1, cx2), y1, x2, y2])
    # 후보 3: 상단 영역
    if y1 < cy1:
        candidates.append([x1, y1, x2, min(y2, cy1)])
    # 후보 4: 하단 영역
    if y2 > cy2:
        candidates.append([x1, max(y1, cy2), x2, y2])
    
    best = None
    best_area = 0
    for cand in candidates:
        cx1_c, cy1_c, cx2_c, cy2_c = cand
        area = (cx2_c - cx1_c) * (cy2_c - cy1_c)
        if area >= min_area_ratio * original_area and area > best_area:
            best = cand
            best_area = area
    return best

def adjust_bbox_for_image2(box, cut, min_area_ratio=0.25):
    """
    이미지2의 bbox에 대해, cut 영역과의 교집합 영역을 반환합니다.
    교집합 면적이 원래 bbox 면적의 min_area_ratio 이상이면 반환하고, 
    그렇지 않으면 None을 반환합니다.
    """
    x1, y1, x2, y2 = box
    cx1, cy1, cx2, cy2 = cut
    inter_x1 = max(x1, cx1)
    inter_y1 = max(y1, cy1)
    inter_x2 = min(x2, cx2)
    inter_y2 = min(y2, cy2)
    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return None
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    original_area = (x2 - x1) * (y2 - y1)
    if inter_area >= min_area_ratio * original_area:
        return [inter_x1, inter_y1, inter_x2, inter_y2]
    return None