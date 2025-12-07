"""
바운딩 박스 관련 유틸리티 함수들
"""
import numpy as np


def calculate_bbox_iou(bbox1, bbox2):
    """
    두 bbox 간의 IoU(Intersection over Union) 계산
    
    Args:
        bbox1, bbox2: [x1, y1, x2, y2] 형식의 바운딩 박스
    
    Returns:
        IoU 값 (0.0 ~ 1.0)
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # 교집합 영역 계산
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # 각 bbox의 면적
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def calculate_bbox_center_distance(bbox1, bbox2):
    """
    두 bbox의 중심점 간 거리 계산
    
    Args:
        bbox1, bbox2: [x1, y1, x2, y2] 형식의 바운딩 박스
    
    Returns:
        중심점 간 유클리드 거리
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    center1_x = (x1_1 + x2_1) / 2
    center1_y = (y1_1 + y2_1) / 2
    center2_x = (x1_2 + x2_2) / 2
    center2_y = (y1_2 + y2_2) / 2
    
    distance = np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
    return distance

def is_same_face_region(bbox1, bbox2, iou_threshold=0.3, distance_threshold=None):
    """
    두 bbox가 같은 얼굴 영역을 가리키는지 판단
    
    Args:
        bbox1, bbox2: [x1, y1, x2, y2] 형식의 바운딩 박스
        iou_threshold: IoU 임계값 (기본 0.3)
        distance_threshold: 중심점 거리 임계값 (None이면 bbox 크기 기반 자동 계산)
    
    Returns:
        같은 얼굴 영역이면 True, 아니면 False
    """
    # IoU 기반 판단
    iou = calculate_bbox_iou(bbox1, bbox2)
    if iou >= iou_threshold:
        return True
    
    # 중심점 거리 기반 판단 (보조)
    if distance_threshold is None:
        # bbox 크기의 평균을 기준으로 임계값 설정
        w1 = bbox1[2] - bbox1[0]
        h1 = bbox1[3] - bbox1[1]
        w2 = bbox2[2] - bbox2[0]
        h2 = bbox2[3] - bbox2[1]
        avg_size = (w1 + h1 + w2 + h2) / 4
        distance_threshold = avg_size * 0.5  # bbox 크기의 50% 이내면 같은 얼굴로 간주
    
    distance = calculate_bbox_center_distance(bbox1, bbox2)
    if distance <= distance_threshold:
        return True
    
    return False
