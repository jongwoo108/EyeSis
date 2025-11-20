# src/utils/mask_detector.py
"""
마스크 감지 유틸리티
얼굴의 하반부(코, 입 영역)가 가려져 있는지 간단히 판단합니다.
"""
import cv2
import numpy as np
from typing import Tuple

def detect_mask_simple(face_bbox: Tuple[int, int, int, int], img_shape: Tuple[int, int]) -> bool:
    """
    간단한 마스크 감지 (bbox 기반 추정)
    
    Args:
        face_bbox: (x1, y1, x2, y2) 얼굴 bounding box
        img_shape: (height, width) 이미지 크기
    
    Returns:
        마스크 착용 여부 (추정)
    """
    x1, y1, x2, y2 = face_bbox
    face_height = y2 - y1
    face_width = x2 - x1
    
    # 얼굴의 하반부(아래 40%) 영역이 가려져 있을 가능성
    # 실제로는 더 정교한 방법이 필요하지만, 간단한 추정으로 사용
    # 마스크는 보통 얼굴의 중간~하반부를 가림
    
    # 얼굴 비율로 추정 (너무 단순하지만 기본적인 힌트)
    aspect_ratio = face_width / face_height if face_height > 0 else 1.0
    
    # 실제로는 얼굴 랜드마크나 이미지 분석이 필요하지만,
    # 여기서는 임베딩 기반으로 판단하는 것이 더 정확할 수 있음
    # 이 함수는 placeholder로 사용
    
    return False  # 기본값: 마스크 감지 안함 (더 정교한 방법 필요)

def estimate_mask_from_embedding(embedding: np.ndarray, gallery_embeddings: dict) -> Tuple[float, float]:
    """
    임베딩 유사도로 마스크 착용 가능성 추정 및 적응형 임계값 계산
    
    마스크를 쓴 얼굴은 일반 얼굴보다 유사도가 낮게 나오는 경향이 있음.
    기존 등록 이미지(마스크 없음)만으로 마스크 쓴 얼굴도 인식하기 위해
    유사도가 낮을 때 마스크 가능성으로 판단하고 낮은 임계값을 적용합니다.
    
    Args:
        embedding: 현재 얼굴 임베딩
        gallery_embeddings: 갤러리 임베딩들
    
    Returns:
        (best_similarity, mask_probability) 튜플
        - best_similarity: 가장 높은 유사도
        - mask_probability: 마스크 착용 가능성 (0.0 ~ 1.0)
    """
    # 가장 높은 유사도 찾기
    best_sim = -1.0
    for ref_emb in gallery_embeddings.values():
        if ref_emb.ndim == 2:  # Bank
            sims = ref_emb @ embedding
            sim = float(np.max(sims))
        else:  # Centroid
            sim = float(np.dot(ref_emb, embedding))
        
        if sim > best_sim:
            best_sim = sim
    
    # 유사도가 낮으면 마스크 가능성 높음
    # 마스크를 쓴 얼굴은 보통 0.25~0.35 사이의 유사도를 보임
    if best_sim < 0.25:
        mask_prob = 0.9  # 매우 높은 가능성
    elif best_sim < 0.28:
        mask_prob = 0.7  # 높은 가능성
    elif best_sim < 0.32:
        mask_prob = 0.5  # 중간 가능성
    elif best_sim < 0.35:
        mask_prob = 0.3  # 낮은 가능성
    else:
        mask_prob = 0.0  # 마스크 아님
    
    return best_sim, mask_prob

def estimate_mask_from_similarity(similarity: float) -> float:
    """
    실제 매칭 유사도로 마스크 착용 가능성 추정
    
    Args:
        similarity: 실제 매칭 결과의 유사도
    
    Returns:
        마스크 착용 가능성 (0.0 ~ 1.0)
    """
    # 마스크를 쓴 얼굴은 보통 0.25~0.35 사이의 유사도를 보임
    if similarity < 0.25:
        return 0.9  # 매우 높은 가능성
    elif similarity < 0.28:
        return 0.7  # 높은 가능성
    elif similarity < 0.32:
        return 0.5  # 중간 가능성
    elif similarity < 0.35:
        return 0.3  # 낮은 가능성
    else:
        return 0.0  # 마스크 아님

def estimate_face_quality(face_bbox: Tuple[int, int, int, int], img_shape: Tuple[int, int]) -> str:
    """
    얼굴 크기와 이미지 크기를 기반으로 화질 추정
    
    얼굴이 크고 이미지 대비 비율이 높으면 화질이 좋다고 판단합니다.
    
    Args:
        face_bbox: (x1, y1, x2, y2) 얼굴 bounding box
        img_shape: (height, width) 이미지 크기
    
    Returns:
        화질 등급: "high", "medium", "low"
    """
    x1, y1, x2, y2 = face_bbox
    face_width = x2 - x1
    face_height = y2 - y1
    face_area = face_width * face_height
    
    img_height, img_width = img_shape
    img_area = img_width * img_height
    
    # 얼굴이 이미지에서 차지하는 비율
    face_ratio = face_area / img_area if img_area > 0 else 0.0
    
    # 얼굴 크기 기준 (픽셀 단위)
    face_size = max(face_width, face_height)
    
    # 화질 판단 기준
    # 1. 얼굴 크기가 150픽셀 이상이고 비율이 0.05 이상 → 고화질
    # 2. 얼굴 크기가 100픽셀 이상이고 비율이 0.02 이상 → 중화질
    # 3. 그 외 → 저화질
    
    if face_size >= 150 and face_ratio >= 0.05:
        return "high"
    elif face_size >= 100 and face_ratio >= 0.02:
        return "medium"
    else:
        return "low"


def get_quality_adjusted_threshold(base_threshold: float, quality: str) -> float:
    """
    화질에 따라 기본 임계값 조정
    
    화질이 좋으면 더 높은 임계값을 사용하여 오탐을 줄이고,
    화질이 낮으면 낮은 임계값을 사용하여 인식률을 유지합니다.
    
    Args:
        base_threshold: 기본 임계값 (예: 0.32)
        quality: 화질 등급 ("high", "medium", "low")
    
    Returns:
        화질에 따라 조정된 임계값
    """
    if quality == "high":
        # 고화질: 임계값 상향 (오탐 방지)
        return base_threshold + 0.03  # 0.32 → 0.35
    elif quality == "medium":
        # 중화질: 기본 임계값 유지
        return base_threshold  # 0.32
    else:  # low
        # 저화질: 임계값 하향 (인식률 유지)
        return max(base_threshold - 0.02, 0.28)  # 0.32 → 0.30 (최소 0.28)


def get_adjusted_threshold(base_threshold: float, mask_probability: float, similarity: float, 
                           face_quality: str = "medium") -> float:
    """
    마스크 착용 가능성, 유사도, 화질에 따라 적응형 임계값 계산
    
    마스크를 쓴 얼굴은 유사도가 낮게 나오므로, 마스크 가능성이 높을 때
    더 낮은 임계값을 적용하여 인식 성능을 향상시킵니다.
    화질이 좋으면 더 높은 임계값을 사용하여 오탐을 줄입니다.
    
    Args:
        base_threshold: 기본 임계값 (예: 0.32)
        mask_probability: 마스크 착용 가능성 (0.0 ~ 1.0)
        similarity: 현재 최고 유사도
        face_quality: 화질 등급 ("high", "medium", "low")
    
    Returns:
        조정된 임계값
    """
    # 먼저 화질에 따라 기본 임계값 조정
    quality_adjusted = get_quality_adjusted_threshold(base_threshold, face_quality)
    
    if mask_probability < 0.3:
        # 마스크 가능성이 낮으면 화질 조정된 임계값 사용
        return quality_adjusted
    
    # 마스크 가능성이 높으면 더 낮은 임계값 적용 (인식률 개선을 위해 조정)
    # 유사도가 0.25~0.30 사이일 때 마스크로 판단하고 0.24~0.28 임계값 사용
    if similarity < 0.28:
        # 매우 낮은 유사도 → 마스크 가능성 매우 높음 → 임계값 낮춤
        adjusted_threshold = quality_adjusted - 0.06
    elif similarity < 0.32:
        # 낮은 유사도 → 마스크 가능성 높음 → 임계값 낮춤
        adjusted_threshold = quality_adjusted - 0.04
    else:
        # 중간 유사도 → 마스크 가능성 중간 → 약간 낮춤
        adjusted_threshold = quality_adjusted - 0.02
    
    # 최소값 보장 (너무 낮으면 오탐 증가)
    # 고화질일 때는 최소 0.28, 중화질/저화질일 때는 최소 0.22
    min_threshold = 0.28 if face_quality == "high" else 0.22
    return max(adjusted_threshold, min_threshold)

