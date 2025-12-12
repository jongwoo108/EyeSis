# backend/utils/mask_detector.py
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
    return False  # 기본값: 마스크 감지 안함

def estimate_mask_from_embedding(embedding: np.ndarray, gallery_embeddings: dict) -> Tuple[float, float]:
    """
    임베딩 유사도로 마스크 착용 가능성 추정
    
    Args:
        embedding: 현재 얼굴 임베딩
        gallery_embeddings: 갤러리 임베딩들
    
    Returns:
        (best_similarity, mask_probability) 튜플
    """
    best_sim = -1.0
    for ref_emb in gallery_embeddings.values():
        if ref_emb.ndim == 2:
            sims = ref_emb @ embedding
            sim = float(np.max(sims))
        else:
            sim = float(np.dot(ref_emb, embedding))
        
        if sim > best_sim:
            best_sim = sim
    
    if best_sim < 0.25:
        mask_prob = 0.9
    elif best_sim < 0.28:
        mask_prob = 0.7
    elif best_sim < 0.32:
        mask_prob = 0.5
    elif best_sim < 0.35:
        mask_prob = 0.3
    else:
        mask_prob = 0.0
    
    return best_sim, mask_prob

def estimate_mask_from_similarity(similarity: float) -> float:
    """
    실제 매칭 유사도로 마스크 착용 가능성 추정
    
    Args:
        similarity: 실제 매칭 결과의 유사도
    
    Returns:
        마스크 착용 가능성 (0.0 ~ 1.0)
    """
    if similarity < 0.25:
        return 0.9
    elif similarity < 0.28:
        return 0.7
    elif similarity < 0.32:
        return 0.5
    elif similarity < 0.35:
        return 0.3
    else:
        return 0.0

def estimate_face_quality(face_bbox: Tuple[int, int, int, int], img_shape: Tuple[int, int]) -> str:
    """
    얼굴 크기와 이미지 크기를 기반으로 화질 추정
    
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
    
    face_ratio = face_area / img_area if img_area > 0 else 0.0
    face_size = max(face_width, face_height)
    
    if face_size >= 150 and face_ratio >= 0.05:
        return "high"
    elif face_size >= 100 and face_ratio >= 0.02:
        return "medium"
    else:
        return "low"


def get_quality_adjusted_threshold(base_threshold: float, quality: str) -> float:
    """
    화질에 따라 기본 임계값 조정
    
    Args:
        base_threshold: 기본 임계값
        quality: 화질 등급
    
    Returns:
        화질에 따라 조정된 임계값
    """
    if quality == "high":
        return base_threshold + 0.03
    elif quality == "medium":
        return base_threshold
    else:
        return max(base_threshold - 0.02, 0.28)


def get_adjusted_threshold(base_threshold: float, mask_probability: float, similarity: float, 
                           face_quality: str = "medium") -> float:
    """
    마스크 착용 가능성, 유사도, 화질에 따라 적응형 임계값 계산
    
    Args:
        base_threshold: 기본 임계값
        mask_probability: 마스크 착용 가능성 (무시됨)
        similarity: 현재 최고 유사도 (무시됨)
        face_quality: 화질 등급
    
    Returns:
        조정된 임계값 (화질 기반만)
    """
    return get_quality_adjusted_threshold(base_threshold, face_quality)
