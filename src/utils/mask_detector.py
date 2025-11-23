# src/utils/mask_detector.py
"""
마스크 감지 유틸리티
얼굴의 하반부(코, 입 영역)가 가려져 있는지 간단히 판단합니다.

TODO: 향후 랜드마크 기반 occlusion 판단 설계 방향
==================================================
현재는 유사도 기반으로 마스크 가능성을 추정하지만, 이는 오인식의 원인이 될 수 있습니다.
향후 제대로 설계할 때는 다음과 같은 방향을 고려해야 합니다:

1. 랜드마크 기반 영역 분리
   - InsightFace에서 제공하는 얼굴 랜드마크(눈, 코, 입 등)를 활용
   - 상반부(눈 영역)와 하반부(코, 입 영역)를 분리하여 각각의 occlusion 비율 계산

2. occlusion 판단
   - 각 영역의 가려짐 정도를 픽셀 단위로 분석
   - 마스크: 하반부 occlusion 높음, 상반부 occlusion 낮음
   - 모자: 상반부 occlusion 높음, 하반부 occlusion 낮음
   - 안경: 눈 영역 일부 occlusion, 나머지 정상

3. 매칭 전략 결정
   - 상반부만 보이는 경우: 상반부(눈) 기반 매칭만 수행
   - 하반부만 보이는 경우: 하반부(코, 입) 기반 매칭만 수행
   - 보이는 영역이 너무 적으면(예: 30% 미만): 아예 unknown으로 처리
   - 일부 케이스에서만 threshold를 아주 제한적으로 조정

4. 현재 로직의 문제점
   - "유사도 낮음 → 마스크일지도? → threshold 내려!" 패턴은 오인식의 주요 원인
   - 유사도가 낮은 이유는 마스크 때문이 아니라 완전히 다른 사람일 수 있음
   - 따라서 현재는 mask_prob를 threshold 조정에 사용하지 않고, 메타데이터/로그용으로만 사용

5. 구현 시 참고사항
   - InsightFace의 face.kps (랜드마크 좌표) 활용
   - 각 랜드마크 영역의 가려짐 정도를 계산하는 함수 필요
   - occlusion 비율에 따른 매칭 전략 선택 로직 필요
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
    
    주의: 이 함수는 현재 threshold 조정에 사용되지 않습니다.
    메타데이터/로그용으로만 사용되며, 향후 랜드마크 기반 occlusion 판단으로 대체될 예정입니다.
    
    Args:
        similarity: 실제 매칭 결과의 유사도
    
    Returns:
        마스크 착용 가능성 (0.0 ~ 1.0) - 메타데이터/로그용
    """
    # 마스크를 쓴 얼굴은 보통 0.25~0.35 사이의 유사도를 보임
    # 하지만 유사도가 낮은 이유가 마스크 때문이 아니라 완전히 다른 사람일 수도 있음
    # 따라서 이 값은 threshold 조정에 사용하지 않고, 단지 메타데이터로만 사용
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
    
    ⚠️ 주의: 이 함수는 현재 사용되지 않습니다.
    마스크 기반 threshold 조정 로직이 제거되었으며, 향후 랜드마크 기반 occlusion 판단으로 대체될 예정입니다.
    
    현재는 화질만으로 threshold를 결정하며, mask_probability는 무시됩니다.
    이 함수는 호환성을 위해 유지되지만, 실제로는 get_quality_adjusted_threshold()를 사용합니다.
    
    Args:
        base_threshold: 기본 임계값 (예: 0.32) - 사용되지 않음
        mask_probability: 마스크 착용 가능성 (0.0 ~ 1.0) - 무시됨
        similarity: 현재 최고 유사도 - 무시됨
        face_quality: 화질 등급 ("high", "medium", "low") - 이것만 사용
    
    Returns:
        조정된 임계값 (화질 기반만)
    """
    # 마스크 기반 threshold 조정 로직 제거
    # 화질만으로 threshold 결정
    return get_quality_adjusted_threshold(base_threshold, face_quality)

