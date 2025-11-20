# src/utils/face_angle_detector.py
"""
얼굴 각도 감지 유틸리티
InsightFace의 랜드마크(keypoints)를 사용하여 얼굴 각도를 추정합니다.
"""
import numpy as np
from typing import Tuple, Optional

def estimate_face_angle(face) -> Tuple[str, float]:
    """
    얼굴 각도를 랜드마크 기반으로 추정
    
    InsightFace의 face 객체는 kps 속성을 가지고 있으며,
    kps는 5개의 랜드마크 포인트를 포함합니다:
    - 왼쪽 눈, 오른쪽 눈, 코, 왼쪽 입꼬리, 오른쪽 입꼬리
    
    Args:
        face: InsightFace의 face 객체
    
    Returns:
        (angle_type, yaw_angle) 튜플
        - angle_type: "front", "left", "right", "left_profile", "right_profile"
        - yaw_angle: 대략적인 yaw 각도 (도 단위, -90 ~ 90)
    """
    if not hasattr(face, 'kps') or face.kps is None:
        return "unknown", 0.0
    
    kps = face.kps  # (5, 2) 형태: [x, y] 좌표
    
    # 랜드마크 포인트
    left_eye = kps[0]   # 왼쪽 눈
    right_eye = kps[1]  # 오른쪽 눈
    nose = kps[2]       # 코
    left_mouth = kps[3] # 왼쪽 입꼬리
    right_mouth = kps[4] # 오른쪽 입꼬리
    
    # 눈의 중심점
    eye_center_x = (left_eye[0] + right_eye[0]) / 2
    eye_center_y = (left_eye[1] + right_eye[1]) / 2
    
    # 코와 눈 중심의 수평 거리로 yaw 각도 추정
    # 코가 눈 중심보다 왼쪽에 있으면 → 얼굴이 오른쪽을 향함 (left)
    # 코가 눈 중심보다 오른쪽에 있으면 → 얼굴이 왼쪽을 향함 (right)
    nose_offset_x = nose[0] - eye_center_x
    
    # 눈 간격으로 정규화
    eye_distance = np.sqrt((right_eye[0] - left_eye[0])**2 + 
                           (right_eye[1] - left_eye[1])**2)
    
    if eye_distance < 1e-6:
        return "unknown", 0.0
    
    # 정규화된 오프셋 (-1 ~ 1)
    normalized_offset = nose_offset_x / eye_distance
    
    # yaw 각도 추정 (대략 -90 ~ 90도)
    # 정면일 때 normalized_offset ≈ 0
    # 왼쪽 프로필일 때 normalized_offset ≈ -1
    # 오른쪽 프로필일 때 normalized_offset ≈ 1
    yaw_angle = normalized_offset * 90.0
    
    # 각도 타입 분류
    if abs(yaw_angle) < 15:
        angle_type = "front"
    elif yaw_angle < -45:
        angle_type = "left_profile"
    elif yaw_angle > 45:
        angle_type = "right_profile"
    elif yaw_angle < 0:
        angle_type = "left"
    else:
        angle_type = "right"
    
    return angle_type, yaw_angle

def is_diverse_angle(collected_angles: list[str], new_angle: str) -> bool:
    """
    새로운 각도가 기존에 수집된 각도와 다른지 확인
    
    Args:
        collected_angles: 이미 수집된 각도 리스트
        new_angle: 새로운 각도
    
    Returns:
        True면 다양한 각도 (추가 가능), False면 중복
    """
    if not collected_angles:
        return True
    
    # 프로필은 항상 추가 (드물기 때문)
    if new_angle in ["left_profile", "right_profile"]:
        return True
    
    # 정면은 최대 2개까지만
    if new_angle == "front":
        front_count = collected_angles.count("front")
        return front_count < 2
    
    # 측면(left, right)은 각각 최대 3개까지
    if new_angle == "left":
        left_count = collected_angles.count("left")
        return left_count < 3
    
    if new_angle == "right":
        right_count = collected_angles.count("right")
        return right_count < 3
    
    return True

def get_angle_priority(angle_type: str) -> int:
    """
    각도의 우선순위 반환 (낮을수록 우선)
    프로필 > 측면 > 정면 순으로 우선순위 높음
    """
    priority_map = {
        "left_profile": 1,
        "right_profile": 1,
        "left": 2,
        "right": 2,
        "front": 3,
        "unknown": 4
    }
    return priority_map.get(angle_type, 4)








