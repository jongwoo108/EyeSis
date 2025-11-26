# src/utils/face_angle_detector.py
"""
얼굴 각도 감지 유틸리티
InsightFace의 랜드마크(keypoints)를 사용하여 얼굴 각도를 추정합니다.
"""
import numpy as np
from typing import Tuple, Optional

def estimate_face_angle(face) -> Tuple[str, float]:
    """
    얼굴 각도를 랜드마크 기반으로 추정 (yaw + pitch)
    
    InsightFace의 face 객체는 kps 속성을 가지고 있으며,
    kps는 5개의 랜드마크 포인트를 포함합니다:
    - 왼쪽 눈, 오른쪽 눈, 코, 왼쪽 입꼬리, 오른쪽 입꼬리
    
    Args:
        face: InsightFace의 face 객체
    
    Returns:
        (angle_type, yaw_angle) 튜플
        - angle_type: "front", "left", "right", "top", "left_profile", "right_profile"
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
    
    # 입의 중심점
    mouth_center_y = (left_mouth[1] + right_mouth[1]) / 2
    
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
    
    # Pitch 각도 추정 (상하 회전)
    # 눈-입 거리 계산
    eye_to_mouth_distance = abs(mouth_center_y - eye_center_y)
    eye_to_nose_distance = abs(nose[1] - eye_center_y)
    
    if eye_to_mouth_distance < 1e-6:
        pitch_angle = 0.0
    else:
        # 코의 상대적 위치로 pitch 계산
        # 정면: 코가 눈과 입의 중간 정도 (비율 약 0.4~0.6)
        # Top (위를 향함): 코가 눈에 가까움 (비율 < 0.3), 입이 눈보다 훨씬 아래
        # Bottom (아래를 향함): 코가 입에 가까움 (비율 > 0.7)
        nose_ratio = eye_to_nose_distance / eye_to_mouth_distance
        
        # Pitch 각도 추정 (대략 -90 ~ 90도)
        # 정면: nose_ratio ≈ 0.5 → pitch ≈ 0
        # Top: nose_ratio < 0.3 → pitch > 0 (양수)
        # Bottom: nose_ratio > 0.7 → pitch < 0 (음수)
        pitch_angle = (0.5 - nose_ratio) * 90.0
    
    # 각도 타입 분류 (pitch 우선, 그 다음 yaw)
    # Top 각도 체크 (pitch > 15도)
    if pitch_angle > 15:
        angle_type = "top"
    # Yaw 각도 체크 (pitch가 작을 때만)
    elif abs(yaw_angle) < 15:
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
    
    # Top은 최대 2개까지만
    if new_angle == "top":
        top_count = collected_angles.count("top")
        return top_count < 2
    
    return True

def get_angle_priority(angle_type: str) -> int:
    """
    각도의 우선순위 반환 (낮을수록 우선)
    프로필 > 측면 > 정면/상하 순으로 우선순위 높음
    """
    priority_map = {
        "left_profile": 1,
        "right_profile": 1,
        "left": 2,
        "right": 2,
        "top": 3,
        "front": 3,
        "unknown": 4
    }
    return priority_map.get(angle_type, 4)


def is_all_angles_collected(collected_angles: list[str]) -> bool:
    """
    모든 필수 각도가 수집되었는지 확인
    
    필수 각도 (정답 데이터 기준):
    - front: 최소 1개
    - left: 최소 1개
    - right: 최소 1개
    - top: 최소 1개
    
    Args:
        collected_angles: 이미 수집된 각도 리스트
    
    Returns:
        True면 모든 필수 각도 수집 완료
    """
    from collections import defaultdict
    
    required_angles = {
        "front": 1,  # 최소 1개
        "left": 1,   # 최소 1개
        "right": 1,  # 최소 1개
        "top": 1     # 최소 1개
    }
    
    angle_counts = defaultdict(int)
    for angle in collected_angles:
        angle_counts[angle] += 1
    
    # 필수 각도 모두 수집되었는지 확인
    for angle, min_count in required_angles.items():
        if angle_counts[angle] < min_count:
            return False
    
    return True











