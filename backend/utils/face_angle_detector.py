# backend/utils/face_angle_detector.py
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
    nose_offset_x = nose[0] - eye_center_x
    
    # 눈 간격으로 정규화
    eye_distance = np.sqrt((right_eye[0] - left_eye[0])**2 + 
                           (right_eye[1] - left_eye[1])**2)
    
    if eye_distance < 1e-6:
        return "unknown", 0.0
    
    # 정규화된 오프셋 (-1 ~ 1)
    normalized_offset = nose_offset_x / eye_distance
    
    # yaw 각도 추정 (대략 -90 ~ 90도)
    yaw_angle = normalized_offset * 90.0
    
    # Pitch 각도 추정 (상하 회전)
    eye_to_mouth_distance = abs(mouth_center_y - eye_center_y)
    eye_to_nose_distance = abs(nose[1] - eye_center_y)
    
    if eye_to_mouth_distance < 1e-6:
        pitch_angle = 0.0
    else:
        nose_ratio = eye_to_nose_distance / eye_to_mouth_distance
        pitch_angle = (0.5 - nose_ratio) * 90.0
    
    # 각도 타입 분류
    if pitch_angle > 15:
        angle_type = "top"
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

def is_diverse_angle(collected_angles: list, new_angle: str) -> bool:
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
    
    # 프로필은 각각 최대 50개까지
    if new_angle == "left_profile":
        left_profile_count = collected_angles.count("left_profile")
        return left_profile_count < 50
    
    if new_angle == "right_profile":
        right_profile_count = collected_angles.count("right_profile")
        return right_profile_count < 50
    
    # 정면은 최대 50개까지
    if new_angle == "front":
        front_count = collected_angles.count("front")
        return front_count < 50
    
    # 측면(left, right)은 각각 최대 50개까지
    if new_angle == "left":
        left_count = collected_angles.count("left")
        return left_count < 50
    
    if new_angle == "right":
        right_count = collected_angles.count("right")
        return right_count < 50
    
    # Top은 최대 50개까지
    if new_angle == "top":
        top_count = collected_angles.count("top")
        return top_count < 50
    
    return True

def get_angle_priority(angle_type: str) -> int:
    """
    각도의 우선순위 반환 (낮을수록 우선)
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


def is_all_angles_collected(collected_angles: list) -> bool:
    """
    모든 필수 각도가 수집되었는지 확인
    
    Args:
        collected_angles: 이미 수집된 각도 리스트
    
    Returns:
        True면 모든 필수 각도 수집 완료
    """
    from collections import defaultdict
    
    required_angles = {
        "front": 1,
        "left": 1,
        "right": 1,
        "top": 1
    }
    
    angle_counts = defaultdict(int)
    for angle in collected_angles:
        angle_counts[angle] += 1
    
    for angle, min_count in required_angles.items():
        if angle_counts[angle] < min_count:
            return False
    
    return True


def check_face_occlusion(face, bbox: Optional[Tuple[int, int, int, int]] = None) -> bool:
    """
    랜드마크 기반으로 얼굴 가림(Occlusion) 여부 확인
    
    Args:
        face: InsightFace의 face 객체 (kps 속성 필요)
        bbox: (x1, y1, x2, y2) 얼굴 bounding box (선택적)
    
    Returns:
        True면 occlusion 없음, False면 occlusion 있음
    """
    if not hasattr(face, 'kps') or face.kps is None:
        return False
    
    kps = face.kps
    
    left_eye = kps[0]
    right_eye = kps[1]
    nose = kps[2]
    left_mouth = kps[3]
    right_mouth = kps[4]
    
    # bbox가 제공된 경우, 랜드마크가 bbox 내에 있는지 확인
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        for i, kp in enumerate(kps):
            kp_x, kp_y = kp[0], kp[1]
            if kp_x < x1 or kp_x > x2 or kp_y < y1 or kp_y > y2:
                return False
    
    # 눈 간격 검증
    eye_distance = np.sqrt((right_eye[0] - left_eye[0])**2 + 
                           (right_eye[1] - left_eye[1])**2)
    
    if eye_distance < 1e-6:
        return False
    
    # 코 위치 검증
    eye_center_x = (left_eye[0] + right_eye[0]) / 2
    eye_center_y = (left_eye[1] + right_eye[1]) / 2
    nose_offset_x = abs(nose[0] - eye_center_x)
    nose_offset_y = abs(nose[1] - eye_center_y)
    
    if nose_offset_x > eye_distance * 0.5 or nose_offset_y > eye_distance * 0.5:
        return False
    
    # 입 위치 검증
    mouth_center_y = (left_mouth[1] + right_mouth[1]) / 2
    
    if mouth_center_y < nose[1]:
        return False
    
    # 입 너비 검증
    mouth_width = np.sqrt((right_mouth[0] - left_mouth[0])**2 + 
                          (right_mouth[1] - left_mouth[1])**2)
    
    if mouth_width < eye_distance * 0.3:
        return False
    
    # 랜드마크 포인트 유효성 검증
    for kp in kps:
        if np.isnan(kp[0]) or np.isnan(kp[1]) or np.isinf(kp[0]) or np.isinf(kp[1]):
            return False
    
    return True
