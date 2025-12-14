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
    elif abs(yaw_angle) < 10:
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
    
    # 프로필은 각각 최대 5개까지
    if new_angle == "left_profile":
        left_profile_count = collected_angles.count("left_profile")
        return left_profile_count < 5
    
    if new_angle == "right_profile":
        right_profile_count = collected_angles.count("right_profile")
        return right_profile_count < 5
    
    # 정면은 최대 2개까지만
    if new_angle == "front":
        front_count = collected_angles.count("front")
        return front_count < 2
    
    # 측면(left, right)은 각각 최대 3개까지
    if new_angle == "left":
        left_count = collected_angles.count("left")
        return left_count < 5
    
    if new_angle == "right":
        right_count = collected_angles.count("right")
        return right_count < 5
    
    # Top은 최대 2개까지만
    if new_angle == "top":
        top_count = collected_angles.count("top")
        return top_count < 5
    
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


def check_face_occlusion(face, bbox: Optional[Tuple[int, int, int, int]] = None) -> bool:
    """
    랜드마크 기반으로 얼굴 가림(Occlusion) 여부 확인
    
    주요 랜드마크(눈, 코, 입)가 모두 선명하게 보이는지 확인합니다.
    마스크나 손으로 가린 상태의 얼굴은 Dynamic Bank에 저장하지 않기 위해 사용됩니다.
    
    Args:
        face: InsightFace의 face 객체 (kps 속성 필요)
        bbox: (x1, y1, x2, y2) 얼굴 bounding box (선택적, 랜드마크 유효성 검증용)
    
    Returns:
        True면 occlusion 없음 (모든 랜드마크가 선명함), False면 occlusion 있음
    """
    if not hasattr(face, 'kps') or face.kps is None:
        # 랜드마크가 없으면 occlusion 상태로 간주
        return False
    
    kps = face.kps  # (5, 2) 형태: [x, y] 좌표
    
    # 랜드마크 포인트
    left_eye = kps[0]   # 왼쪽 눈
    right_eye = kps[1]  # 오른쪽 눈
    nose = kps[2]       # 코
    left_mouth = kps[3] # 왼쪽 입꼬리
    right_mouth = kps[4] # 오른쪽 입꼬리
    
    # bbox가 제공된 경우, 랜드마크가 bbox 내에 있는지 확인
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        for i, kp in enumerate(kps):
            kp_x, kp_y = kp[0], kp[1]
            # 랜드마크가 bbox 밖에 있으면 occlusion으로 간주
            if kp_x < x1 or kp_x > x2 or kp_y < y1 or kp_y > y2:
                return False
    
    # 1. 눈 간격 검증: 두 눈이 너무 가까우면(occlusion) 또는 너무 멀면(비정상) 문제
    eye_distance = np.sqrt((right_eye[0] - left_eye[0])**2 + 
                           (right_eye[1] - left_eye[1])**2)
    
    # 눈 간격이 너무 작으면(0.1 이하) occlusion으로 간주
    # (정상적인 얼굴에서는 눈 간격이 최소한 얼굴 너비의 20% 이상이어야 함)
    if eye_distance < 1e-6:
        return False
    
    # 2. 코 위치 검증: 코가 눈 중심과 너무 멀리 떨어져 있으면 occlusion 가능성
    eye_center_x = (left_eye[0] + right_eye[0]) / 2
    eye_center_y = (left_eye[1] + right_eye[1]) / 2
    nose_offset_x = abs(nose[0] - eye_center_x)
    nose_offset_y = abs(nose[1] - eye_center_y)
    
    # 코가 눈 중심에서 너무 멀리 떨어져 있으면(눈 간격의 50% 이상) occlusion 가능성
    if nose_offset_x > eye_distance * 0.5 or nose_offset_y > eye_distance * 0.5:
        return False
    
    # 3. 입 위치 검증: 입이 눈과 코 사이의 정상적인 위치에 있는지 확인
    mouth_center_x = (left_mouth[0] + right_mouth[0]) / 2
    mouth_center_y = (left_mouth[1] + right_mouth[1]) / 2
    
    # 입이 코보다 위에 있으면(비정상) occlusion 가능성
    if mouth_center_y < nose[1]:
        return False
    
    # 4. 입 너비 검증: 입이 너무 좁으면(occlusion) 문제
    mouth_width = np.sqrt((right_mouth[0] - left_mouth[0])**2 + 
                          (right_mouth[1] - left_mouth[1])**2)
    
    # 입 너비가 눈 간격의 30% 미만이면 occlusion 가능성
    if mouth_width < eye_distance * 0.3:
        return False
    
    # 5. 랜드마크 포인트 유효성 검증: 모든 포인트가 유효한 좌표를 가지고 있는지
    for kp in kps:
        if np.isnan(kp[0]) or np.isnan(kp[1]) or np.isinf(kp[0]) or np.isinf(kp[1]):
            return False
    
    # 모든 검증 통과: occlusion 없음
    return True











