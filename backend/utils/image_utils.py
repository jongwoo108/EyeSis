# backend/utils/image_utils.py
"""
이미지 처리 관련 유틸리티 함수들
"""

import base64
import cv2
import numpy as np
from typing import Optional

def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """벡터를 L2 정규화"""
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

def compute_cosine_similarity(embed1: np.ndarray, embed2: np.ndarray) -> float:
    """두 임베딩 벡터 간의 코사인 유사도 계산"""
    norm1 = np.linalg.norm(embed1)
    norm2 = np.linalg.norm(embed2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(embed1, embed2) / (norm1 * norm2))



def preprocess_image_for_detection(image: np.ndarray, min_size: int = 640) -> np.ndarray:
    """
    저화질 영상 처리를 위한 이미지 전처리
    
    Args:
        image: 입력 이미지 (BGR)
        min_size: 최소 크기 (이보다 작으면 업스케일링)
    
    Returns:
        전처리된 이미지
    """
    height, width = image.shape[:2]
    min_dimension = min(height, width)
    
    # 저화질 이미지 감지 및 업스케일링
    if min_dimension < min_size:
        # 업스케일링 비율 계산 (최소 크기 이상으로)
        scale_factor = min_size / min_dimension
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # 고품질 업스케일링 (INTER_LANCZOS4 사용)
        upscaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        # 샤프닝 필터 적용 (선명도 향상)
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]]) * 0.5
        sharpened = cv2.filter2D(upscaled, -1, kernel)
        
        # 약간의 대비 향상
        lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    return image

def base64_to_image(base64_string: str) -> Optional[np.ndarray]:
    """Base64 문자열을 OpenCV 이미지로 변환"""
    try:
        if "base64," in base64_string:
            base64_string = base64_string.split("base64,")[1]
        image_bytes = base64.b64decode(base64_string)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        print(f"⚠️ 이미지 디코딩 오류: {e}")
        return None

def image_to_base64(image: np.ndarray) -> str:
    """OpenCV 이미지를 Base64 문자열로 변환"""
    _, buffer = cv2.imencode('.jpg', image)
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')


# ==========================================
# 얼굴 각도 및 품질 추정 함수들
# ==========================================

def estimate_face_angle(face) -> tuple:
    """
    얼굴 각도를 랜드마크 기반으로 추정 (yaw + pitch)
    
    Args:
        face: InsightFace의 face 객체
    
    Returns:
        (angle_type, yaw_angle) 튜플
    """
    if not hasattr(face, 'kps') or face.kps is None:
        return "unknown", 0.0
    
    kps = face.kps  # (5, 2) 형태: [x, y] 좌표
    
    left_eye = kps[0]
    right_eye = kps[1]
    nose = kps[2]
    left_mouth = kps[3]
    right_mouth = kps[4]
    
    eye_center_x = (left_eye[0] + right_eye[0]) / 2
    eye_center_y = (left_eye[1] + right_eye[1]) / 2
    mouth_center_y = (left_mouth[1] + right_mouth[1]) / 2
    
    nose_offset_x = nose[0] - eye_center_x
    eye_distance = np.sqrt((right_eye[0] - left_eye[0])**2 + (right_eye[1] - left_eye[1])**2)
    
    if eye_distance < 1e-6:
        return "unknown", 0.0
    
    normalized_offset = nose_offset_x / eye_distance
    yaw_angle = normalized_offset * 90.0
    
    eye_to_mouth_distance = abs(mouth_center_y - eye_center_y)
    eye_to_nose_distance = abs(nose[1] - eye_center_y)
    
    if eye_to_mouth_distance < 1e-6:
        pitch_angle = 0.0
    else:
        nose_ratio = eye_to_nose_distance / eye_to_mouth_distance
        pitch_angle = (0.5 - nose_ratio) * 90.0
    
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


def check_face_occlusion(face, bbox: Optional[tuple] = None) -> bool:
    """
    랜드마크 기반으로 얼굴 가림(Occlusion) 여부 확인
    
    Args:
        face: InsightFace의 face 객체
        bbox: (x1, y1, x2, y2) 얼굴 bounding box
    
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
    
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        for kp in kps:
            kp_x, kp_y = kp[0], kp[1]
            if kp_x < x1 or kp_x > x2 or kp_y < y1 or kp_y > y2:
                return False
    
    eye_distance = np.sqrt((right_eye[0] - left_eye[0])**2 + (right_eye[1] - left_eye[1])**2)
    if eye_distance < 1e-6:
        return False
    
    eye_center_x = (left_eye[0] + right_eye[0]) / 2
    eye_center_y = (left_eye[1] + right_eye[1]) / 2
    nose_offset_x = abs(nose[0] - eye_center_x)
    nose_offset_y = abs(nose[1] - eye_center_y)
    
    if nose_offset_x > eye_distance * 0.5 or nose_offset_y > eye_distance * 0.5:
        return False
    
    mouth_center_x = (left_mouth[0] + right_mouth[0]) / 2
    mouth_center_y = (left_mouth[1] + right_mouth[1]) / 2
    
    if mouth_center_y < nose[1]:
        return False
    
    mouth_width = np.sqrt((right_mouth[0] - left_mouth[0])**2 + (right_mouth[1] - left_mouth[1])**2)
    if mouth_width < eye_distance * 0.3:
        return False
    
    for kp in kps:
        if np.isnan(kp[0]) or np.isnan(kp[1]) or np.isinf(kp[0]) or np.isinf(kp[1]):
            return False
    
    return True


def estimate_face_quality(face_bbox: tuple, img_shape: tuple) -> str:
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



def is_diverse_angle(collected_angles, new_angle):
    """Check if new angle is diverse from collected angles"""
    if not collected_angles:
        return True
    max_counts = {
        "left_profile": 50,
        "right_profile": 50,
        "front": 50,
        "left": 50,
        "right": 50,
        "top": 50
    }
    return collected_angles.count(new_angle) < max_counts.get(new_angle, 50)


def is_all_angles_collected(collected_angles):
    """Check if all required angles have been collected"""
    from collections import defaultdict
    required = {"front": 1, "left": 1, "right": 1, "top": 1}
    counts = defaultdict(int)
    for a in collected_angles:
        counts[a] += 1
    return all(counts[a] >= required[a] for a in required)