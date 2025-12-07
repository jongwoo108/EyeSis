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