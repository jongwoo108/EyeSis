"""
FaceWatch FastAPI 백엔드 서버
웹 프론트엔드와 연동하여 실시간 얼굴 인식 서비스 제공
PostgreSQL 데이터베이스 사용
"""
import base64
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Set
from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
import json
import asyncio
import subprocess
import tempfile
import os

import sys

# 프로젝트 루트를 Python 경로에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# CUDA 경로를 먼저 설정 (가장 먼저 import)
from src.utils.device_config import _ensure_cuda_in_path
_ensure_cuda_in_path()

from insightface.app import FaceAnalysis
from src.utils.device_config import get_device_id, safe_prepare_insightface
from src.utils.gallery_loader import load_gallery, match_with_bank, match_with_bank_detailed
from src.utils.face_angle_detector import estimate_face_angle
from src.utils.mask_detector import estimate_mask_from_similarity, get_adjusted_threshold, estimate_face_quality

# PostgreSQL 데이터베이스 모듈
from backend.database import (
    get_db, get_all_persons, get_person_by_id,
    log_detection, init_db as db_init, Person
)

# ==========================================
# 1. 설정 및 경로
# ==========================================

EMBEDDINGS_DIR = PROJECT_ROOT / "outputs" / "embeddings"

# Masked Bank 관련 설정
MASKED_BANK_MASK_PROB_THRESHOLD = 0.5  # mask_prob >= 0.5이면 masked bank로 분류 (완화: 0.7 → 0.5)
MASKED_CANDIDATE_MIN_SIM = 0.25  # base_sim >= 0.25 이상이어야 masked candidate로 판단 (완화: 0.30 → 0.25)
MASKED_CANDIDATE_MIN_FRAMES = 3  # 연속 N 프레임 이상 조건 충족 시 masked bank에 추가 (완화: 5 → 3)
MASKED_TRACKING_IOU_THRESHOLD = 0.5  # bbox tracking을 위한 IoU 임계값

# ==========================================
# 2. FastAPI 앱 초기화
# ==========================================

app = FastAPI(title="FaceWatch API", version="1.0.0")

# CORS 허용 (프론트엔드 접근 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 3. InsightFace 모델 초기화 (device_config 사용)
# ==========================================

print("=" * 70)
print("🔧 InsightFace 모델 초기화 중...")
print("=" * 70)

device_id = get_device_id()
device_type = "GPU" if device_id >= 0 else "CPU"
print(f"디바이스: {device_type} (ctx_id={device_id})")

model = FaceAnalysis(name="buffalo_l")
actual_device_id = safe_prepare_insightface(model, device_id, det_size=(640, 640))
if actual_device_id != device_id:
    print(f"   (실제 사용: {'GPU' if actual_device_id >= 0 else 'CPU'})")
print()

# ==========================================
# 4. 데이터베이스 초기화 및 캐시
# ==========================================

# 메모리 캐시 (성능 향상을 위해)
persons_cache: List[Dict] = []
# base(마스크 없음) / masked(마스크 얼굴)를 분리해서 관리
gallery_base_cache: Dict[str, np.ndarray] = {}  # base bank (정면, 측면, 마스크 없는 얼굴)
gallery_masked_cache: Dict[str, np.ndarray] = {}  # masked bank (마스크 쓴 얼굴)

def load_persons_from_db(db: Session):
    """PostgreSQL에서 인물 정보 로드 및 캐시 (Bank 데이터 포함 - base/masked 분리)"""
    global persons_cache, gallery_base_cache, gallery_masked_cache
    
    persons = get_all_persons(db)
    
    persons_cache = []
    gallery_base_cache = {}
    gallery_masked_cache = {}
    
    for person in persons:
        person_id = person.person_id
        
        # outputs/embeddings 폴더에서 Bank 데이터 확인
        person_dir = EMBEDDINGS_DIR / person_id
        base_bank_path = person_dir / "bank_base.npy"
        masked_bank_path = person_dir / "bank_masked.npy"
        centroid_path = person_dir / "centroid.npy"
        
        # Backward compatibility: 기존 bank.npy, centroid.npy
        legacy_bank_path = person_dir / "bank.npy"
        legacy_centroid_path = person_dir / "centroid.npy"
        
        base_bank = None
        masked_bank = None
        
        # ===== Base Bank 로딩 (우선순위 순) =====
        # 1. bank_base.npy (새 구조)
        if base_bank_path.exists():
            try:
                base_bank = np.load(base_bank_path)
                if base_bank.ndim == 1:
                    base_bank = base_bank.reshape(1, -1)
                # L2 정규화
                base_bank = base_bank / (np.linalg.norm(base_bank, axis=1, keepdims=True) + 1e-6)
            except Exception as e:
                print(f"  ⚠️ Base Bank 로드 실패 ({person_id}): {e}")
                base_bank = None
        
        # 2. 기존 bank.npy (backward compatibility - read-only로 사용)
        if base_bank is None and legacy_bank_path.exists():
            try:
                base_bank = np.load(legacy_bank_path)
                if base_bank.ndim == 1:
                    base_bank = base_bank.reshape(1, -1)
                base_bank = base_bank / (np.linalg.norm(base_bank, axis=1, keepdims=True) + 1e-6)
            except Exception as e:
                print(f"  ⚠️ Legacy Bank 로드 실패 ({person_id}): {e}")
                base_bank = None
        
        # 3. 기존 centroid.npy (backward compatibility)
        if base_bank is None and legacy_centroid_path.exists():
            try:
                centroid_data = np.load(legacy_centroid_path)
                centroid_data = l2_normalize(centroid_data)
                base_bank = centroid_data.reshape(1, -1)
            except Exception as e:
                print(f"  ⚠️ Legacy Centroid 로드 실패 ({person_id}): {e}")
                base_bank = None
        
        # 4. DB 임베딩 사용
        if base_bank is None:
            try:
                db_embedding = person.get_embedding()
                db_embedding = l2_normalize(db_embedding)
                base_bank = db_embedding.reshape(1, -1)
            except Exception as e:
                print(f"  ⚠️ DB 임베딩 로드 실패 ({person_id}): {e}")
                base_bank = None
        
        # Base가 없으면 스킵
        if base_bank is None:
            print(f"  ❌ Base Bank를 찾을 수 없음: {person.name} (ID: {person_id}), 스킵")
            continue
        
        # ===== Masked Bank 로딩 =====
        if masked_bank_path.exists():
            try:
                masked_bank = np.load(masked_bank_path)
                if masked_bank.ndim == 1:
                    masked_bank = masked_bank.reshape(1, -1)
                if masked_bank.shape[0] > 0:
                    # L2 정규화
                    masked_bank = masked_bank / (np.linalg.norm(masked_bank, axis=1, keepdims=True) + 1e-6)
                else:
                    masked_bank = None
            except Exception as e:
                print(f"  ⚠️ Masked Bank 로드 실패 ({person_id}): {e}")
                masked_bank = None
        else:
            # Masked Bank가 없으면 None (빈 상태)
            masked_bank = None
        
        # gallery_base_cache와 gallery_masked_cache에 저장
        gallery_base_cache[person_id] = base_bank
        if masked_bank is not None:
            gallery_masked_cache[person_id] = masked_bank
        
        # persons_cache에는 base의 첫 번째 임베딩 사용 (표시용)
        first_embedding = base_bank[0] if base_bank.ndim == 2 else base_bank.flatten()
        
        person_data = {
            "id": person_id,
            "name": person.name,
            "is_criminal": person.is_criminal,
            "info": person.info or {},
            "embedding": first_embedding
        }
        persons_cache.append(person_data)
        
        # 로드 결과 출력
        masked_count = masked_bank.shape[0] if masked_bank is not None else 0
        masked_file_path = str(masked_bank_path.relative_to(PROJECT_ROOT)) if masked_bank_path.exists() else "없음"
        print(f"  ✅ Bank 로드: {person.name} (ID: {person_id}, base: {base_bank.shape[0]}개, masked: {masked_count}개) [masked 파일: {masked_file_path}]")
    
    print(f"📂 데이터베이스 로딩 완료 ({len(persons_cache)}명, Base/Masked Bank 분리 구조)\n")

def load_persons_from_embeddings():
    """outputs/embeddings에서 gallery 로드 (fallback - base/masked 분리 구조)"""
    global gallery_base_cache, gallery_masked_cache, persons_cache
    
    if not EMBEDDINGS_DIR.exists():
        print(f"⚠️ embeddings 폴더를 찾을 수 없습니다: {EMBEDDINGS_DIR}")
        return
    
    try:
        gallery_base_cache = {}
        gallery_masked_cache = {}
        persons_cache = []
        
        # 사람별 폴더 구조 확인
        person_dirs = [d for d in EMBEDDINGS_DIR.iterdir() if d.is_dir()]
        
        for person_dir in person_dirs:
            person_id = person_dir.name
            
            base_bank_path = person_dir / "bank_base.npy"
            masked_bank_path = person_dir / "bank_masked.npy"
            legacy_bank_path = person_dir / "bank.npy"
            legacy_centroid_path = person_dir / "centroid.npy"
            
            base_bank = None
            masked_bank = None
            
            # Base Bank 로딩
            if base_bank_path.exists():
                try:
                    base_bank = np.load(base_bank_path)
                    if base_bank.ndim == 1:
                        base_bank = base_bank.reshape(1, -1)
                    base_bank = base_bank / (np.linalg.norm(base_bank, axis=1, keepdims=True) + 1e-6)
                except Exception as e:
                    print(f"  ⚠️ Base Bank 로드 실패 ({person_id}): {e}")
                    base_bank = None
            
            # Backward compatibility: 기존 bank.npy
            if base_bank is None and legacy_bank_path.exists():
                try:
                    base_bank = np.load(legacy_bank_path)
                    if base_bank.ndim == 1:
                        base_bank = base_bank.reshape(1, -1)
                    base_bank = base_bank / (np.linalg.norm(base_bank, axis=1, keepdims=True) + 1e-6)
                    print(f"  ⚠️ Legacy Bank를 Base로 사용: {person_id}")
                except Exception as e:
                    print(f"  ⚠️ Legacy Bank 로드 실패 ({person_id}): {e}")
                    base_bank = None
            
            # Backward compatibility: 기존 centroid.npy
            if base_bank is None and legacy_centroid_path.exists():
                try:
                    centroid_data = np.load(legacy_centroid_path)
                    centroid_data = l2_normalize(centroid_data)
                    base_bank = centroid_data.reshape(1, -1)
                    print(f"  ⚠️ Legacy Centroid를 Base로 사용: {person_id}")
                except Exception as e:
                    print(f"  ⚠️ Legacy Centroid 로드 실패 ({person_id}): {e}")
                    base_bank = None
            
            if base_bank is None:
                continue  # Base가 없으면 스킵
            
            # Masked Bank 로딩
            if masked_bank_path.exists():
                try:
                    masked_bank = np.load(masked_bank_path)
                    if masked_bank.ndim == 1:
                        masked_bank = masked_bank.reshape(1, -1)
                    if masked_bank.shape[0] > 0:
                        masked_bank = masked_bank / (np.linalg.norm(masked_bank, axis=1, keepdims=True) + 1e-6)
                    else:
                        masked_bank = None
                except Exception as e:
                    print(f"  ⚠️ Masked Bank 로드 실패 ({person_id}): {e}")
                    masked_bank = None
            
            # gallery_base_cache와 gallery_masked_cache에 저장
            gallery_base_cache[person_id] = base_bank
            if masked_bank is not None:
                gallery_masked_cache[person_id] = masked_bank
            
            # persons_cache에 추가
            first_embedding = base_bank[0] if base_bank.ndim == 2 else base_bank.flatten()
            persons_cache.append({
                "id": person_id,
                "name": person_id,  # 이름이 없으면 ID 사용
                "is_criminal": person_id == "criminal",
                "info": {},
                "embedding": first_embedding
            })
            masked_count = masked_bank.shape[0] if masked_bank is not None else 0
            print(f"  - {person_id} (base: {base_bank.shape[0]}개, masked: {masked_count}개)")
        
        print(f"📂 Gallery 로딩 완료 ({len(gallery_base_cache)}명, Base/Masked Bank 분리 구조)\n")
    except Exception as e:
        print(f"⚠️ Gallery 로딩 실패: {e}\n")
        import traceback
        traceback.print_exc()

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 데이터베이스 초기화 및 데이터 로드"""
    print("=" * 70)
    print("🚀 FaceWatch 서버 시작")
    print("=" * 70)
    print("📡 WebSocket 엔드포인트:")
    print("   - /ws/detect (메인 감지 엔드포인트)")
    print("   - /ws/test (테스트 엔드포인트)")
    print("=" * 70)
    
    # 1. 데이터베이스 테이블 생성 (없으면 생성)
    try:
        db_init()
    except Exception as e:
        print(f"⚠️ 데이터베이스 초기화 오류: {e}")
        print("   outputs/embeddings를 사용합니다.")
    
    # 2. PostgreSQL에서 데이터 로드 시도
    try:
        db = next(get_db())
        try:
            load_persons_from_db(db)
        finally:
            db.close()
    except Exception as e:
        print(f"⚠️ PostgreSQL 연결 실패: {e}")
        print("   outputs/embeddings를 사용합니다.")
        # Fallback: outputs/embeddings 사용
        load_persons_from_embeddings()
    
    # 3. 데이터가 없으면 경고
    if not gallery_base_cache and not persons_cache:
        print("⚠️ 경고: 등록된 얼굴 데이터가 없습니다!")
        print("   face_enroll.py를 실행하여 인물을 등록하거나,")
        print("   python backend/init_db.py를 실행하여 데이터를 마이그레이션해주세요.\n")

# ==========================================
# 5. 유틸리티 함수
# ==========================================

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
        # 얼굴 대각선 길이 기준으로 임계값 설정 (CCTV 환경의 bbox 떨림 고려)
        w1 = bbox1[2] - bbox1[0]
        h1 = bbox1[3] - bbox1[1]
        w2 = bbox2[2] - bbox2[0]
        h2 = bbox2[3] - bbox2[1]
        avg_w = (w1 + w2) / 2
        avg_h = (h1 + h2) / 2
        face_diag = (avg_w ** 2 + avg_h ** 2) ** 0.5
        distance_threshold = face_diag * 0.6  # 대각선의 60% 이내면 같은 얼굴로 간주
    
    distance = calculate_bbox_center_distance(bbox1, bbox2)
    if distance <= distance_threshold:
        return True
    
    return False

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

def find_person_info(person_id: str) -> Optional[Dict]:
    """person_id로 인물 정보 찾기"""
    for person in persons_cache:
        if person["id"] == person_id:
            return person
    return None

# ==========================================
# 6. WebSocket 연결 관리
# ==========================================

# 활성 WebSocket 연결 추적
active_connections: Set[WebSocket] = set()

# 연결별 상태 관리
connection_states: Dict[WebSocket, Dict] = {}

async def register_connection(websocket: WebSocket):
    """WebSocket 연결 등록"""
    try:
        await websocket.accept()
        active_connections.add(websocket)
        connection_states[websocket] = {
            "suspect_ids": [],  # 여러 명 선택 가능
            "connected_at": asyncio.get_event_loop().time()
        }
        print(f"✅ WebSocket 연결됨 (총 {len(active_connections)}개 연결)")
    except Exception as e:
        print(f"❌ WebSocket 연결 등록 실패: {e}")
        import traceback
        traceback.print_exc()
        raise

def unregister_connection(websocket: WebSocket):
    """WebSocket 연결 해제"""
    active_connections.discard(websocket)
    connection_states.pop(websocket, None)
    print(f"❌ WebSocket 연결 해제됨 (남은 연결: {len(active_connections)}개)")

# ==========================================
# 6.5. Bank 자동 추가 함수
# ==========================================

async def add_embedding_to_bank_async(person_id: str, embedding: np.ndarray, 
                                      angle_type: str = None, yaw_angle: float = None,
                                      bank_type: str = "base"):
    """
    Bank에 임베딩을 비동기로 추가 (파일 저장)
    
    주의: bank_base.npy는 절대 수정하지 않습니다. bank_masked.npy에만 추가합니다.
    base bank에는 마스크 없는 얼굴만, masked bank에는 마스크 쓴 얼굴만 저장합니다.
    
    Args:
        person_id: 인물 ID
        embedding: 추가할 임베딩 (512차원, L2 정규화됨)
        angle_type: 얼굴 각도 타입
        yaw_angle: yaw 각도 값
        bank_type: "base" 또는 "masked"
    
    Returns:
        추가 성공 여부 (True: 추가됨, False: 중복으로 스킵)
    """
    import json
    
    person_dir = EMBEDDINGS_DIR / person_id
    base_bank_path = person_dir / "bank_base.npy"
    masked_bank_path = person_dir / "bank_masked.npy"
    
    # bank_type에 따라 파일 경로 결정
    if bank_type == "masked":
        target_bank_path = masked_bank_path
        angles_path = person_dir / "angles_masked.json"  # masked용 각도 정보
    else:  # base
        # base bank는 자동 학습으로 추가하지 않음 (read-only)
        # 하지만 호환성을 위해 함수는 동작하도록 함
        target_bank_path = base_bank_path
        angles_path = person_dir / "angles_base.json"
    
    # Backward compatibility: 기존 bank.npy를 base로 사용
    legacy_bank_path = person_dir / "bank.npy"
    
    # Base Bank 로드 (중복 체크용, read-only)
    base_bank = None
    if base_bank_path.exists():
        try:
            base_bank = np.load(base_bank_path)
            if base_bank.ndim == 1:
                base_bank = base_bank.reshape(1, -1)
        except Exception as e:
            print(f"  ⚠️ Base Bank 로드 실패 ({person_id}): {e}")
            base_bank = None
    
    # Backward compatibility: 기존 bank.npy를 base로 사용
    if base_bank is None and legacy_bank_path.exists():
        try:
            base_bank = np.load(legacy_bank_path)
            if base_bank.ndim == 1:
                base_bank = base_bank.reshape(1, -1)
        except Exception as e:
            print(f"  ⚠️ Legacy Bank 로드 실패 ({person_id}): {e}")
            base_bank = None
    
    # Masked Bank 로드 (중복 체크용)
    masked_bank = None
    if masked_bank_path.exists():
        try:
            masked_bank = np.load(masked_bank_path)
            if masked_bank.ndim == 1:
                masked_bank = masked_bank.reshape(1, -1)
        except Exception as e:
            print(f"  ⚠️ Masked Bank 로드 실패 ({person_id}): {e}")
            masked_bank = None
    
    # Target Bank 로드 (추가할 bank)
    if target_bank_path.exists():
        try:
            target_bank = np.load(target_bank_path)
            if target_bank.ndim == 1:
                target_bank = target_bank.reshape(1, -1)
        except Exception as e:
            print(f"  ⚠️ Target Bank 로드 실패 ({person_id}): {e}")
            target_bank = np.empty((0, 512), dtype=np.float32)
    else:
        target_bank = np.empty((0, 512), dtype=np.float32)
    
    # 중복 체크: base + masked 전체를 대상으로
    BANK_DUPLICATE_THRESHOLD = 0.95
    all_bank_list = []
    if base_bank is not None:
        all_bank_list.append(base_bank)
    if masked_bank is not None and masked_bank.shape[0] > 0:
        all_bank_list.append(masked_bank)
    
    if all_bank_list:
        all_bank = np.vstack(all_bank_list)
        max_sim = float(np.max(all_bank @ embedding))
        if max_sim >= BANK_DUPLICATE_THRESHOLD:
            return False  # 중복으로 스킵
    
    # Target Bank에 추가
    new_emb = embedding.reshape(1, -1)
    updated_target_bank = np.vstack([target_bank, new_emb])
    
    # 각도 정보 로드 및 추가
    if angles_path.exists():
        try:
            with open(angles_path, 'r', encoding='utf-8') as f:
                angles_info = json.load(f)
        except Exception as e:
            print(f"  ⚠️ 각도 정보 로드 실패 ({person_id}): {e}")
            angles_info = {"angle_types": [], "yaw_angles": [], "bank_types": []}
    else:
        angles_info = {"angle_types": [], "yaw_angles": [], "bank_types": []}
    
    angles_info["angle_types"].append(angle_type if angle_type else "unknown")
    angles_info["yaw_angles"].append(float(yaw_angle) if yaw_angle is not None else 0.0)
    angles_info["bank_types"].append(bank_type)  # bank_type 정보 추가
    
    # 파일 저장 (비동기로 처리)
    person_dir.mkdir(parents=True, exist_ok=True)
    np.save(target_bank_path, updated_target_bank)
    
    with open(angles_path, 'w', encoding='utf-8') as f:
        json.dump(angles_info, f, indent=2, ensure_ascii=False)
    
    bank_name = "Masked" if bank_type == "masked" else "Base"
    file_path_str = str(target_bank_path.relative_to(PROJECT_ROOT)) if target_bank_path.exists() else str(target_bank_path)
    print(f"  ✅ [{bank_name} BANK] 파일 저장: {file_path_str} (총 {updated_target_bank.shape[0]}개 임베딩, angle: {angle_type})")
    
    return True

def update_gallery_cache_in_memory(person_id: str, embedding: np.ndarray, bank_type: str = "base"):
    """
    gallery_cache를 메모리에서 즉시 업데이트 (실시간 반영)
    
    주의: base bank는 절대 수정하지 않습니다. masked bank에만 추가합니다.
    base bank에는 마스크 없는 얼굴만, masked bank에는 마스크 쓴 얼굴만 저장합니다.
    
    Args:
        person_id: 인물 ID
        embedding: 추가할 임베딩 (512차원, L2 정규화됨)
        bank_type: "base" 또는 "masked"
    
    Returns:
        추가 성공 여부 (True: 추가됨, False: 중복으로 스킵)
    """
    global gallery_base_cache, gallery_masked_cache
    
    embedding = l2_normalize(embedding.astype("float32"))
    
    BANK_DUPLICATE_THRESHOLD = 0.95
    
    # Base Bank와 Masked Bank 모두 확인 (중복 체크용)
    base_bank = gallery_base_cache.get(person_id)
    masked_bank = gallery_masked_cache.get(person_id)
    
    # 중복 체크: base + masked 전체를 대상으로
    all_bank_list = []
    if base_bank is not None:
        all_bank_list.append(base_bank)
    if masked_bank is not None:
        all_bank_list.append(masked_bank)
    
    if all_bank_list:
        all_bank = np.vstack(all_bank_list)
        max_sim = float(np.max(all_bank @ embedding))
        if max_sim >= BANK_DUPLICATE_THRESHOLD:
            return False  # 중복으로 스킵
    
    # bank_type에 따라 적절한 캐시에 추가
    if bank_type == "masked":
        # Masked Bank에 추가
        if masked_bank is None:
            masked_bank = np.empty((0, 512), dtype=np.float32)
        
        new_emb = embedding.reshape(1, -1)
        updated_masked_bank = np.vstack([masked_bank, new_emb])
        gallery_masked_cache[person_id] = updated_masked_bank
    else:
        # Base Bank는 자동 학습으로 추가하지 않음 (read-only)
        # 하지만 호환성을 위해 함수는 동작하도록 함
        if base_bank is None:
            print(f"  ⚠️ Base Bank가 없는 상태에서 Base 추가 시도: {person_id}")
            base_bank = np.empty((0, 512), dtype=np.float32)
        
        new_emb = embedding.reshape(1, -1)
        updated_base_bank = np.vstack([base_bank, new_emb])
        gallery_base_cache[person_id] = updated_base_bank
    
    return True

# ==========================================
# 6.6. Temporal Consistency 필터 함수
# ==========================================

def apply_temporal_filter(websocket: WebSocket, result: Dict) -> Dict:
    """
    연속 프레임 기반 매칭 확정 로직 적용
    
    최소 3프레임 이상 연속으로 동일 인물 매칭이 나왔을 때만 "확정 match"로 처리합니다.
    그 이전까지는 "candidate match(후보)" 상태로 두고, status를 "unknown"으로 유지합니다.
    
    Args:
        websocket: WebSocket 연결 객체
        result: process_detection의 반환값
    
    Returns:
        temporal filter가 적용된 result
    """
    MIN_STABLE_FRAMES = 3  # 최소 연속 프레임 수
    
    if websocket not in connection_states:
        # 연결 상태가 없으면 그대로 반환
        return result
    
    state = connection_states[websocket]
    match_counters = state.get("match_counters", {})
    
    # 현재 프레임의 모든 person_id 수집
    current_person_ids = set()
    for det in result.get("detections", []):
        person_id = det.get("person_id")
        if person_id:
            current_person_ids.add(person_id)
    
    # 이전 프레임에서 매칭되었지만 현재 프레임에서 사라진 person_id는 카운터 초기화
    for person_id in list(match_counters.keys()):
        if person_id not in current_person_ids:
            del match_counters[person_id]
    
    # 각 detection에 대해 temporal filter 적용
    filtered_detections = []
    alert_triggered = False
    detected_metadata = result.get("metadata", {"name": "미상", "confidence": 0, "status": "unknown"})
    
    for det in result.get("detections", []):
        person_id = det.get("person_id")
        status = det.get("status", "unknown")
        
        # criminal 또는 normal 상태이고 person_id가 있는 경우만 temporal filter 적용
        if status in ["criminal", "normal"] and person_id:
            # 카운터 증가
            if person_id not in match_counters:
                match_counters[person_id] = 0
            match_counters[person_id] += 1
            
            # 최소 프레임 수에 도달하지 않았으면 unknown으로 변경
            if match_counters[person_id] < MIN_STABLE_FRAMES:
                # 후보 상태로 표시 (unknown)
                filtered_det = det.copy()
                filtered_det["status"] = "unknown"
                filtered_det["color"] = "yellow"
                filtered_det["name"] = "Unknown"
                filtered_detections.append(filtered_det)
            else:
                # 확정 매칭 - 원래 상태 유지
                filtered_detections.append(det)
                if status == "criminal":
                    alert_triggered = True
                    detected_metadata = {
                        "name": det.get("name", "Unknown"),
                        "confidence": det.get("confidence", 0),
                        "status": "criminal"
                    }
                elif not alert_triggered:
                    detected_metadata = {
                        "name": det.get("name", "Unknown"),
                        "confidence": det.get("confidence", 0),
                        "status": "normal"
                    }
        else:
            # unknown 상태는 그대로 유지
            filtered_detections.append(det)
    
    # match_counters 업데이트
    state["match_counters"] = match_counters
    
    return {
        "detections": filtered_detections,
        "alert": alert_triggered,
        "metadata": detected_metadata,
        "learning_events": result.get("learning_events", [])
    }

# ==========================================
# 7. 공통 감지 로직 함수
# ==========================================

def process_detection(frame: np.ndarray, suspect_id: Optional[str] = None, suspect_ids: Optional[List[str]] = None, db: Optional[Session] = None, tracking_state: Optional[Dict] = None) -> Dict:
    """
    공통 얼굴 감지 및 인식 로직
    
    Args:
        frame: BGR 이미지 (numpy array)
        suspect_id: 선택적 타겟 ID (단일, 호환성 유지)
        suspect_ids: 선택적 타겟 ID 배열 (여러 명 선택 시)
        db: 데이터베이스 세션 (로그 저장용, None이면 로그 저장 안함)
        tracking_state: bbox tracking 상태 (None이면 자동 생성)
    
    Returns:
        {
            "detections": [...],  # 박스 좌표 및 메타데이터 배열
            "alert": bool,        # 범죄자 감지 여부
            "metadata": {...}      # 주요 감지 정보
        }
    """
    # suspect_ids가 없으면 suspect_id를 배열로 변환
    if suspect_ids is None:
        suspect_ids = [suspect_id] if suspect_id else []
    
    # tracking_state 초기화 (없으면 생성)
    if tracking_state is None:
        tracking_state = {
            "tracks": {}  # {track_id: {"bbox": [...], "person_id": str, "frames": int, "embeddings": [...], "last_frame": int}}
        }
    
    # 1. 저화질 영상 전처리 (업스케일링 및 샤프닝)
    original_height, original_width = frame.shape[:2]
    processed_frame = preprocess_image_for_detection(frame, min_size=640)
    processed_height, processed_width = processed_frame.shape[:2]
    
    # 스케일 비율 계산 (박스 좌표 변환용)
    scale_x = original_width / processed_width
    scale_y = original_height / processed_height

    # 2. InsightFace로 얼굴 탐지 및 특징 추출 (전처리된 이미지 사용)
    faces = model.get(processed_frame)
    
    # 얼굴 감지 개수 로그 출력 (디버깅용)
    print(f"🔍 [얼굴 감지] 감지된 얼굴 개수: {len(faces)}")
    if suspect_ids:
        print(f"   - suspect_ids 모드: {suspect_ids}")
    else:
        print(f"   - 전체 갤러리 모드")
    
    alert_triggered = False
    detected_metadata = {"name": "미상", "confidence": 0, "status": "unknown"}
    detections = []  # 박스 좌표 및 메타데이터 배열
    learning_events = []  # 학습 이벤트 (UI 피드백용)

    # 3. 먼저 모든 얼굴에 대해 매칭 결과 수집 (오인식 방지 필터링을 위해)
    face_results = []
    for face in faces:
        # 바운딩 박스 좌표 (정수형 변환)
        # 전처리된 이미지의 좌표를 원본 이미지 좌표로 변환
        box = face.bbox.astype(float)
        box[0] *= scale_x  # x1
        box[1] *= scale_y  # y1
        box[2] *= scale_x  # x2
        box[3] *= scale_y  # y2
        box = box.astype(int)
        
        embedding = face.embedding.astype("float32")
        embedding_normalized = l2_normalize(embedding)
        
        # 얼굴 각도 추정
        angle_type, yaw_angle = estimate_face_angle(face)
        
        # 화질 추정
        face_quality = estimate_face_quality(box, (original_height, original_width))
        
        # Base Bank와 Masked Bank 각각 매칭 (분리 계산)
        base_sim = 0.0
        masked_sim = 0.0
        best_base_person_id = "unknown"
        best_mask_person_id = "unknown"
        second_base_sim = -1.0
        second_mask_sim = -1.0
        
        # suspect_ids가 지정된 경우: 선택된 용의자들만 검색 (전체 DB 검색 안 함)
        if suspect_ids:
            # 선택된 용의자들만 포함한 base/masked 갤러리 생성
            target_base_gallery = {}
            target_masked_gallery = {}
            for sid in suspect_ids:
                if sid in gallery_base_cache:
                    target_base_gallery[sid] = gallery_base_cache[sid]
                if sid in gallery_masked_cache:
                    target_masked_gallery[sid] = gallery_masked_cache[sid]
            
            # Base Bank 매칭
            if target_base_gallery:
                best_base_person_id, base_sim, second_base_sim = match_with_bank_detailed(embedding, target_base_gallery)
            
            # Masked Bank 매칭
            if target_masked_gallery:
                best_mask_person_id, masked_sim, second_mask_sim = match_with_bank_detailed(embedding, target_masked_gallery)
        
        # 전체 DB 검색 (suspect_ids가 없는 경우에만 수행)
        else:
            if gallery_base_cache:
                best_base_person_id, base_sim, second_base_sim = match_with_bank_detailed(embedding, gallery_base_cache)
            
            if gallery_masked_cache:
                best_mask_person_id, masked_sim, second_mask_sim = match_with_bank_detailed(embedding, gallery_masked_cache)
        
        # 두 결과 중 더 좋은 후보 선택 (best_sim)
        if base_sim > masked_sim:
            best_person_id = best_base_person_id
            max_similarity = base_sim
            second_similarity = second_base_sim if second_base_sim > 0 else 0.0
            bank_type = "base"
        else:
            best_person_id = best_mask_person_id
            max_similarity = masked_sim
            second_similarity = second_mask_sim if second_mask_sim > 0 else 0.0
            bank_type = "masked"
        
        # best_match 찾기
        if best_person_id != "unknown" and max_similarity > 0:
            best_match = find_person_info(best_person_id)
        else:
            # 직접 비교 (fallback)
            similarities = []
            for person in persons_cache:
                sim = compute_cosine_similarity(embedding, person["embedding"])
                similarities.append((sim, person))
            
            # 유사도 순으로 정렬
            similarities.sort(key=lambda x: x[0], reverse=True)
            if similarities:
                max_similarity = similarities[0][0]
                second_similarity = similarities[1][0] if len(similarities) > 1 else 0.0
                best_match = similarities[0][1]
                best_person_id = best_match["id"]
                base_sim = max_similarity  # fallback에서는 base_sim으로 간주
                masked_sim = 0.0
        
        # best_match가 None인 경우 처리 (suspect_ids 모드 또는 전체 DB 검색 실패)
        if not best_match:
            # 화질 기반 기본값 설정
            if face_quality == "high":
                main_threshold = 0.42
                gap_margin = 0.12
            elif face_quality == "medium":
                main_threshold = 0.40
                gap_margin = 0.10
            else:
                main_threshold = 0.38
                gap_margin = 0.08
            
            # unknown 상태로 face_results에 추가 (나중에 detections에 포함됨)
            face_results.append({
                "bbox": box.tolist(),
                "embedding": embedding_normalized,
                "angle_type": angle_type,
                "yaw_angle": float(yaw_angle) if yaw_angle is not None else 0.0,
                "face_quality": face_quality,
                "max_similarity": 0.0,
                "second_similarity": 0.0,
                "sim_gap": 0.0,
                "main_threshold": main_threshold,
                "gap_margin": gap_margin,
                "is_match": False,
                "best_match": None,
                "best_person_id": None,
                "mask_prob": 0.0
            })
            continue  # 다음 얼굴로 진행
        
        # 화질 기반 절대 임계값 설정 (마스크와 무관하게)
        # 마스크 기반 threshold 조정 로직 제거: "유사도 낮음 → 마스크겠지 → threshold 내려!" 패턴 폐기
        # 
        # 튜닝 가이드:
        # - False Positive가 많으면 threshold/gap을 높이기 (+0.01 ~ +0.02)
        # - True Positive가 적으면 threshold/gap을 낮추기 (-0.01 ~ -0.02)
        # - 특정 화질에서만 문제가 있으면 해당 화질만 조정
        # - 자세한 튜닝 가이드: python scripts/tune_threshold_gap.py --guide
        if face_quality == "high":
            main_threshold = 0.42  # 튜닝 가능: False Positive 많으면 +0.01~+0.02, True Positive 적으면 -0.01~-0.02
            gap_margin = 0.12      # 튜닝 가능: False Positive 많으면 +0.01~+0.02, True Positive 적으면 -0.01~-0.02
        elif face_quality == "medium":
            main_threshold = 0.40  # 튜닝 가능: False Positive 많으면 +0.01~+0.02, True Positive 적으면 -0.01~-0.02
            gap_margin = 0.10      # 튜닝 가능: False Positive 많으면 +0.01~+0.02, True Positive 적으면 -0.01~-0.02
        else:  # low
            main_threshold = 0.38  # 튜닝 가능: False Positive 많으면 +0.01~+0.02, True Positive 적으면 -0.01~-0.02
            gap_margin = 0.08      # 튜닝 가능: False Positive 많으면 +0.01~+0.02, True Positive 적으면 -0.01~-0.02
        
        # suspect_ids 모드에서 threshold 강화 (더 보수적으로 판단)
        if suspect_ids:
            main_threshold += 0.02  # threshold 상향
            gap_margin += 0.03  # gap 기준 더 엄격하게
        
        # 두 번째 유사도와의 차이 계산 (오인식 방지)
        sim_gap = max_similarity - second_similarity if second_similarity > 0 else max_similarity
        
        # 마스크 가능성 추정 (base_sim 기반으로 판단)
        # base_sim이 낮으면 마스크 가능성이 높음
        mask_prob = estimate_mask_from_similarity(base_sim)
        
        # Masked candidate frame 판단
        # 조건: base_sim < threshold AND base_sim >= 0.25 AND mask_prob >= 0.5
        # 주의: best_person_id가 있어야 tracking 가능 (base_sim이 낮아도 매칭된 인물이 있어야 함)
        is_masked_candidate = False
        if best_person_id != "unknown":  # 매칭된 인물이 있어야 masked candidate로 판단
            # 모든 조건 체크 및 상세 로그
            cond1 = base_sim < main_threshold
            cond2 = base_sim >= MASKED_CANDIDATE_MIN_SIM
            cond3 = mask_prob >= MASKED_BANK_MASK_PROB_THRESHOLD
            
            if cond1 and cond2 and cond3:
                is_masked_candidate = True
                print(f"🎭 [MASKED CAND] ✅ 감지됨! person_id={best_person_id}, base_sim={base_sim:.3f}, mask_prob={mask_prob:.3f}, threshold={main_threshold:.3f}")
            else:
                # 조건 미충족 이유 상세 로그
                reasons = []
                if not cond1:
                    reasons.append(f"base_sim({base_sim:.3f}) >= threshold({main_threshold:.3f})")
                if not cond2:
                    reasons.append(f"base_sim({base_sim:.3f}) < min({MASKED_CANDIDATE_MIN_SIM:.3f})")
                if not cond3:
                    reasons.append(f"mask_prob({mask_prob:.3f}) < min({MASKED_BANK_MASK_PROB_THRESHOLD:.3f})")
                print(f"🎭 [MASKED CAND] ❌ 조건 미충족: person_id={best_person_id}, base_sim={base_sim:.3f}, mask_prob={mask_prob:.3f} | 이유: {', '.join(reasons)}")
        else:
            # best_person_id가 unknown인 경우도 로그 출력 (디버깅용)
            if base_sim > 0:  # base_sim이 0보다 크면 매칭 시도는 했지만 실패한 경우
                print(f"🎭 [MASKED CAND] ⚠️ 매칭 실패: best_person_id=unknown, base_sim={base_sim:.3f}, mask_prob={mask_prob:.3f}")
        
        # 박스 정보 초기화
        box_info = {
            "bbox": box.tolist(),  # [x1, y1, x2, y2]
            "status": "unknown",
            "name": "Unknown",
            "confidence": int(max_similarity * 100),
            "color": "yellow",  # 기본값: 노란색 (미확인)
            "angle_type": angle_type,  # 각도 정보 추가
            "yaw_angle": float(yaw_angle) if yaw_angle is not None else 0.0
        }
        
        # Bank 자동 추가 여부 결정
        AUTO_ADD_TO_BANK = True  # 자동 학습 활성화
        BANK_DUPLICATE_THRESHOLD = 0.95
        bank_added = False
        
        # 강화된 매칭 조건: 세 가지 조건을 모두 만족해야 match 인정
        # 1) 절대 유사도 기준: main_threshold 이상
        # 2) gap 기준: sim_gap >= gap_margin
        # 3) 두 번째 후보 상한: second_similarity < main_threshold - 0.02
        #    (두 번째 후보도 꽤 높으면 애매하니 unknown 처리)
        is_match = False
        if max_similarity >= main_threshold:
            # 두 번째 후보가 너무 비슷하면 match 포기
            if second_similarity > 0 and second_similarity >= (main_threshold - 0.02):
                is_match = False
            else:
                # gap이 충분히 벌어졌을 때만 match 인정
                if sim_gap >= gap_margin:
                    is_match = True
        
        # suspect_ids가 지정된 경우: 추가 강화 규칙 적용
        if suspect_ids:
            # best_match가 이미 선택된 용의자 중 하나임을 보장
            if not best_match:
                is_match = False
            # 절대값 0.45 미만이면 match 포기 (suspect_ids 모드에서 더 보수적으로)
            elif max_similarity < 0.45:
                is_match = False
        else:
            # 전체 갤러리 모드에서도 best_match가 없으면 match 불가
            if not best_match:
                is_match = False
        
        # Bbox tracking 기반 multi-frame 확인 (masked candidate인 경우)
        track_id = None
        candidate_frames_count = 0
        
        if is_masked_candidate:
            # 기존 track 찾기 (IoU 기반)
            best_iou = 0.0
            for tid, track in tracking_state["tracks"].items():
                if track["person_id"] == best_person_id:
                    # 마지막 bbox와 현재 bbox의 IoU 계산
                    last_bbox = track["bbox"]
                    iou = calculate_bbox_iou(box.tolist(), last_bbox)
                    if iou > best_iou and iou >= MASKED_TRACKING_IOU_THRESHOLD:
                        best_iou = iou
                        track_id = tid
            
            # 기존 track이 있으면 업데이트, 없으면 새로 생성
            if track_id is not None:
                track = tracking_state["tracks"][track_id]
                track["bbox"] = box.tolist()
                track["frames"] += 1
                track["embeddings"].append(embedding_normalized)
                candidate_frames_count = track["frames"]
                
                # 연속 N 프레임 이상 조건 충족 시 masked bank에 추가
                if track["frames"] >= MASKED_CANDIDATE_MIN_FRAMES:
                    # masked bank에 추가 (중복 체크 포함)
                    added = update_gallery_cache_in_memory(best_person_id, embedding_normalized, bank_type="masked")
                    if added:
                        learning_events.append({
                            "person_id": best_person_id,
                            "person_name": best_match["name"] if best_match else "Unknown",
                            "angle_type": angle_type,
                            "yaw_angle": yaw_angle,
                            "embedding": embedding_normalized.tolist(),
                            "bank_type": "masked",
                            "track_frames": track["frames"]
                        })
                        print(f"  ✅ [MASKED BANK] 자동 추가 성공: {best_person_id} (연속 {track['frames']}프레임, base_sim={base_sim:.3f}, mask_prob={mask_prob:.3f})")
                    else:
                        print(f"  ⚠️ [MASKED BANK] 중복으로 스킵: {best_person_id} (연속 {track['frames']}프레임)")
                else:
                    print(f"  📊 [MASKED CAND] 추적 중: {best_person_id} ({track['frames']}/{MASKED_CANDIDATE_MIN_FRAMES}프레임, base_sim={base_sim:.3f})")
            else:
                # 새 track 생성
                track_id = f"track_{len(tracking_state['tracks'])}"
                tracking_state["tracks"][track_id] = {
                    "bbox": box.tolist(),
                    "person_id": best_person_id,
                    "frames": 1,
                    "embeddings": [embedding_normalized],
                    "last_frame": 0  # 프레임 번호는 나중에 업데이트
                }
                candidate_frames_count = 1
                print(f"  🆕 [MASKED CAND] 새 track 생성: {best_person_id} (track_id={track_id}, base_sim={base_sim:.3f})")
        
        # 결과 저장 (나중에 필터링)
        face_results.append({
            "bbox": box.tolist(),
            "embedding": embedding_normalized,
            "angle_type": angle_type,
            "yaw_angle": float(yaw_angle) if yaw_angle is not None else 0.0,
            "face_quality": face_quality,
            "max_similarity": max_similarity,
            "base_sim": base_sim,  # base bank 유사도
            "masked_sim": masked_sim,  # masked bank 유사도
            "second_similarity": second_similarity,
            "sim_gap": sim_gap,
            "main_threshold": main_threshold,
            "gap_margin": gap_margin,
            "is_match": is_match,
            "best_match": best_match,
            "best_person_id": best_person_id,
            "mask_prob": mask_prob,
            "bank_type": bank_type,
            "is_masked_candidate": is_masked_candidate,
            "candidate_frames_count": candidate_frames_count,
            "track_id": track_id
        })
    
    # 4. 같은 얼굴 영역에서 여러 인물로 매칭되는 경우 필터링 (오인식 방지)
    print(f"🔍 [필터링 전] face_results 개수: {len(face_results)}")
    filtered_results = []
    used_indices = set()
    
    for i, r1 in enumerate(face_results):
        if i in used_indices:
            continue
        
        # 같은 얼굴 영역 그룹 찾기
        group = [r1]
        used_indices.add(i)
        
        for j, r2 in enumerate(face_results):
            if j <= i or j in used_indices:
                continue
            
            if is_same_face_region(r1["bbox"], r2["bbox"]):
                group.append(r2)
                used_indices.add(j)
        
        # 그룹 처리
        if len(group) == 1:
            # 단일 매칭: 그대로 유지
            filtered_results.append(group[0])
        else:
            # 같은 얼굴 영역에서 여러 인물로 매칭됨 → 오인식 가능성 높음
            # 유사도 순으로 정렬
            group.sort(key=lambda x: x["max_similarity"], reverse=True)
            
            best_match = group[0]
            second_match = group[1] if len(group) > 1 else None
            
            # 더 엄격한 기준 적용 (오인식 방지)
            # 새로운 강화된 매칭 조건 사용
            quality = best_match["face_quality"]
            main_threshold = best_match.get("main_threshold", 0.40)
            gap_margin = best_match.get("gap_margin", 0.10)
            
            # 강화된 조건 재검증
            max_sim = best_match["max_similarity"]
            second_sim = best_match.get("second_similarity", 0.0)
            sim_gap = best_match["sim_gap"]
            
            is_match = False
            if max_sim >= main_threshold:
                if second_sim > 0 and second_sim >= (main_threshold - 0.02):
                    is_match = False
                else:
                    if sim_gap >= gap_margin:
                        is_match = True
            
            if is_match:
                # 확신 있는 매칭
                best_match["is_match"] = True
                filtered_results.append(best_match)
            else:
                # 조건을 만족하지 않으면 매칭 해제 (오인식 방지)
                # 하지만 unknown 상태로라도 detections에 포함되어야 함
                best_match["is_match"] = False
                best_match["best_match"] = None  # 매칭 해제
                filtered_results.append(best_match)  # unknown 상태로 추가
                print(f"  ⚠️ 같은 얼굴 영역에서 여러 인물 매칭됨 → 매칭 해제 (sim={max_sim:.3f} < {main_threshold:.3f} 또는 gap={sim_gap:.3f} < {gap_margin:.3f} 또는 second_sim={second_sim:.3f} >= {main_threshold - 0.02:.3f})")
    
    print(f"🔍 [필터링 후] filtered_results 개수: {len(filtered_results)}")
    
    # 5. 최종 결과 생성
    for result in filtered_results:
        # 최종 결과 생성
        box = result["bbox"]
        max_similarity = result["max_similarity"]
        best_match = result["best_match"]
        is_match = result["is_match"]
        angle_type = result["angle_type"]
        yaw_angle = result["yaw_angle"]
        main_threshold = result.get("main_threshold", 0.40)
        gap_margin = result.get("gap_margin", 0.10)
        sim_gap = result["sim_gap"]
        second_similarity = result.get("second_similarity", 0.0)
        mask_prob = result.get("mask_prob", 0.0)
        bank_type_result = result.get("bank_type", "base")
        
        # 디버깅: 매칭 조건 상세 정보 출력
        bank_type_result = result.get("bank_type", "base")
        base_sim_result = result.get("base_sim", 0.0)
        masked_sim_result = result.get("masked_sim", 0.0)
        mask_prob_result = result.get("mask_prob", 0.0)
        is_masked_candidate_result = result.get("is_masked_candidate", False)
        candidate_frames_count_result = result.get("candidate_frames_count", 0)
        
        print(f"🎯 [매칭 디버깅] bank={bank_type_result}, base_sim={base_sim_result:.3f}, masked_sim={masked_sim_result:.3f}, best_sim={max_similarity:.3f}")
        print(f"   - main_threshold={main_threshold:.3f}, sim_gap={sim_gap:.3f}, gap_margin={gap_margin:.3f}, 매칭={is_match}")
        print(f"   - mask_prob={mask_prob_result:.3f}, masked_candidate={is_masked_candidate_result}, candidate_frames={candidate_frames_count_result}")
        print(f"   - 유사도 >= main_threshold: {max_similarity:.3f} >= {main_threshold:.3f} = {max_similarity >= main_threshold}")
        print(f"   - sim_gap >= gap_margin: {sim_gap:.3f} >= {gap_margin:.3f} = {sim_gap >= gap_margin}")
        
        if is_match:
            # 매칭 성공
            name = best_match["name"]
            person_id = best_match["id"]
            is_criminal = best_match["is_criminal"]
            embedding_normalized = result["embedding"]
            
            # 감지 로그 저장 (PostgreSQL) - db가 제공된 경우에만
            if db is not None:
                try:
                    log_detection(
                        db=db,
                        person_id=person_id,
                        person_name=name,
                        similarity=max_similarity,
                        is_criminal=is_criminal,
                        status="criminal" if is_criminal else "normal",
                        metadata={
                            "bbox": box,
                            "threshold": main_threshold
                        }
                    )
                except Exception as e:
                    print(f"⚠️ 로그 저장 실패: {e}")
            
            # Bank 자동 추가 (매칭 성공 시) - base bank는 절대 자동 추가하지 않음
            # masked bank는 이미 bbox tracking 로직에서 처리됨
            # 여기서는 일반적인 측면/프로파일 각도 학습만 처리 (masked bank에만)
            AUTO_ADD_TO_BANK = True
            
            # Bank 자동 학습 안정화 조건:
            # 1) 정면은 제외 (측면/프로파일만)
            # 2) 고화질 + 고유사도 조건 (확신도 높은 프레임만)
            # 3) base bank는 절대 자동 추가하지 않음 (오염 방지)
            important_angles = ["left_profile", "right_profile", "left", "right"]
            
            if AUTO_ADD_TO_BANK:
                # 조건 1: 측면/프로파일 각도만 허용 (정면 제외)
                is_profile_angle = angle_type in important_angles
                
                # 조건 2: 고화질 + 고유사도 (main_threshold + 0.05 이상)
                is_high_confidence = (face_quality == "high" and 
                                     max_similarity >= (main_threshold + 0.05))
                
                # masked bank에만 추가 (base bank는 절대 추가하지 않음)
                if is_profile_angle and is_high_confidence and bank_type == "masked":
                    # 메모리에서 즉시 업데이트 (실시간 반영)
                    added = update_gallery_cache_in_memory(person_id, embedding_normalized, bank_type="masked")
                    if added:
                        # 학습 이벤트 기록 (masked bank만)
                        learning_events.append({
                            "person_id": person_id,
                            "person_name": name,
                            "angle_type": angle_type,
                            "yaw_angle": yaw_angle,
                            "embedding": embedding_normalized.tolist(),  # 파일 저장용
                            "bank_type": "masked"
                        })
            
            # 박스 정보 설정 (person_id 포함)
            box_info = {
                "bbox": box,
                "status": "criminal" if is_criminal else "normal",
                "name": name,
                "person_id": person_id,  # person_id 필드 추가 (temporal filter용)
                "confidence": int(max_similarity * 100),
                "color": "red" if is_criminal else "green",
                "angle_type": angle_type,
                "yaw_angle": yaw_angle,
                "bank_type": bank_type  # base 또는 masked
            }
            
            if is_criminal:
                # [범죄자 발견] 빨간색 박스
                alert_triggered = True
                detected_metadata = {
                    "name": name,
                    "confidence": int(max_similarity * 100),
                    "status": "criminal"
                }
            else:
                # [일반인] 초록색 박스
                # 현재 화면에 범죄자가 없다면 일반인 정보 표시
                if not alert_triggered:
                    detected_metadata = {
                        "name": name,
                        "confidence": int(max_similarity * 100),
                        "status": "normal"
                    }
        else:
            # [미확인] 노란색 박스 (person_id는 None)
            box_info = {
                "bbox": box,
                "status": "unknown",
                "name": "Unknown",
                "person_id": None,  # person_id 필드 추가 (temporal filter용)
                "confidence": int(max_similarity * 100),
                "color": "yellow",
                "angle_type": angle_type,
                "yaw_angle": yaw_angle
            }
            
            # 미확인 감지도 로그 저장 - db가 제공된 경우에만
            if db is not None:
                try:
                    log_detection(
                        db=db,
                        similarity=max_similarity,
                        status="unknown",
                        metadata={
                            "bbox": box,
                            "threshold": main_threshold
                        }
                    )
                except Exception as e:
                    print(f"⚠️ 로그 저장 실패: {e}")
        
        detections.append(box_info)

    # 최종 결과 로그 출력 (디버깅용)
    print(f"📊 [최종 결과] detections 개수: {len(detections)}, alert: {alert_triggered}")
    if detections:
        for i, det in enumerate(detections):
            print(f"   - [{i+1}] {det.get('name', 'Unknown')} ({det.get('status', 'unknown')}), confidence: {det.get('confidence', 0)}%")

    return {
        "detections": detections,
        "alert": alert_triggered,
        "metadata": detected_metadata,
        "learning_events": learning_events  # 학습 이벤트 (UI 피드백용)
    }

# ==========================================
# 8. API 엔드포인트
# ==========================================

class DetectionRequest(BaseModel):
    image: str       # Base64 이미지
    suspect_id: Optional[str] = None  # (선택적) 특정 타겟 ID (호환성 유지)
    suspect_ids: Optional[List[str]] = None  # (선택적) 여러 타겟 ID

@app.post("/api/detect")
async def detect_faces(request: DetectionRequest, db: Session = Depends(get_db)):
    """
    얼굴 감지 및 인식 (HTTP API - 호환성 유지)
    
    Args:
        request: DetectionRequest (image: Base64, suspect_id: 선택적)
        db: 데이터베이스 세션
    
    Returns:
        {
            "success": bool,
            "detections": [...],  # 박스 좌표 및 메타데이터 배열
            "alert": bool,
            "metadata": {...}
        }
    """
    # 1. 이미지 디코딩
    frame = base64_to_image(request.image)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image data")
    
    # 2. 공통 감지 로직 사용 (suspect_ids 우선, 없으면 suspect_id 사용)
    result = process_detection(
        frame, 
        suspect_id=request.suspect_id, 
        suspect_ids=request.suspect_ids,
        db=db
    )
    
    # 3. 범죄자 감지 시 스냅샷 Base64 인코딩 추가 (HTTP API용)
    snapshot_base64 = None
    video_timestamp = None
    
    if result.get("alert"):  # 범죄자 감지됨
        print(f"🚨 HTTP API: 범죄자 감지됨! 스냅샷 생성 중...")
        try:
            # 프레임을 JPEG로 인코딩하여 Base64 생성
            success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if success and buffer is not None and len(buffer) > 0:
                snapshot_base64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')
                print(f"✅ HTTP API: 스냅샷 생성 완료: 크기={len(snapshot_base64)} bytes")
            else:
                print(f"⚠️ HTTP API: 스냅샷 인코딩 실패 (success={success}, buffer={buffer is not None})")
        except Exception as e:
            print(f"❌ HTTP API: 스냅샷 생성 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
    
    # 4. 결과 반환
    response = {
        "success": True,
        **result
    }
    
    # 범죄자 감지 시 스냅샷 추가
    if snapshot_base64:
        response["snapshot_base64"] = snapshot_base64
        response["video_timestamp"] = video_timestamp  # None이지만 필드 추가
        print(f"📤 HTTP API 응답에 스냅샷 포함: {len(snapshot_base64)} bytes")
    
    return response

@app.websocket("/ws/detect", name="websocket_detect")
async def websocket_detect(websocket: WebSocket):
    """
    WebSocket을 통한 실시간 얼굴 감지 및 인식
    
    메시지 형식:
    - 클라이언트 → 서버:
        {
            "type": "frame",
            "data": {
                "image": "base64_string",
                "suspect_id": "optional_id",
                "frame_id": 123
            }
        }
        또는
        {
            "type": "config",
            "suspect_id": "optional_id"
        }
    
    - 서버 → 클라이언트:
        {
            "type": "detection",
            "data": {
                "frame_id": 123,
                "detections": [...],
                "alert": false,
                "metadata": {...}
            }
        }
        또는
        {
            "type": "error",
            "message": "error message"
        }
    """
    # WebSocket 연결 수락 (CORS 허용)
    try:
        print(f"🔌 [메인] WebSocket 연결 시도: {websocket.client}")
        print(f"   URL: {websocket.url}")
        print(f"   Path: {websocket.url.path}")
        origin = websocket.headers.get("origin")
        print(f"   Origin: {origin}")
        print(f"   Headers: {dict(websocket.headers)}")
        
        # WebSocket 연결 수락 (모든 origin 허용)
        await websocket.accept()
        print(f"✅ [메인] WebSocket 연결 수락됨")
        
        # 연결 등록
        active_connections.add(websocket)
        connection_states[websocket] = {
            "suspect_ids": [],  # 여러 명 선택 가능
            "connected_at": asyncio.get_event_loop().time(),
            "match_counters": {},  # person_id별 연속 매칭 프레임 카운터 (temporal consistency용)
            "tracking_state": {
                "tracks": {}  # bbox tracking 상태
            }
        }
        print(f"✅ [메인] WebSocket 연결됨 (총 {len(active_connections)}개 연결)")
        
    except Exception as e:
        print(f"❌ [메인] WebSocket 연결 수락 실패: {e}")
        import traceback
        traceback.print_exc()
        try:
            await websocket.close()
        except:
            pass
        return
    
    try:
        while True:
            # 클라이언트로부터 메시지 수신
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                msg_type = message.get("type")
                
                if msg_type == "frame":
                    # 프레임 처리 요청
                    frame_data = message.get("data", {})
                    image_base64 = frame_data.get("image")
                    suspect_ids = frame_data.get("suspect_ids")  # 배열로 받음
                    suspect_id = frame_data.get("suspect_id")  # 호환성 유지 (단일)
                    frame_id = frame_data.get("frame_id", 0)
                    video_time = frame_data.get("video_time")  # 비디오 시간 (초 단위)
                    
                    # 연결 상태에서 suspect_ids 업데이트
                    if suspect_ids is not None:
                        connection_states[websocket]["suspect_ids"] = suspect_ids
                    elif suspect_id is not None:
                        # 단일 suspect_id를 배열로 변환 (호환성)
                        connection_states[websocket]["suspect_ids"] = [suspect_id]
                    else:
                        # 연결 상태에서 suspect_ids 사용
                        suspect_ids = connection_states[websocket].get("suspect_ids", [])
                    
                    if not image_base64:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Missing image data"
                        })
                        continue
                    
                    # 이미지 디코딩
                    frame = base64_to_image(image_base64)
                    if frame is None:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Invalid image data"
                        })
                        continue
                    
                    # 각 요청마다 새로운 DB 세션 생성 (연결 유지 시 세션 문제 방지)
                    db = next(get_db())
                    try:
                        # tracking_state 가져오기
                        tracking_state = connection_states[websocket].get("tracking_state", {"tracks": {}})
                        
                        # 공통 감지 로직 사용 (suspect_ids 우선)
                        result = process_detection(
                            frame, 
                            suspect_id=suspect_id if not suspect_ids else None,
                            suspect_ids=suspect_ids if suspect_ids else None,
                            db=db,
                            tracking_state=tracking_state
                        )
                        
                        # tracking_state 업데이트
                        connection_states[websocket]["tracking_state"] = tracking_state
                    finally:
                        db.close()
                    
                    # Temporal Consistency 필터 적용 (연속 프레임 기반 매칭 확정)
                    result = apply_temporal_filter(websocket, result)
                    
                    # 범죄자 감지 시 스냅샷 Base64 인코딩 추가
                    snapshot_base64 = None
                    video_timestamp = None
                    
                    print(f"🔍 WebSocket 감지 결과: alert={result.get('alert')}, detections={len(result.get('detections', []))}")
                    
                    if result.get("alert"):  # 범죄자 감지됨
                        print(f"🚨 범죄자 감지됨! 스냅샷 생성 중...")
                        try:
                            # 프레임을 JPEG로 인코딩하여 Base64 생성
                            success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                            if success and buffer is not None and len(buffer) > 0:
                                snapshot_base64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')
                                
                                # 비디오 타임스탬프 계산 (초 단위)
                                # 클라이언트에서 전송한 video_time이 있으면 사용, 없으면 프레임 ID 기반 계산
                                if video_time is not None:
                                    video_timestamp = float(video_time)
                                else:
                                    # 프레임 ID를 사용하여 대략적인 타임스탬프 계산 (10 FPS 가정)
                                    video_timestamp = frame_id / 10.0
                                print(f"✅ 스냅샷 생성 완료: 크기={len(snapshot_base64)} bytes, 타임스탬프={video_timestamp:.1f}s")
                            else:
                                print(f"⚠️ WebSocket: 스냅샷 인코딩 실패 (success={success}, buffer={buffer is not None})")
                        except Exception as e:
                            print(f"❌ WebSocket: 스냅샷 생성 중 오류 발생: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    # 결과 전송 (응답 먼저 - 성능 최우선)
                    response_data = {
                        "type": "detection",
                        "data": {
                            "frame_id": frame_id,
                            **result
                        }
                    }
                    
                    # 범죄자 감지 시 스냅샷 추가
                    if snapshot_base64:
                        response_data["data"]["snapshot_base64"] = snapshot_base64
                        response_data["data"]["video_timestamp"] = video_timestamp
                        print(f"📤 WebSocket 응답에 스냅샷 포함: {len(snapshot_base64)} bytes")
                    
                    await websocket.send_json(response_data)

                    
                    # 학습 이벤트가 있으면 파일 저장 (비동기, 응답 후)
                    learning_events = result.get("learning_events", [])
                    for event in learning_events:
                        # 임베딩을 numpy 배열로 변환
                        embedding_array = np.array(event["embedding"], dtype=np.float32)
                        bank_type = event.get("bank_type", "base")
                        # 파일 저장은 백그라운드에서 비동기 처리 (응답 지연 없음)
                        asyncio.create_task(add_embedding_to_bank_async(
                            event["person_id"],
                            embedding_array,
                            event.get("angle_type"),
                            event.get("yaw_angle"),
                            bank_type=bank_type
                        ))
                
                elif msg_type == "config":
                    # 설정 변경 (suspect_ids 등)
                    suspect_ids = message.get("suspect_ids")  # 배열로 받음
                    suspect_id = message.get("suspect_id")  # 호환성 유지 (단일)
                    
                    if suspect_ids is not None:
                        connection_states[websocket]["suspect_ids"] = suspect_ids
                    elif suspect_id is not None:
                        # 단일 suspect_id를 배열로 변환 (호환성)
                        connection_states[websocket]["suspect_ids"] = [suspect_id]
                    
                    await websocket.send_json({
                        "type": "config_updated",
                        "suspect_ids": connection_states[websocket].get("suspect_ids", [])
                    })
                
                elif msg_type == "ping":
                    # 연결 확인
                    await websocket.send_json({
                        "type": "pong"
                    })
                
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Unknown message type: {msg_type}"
                    })
            
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON format"
                })
            except Exception as e:
                print(f"⚠️ WebSocket 처리 오류: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
    
    except WebSocketDisconnect:
        print("WebSocket 연결이 끊어졌습니다")
    except Exception as e:
        print(f"⚠️ WebSocket 오류: {e}")
    finally:
        unregister_connection(websocket)

@app.get("/api/health")
async def health_check():
    """서버 상태 확인 (WebSocket 연결 테스트용)"""
    return {
        "status": "ok",
        "websocket_endpoint": "/ws/detect",
        "active_connections": len(active_connections),
        "websocket_url": "ws://localhost:5000/ws/detect"
    }

@app.websocket("/ws/test")
async def websocket_test(websocket: WebSocket):
    """WebSocket 연결 테스트용 간단한 엔드포인트"""
    try:
        print(f"🔌 [테스트] WebSocket 연결 시도: {websocket.client}")
        await websocket.accept()
        print(f"✅ [테스트] WebSocket 연결됨")
        
        await websocket.send_json({
            "type": "test",
            "message": "WebSocket 연결 성공!"
        })
        
        # 간단한 에코 테스트
        while True:
            data = await websocket.receive_text()
            await websocket.send_json({
                "type": "echo",
                "message": f"받은 메시지: {data}"
            })
    except WebSocketDisconnect:
        print("⚠️ [테스트] WebSocket 연결 종료")
    except Exception as e:
        print(f"❌ [테스트] WebSocket 오류: {e}")
        import traceback
        traceback.print_exc()

@app.get("/api/persons")
async def get_persons(db: Session = Depends(get_db)):
    """등록된 모든 인물 목록 조회"""
    global persons_cache, gallery_base_cache, gallery_masked_cache
    
    print(f"🔍 [API /persons] 요청 받음 - persons_cache 길이: {len(persons_cache) if persons_cache else 0}")
    
    # 캐시에서 반환 (성능 향상)
    if persons_cache and len(persons_cache) > 0:
        print(f"📋 [API] persons_cache에서 반환: {len(persons_cache)}명")
        result = {
            "success": True,
            "count": len(persons_cache),
            "persons": [
                {
                    "id": p["id"],
                    "name": p["name"],
                    "is_criminal": p["is_criminal"],
                    "info": p.get("info", {})
                }
                for p in persons_cache
            ]
        }
        print(f"✅ [API] 응답 전송: success={result['success']}, count={result['count']}")
        return result
    
    # 캐시가 없으면 DB에서 직접 조회
    print(f"⚠️ [API] persons_cache가 비어있음, DB에서 직접 조회 시도")
    try:
        persons = get_all_persons(db)
        print(f"📋 [API] DB에서 조회: {len(persons)}명")
        
        # DB에서 조회한 데이터로 캐시 갱신 (다음 요청을 위해)
        if persons:
            # 캐시 갱신을 위해 load_persons_from_db 호출
            try:
                load_persons_from_db(db)
                print(f"✅ [API] 캐시 갱신 완료: {len(persons_cache)}명")
            except Exception as cache_error:
                print(f"⚠️ [API] 캐시 갱신 실패: {cache_error}")
                import traceback
                traceback.print_exc()
        
        result = {
            "success": True,
            "count": len(persons),
            "persons": [
                {
                    "id": p.person_id,
                    "name": p.name,
                    "is_criminal": p.is_criminal,
                    "info": p.info or {}
                }
                for p in persons
            ]
        }
        print(f"✅ [API] 응답 전송: success={result['success']}, count={result['count']}")
        return result
    except Exception as e:
        print(f"❌ [API] DB 조회 실패: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "count": 0,
            "persons": []
        }

@app.get("/api/logs")
async def get_logs(limit: int = 100, db: Session = Depends(get_db)):
    """감지 로그 조회"""
    from backend.database import DetectionLog
    try:
        logs = db.query(DetectionLog).order_by(DetectionLog.detected_at.desc()).limit(limit).all()
        return {
            "success": True,
            "count": len(logs),
            "logs": [
                {
                    "id": log.id,
                    "person_id": log.person_id,
                    "person_name": log.person_name,
                    "similarity": log.similarity,
                    "is_criminal": log.is_criminal,
                    "status": log.status,
                    "detected_at": log.detected_at.isoformat(),
                    "metadata": log.detection_metadata
                }
                for log in logs
            ]
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "count": 0,
            "logs": []
        }

@app.post("/api/extract_clip")
async def extract_clip(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    start_time: float = Form(...),
    end_time: float = Form(...),
    person_name: str = Form("Unknown")
):
    """
    비디오 파일에서 특정 구간을 추출하여 클립 생성
    
    Args:
        video: 비디오 파일
        start_time: 시작 시간 (초)
        end_time: 종료 시간 (초)
        person_name: 범죄자 이름
    
    Returns:
        추출된 클립 파일
    """
    try:
        # 임시 파일 생성
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as input_file:
            input_path = input_file.name
            # 업로드된 비디오 파일 저장
            content = await video.read()
            input_file.write(content)
        
        # 출력 파일 경로
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        
        # ffmpeg를 사용하여 클립 추출
        duration = end_time - start_time
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-ss', str(start_time),
            '-t', str(duration),
            '-c', 'copy',  # 재인코딩 없이 복사 (빠름)
            '-avoid_negative_ts', 'make_zero',
            '-y',  # 덮어쓰기
            output_path
        ]
        
        print(f"🎬 클립 추출 시작: {person_name} ({start_time:.1f}s - {end_time:.1f}s)")
        print(f"📝 명령어: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60  # 60초 타임아웃
        )
        
        if result.returncode != 0:
            print(f"❌ ffmpeg 오류: {result.stderr}")
            raise HTTPException(status_code=500, detail=f"클립 추출 실패: {result.stderr}")
        
        # 임시 입력 파일 삭제
        try:
            os.unlink(input_path)
        except:
            pass
        
        print(f"✅ 클립 추출 완료: {output_path}")
        
        # 응답 후 파일 삭제를 BackgroundTasks로 등록
        background_tasks.add_task(os.unlink, output_path)
        
        # 파일 응답 반환
        return FileResponse(
            output_path,
            media_type='video/mp4',
            filename=f"clip_{person_name}_{start_time:.1f}s-{end_time:.1f}s.mp4"
        )
        
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="클립 추출 시간 초과")
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="ffmpeg가 설치되지 않았습니다. ffmpeg를 설치해주세요.")
    except Exception as e:
        print(f"❌ 클립 추출 오류: {e}")
        raise HTTPException(status_code=500, detail=f"클립 추출 실패: {str(e)}")
    finally:
        # 임시 파일 정리
        try:
            if 'input_path' in locals():
                os.unlink(input_path)
        except:
            pass

# 실행 명령: uvicorn backend.main:app --reload --host 0.0.0.0 --port 5000
