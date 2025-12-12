"""
FaceWatch FastAPI 백엔드 서버
웹 프론트엔드와 연동하여 실시간 얼굴 인식 서비스 제공
PostgreSQL 데이터베이스 사용
"""
import base64
import cv2
import numpy as np

import shutil
from typing import Optional, List, Dict, Set
from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
from sqlalchemy.orm import Session

import json
import asyncio
import subprocess
import tempfile
import os
import time

import sys

from backend.utils.bbox_utils import (
    calculate_bbox_iou,
    calculate_bbox_center_distance,
    is_same_face_region
)
from backend.utils.image_utils import (
    l2_normalize,
    compute_cosine_similarity,
    preprocess_image_for_detection,
    base64_to_image,
    image_to_base64
)
from backend.utils.websocket_manager import(
    active_connections,
    connection_states,
    register_connection,
    unregister_connection
)
from backend.services import data_loader
from backend.services.data_loader import (
    load_persons_from_db,
    load_persons_from_embeddings,
    load_persons_from_legacy_files,
    find_person_info
)
from backend.services.bank_manager import(
    save_angle_separated_banks,
    add_embedding_to_bank_async,
    add_embedding_to_dynamic_bank_async,
    update_gallery_cache_in_memory
)
from backend.services.temporal_filter import apply_temporal_filter
from backend.services.face_detection import process_detection
from backend.models.schemas import DetectionRequest
from backend.api import persons


# 프로젝트 루트를 Python 경로에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# CUDA 경로를 먼저 설정 (가장 먼저 import)
from src.utils.device_config import _ensure_cuda_in_path
_ensure_cuda_in_path()

from insightface.app import FaceAnalysis
from src.utils.device_config import get_device_id, safe_prepare_insightface
from src.utils.gallery_loader import load_gallery, match_with_bank, match_with_bank_detailed
from src.utils.face_angle_detector import estimate_face_angle, is_diverse_angle, is_all_angles_collected, check_face_occlusion
from src.utils.mask_detector import estimate_mask_from_similarity, get_adjusted_threshold, estimate_face_quality
from src.face_enroll import get_main_face_embedding, save_embeddings, l2_normalize

# PostgreSQL 데이터베이스 모듈
from backend.database import (
    get_db, get_all_persons, get_person_by_id,
    log_detection, init_db as db_init, Person
)

# ==========================================
# 1. 설정 및 경로
# ==========================================


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

# API 라우터 등록
from backend.api import detection
app.include_router(detection.router, tags=["detection"])
from backend.api import persons
app.include_router(persons.router, tags=["persons"])
from backend.api import video
app.include_router(video.router, tags=["video"])

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

# Inject model into face_detection module
from backend.services import face_detection
face_detection.set_model(model)



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
    if not data_loader.gallery_base_cache and not data_loader.persons_cache:
        print("⚠️ 경고: 등록된 얼굴 데이터가 없습니다!")
        print("   face_enroll.py를 실행하여 인물을 등록하거나,")
        print("   python backend/init_db.py를 실행하여 데이터를 마이그레이션해주세요.\n")




# ==========================================
# 7. 공통 감지 로직 함수
# ==========================================


# ==========================================
# 8. API 엔드포인트
# ==========================================










@app.get("/api/images/enroll/{person_id}/{filename}")
async def get_person_image(person_id: str, filename: str):
    """등록된 인물의 이미지 제공"""
    image_path = PROJECT_ROOT / "images" / "enroll" / person_id / filename
    
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="이미지를 찾을 수 없습니다.")
    
    # 보안 체크: person_id와 filename이 일치하는지 확인
    if image_path.parent.name != person_id:
        raise HTTPException(status_code=403, detail="접근 권한이 없습니다.")
    
    return FileResponse(image_path)



# ==========================================
# Static Files 마운트 (프론트엔드 서빙)
# ==========================================
# web 폴더의 정적 파일들을 루트 경로로 서빙
# 이렇게 하면 ngrok으로 외부 접속 시에도 하나의 URL로 통합 가능
web_dir = PROJECT_ROOT / "web"
app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="static")

# 실행 명령: uvicorn backend.main:app --reload --host 0.0.0.0 --port 5000
