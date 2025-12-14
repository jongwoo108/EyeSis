"""
EyeSis FastAPI 백엔드 서버
웹 프론트엔드와 연동하여 실시간 얼굴 인식 서비스 제공
PostgreSQL 데이터베이스 사용
"""
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# 프로젝트 루트를 Python 경로에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# CUDA 경로를 먼저 설정
from backend.utils.device_config import _ensure_cuda_in_path
_ensure_cuda_in_path()

# InsightFace 및 유틸리티
from insightface.app import FaceAnalysis
from backend.utils.device_config import get_device_id, safe_prepare_insightface

# 데이터 로딩
from backend.services import data_loader
from backend.services.data_loader import load_persons_from_db, load_persons_from_embeddings
from backend.database import get_db, init_db as db_init

# ==========================================
# FastAPI 앱 초기화
# ==========================================

app = FastAPI(title="EyeSis API", version="1.0.0")

# CORS 허용 (프론트엔드 접근 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 라우터 등록
from backend.api import detection, persons, video
app.include_router(detection.router, tags=["detection"])
app.include_router(persons.router, tags=["persons"])
app.include_router(video.router, tags=["video"])

# ==========================================
# InsightFace 모델 초기화
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

# 모듈에 모델 주입
from backend.services import face_detection
from backend.api import persons as persons_api
face_detection.set_model(model)
persons_api.set_model(model)

# ==========================================
# 서버 시작 이벤트
# ==========================================

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 데이터베이스 초기화 및 데이터 로드"""
    print("=" * 70)
    print("🚀 EyeSis 서버 시작")
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
        load_persons_from_embeddings()
    
    # 3. 데이터가 없으면 경고
    if not data_loader.gallery_base_cache and not data_loader.persons_cache:
        print("⚠️ 경고: 등록된 얼굴 데이터가 없습니다!")
        print("   face_enroll.py를 실행하여 인물을 등록하거나,")
        print("   python backend/init_db.py를 실행하여 데이터를 마이그레이션해주세요.\n")

# ==========================================
# 이미지 서빙 API (라우터에 포함시키기 어려운 경로 패턴)
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
web_dir = PROJECT_ROOT / "web"
app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="static")

# 실행 명령: uvicorn backend.main:app --reload --host 0.0.0.0 --port 5000
