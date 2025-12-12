# backend/config.py
"""
FaceWatch 백엔드 설정
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 경로 설정
# ==========================================
PROJECT_ROOT = Path(__file__).parent.parent
EMBEDDINGS_DIR = PROJECT_ROOT / "outputs" / "embeddings"
ENROLL_IMAGES_DIR = PROJECT_ROOT / "images" / "enroll"
WEB_DIR = PROJECT_ROOT / "web"

# ==========================================
# 데이터베이스 설정
# ==========================================
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/facewatch"
)

# ==========================================
# 서버 설정
# ==========================================
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 5000))
WEBSOCKET_URL = f"ws://localhost:{PORT}/ws/detect"

# ==========================================
# InsightFace 모델 설정
# ==========================================
INSIGHTFACE_MODEL = os.getenv("INSIGHTFACE_MODEL", "buffalo_l")
INSIGHTFACE_CTX_ID = int(os.getenv("INSIGHTFACE_CTX_ID", 0))  # GPU: 0, CPU: -1
INSIGHTFACE_DET_SIZE = (640, 640)

# ==========================================
# 얼굴 인식 임계값
# ==========================================
# 기본 매칭 임계값
MAIN_THRESHOLD = 0.45  # 기본 유사도 임계값
SUSPECT_THRESHOLD = 0.48  # 용의자 모드 임계값

# Masked Bank 관련 설정
MASKED_BANK_MASK_PROB_THRESHOLD = 0.5  # mask_prob >= 0.5이면 masked bank로 분류
MASKED_CANDIDATE_MIN_SIM = 0.25  # base_sim >= 0.25 이상이어야 masked candidate로 판단
MASKED_CANDIDATE_MIN_FRAMES = 3  # 연속 N 프레임 이상 조건 충족 시 masked bank에 추가
MASKED_TRACKING_IOU_THRESHOLD = 0.5  # bbox tracking을 위한 IoU 임계값

# Dynamic Bank 관련 설정
DYNAMIC_BANK_SIMILARITY_THRESHOLD = 0.9  # 중복 체크 임계값
BANK_DUPLICATE_THRESHOLD = 0.95  # 등록 시 중복 체크 임계값

# ==========================================
# Temporal Filter 설정
# ==========================================
TEMPORAL_FILTER_WINDOW = 5  # 최근 N 프레임을 고려
TEMPORAL_FILTER_MIN_MATCHES = 3  # 최소 매칭 프레임 수

# ==========================================
# API 설정
# ==========================================
DEFAULT_LOG_LIMIT = 100  # 기본 로그 조회 수
JPEG_QUALITY = 85  # 스냅샷 JPEG 품질
