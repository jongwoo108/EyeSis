"""
설정 파일
"""
import os
from dotenv import load_dotenv

load_dotenv()

# 데이터베이스 설정
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/facewatch"
)

# 서버 설정
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 5000))

# InsightFace 설정
INSIGHTFACE_MODEL = os.getenv("INSIGHTFACE_MODEL", "buffalo_l")
INSIGHTFACE_CTX_ID = int(os.getenv("INSIGHTFACE_CTX_ID", 0))  # GPU: 0, CPU: -1






