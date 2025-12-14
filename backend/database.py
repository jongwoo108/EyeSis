"""
데이터베이스 연결 및 모델 정의
"""
import os
from pathlib import Path
from urllib.parse import urlparse
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from dotenv import load_dotenv
import numpy as np
import json

# .env 파일 로드
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=str(env_path), encoding='utf-8')
else:
    load_dotenv(encoding='utf-8')

# 데이터베이스 연결 설정
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/eyesis"
)

# PostgreSQL 연결 시 인코딩 명시
def create_database_engine():
    """데이터베이스 엔진 생성"""
    import psycopg2
    
    try:
        parsed = urlparse(DATABASE_URL)
        
        if "postgresql" in (parsed.scheme or '').lower():
            # 연결 정보 추출
            username = parsed.username or 'postgres'
            password = parsed.password or 'postgres'
            hostname = parsed.hostname or 'localhost'
            port = parsed.port or 5432
            database = parsed.path.lstrip('/') if parsed.path else 'eyesis'
            
            # 커스텀 연결 함수
            def create_connection():
                """커스텀 연결 함수"""
                return psycopg2.connect(
                    host=hostname,
                    port=port,
                    dbname=database,
                    user=username,
                    password=password,
                    client_encoding='UTF8'
                )
            
            # SQLAlchemy 엔진 생성
            safe_url = f"postgresql://{hostname}:{port}/{database}"
            engine = create_engine(
                safe_url,
                pool_pre_ping=True,
                creator=create_connection
            )
        else:
            engine = create_engine(
                DATABASE_URL,
                pool_pre_ping=True
            )
        
        return engine
    except Exception as e:
        print(f"⚠️ 데이터베이스 엔진 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        raise

try:
    engine = create_database_engine()
except Exception as e:
    print(f"⚠️ 데이터베이스 엔진 생성 실패: {e}")
    import traceback
    traceback.print_exc()
    raise
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ==========================================
# 데이터베이스 모델
# ==========================================

class Person(Base):
    """인물 정보 테이블"""
    __tablename__ = "persons"
    
    id = Column(Integer, primary_key=True, index=True)
    person_id = Column(String, unique=True, index=True, nullable=False)  # 고유 ID (예: "criminal", "hani")
    name = Column(String, nullable=False)  # 이름
    is_criminal = Column(Boolean, default=False)  # 범죄자 여부
    info = Column(JSON, default={})  # 추가 정보 (JSON 형태)
    embedding = Column(Text, nullable=False)  # 임베딩 벡터 (JSON 문자열로 저장)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def set_embedding(self, embedding_array: np.ndarray):
        """numpy 배열을 JSON 문자열로 변환하여 저장"""
        self.embedding = json.dumps(embedding_array.tolist())
    
    def get_embedding(self) -> np.ndarray:
        """JSON 문자열을 numpy 배열로 변환하여 반환"""
        return np.array(json.loads(self.embedding), dtype=np.float32)


class DetectionLog(Base):
    """감지 로그 테이블"""
    __tablename__ = "detection_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    person_id = Column(String, index=True, nullable=True)  # 감지된 인물 ID (없으면 None)
    person_name = Column(String, nullable=True)  # 감지된 인물 이름
    similarity = Column(Float, nullable=False)  # 유사도 (0.0 ~ 1.0)
    is_criminal = Column(Boolean, default=False)  # 범죄자 여부
    status = Column(String, nullable=False)  # "criminal", "normal", "unknown"
    frame_timestamp = Column(DateTime, nullable=True)  # 프레임 타임스탬프
    detected_at = Column(DateTime, default=datetime.utcnow)  # 감지 시간
    detection_metadata = Column(JSON, default={})  # 추가 메타데이터 (bbox, 각도 등) - metadata는 SQLAlchemy 예약어라서 변경


# ==========================================
# 데이터베이스 유틸리티 함수
# ==========================================

def get_db():
    """데이터베이스 세션 생성 (의존성 주입용)"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """데이터베이스 테이블 생성"""
    try:
        Base.metadata.create_all(bind=engine)
        print("✅ 데이터베이스 테이블 생성 완료")
    except Exception as e:
        import traceback
        print(f"❌ 데이터베이스 테이블 생성 실패: {e}")
        traceback.print_exc()
        raise


def get_person_by_id(db, person_id: str):
    """person_id로 인물 정보 조회"""
    return db.query(Person).filter(Person.person_id == person_id).first()


def get_all_persons(db):
    """모든 인물 정보 조회"""
    return db.query(Person).all()


def create_person(db, person_id: str, name: str, embedding: np.ndarray, 
                  is_criminal: bool = False, info: dict = None):
    """새 인물 등록"""
    person = Person(
        person_id=person_id,
        name=name,
        is_criminal=is_criminal,
        info=info or {}
    )
    person.set_embedding(embedding)
    db.add(person)
    db.commit()
    db.refresh(person)
    return person


def log_detection(db, person_id: str = None, person_name: str = None,
                  similarity: float = 0.0, is_criminal: bool = False,
                  status: str = "unknown", metadata: dict = None):
    """감지 로그 저장"""
    log = DetectionLog(
        person_id=person_id,
        person_name=person_name,
        similarity=similarity,
        is_criminal=is_criminal,
        status=status,
        detection_metadata=metadata or {}  # metadata -> detection_metadata로 변경
    )
    db.add(log)
    db.commit()
    return log

