# EyeSis Backend

FastAPI 기반 백엔드 서버 (PostgreSQL 사용)

## 설치 및 설정

### 1. PostgreSQL 설치 및 데이터베이스 생성

자세한 내용은 `backend/SETUP.md`를 참조하세요.

**간단 요약:**
```bash
# PostgreSQL 접속
psql -U postgres

# 데이터베이스 생성
CREATE DATABASE eyesis;
\q
```

### 2. 환경 변수 설정

`backend/.env` 파일을 생성하고 다음 내용을 추가:

```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/eyesis
HOST=0.0.0.0
PORT=5000
INSIGHTFACE_MODEL=buffalo_l
INSIGHTFACE_CTX_ID=0
```

**주의**: 사용자명과 비밀번호를 실제 PostgreSQL 설정에 맞게 변경하세요.

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

### 4. 데이터베이스 초기화

기존 데이터를 PostgreSQL로 마이그레이션:

```bash
# 프로젝트 루트에서 실행
python backend/init_db.py
```

이 스크립트는 다음 순서로 데이터를 로드합니다:
1. `outputs/embeddings/<person>/bank.npy` 또는 `centroid.npy` (우선)
2. `backend/database/*.json` (fallback)

### 5. 서버 실행

```bash
# 프로젝트 루트에서 실행
uvicorn backend.main:app --reload --host 0.0.0.0 --port 5000
```

**Fallback 동작**: PostgreSQL 연결 실패 시 자동으로 `outputs/embeddings`를 사용합니다.

## API 엔드포인트

### POST `/api/detect`
얼굴 감지 및 인식

**Request:**
```json
{
  "image": "data:image/jpeg;base64,...",
  "suspect_id": "criminal"
}
```

**Response:**
```json
{
  "success": true,
  "processed_image": "data:image/jpeg;base64,...",
  "alert": false,
  "metadata": {
    "name": "홍길동",
    "confidence": 85,
    "status": "normal"
  }
}
```

### GET `/api/persons`
등록된 모든 인물 목록 조회

### GET `/api/logs?limit=100`
감지 로그 조회

## 데이터베이스 구조

### `persons` 테이블
- `id`: Primary Key
- `person_id`: 고유 ID (예: "criminal", "hani")
- `name`: 이름
- `is_criminal`: 범죄자 여부
- `info`: 추가 정보 (JSON)
- `embedding`: 임베딩 벡터 (JSON 문자열)
- `created_at`, `updated_at`: 타임스탬프

### `detection_logs` 테이블
- `id`: Primary Key
- `person_id`: 감지된 인물 ID
- `person_name`: 감지된 인물 이름
- `similarity`: 유사도
- `is_criminal`: 범죄자 여부
- `status`: 상태 ("criminal", "normal", "unknown")
- `detected_at`: 감지 시간
- `metadata`: 추가 메타데이터 (JSON)

