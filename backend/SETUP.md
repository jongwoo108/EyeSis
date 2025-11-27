# PostgreSQL 데이터베이스 설정 가이드

## 1. PostgreSQL 설치

### Windows
1. PostgreSQL 공식 사이트에서 다운로드: https://www.postgresql.org/download/windows/
2. 설치 프로그램 실행 및 기본 설정으로 설치
3. 설치 중 비밀번호 설정 (기본값: `postgres`)

### macOS
```bash
brew install postgresql
brew services start postgresql
```

### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
```

## 2. 데이터베이스 생성

PostgreSQL에 접속하여 데이터베이스 생성:

```bash
# PostgreSQL 접속
psql -U postgres

# 데이터베이스 생성
CREATE DATABASE facewatch;

# 확인
\l

# 종료
\q
```

## 3. 환경 변수 설정

`backend/.env` 파일을 생성하고 다음 내용을 추가:

```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/facewatch
HOST=0.0.0.0
PORT=5000
INSIGHTFACE_MODEL=buffalo_l
INSIGHTFACE_CTX_ID=0
```

**주의**: `postgres:postgres` 부분을 실제 PostgreSQL 사용자명과 비밀번호로 변경하세요.

## 4. 데이터 마이그레이션

기존 데이터를 PostgreSQL로 마이그레이션:

```bash
# 프로젝트 루트에서 실행
python backend/init_db.py
```

이 스크립트는 다음 순서로 데이터를 로드합니다:
1. `outputs/embeddings/<person>/bank.npy` 또는 `centroid.npy` (우선)
2. `backend/database/*.json` (fallback)

## 5. 서버 실행

```bash
# 백엔드 서버 실행
uvicorn backend.main:app --reload --host 0.0.0.0 --port 5000
```

서버가 시작되면 PostgreSQL에서 데이터를 자동으로 로드합니다.

## 6. 문제 해결

### PostgreSQL 연결 실패
- PostgreSQL이 실행 중인지 확인: `pg_isready` 또는 서비스 상태 확인
- 데이터베이스가 생성되었는지 확인: `psql -U postgres -l`
- `.env` 파일의 `DATABASE_URL`이 올바른지 확인

### 데이터가 없음
- `python backend/init_db.py` 실행하여 데이터 마이그레이션
- 또는 `python src/face_enroll.py` 실행하여 새로 등록

### Fallback 동작
PostgreSQL 연결 실패 시 자동으로 `outputs/embeddings`를 사용합니다.
서버 로그에서 확인할 수 있습니다.

## 7. 데이터베이스 구조

### `persons` 테이블
- `id`: Primary Key
- `person_id`: 고유 ID (예: "hani", "criminal")
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












