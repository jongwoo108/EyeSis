# PostgreSQL 도입 완료 보고서

## 작업 완료 사항

### 1. 백엔드 코드 수정

**파일**: `backend/main.py`
- PostgreSQL 데이터베이스 사용하도록 완전 재작성
- `backend.database` 모듈을 통한 데이터베이스 연동
- Fallback 메커니즘: PostgreSQL 실패 시 `outputs/embeddings` 자동 사용
- 감지 로그 자동 저장 (`detection_logs` 테이블)

**주요 변경점**:
- `get_db()` 의존성 주입으로 데이터베이스 세션 관리
- `load_persons_from_db()`: PostgreSQL에서 인물 정보 로드
- `load_persons_from_embeddings()`: Fallback용 embeddings 로드
- 모든 API 엔드포인트에 `db: Session = Depends(get_db)` 추가

### 2. 데이터 마이그레이션 스크립트 개선

**파일**: `backend/init_db.py`
- `outputs/embeddings` 폴더 지원 추가
- 우선순위: embeddings > JSON
- 중복 체크 및 스킵 기능
- 상세한 진행 상황 출력

**사용법**:
```bash
python backend/init_db.py
```

### 3. 데이터베이스 모델

**파일**: `backend/database.py`
- `Person` 모델: 인물 정보 및 임베딩 저장
- `DetectionLog` 모델: 감지 로그 저장
- 유틸리티 함수: CRUD 작업 지원

### 4. 설정 파일

**파일**: `backend/.env` (생성됨)
- 데이터베이스 연결 정보
- 서버 설정
- InsightFace 설정

**파일**: `backend/SETUP.md` (생성됨)
- PostgreSQL 설치 가이드
- 데이터베이스 설정 방법
- 문제 해결 가이드

### 5. 문서 업데이트

**파일**: `README.md`
- 웹 UI 섹션에 PostgreSQL 관련 내용 추가
- 데이터 소스 우선순위 명시
- API 엔드포인트 문서화

**파일**: `backend/README.md`
- PostgreSQL 설정 가이드 업데이트
- Fallback 동작 설명 추가

## 데이터 흐름

### 서버 시작 시
```
1. PostgreSQL 연결 시도
   ↓ 성공
2. persons 테이블에서 데이터 로드
   ↓ 실패
3. outputs/embeddings에서 로드 (fallback)
```

### API 요청 시
```
POST /api/detect
  ↓
1. 이미지 디코딩
2. InsightFace로 얼굴 감지
3. PostgreSQL 캐시 또는 embeddings와 매칭
4. 결과 반환
5. 감지 로그를 PostgreSQL에 저장
```

## 사용 방법

### 1. PostgreSQL 설치 및 데이터베이스 생성

```bash
psql -U postgres
CREATE DATABASE facewatch;
\q
```

### 2. 환경 변수 설정

`backend/.env` 파일 생성:
```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/facewatch
```

### 3. 데이터 마이그레이션

```bash
python backend/init_db.py
```

### 4. 서버 실행

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 5000
```

## Fallback 동작

PostgreSQL 연결 실패 시:
1. 서버는 정상적으로 시작됨
2. `outputs/embeddings`에서 데이터 로드
3. 감지 로그는 저장되지 않음 (PostgreSQL 필요)
4. 서버 로그에 경고 메시지 출력

## API 엔드포인트

### POST `/api/detect`
- 얼굴 감지 및 인식
- 감지 로그 자동 저장 (PostgreSQL)

### GET `/api/persons`
- 등록된 모든 인물 목록 조회
- PostgreSQL 또는 캐시에서 반환

### GET `/api/logs?limit=100`
- 감지 로그 조회
- PostgreSQL에서 최근 N개 반환

## 데이터베이스 구조

### `persons` 테이블
- 인물 정보 및 임베딩 벡터 저장
- `person_id`로 고유 식별
- `is_criminal` 플래그로 범죄자 구분

### `detection_logs` 테이블
- 모든 감지 이벤트 저장
- 타임스탬프, 유사도, 메타데이터 포함
- 분석 및 통계에 활용 가능

## 향후 개선 사항

1. **인물 정보 업데이트 API**: 웹 UI에서 인물 정보 수정
2. **통계 대시보드**: 감지 로그 기반 통계 제공
3. **이미지 저장**: 감지된 얼굴 이미지를 데이터베이스에 저장
4. **인물 그룹 관리**: 범죄자/일반인 그룹 관리 기능

## 주의사항

1. **PostgreSQL 필수 아님**: Fallback으로 `outputs/embeddings` 사용 가능
2. **데이터 마이그레이션**: 첫 실행 시 `init_db.py` 실행 필요
3. **환경 변수**: `.env` 파일의 `DATABASE_URL` 확인 필수
4. **감지 로그**: PostgreSQL 없으면 로그 저장 안 됨 (기능은 정상 작동)

