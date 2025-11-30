# FaceWatch 프로젝트 구조 정리 및 통합 작업

## 변경 사항 요약

### 1. 폴더 구조 정리 및 경로 일관성 확보

#### 생성/수정된 파일:
- `backend/database/` 폴더 생성 (레거시 JSON 파일용)
- `backend/main.py` 완전 재작성

#### 주요 변경:
- 프로젝트 루트 경로를 `Path(__file__).parent.parent`로 안전하게 처리
- `backend/database/*.json` 경로를 `DATABASE_DIR` 상수로 관리
- `outputs/embeddings` 경로를 `EMBEDDINGS_DIR` 상수로 관리

### 2. FastAPI 서버와 기존 FaceWatch 코어 로직 통합

#### 변경된 파일:
- `backend/main.py`

#### 주요 변경:
- **InsightFace 초기화**: `src/utils/device_config.py`의 로직 사용
  - `_ensure_cuda_in_path()` 먼저 호출
  - `get_device_id()`로 GPU/CPU 자동 감지
  - `safe_prepare_insightface()`로 안전한 초기화 (GPU 실패 시 CPU fallback)
  
- **갤러리 로더 통합**: `src/utils/gallery_loader.py` 사용
  - `load_gallery()`로 `outputs/embeddings`에서 bank/centroid 로드 (권장)
  - `match_with_bank()`로 매칭 수행
  - 레거시 지원: `backend/database/*.json`도 여전히 지원

- **이중 데이터 소스 지원**:
  1. 우선순위 1: `outputs/embeddings/<person>/bank.npy` 또는 `centroid.npy` (gallery_loader 사용)
  2. 우선순위 2: `backend/database/*.json` (레거시 지원)

### 3. 프론트엔드-백엔드 I/O 계약 점검 및 수정

#### 변경된 파일:
- `backend/main.py` (API 응답 형식)
- `web/index.html` (HTML 오타 수정)
- `web/script.js` (화면 전환 코드 보완)

#### I/O 계약 확인:
- **Request 형식** (변경 없음):
  ```json
  {
    "image": "data:image/jpeg;base64,...",
    "suspect_id": "hani"  // 선택적
  }
  ```

- **Response 형식** (변경 없음):
  ```json
  {
    "success": true,
    "processed_image": "data:image/jpeg;base64,...",
    "alert": false,
    "metadata": {
      "name": "홍길동",
      "confidence": 85,
      "status": "normal"  // "criminal" | "normal" | "unknown"
    }
  }
  ```

- **프론트엔드 사용** (변경 없음):
  - `result.processed_image` → `UI.processedFrame.src`
  - `result.alert` → `alert-border` 클래스 토글
  - `result.metadata` → `updateDetectionPanel()`에서 표시

#### 수정 사항:
- `web/index.html` 116줄: `<img>` 태그 닫힘 오타 수정
- `web/script.js`: 화면 전환 코드 보완 (용의자 선택 → 대시보드)

### 4. suspect_id 활용 개선

#### 변경된 파일:
- `backend/main.py`

#### 구현 방식 (B 방식 선택):
- `suspect_id`가 지정된 경우:
  1. 먼저 해당 인물과의 유사도 확인 (우선 검색)
  2. 매칭 실패 시 전체 DB를 fallback으로 검색
  
- `suspect_id`가 없는 경우:
  - 전체 DB 검색 (기존 동작 유지)

#### 코드 위치:
```python
# suspect_id가 지정된 경우 우선 확인
if request.suspect_id:
    if gallery_cache:
        # Gallery 기반 매칭 (권장)
        if request.suspect_id in gallery_cache:
            person_id, sim = match_with_bank(embedding, {request.suspect_id: gallery_cache[request.suspect_id]})
            # ...
    else:
        # JSON 기반 매칭 (레거시)
        person_info = find_person_info(request.suspect_id)
        # ...

# 전체 DB 검색 (suspect_id가 없거나 매칭 실패 시)
if not best_match:
    # ...
```

### 5. 코드 스타일 및 리팩토링

#### 변경된 파일:
- `backend/main.py` (전체 리팩토링)

#### 주요 개선:
- 타입 힌트 추가 (`Optional`, `List`, `Dict` 등)
- 함수별 독스트링 추가 (한국어)
- 상수 분리 (`DATABASE_DIR`, `EMBEDDINGS_DIR`, `THRESHOLD` 등)
- 에러 처리 개선 (`try-except` 블록)
- 주석 정리 및 한국어 유지

#### 알고리즘 유지:
- 유사도 계산 로직 유지 (`compute_cosine_similarity`)
- 임계값 0.5 유지
- 매칭 알고리즘 변경 없음

### 6. 추가 API 엔드포인트

#### 변경된 파일:
- `backend/main.py`

#### 추가된 엔드포인트:
- `GET /api/persons`: 등록된 모든 인물 목록 조회
  ```json
  {
    "success": true,
    "count": 5,
    "persons": [
      {
        "id": "hani",
        "name": "하니",
        "is_criminal": false,
        "info": {}
      },
      ...
    ]
  }
  ```

## 사용 방법

### 1. 백엔드 서버 실행

```bash
# 프로젝트 루트에서 실행
uvicorn backend.main:app --reload --host 0.0.0.0 --port 5000
```

### 2. 프론트엔드 실행

#### 방법 1: VSCode Live Server
1. VSCode에서 `web/index.html` 열기
2. 우클릭 → "Open with Live Server"

#### 방법 2: Python HTTP 서버
```bash
# 프로젝트 루트에서 실행
cd web
python -m http.server 5500
```

#### 방법 3: Node.js http-server
```bash
npx http-server web -p 5500
```

### 3. 브라우저에서 접근

- 프론트엔드: http://localhost:5500/index.html
- 백엔드 API: http://localhost:5000/api/detect
- API 문서: http://localhost:5000/docs (FastAPI 자동 생성)

## 주의사항

1. **데이터 소스 우선순위**:
   - `outputs/embeddings`가 있으면 우선 사용 (gallery_loader)
   - 없으면 `backend/database/*.json` 사용 (레거시)

2. **인물 등록**:
   - 권장: `python src/face_enroll.py` 실행하여 `outputs/embeddings`에 등록
   - 레거시: `backend/database/*.json` 파일 직접 생성

3. **suspect_id 매핑**:
   - 프론트엔드의 `data-suspect-id`는 실제 `person_id`와 일치해야 함
   - 예: `data-suspect-id="hani"` → `outputs/embeddings/hani/` 또는 `database/hani.json`

## 향후 개선 사항

1. 프론트엔드에서 `/api/persons`를 호출하여 동적으로 용의자 카드 생성
2. PostgreSQL 도입 시 `backend/database.py`와 통합
3. 설정 파일(`config.py`)로 경로 및 임계값 관리
4. 로깅 시스템 추가 (감지 이벤트 저장)




















