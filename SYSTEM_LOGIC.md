# FaceWatch 시스템 로직 및 사용자 흐름 문서

> 프로젝트의 전체 사용자 흐름, 얼굴 인식 프로세스, 하이퍼파라미터, 연결된 기능들의 상세 로직을 정리한 문서

---

## 목차

1. [사용자 흐름 (User Flow)](#1-사용자-흐름-user-flow)
2. [얼굴 인식 프로세스](#2-얼굴-인식-프로세스)
3. [하이퍼파라미터](#3-하이퍼파라미터)
4. [핵심 로직 상세](#4-핵심-로직-상세)
5. [데이터 흐름](#5-데이터-흐름)
6. [의존성 및 연결 관계](#6-의존성-및-연결-관계)
7. [잠재적 문제점 및 개선 방향](#7-잠재적-문제점-및-개선-방향)

---

## 1. 사용자 흐름 (User Flow)

### 1.1 웹 UI 사용 흐름

```
[1단계: 영상 업로드]
사용자 → "영상 업로드" 버튼 클릭 또는 중앙 영역 클릭
  ↓
시스템 파일 탐색기 열림
  ↓
영상 파일 선택 (MP4, AVI, MOV 등)
  ↓
파일 검증 (형식, 크기)
  ↓
비디오 플레이어에 로드 (일시 정지 상태)
  ↓
빈 상태 카드 숨김

[2단계: 인물 선택]
사용자 → "인물 선택" 버튼 클릭
  ↓
인물 목록 로드 (PostgreSQL 또는 outputs/embeddings)
  ↓
인물 카드 표시 (이름, 사진, 범죄자/실종자 구분)
  ↓
사용자 → 감지할 인물 카드 클릭 (여러 명 선택 가능)
  ↓
선택된 인물 정보 표시
  ↓
"모니터링 시작" 버튼 활성화

[3단계: 모니터링 시작]
사용자 → "모니터링 시작" 버튼 클릭
  ↓
타임라인 트랙 초기화 (선택된 인물별)
  ↓
비디오 재생 준비 (일시 정지 상태 유지)
  ↓
WebSocket 연결 시도 (백그라운드)
  ↓
suspect_ids 전송 (선택된 인물 ID 배열)

[4단계: AI 감지 활성화]
사용자 → 우측 제어판 "AI 감지 활성화" 토글 ON
  ↓
프레임 캡처 시작 (0.1초 간격, 10fps)
  ↓
각 프레임을 Base64로 인코딩
  ↓
WebSocket 또는 HTTP로 백엔드 전송
  ↓
백엔드에서 얼굴 감지 및 인식 수행
  ↓
결과 수신 및 UI 업데이트

[5단계: 결과 확인]
실시간 영상에 얼굴 박스 표시
  ↓
범죄자/실종자 감지 시 색상 구분
  ↓
인물 감지 타임라인에 구간 표시 (병합 로직 적용)
  ↓
감지 로그에 항목 추가 (5초 쿨타임)
  ↓
스냅샷 자동 저장 (Base64)
  ↓
클립 자동 추적 (시작/종료 시간)

[6단계: 결과 다운로드]
사용자 → "스냅샷 보기" 버튼 클릭
  ↓
인물별 필터링 탭 표시
  ↓
스냅샷 갤러리 표시
  ↓
선택 다운로드 또는 전체 다운로드

사용자 → "클립 보기" 버튼 클릭
  ↓
감지 구간별 영상 클립 목록 표시
  ↓
선택 다운로드 (ffmpeg로 영상 추출)

사용자 → "CSV 저장" 버튼 클릭
  ↓
감지 로그를 CSV 파일로 다운로드 (한글 BOM 포함)
```

### 1.2 CLI 사용 흐름

```
[1단계: 인물 등록]
python src/face_enroll.py
  ↓
MODE 선택 (1: 기본 등록, 2: 수동 추가)
  ↓
images/enroll/{person_id}/*.jpg 읽기
  ↓
각 이미지에서 얼굴 감지 및 임베딩 추출
  ↓
Bank 생성 (N×512 배열)
  ↓
Centroid 계산 (평균값)
  ↓
outputs/embeddings/{person_id}/ 저장

[2단계: 영상 분석]
python src/face_match_cctv.py
  ↓
입력 파일 지정 (videos/source/{filename})
  ↓
프레임별 처리 루프
  ↓
얼굴 감지 → 임베딩 추출 → Bank 매칭
  ↓
오탐 방지 필터링
  ↓
결과 분류 (Match / Review / Unknown)
  ↓
스냅샷 저장, CSV 로그 기록
  ↓
임베딩 자동 수집 (Dynamic Bank에 추가)
```

---

## 2. 얼굴 인식 프로세스

### 2.1 전체 파이프라인

```
[입력: 비디오 프레임 또는 이미지]
  ↓
[전처리]
  - 저화질 영상 업스케일링 (최소 640px)
  - 샤프닝 적용 (선택적)
  - 스케일 비율 계산 (박스 좌표 변환용)
  ↓
[얼굴 감지]
  - InsightFace RetinaFace 모델 사용
  - det_size=(640, 640)
  - 감지된 얼굴 개수 확인
  ↓
[각 얼굴별 처리]
  FOR each face in faces:
    ↓
    [임베딩 추출]
    - InsightFace buffalo_l 모델 사용
    - 512차원 벡터 추출
    - L2 정규화 (norm=1)
    ↓
    [메타데이터 추출]
    - 얼굴 각도 추정 (yaw, pitch)
    - 화질 추정 (high/medium/low)
    - 바운딩 박스 좌표 (원본 이미지 좌표로 변환)
    ↓
    [Bank 매칭]
    - Base Bank 매칭 (정면, 측면, 마스크 없는 얼굴)
    - Masked Bank 매칭 (마스크 쓴 얼굴)
    - Dynamic Bank 매칭 (CCTV에서 수집한 다양한 각도)
    - 각 Bank에서 최고 유사도 선택
    - 최종 최고 유사도 = max(base_sim, masked_sim, dynamic_sim)
    ↓
    [두 번째 유사도 계산]
    - 모든 인물 중 두 번째로 높은 유사도
    - sim_gap = best_sim - second_sim
    ↓
    [임계값 결정]
    - 화질 기반 기본 임계값 조정
      * high: 0.42
      * medium: 0.40
      * low: 0.38
    - suspect_ids 모드: +0.02 추가
    - 절대 최소값: 0.45 (suspect_ids 모드)
    ↓
    [매칭 판단]
    조건 1: best_sim >= main_threshold
    조건 2: sim_gap >= gap_margin
      * high: 0.12
      * medium: 0.10
      * low: 0.08
    조건 3: second_sim < main_threshold - 0.02
    조건 4: (suspect_ids 모드) best_sim >= 0.45
    ↓
    [오탐 방지 필터링]
    - bbox 기반 다중 매칭 필터링
    - 프레임 간 연속성 체크
    - 중복 얼굴 필터링
    ↓
    [결과 분류]
    - Match: 모든 조건 만족
    - Review: 일부 조건 만족 (의심스러운 경우)
    - Unknown: 조건 불만족
    ↓
    [임베딩 자동 수집]
    - Match 성공 시 해당 얼굴 임베딩 저장
    - 중복 체크 (유사도 >= 0.95면 스킵)
    - Dynamic Bank에 추가
    - 각도 정보와 함께 저장
```

### 2.2 Bank 매칭 상세 로직

```python
# backend/main.py:1226-1280

# 1. Base Bank 매칭
base_sim = 0.0
best_base_person_id = "unknown"
if gallery_base_cache:
    for person_id, base_bank in gallery_base_cache.items():
        if base_bank.ndim == 2:  # (N, 512)
            sims = np.dot(base_bank, face_emb_normalized)  # (N,)
            max_sim = float(np.max(sims))
        else:  # (512,)
            max_sim = float(np.dot(base_bank, face_emb_normalized))
        
        if max_sim > base_sim:
            base_sim = max_sim
            best_base_person_id = person_id

# 2. Masked Bank 매칭
masked_sim = 0.0
best_mask_person_id = "unknown"
mask_prob = estimate_mask_from_similarity(base_sim)
if mask_prob >= MASKED_BANK_MASK_PROB_THRESHOLD:  # 0.5
    if gallery_masked_cache:
        for person_id, masked_bank in gallery_masked_cache.items():
            if masked_bank.ndim == 2:
                sims = np.dot(masked_bank, face_emb_normalized)
                max_sim = float(np.max(sims))
            else:
                max_sim = float(np.dot(masked_bank, face_emb_normalized))
            
            if max_sim > masked_sim:
                masked_sim = max_sim
                best_mask_person_id = person_id

# 3. Dynamic Bank 매칭
dynamic_sim = 0.0
best_dynamic_person_id = "unknown"
if gallery_dynamic_cache:
    for person_id, dynamic_bank in gallery_dynamic_cache.items():
        if dynamic_bank.ndim == 2 and dynamic_bank.shape[0] > 0:
            sims = np.dot(dynamic_bank, face_emb_normalized)
            max_sim = float(np.max(sims))
        else:
            max_sim = 0.0
        
        if max_sim > dynamic_sim:
            dynamic_sim = max_sim
            best_dynamic_person_id = person_id

# 4. 최종 최고 유사도 선택
max_similarity = max(base_sim, masked_sim, dynamic_sim)
best_person_id = best_base_person_id  # 우선순위: Base > Masked > Dynamic
if masked_sim > base_sim:
    best_person_id = best_mask_person_id
if dynamic_sim > max(base_sim, masked_sim):
    best_person_id = best_dynamic_person_id
```

### 2.3 오탐 방지 필터링

```python
# backend/main.py:1282-1350

# 1. bbox 기반 다중 매칭 필터링
# 같은 얼굴 영역에서 여러 인물로 매칭된 경우 처리
for i, face_result in enumerate(face_results):
    for j, other_result in enumerate(face_results):
        if i == j:
            continue
        
        # IoU 및 중심점 거리로 같은 얼굴 영역 판단
        if is_same_face_region(face_result['bbox'], other_result['bbox']):
            # sim_gap이 충분히 크면(>=0.10) 가장 높은 유사도만 인정
            if face_result['sim_gap'] >= 0.10:
                # 다른 결과 제거 또는 검토 대상으로 분리
                other_result['review_reason'] = 'same_face_multiple_persons'

# 2. 프레임 간 연속성 체크
# tracking_state를 사용하여 최근 5프레임 내 같은 인물 매칭 여부 확인
for face_result in face_results:
    track_id = face_result.get('track_id')
    if track_id and track_id in tracking_state['tracks']:
        track = tracking_state['tracks'][track_id]
        # 최근 5프레임 내 같은 인물 매칭 여부
        has_continuity = check_frame_continuity(track, best_person_id)
        
        if not has_continuity and face_result['similarity'] < continuity_threshold:
            face_result['review_reason'] = 'no_continuity'

# 3. sim_gap 체크
# 최고 유사도와 두 번째 유사도의 차이 확인
if sim_gap < gap_margin:
    face_result['review_reason'] = 'ambiguous_match'

# 4. 중복 얼굴 필터링
# 같은 프레임 내 동일 인물 중복 감지 방지
seen_persons = {}
for face_result in face_results:
    person_id = face_result['best_person_id']
    if person_id in seen_persons:
        # 이미 본 인물이면 유사도가 더 높은 것만 유지
        if face_result['similarity'] > seen_persons[person_id]['similarity']:
            seen_persons[person_id] = face_result
        else:
            face_result['review_reason'] = 'duplicate_face'
    else:
        seen_persons[person_id] = face_result
```

---

## 3. 하이퍼파라미터

### 3.1 기본 임계값

| 파라미터 | 값 | 위치 | 설명 |
|----------|-----|------|------|
| `BASE_THRESH` | 0.32 | `src/face_match_cctv.py:606` | 기본 임계값 (조정 전) |
| `BANK_DUPLICATE_THRESHOLD` | 0.95 | `backend/main.py:1416` | Bank 중복 체크 (유사도 >= 0.95면 중복) |

### 3.2 화질별 임계값

| 화질 | main_threshold | gap_margin | 조건 |
|------|----------------|------------|------|
| **high** | **0.42** | **0.12** | 얼굴 너비≥150px AND 비율≥5% |
| **medium** | **0.40** | **0.10** | 얼굴 너비≥100px AND 비율≥2% |
| **low** | **0.38** | **0.08** | 그 외 |

**위치**: `backend/main.py:1353-1361`

### 3.3 suspect_ids 모드 강화

```python
# backend/main.py:1363-1366
if suspect_ids:
    main_threshold += 0.02  # 임계값 상향 (더 보수적으로)
    gap_margin += 0.03      # Gap 기준 강화
    min_absolute = 0.45      # 절대 최소값
```

| 화질 | 일반 모드 threshold | suspect_ids 모드 threshold |
|------|---------------------|----------------------------|
| **high** | 0.42 | **0.44** |
| **medium** | 0.40 | **0.42** |
| **low** | 0.38 | **0.40** |

### 3.4 마스크 관련 파라미터

| 파라미터 | 값 | 위치 | 설명 |
|----------|-----|------|------|
| `MASKED_BANK_MASK_PROB_THRESHOLD` | 0.5 | `backend/main.py:54` | mask_prob >= 0.5이면 masked bank로 분류 |
| `MASKED_CANDIDATE_MIN_SIM` | 0.25 | `backend/main.py:55` | base_sim >= 0.25 이상이어야 masked candidate로 판단 |
| `MASKED_CANDIDATE_MIN_FRAMES` | 3 | `backend/main.py:56` | 연속 N 프레임 이상 조건 충족 시 masked bank에 추가 |
| `MASKED_TRACKING_IOU_THRESHOLD` | 0.5 | `backend/main.py:57` | bbox tracking을 위한 IoU 임계값 |

**주의**: 현재 `mask_prob`는 threshold 조정에 사용되지 않고 메타데이터/로그용으로만 사용됩니다. (`src/utils/mask_detector.py:113-138`)

### 3.5 얼굴 필터링 파라미터

| 파라미터 | 값 | 위치 | 설명 |
|----------|-----|------|------|
| `is_same_face_region` IoU | 0.3 | `src/face_match_cctv.py:355` | 같은 얼굴 영역 판단 IoU 임계값 |
| `distance_threshold` | `face_diag * 0.6` | `src/face_match_cctv.py:380` | 중심점 거리 임계값 (대각선의 60%) |
| `min_gap` | 0.05 | `src/face_match_cctv.py:423` | 최소 sim_gap (5% 이상 차이 필요) |

### 3.6 매칭 조건 요약

**매칭 성공 조건 (모두 만족해야 함):**
1. `max_similarity >= main_threshold` (화질별 임계값)
2. `sim_gap >= gap_margin` (최고 유사도와 두 번째 유사도의 차이)
3. `second_similarity < main_threshold - 0.02` (두 번째 후보가 너무 높으면 미매칭)
4. (suspect_ids 모드) `max_similarity >= 0.45` (절대값 최소 기준)

---

## 4. 핵심 로직 상세

### 4.1 화질 추정 로직

```python
# src/utils/mask_detector.py:140-177

def estimate_face_quality(face_bbox, img_shape):
    face_width = bbox[2] - bbox[0]
    face_height = bbox[3] - bbox[1]
    face_size = max(face_width, face_height)
    face_ratio = (face_width * face_height) / (img_width * img_height)
    
    if face_size >= 150 and face_ratio >= 0.05:
        return "high"
    elif face_size >= 100 and face_ratio >= 0.02:
        return "medium"
    else:
        return "low"
```

### 4.2 얼굴 각도 추정 로직

```python
# src/utils/face_angle_detector.py:9-101

def estimate_face_angle(face):
    # InsightFace 랜드마크 사용 (5개 포인트)
    kps = face.kps  # (5, 2): [왼쪽 눈, 오른쪽 눈, 코, 왼쪽 입꼬리, 오른쪽 입꼬리]
    
    # Yaw 각도 추정 (좌우 회전)
    eye_center_x = (left_eye[0] + right_eye[0]) / 2
    nose_offset_x = nose[0] - eye_center_x
    eye_distance = sqrt((right_eye[0] - left_eye[0])^2 + (right_eye[1] - left_eye[1])^2)
    normalized_offset = nose_offset_x / eye_distance
    yaw_angle = normalized_offset * 90.0  # -90 ~ 90도
    
    # Pitch 각도 추정 (상하 회전)
    eye_to_mouth_distance = abs(mouth_center_y - eye_center_y)
    eye_to_nose_distance = abs(nose[1] - eye_center_y)
    nose_ratio = eye_to_nose_distance / eye_to_mouth_distance
    pitch_angle = (0.5 - nose_ratio) * 90.0  # -90 ~ 90도
    
    # 각도 타입 분류
    if pitch_angle > 15:
        return "top", yaw_angle
    elif abs(yaw_angle) < 15:
        return "front", yaw_angle
    elif yaw_angle < -45:
        return "left_profile", yaw_angle
    elif yaw_angle > 45:
        return "right_profile", yaw_angle
    elif yaw_angle < 0:
        return "left", yaw_angle
    else:
        return "right", yaw_angle
```

### 4.3 임베딩 자동 수집 로직

```python
# backend/main.py:1400-1500

# 매칭 성공 시 Dynamic Bank에 임베딩 추가
if is_match and best_person_id != "unknown":
    # 중복 체크
    if best_person_id in gallery_dynamic_cache:
        dynamic_bank = gallery_dynamic_cache[best_person_id]
        if dynamic_bank.ndim == 2 and dynamic_bank.shape[0] > 0:
            # 기존 임베딩과 유사도 계산
            similarities = np.dot(dynamic_bank, face_emb_normalized)
            max_sim = float(np.max(similarities))
            
            # 중복 체크 (유사도 >= 0.95면 스킵)
            if max_sim >= BANK_DUPLICATE_THRESHOLD:  # 0.95
                continue  # 중복이므로 스킵
    
    # 각도 정보 추출
    angle_type, yaw_angle = estimate_face_angle(face)
    
    # 다양한 각도 수집 확인
    if is_diverse_angle(collected_angles, angle_type):
        # Dynamic Bank에 추가
        if best_person_id not in gallery_dynamic_cache:
            gallery_dynamic_cache[best_person_id] = np.array([face_emb_normalized])
        else:
            gallery_dynamic_cache[best_person_id] = np.vstack([
                gallery_dynamic_cache[best_person_id],
                face_emb_normalized
            ])
        
        # 각도 정보 저장
        collected_angles.append(angle_type)
        
        # 파일 저장 (비동기)
        save_embeddings_async(best_person_id, face_emb_normalized, angle_type)
```

### 4.4 Temporal Consistency 필터

```python
# backend/main.py:1060-1147

def apply_temporal_filter(websocket, result):
    """
    프레임 간 연속성을 체크하여 일시적 오탐 제거
    """
    connection_id = id(websocket)
    if connection_id not in connection_states:
        return result
    
    tracking_state = connection_states[connection_id].get("tracking_state", {})
    
    for detection in result.get("detections", []):
        track_id = detection.get("track_id")
        person_id = detection.get("person_id")
        
        if track_id and track_id in tracking_state["tracks"]:
            track = tracking_state["tracks"][track_id]
            
            # 최근 5프레임 내 같은 인물 매칭 여부 확인
            recent_matches = track.get("recent_person_ids", [])
            has_continuity = person_id in recent_matches[-5:]
            
            if not has_continuity:
                # 연속성이 없으면 검토 대상으로 분리
                detection["review_reason"] = "no_continuity"
                detection["is_match"] = False
    
    return result
```

---

## 5. 데이터 흐름

### 5.1 웹 UI → 백엔드

```
[프론트엔드]
비디오 프레임 캡처 (Canvas API)
  ↓
Base64 인코딩
  ↓
WebSocket 또는 HTTP POST 전송
  ↓
{
  "type": "detect",
  "image": "data:image/jpeg;base64,...",
  "suspect_ids": ["person1", "person2"],
  "frame_id": 123
}

[백엔드]
Base64 디코딩 → numpy array (BGR)
  ↓
process_detection() 호출
  ↓
얼굴 감지 및 인식 수행
  ↓
결과 반환
  ↓
{
  "type": "detection",
  "alert": true/false,
  "detections": [
    {
      "bbox": [x1, y1, x2, y2],
      "person_id": "person1",
      "name": "홍길동",
      "similarity": 0.85,
      "status": "criminal",
      "metadata": {...}
    }
  ],
  "snapshot_base64": "data:image/jpeg;base64,...",
  "video_timestamp": 12.5
}

[프론트엔드]
결과 수신
  ↓
UI 업데이트 (박스 그리기, 로그 추가, 타임라인 업데이트)
```

### 5.2 임베딩 저장 흐름

```
[인물 등록]
images/enroll/{person_id}/*.jpg
  ↓
face_enroll.py 실행
  ↓
각 이미지에서 얼굴 감지 및 임베딩 추출
  ↓
Bank 생성 (N×512 배열)
  ↓
outputs/embeddings/{person_id}/bank_base.npy 저장

[영상 분석 중 자동 수집]
CCTV 영상 프레임
  ↓
얼굴 감지 및 매칭 성공
  ↓
임베딩 추출
  ↓
중복 체크 (유사도 >= 0.95면 스킵)
  ↓
각도 정보 추출
  ↓
Dynamic Bank에 추가
  ↓
outputs/embeddings/{person_id}/bank_dynamic.npy 저장
  ↓
메모리 캐시 즉시 갱신 (다음 프레임부터 반영)
```

### 5.3 데이터베이스 흐름

```
[초기화]
python backend/init_db.py
  ↓
outputs/embeddings 또는 backend/database/*.json 읽기
  ↓
PostgreSQL persons 테이블에 저장
  ↓
메모리 캐시 갱신

[실시간 감지]
프레임 처리
  ↓
매칭 결과 생성
  ↓
PostgreSQL detection_logs 테이블에 저장
  ↓
{
  "person_id": "person1",
  "person_name": "홍길동",
  "similarity": 0.85,
  "is_criminal": true,
  "status": "criminal",
  "detection_metadata": {
    "bbox": [x1, y1, x2, y2],
    "angle_type": "front",
    "face_quality": "high",
    "mask_prob": 0.0
  }
}
```

---

## 6. 의존성 및 연결 관계

### 6.1 모듈 의존성

```
backend/main.py
  ├─ src/utils/device_config.py (GPU/CPU 설정)
  ├─ src/utils/gallery_loader.py (갤러리 로드 및 매칭)
  ├─ src/utils/face_angle_detector.py (얼굴 각도 추정)
  ├─ src/utils/mask_detector.py (화질 추정, 임계값 조정)
  ├─ src/face_enroll.py (임베딩 저장)
  └─ backend/database.py (PostgreSQL 연동)

src/face_match_cctv.py
  ├─ src/utils/gallery_loader.py
  ├─ src/utils/face_angle_detector.py
  └─ src/utils/mask_detector.py

web/script.js
  ├─ WebSocket 연결 (ws:// 또는 wss://)
  ├─ HTTP API 호출 (/api/detect, /api/persons)
  └─ snapshot_functions.js (타임라인, 스냅샷 관리)
```

### 6.2 데이터 소스 우선순위

```
[백엔드 갤러리 로드]
1. PostgreSQL persons 테이블 (우선순위 1)
   ↓ 실패 시
2. outputs/embeddings/{person_id}/bank*.npy (우선순위 2)
   ↓ 실패 시
3. backend/database/*.json (우선순위 3, 레거시)

[Bank 우선순위]
1. bank_base.npy (정면, 측면, 마스크 없는 얼굴)
2. bank_dynamic.npy (CCTV에서 수집한 다양한 각도)
3. bank.npy (Legacy 호환)
4. centroid*.npy (Bank가 없을 때만 사용)
```

### 6.3 WebSocket vs HTTP

```
[WebSocket 모드 (기본)]
프론트엔드 → WebSocket 연결 시도
  ↓
백엔드 → 연결 수락
  ↓
프론트엔드 → suspect_ids 전송
  ↓
백엔드 → 설정 완료 확인
  ↓
프론트엔드 → 프레임 전송 (0.1초 간격)
  ↓
백엔드 → 실시간 결과 반환
  ↓
양방향 통신 유지

[HTTP 모드 (폴백)]
WebSocket 연결 실패 시
  ↓
프론트엔드 → HTTP POST /api/detect (0.1초 간격)
  ↓
백엔드 → 결과 반환
  ↓
단방향 통신 (폴링 방식)
```

---

## 7. 잠재적 문제점 및 개선 방향

### 7.1 현재 로직의 문제점

#### 7.1.1 마스크 감지 로직
- **문제**: 유사도 기반으로 마스크 가능성을 추정하는 것은 부정확함
- **원인**: 유사도가 낮은 이유가 마스크 때문이 아니라 완전히 다른 사람일 수 있음
- **현재 상태**: `mask_prob`는 메타데이터/로그용으로만 사용, threshold 조정에는 사용 안 함
- **개선 방향**: 랜드마크 기반 occlusion 판단으로 대체 필요 (`src/utils/mask_detector.py:6-36` 참고)

#### 7.1.2 임계값 조정 로직
- **문제**: 화질 기반 임계값 조정이 단순함
- **현재 상태**: 화질만으로 임계값 결정 (high: +0.03, medium: 0, low: -0.02)
- **개선 방향**: 얼굴 크기, 해상도, 조명 조건 등을 종합적으로 고려

#### 7.1.3 Bank 매칭 우선순위
- **문제**: Base, Masked, Dynamic Bank 중 어떤 것을 우선할지 명확하지 않음
- **현재 상태**: 최고 유사도만 선택 (우선순위: Base > Masked > Dynamic)
- **개선 방향**: 각 Bank의 신뢰도나 컨텍스트를 고려한 가중치 적용

#### 7.1.4 Temporal Consistency 필터
- **문제**: 최근 5프레임만 체크하는 것이 충분한지 불명확
- **현재 상태**: 고정된 프레임 수 (5프레임)
- **개선 방향**: 동적 프레임 수 조정 (영상 속도, 프레임레이트 고려)

### 7.2 데이터 일관성 문제

#### 7.2.1 메모리 캐시 vs 파일 시스템
- **문제**: Dynamic Bank에 임베딩 추가 시 메모리 캐시는 즉시 갱신되지만, 파일 저장은 비동기
- **영향**: 서버 재시작 시 최신 임베딩이 손실될 수 있음
- **개선 방향**: 동기 저장 또는 트랜잭션 로깅

#### 7.2.2 PostgreSQL vs 파일 시스템
- **문제**: 두 데이터 소스 간 동기화 문제
- **현재 상태**: PostgreSQL 우선, 실패 시 파일 시스템 사용
- **개선 방향**: 단일 소스로 통일 또는 자동 동기화 메커니즘

### 7.3 성능 최적화 필요 영역

#### 7.3.1 프레임 처리 속도
- **현재**: 0.1초 간격 (10fps)
- **병목**: InsightFace 모델 추론 시간
- **개선 방향**: 배치 처리, 모델 최적화, GPU 활용 최대화

#### 7.3.2 Bank 크기 관리
- **문제**: Dynamic Bank가 계속 커지면 매칭 속도 저하
- **현재 상태**: 중복 체크만 수행, 크기 제한 없음
- **개선 방향**: 최대 크기 제한, 오래된 임베딩 제거, 샘플링 전략

### 7.4 로직 검증 필요 사항

#### 7.4.1 suspect_ids 모드 로직
- **검증 필요**: suspect_ids 모드에서 임계값을 +0.02 올리는 것이 적절한지
- **의문**: 더 엄격하게 하면 오히려 인식률이 떨어질 수 있음
- **테스트 필요**: 실제 데이터로 False Positive/False Negative 측정

#### 7.4.2 sim_gap 기준
- **검증 필요**: gap_margin (0.08~0.12)이 적절한지
- **의문**: 너무 엄격하면 유사한 사람을 구분하지 못할 수 있음
- **테스트 필요**: 유사한 사람 쌍에 대한 sim_gap 분포 분석

#### 7.4.3 화질 추정 정확도
- **검증 필요**: 얼굴 크기와 비율만으로 화질을 정확히 추정할 수 있는지
- **의문**: 조명, 노이즈, 블러 등도 화질에 영향을 줌
- **개선 방향**: 더 정교한 화질 평가 지표 도입

### 7.5 핵심 문제점: 오인식 확산 및 임베딩 오염 (치명적 약점)

#### 7.5.1 문제 1: 방어력 없는 Dynamic Bank 업데이트 로직

**위치**: `backend/main.py:1657-1672` (임베딩 자동 수집 로직)

**현재 로직**:
```python
# 매칭 성공(is_match=True)하고 각도가 다양하면
if AUTO_ADD_TO_DYNAMIC_BANK:
    # 중복 체크(>=0.95)와 각도 다양성만 확인
    learning_events.append({
        "person_id": person_id,
        "embedding": embedding_normalized.tolist(),
        "bank_type": "dynamic"
    })
```

**문제점**:
1. **오인식 확산**: 한 번이라도 오인식(False Positive)이 발생하여 잘못된 사람과 매칭 성공 판정을 내리면, 그 잘못된 임베딩이 Dynamic Bank에 영구적으로 저장됩니다.
2. **오염된 임베딩의 영구적 영향**: 이후에는 이 '오염된 임베딩' 때문에 계속해서 오인식이 발생하는 악순환이 반복됩니다.
3. **검증 부재**: 
   - Base Bank와의 최소 유사도 검증 없음
   - 화질 검증이 약함 (현재는 얼굴 크기만 확인)
   - Occlusion(가림) 검증 없음 (마스크나 손으로 가린 상태도 저장됨)

**영향**:
- 이미지 속 상황처럼 Base Bank(정면)와는 전혀 닮지 않았는데, 오염된 Dynamic Bank(측면) 데이터만 믿고 79% 유사도가 나오는 상황 발생
- 시스템이 한 번 오인식을 하면 그 오인식이 계속 확산됨

**개선 방안 (1단계: 가장 시급한 조치)**:
1. **Base Bank와의 최소 유사도 검증 추가**:
   - 새로 들어온 임베딩이 현재 매칭된 인물의 Base Bank와 비교했을 때 최소한의 유사도(예: 0.6 이상)는 넘어야 Dynamic Bank에 추가 허용
   - 목적: 정면 얼굴(Base)과 너무 다르게 생긴 임베딩이 들어오는 것을 막음

2. **고화질 검증 강화**:
   - 현재 화질 추정(`estimate_face_quality`)에서 확실한 "high" 등급일 때만 수집
   - 얼굴 크기 최소 200px 이상 등 기준 상향 필요

3. **Occlusion 없는 상태 검증**:
   - 현재의 유사도 기반 마스크 추정은 부정확함 (7.1.1절 문제점 참고)
   - 랜드마크 기반 Occlusion 판단 도입: 주요 랜드마크(눈, 코, 입)가 모두 선명하게 보일 때만 Dynamic Bank에 추가
   - 마스크나 손으로 가린 상태의 임베딩은 실시간 매칭에만 쓰고 저장하지 않음

**예상 효과**:
- "쓰레기가 들어오면 쓰레기가 나간다(Garbage In, Garbage Out)" 문제 해결
- 오인식으로 인한 임베딩 오염 방지

---

#### 7.5.2 문제 2: "승자 독식" 방식의 Bank 매칭 로직

**위치**: `backend/main.py:1273-1285` (Bank 매칭 상세 로직)

**현재 로직**:
```python
# Base, Masked, Dynamic Bank 각각 매칭 후
max_similarity = max(base_sim, masked_sim, dynamic_sim)
best_person_id = best_base_person_id  # 우선순위: Base > Masked > Dynamic
if dynamic_sim > max(base_sim, masked_sim):
    best_person_id = best_dynamic_person_id
```

**문제점**:
1. **신뢰도 무시**: Base Bank(등록된 정면 사진)와는 전혀 닮지 않았더라도(예: 유사도 0.2), 오염된 Dynamic Bank에 우연히 닮은 임베딩이 있어 높은 점수(예: 0.79)가 나오면, 시스템은 최종적으로 0.79라고 판단해버립니다.
2. **데이터 소스 동등 취급**: 신뢰도가 다른 데이터 소스를 동등하게 취급하는 것이 문제입니다.
   - Base Bank: 가장 신뢰 (등록된 정면 사진)
   - Dynamic Bank: 중간 신뢰 (오염 가능성 있음)
   - Masked Bank: 낮은 신뢰 (불확실성 높음)

**영향**:
- Base Bank와는 안 닮았는데 오염된 Dynamic Bank 데이터만 믿고 높은 유사도가 나오는 상황
- 오인식 확산의 직접적 원인

**개선 방안 (2단계: 핵심 로직 변경)**:
1. **가중치 기반 매칭 (Weighted Voting)**:
   - `max()` 함수를 신뢰도 기반 가중치 합산으로 변경
   - 각 Bank의 신뢰도 가중치 설정:
     - `W_BASE = 1.0` (가장 신뢰)
     - `W_DYNAMIC = 0.8` (중간 신뢰, 오염 가능성 있음)
     - `W_MASKED = 0.6` (낮은 신뢰, 불확실성 높음)

2. **Base 점수 기준 보정**:
   ```python
   # Base 점수가 너무 낮으면 다른 Bank 점수가 아무리 높아도 신뢰하지 않음
   if base_sim < 0.4:
       # Base와 너무 다르면 Dynamic/Masked 점수를 크게 깎아서 반영
       confident_dynamic = dynamic_sim * 0.7
       confident_masked = masked_sim * 0.6
       final_sim = max(base_sim, confident_dynamic, confident_masked)
   else:
       # Base와 어느 정도 닮았다면, 다른 Bank의 높은 점수를 인정해주되 가중치 적용
       final_sim = max(
           base_sim * W_BASE,
           dynamic_sim * W_DYNAMIC,
           masked_sim * W_MASKED
       )
   final_sim = min(final_sim, 1.0)  # 최종 유사도는 1.0을 넘을 수 없음
   ```

**예상 효과**:
- Base Bank와는 안 닮았는데 오염된 Dynamic Bank 데이터만 믿고 높은 점수가 나오는 상황 방지
- Base 점수가 낮으면 최종 점수도 강제로 낮아짐

---

#### 7.5.3 문제 3: 오염된 Bank 정화 메커니즘 부재

**현재 상태**:
- Dynamic Bank에 이미 오염된 데이터가 들어간 경우, 이를 정리하는 메커니즘이 없음
- Bank 크기 관리도 없어서 오래된 오염 데이터가 계속 남아있음

**개선 방안 (3단계: 유지보수)**:
1. **전수 검사 스크립트 실행**:
   - 현재 저장된 모든 `bank_dynamic.npy` 파일을 로드
   - Dynamic Bank에 있는 각 임베딩을 해당 인물의 Base Bank와 다시 비교
   - Base Bank와의 유사도가 특정 기준(예: 0.5) 이하인 임베딩은 오염된 것으로 간주하고 삭제

2. **각도별 최대 개수 제한**:
   - 특정 각도(예: 오른쪽 프로필)에 너무 많은 임베딩이 쌓이지 않도록, 각도별로 가장 품질이 좋은 상위 N개(예: 5개)만 남기고 나머지는 삭제하는 로직 추가

3. **정기적 정화 스케줄**:
   - 주기적으로(예: 매주) Dynamic Bank를 검사하여 오염 데이터 제거

**예상 효과**:
- 이미 오염된 현재의 Dynamic Bank 데이터 정리
- 오염 데이터의 재발 방지

---

### 7.6 개선 실행 계획 (우선순위별)

#### 1단계: 가장 시급한 조치 - Dynamic Bank 입력 필터 강화 (Hygiene Check)
- **목표**: "쓰레기가 들어오면 쓰레기가 나간다" 문제 해결
- **수정 대상**: `backend/main.py:1657-1672` (임베딩 자동 수집 로직)
- **적용 로직**:
  1. Base Bank와의 최소 유사도 검증 (>= 0.6)
  2. 고화질 검증 강화 (얼굴 크기 >= 200px)
  3. Occlusion 없는 상태 검증 (랜드마크 기반)
- **예상 소요 시간**: 2-3시간
- **위험도**: 낮음 (기존 로직에 검증만 추가)

#### 2단계: 핵심 로직 변경 - 가중치 기반 매칭 (Weighted Voting)
- **목표**: "승자 독식" 방식 개선
- **수정 대상**: `backend/main.py:1273-1285` (Bank 매칭 상세 로직)
- **적용 로직**:
  1. 가중치 기반 최종 유사도 계산
  2. Base 점수 기준 보정 로직 추가
- **예상 소요 시간**: 3-4시간
- **위험도**: 중간 (핵심 로직 변경이므로 테스트 필요)

#### 3단계: 유지보수 - 오염된 Bank 정화 (Cleanup)
- **목표**: 이미 오염된 현재의 Dynamic Bank 데이터 정리
- **수정 대상**: 별도 유지보수 스크립트 생성
- **적용 로직**:
  1. 전수 검사 스크립트 실행
  2. Base Bank와 역검증하여 오염 데이터 삭제
  3. 각도별 최대 개수 제한 로직 추가
- **예상 소요 시간**: 4-5시간
- **위험도**: 낮음 (별도 스크립트이므로 기존 로직에 영향 없음)

**전체 예상 소요 시간**: 9-12시간
**권장 실행 순서**: 1단계 → 테스트 → 2단계 → 테스트 → 3단계

---

## 부록: 주요 함수 호출 체인

### A.1 웹 UI 감지 흐름

```
UI.video (비디오 재생)
  ↓
setInterval(processRealtimeDetection, 100)  // 0.1초마다
  ↓
captureVideoFrame()  // Canvas에서 프레임 캡처
  ↓
toDataURL('image/jpeg')  // Base64 인코딩
  ↓
WebSocket.send() 또는 fetch('/api/detect')
  ↓
handleWebSocketMessage() 또는 handleHTTPResponse()
  ↓
drawDetections()  // 박스 그리기
  ↓
addDetectionLogItem()  // 로그 추가
  ↓
addTimelineMarkerDirect()  // 타임라인 업데이트
  ↓
renderTimelineWithMerging()  // 타임라인 병합 및 렌더링
```

### A.2 백엔드 감지 흐름

```
WebSocket.receive() 또는 POST /api/detect
  ↓
base64_to_image()  // Base64 디코딩
  ↓
process_detection()  // 공통 감지 로직
  ↓
preprocess_image_for_detection()  // 전처리
  ↓
model.get()  // InsightFace 얼굴 감지
  ↓
FOR each face:
    estimate_face_angle()  // 각도 추정
    estimate_face_quality()  // 화질 추정
    match_with_bank_detailed()  // Bank 매칭
    get_adjusted_threshold()  // 임계값 조정
    오탐 방지 필터링
    임베딩 자동 수집
  ↓
apply_temporal_filter()  // Temporal Consistency
  ↓
log_detection()  // PostgreSQL 저장
  ↓
결과 반환
```

---

**작성일**: 2024년
**버전**: 1.0
**작성자**: FaceWatch 개발팀

