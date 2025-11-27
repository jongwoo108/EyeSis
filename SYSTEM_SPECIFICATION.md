# FaceWatch 시스템 전체 명세서

## 📑 목차

1. [시스템 개요](#1-시스템-개요)
2. [아키텍처](#2-아키텍처)
3. [핵심 기능](#3-핵심-기능)
4. [처리 플로우](#4-처리-플로우)
5. [임계값 및 파라미터](#5-임계값-및-파라미터)
6. [임베딩 시스템](#6-임베딩-시스템)
7. [매칭 및 판정 로직](#7-매칭-및-판정-로직)
8. [오탐 방지 메커니즘](#8-오탐-방지-메커니즘)
9. [API 명세](#9-api-명세)
10. [데이터 구조](#10-데이터-구조)

---

## 1. 시스템 개요

### 1.1 시스템 목적
FaceWatch는 CCTV 영상, 이미지, 실시간 스트림에서 특정 인물을 자동으로 식별하고 추적하는 엔터프라이즈급 얼굴 인식 시스템입니다.

### 1.2 핵심 기술
- **얼굴 인식 모델**: InsightFace buffalo_l (512차원 임베딩)
- **얼굴 검출**: RetinaFace (InsightFace 내장)
- **매칭 방법**: 코사인 유사도 (Cosine Similarity)
- **통신 프로토콜**: WebSocket (실시간) + HTTP (폴백)
- **데이터베이스**: PostgreSQL (메인) + Bank 임베딩 시스템

### 1.3 주요 특징
- ✅ 실시간 처리 (50-150ms 지연)
- ✅ 다양한 각도 지원 (정면 ~ 프로필)
- ✅ 마스크 착용자 인식
- ✅ 자동 학습 (Dynamic Bank)
- ✅ 3단계 결과 분류 (Match / Review / Unknown)

---

## 2. 아키텍처

### 2.1 시스템 계층 구조

```
┌─────────────────────────────────────────────────────────┐
│                  클라이언트 레이어                        │
│  • HTML5 Video Player                                   │
│  • Canvas API (박스 렌더링)                              │
│  • WebSocket Client                                     │
└──────────────────┬──────────────────────────────────────┘
                   │ WebSocket / HTTP
                   │ (Base64 인코딩 이미지 + 메타데이터)
                   ▼
┌─────────────────────────────────────────────────────────┐
│              애플리케이션 레이어 (FastAPI)                │
│  ┌─────────────────────────────────────────────────┐   │
│  │ WebSocket 엔드포인트 (/ws/detect)              │   │
│  │ HTTP API (/api/detect, /api/persons 등)        │   │
│  └─────────────────────────────────────────────────┘   │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│                   AI 추론 레이어                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │ InsightFace (buffalo_l)                        │   │
│  │ • 얼굴 검출 (RetinaFace)                        │   │
│  │ • 임베딩 추출 (512차원)                          │   │
│  │ • 랜드마크 추출 (각도 계산용)                    │   │
│  └─────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────┐   │
│  │ 매칭 엔진                                        │   │
│  │ • Bank 매칭 (코사인 유사도)                      │   │
│  │ • 적응형 임계값 적용                             │   │
│  │ • 오탐 필터링                                    │   │
│  └─────────────────────────────────────────────────┘   │
└──────────────────┬──────────────────────────────────────┘
                   │
      ┌────────────┼────────────┐
      │            │            │
      ▼            ▼            ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│PostgreSQL│ │  Bank    │ │   JSON   │
│  (주)    │ │임베딩(보조)│ │(레거시) │
└──────────┘ └──────────┘ └──────────┘
```

### 2.2 디렉토리 구조

```
FaceWatch/
├── backend/                    # FastAPI 백엔드
│   ├── main.py                 # 메인 서버 (2760줄)
│   ├── database.py             # PostgreSQL ORM
│   └── init_db.py              # DB 초기화
│
├── web/                        # 프론트엔드
│   ├── index.html              # 메인 UI
│   ├── script.js               # WebSocket 클라이언트
│   ├── snapshot_functions.js  # 스냅샷 관리
│   └── style.css               # 스타일
│
├── src/                        # 코어 로직
│   ├── face_enroll.py          # 인물 등록
│   ├── face_match_cctv.py     # CCTV 분석
│   └── utils/                  # 유틸리티
│       ├── gallery_loader.py   # Bank 로더
│       ├── face_angle_detector.py # 각도 감지
│       ├── mask_detector.py    # 마스크 감지
│       └── device_config.py    # GPU/CPU 설정
│
└── outputs/                    # 출력
    ├── embeddings/             # Bank 임베딩
    │   └── <person_id>/
    │       ├── bank_base.npy      # Base Bank
    │       ├── bank_masked.npy    # Masked Bank
    │       ├── bank_dynamic.npy   # Dynamic Bank
    │       ├── centroid.npy       # Centroid
    │       └── angles_dynamic.json # 각도 메타데이터
    └── results/                # 분석 결과
```

---

## 3. 핵심 기능

### 3.1 얼굴 감지 및 인식

#### 3.1.1 감지 프로세스
1. **얼굴 검출**: RetinaFace로 bbox 추출
2. **임베딩 추출**: buffalo_l 모델로 512차원 벡터 생성
3. **L2 정규화**: 코사인 유사도 계산을 위한 정규화
4. **각도 추정**: 랜드마크 기반 Yaw 각도 계산
5. **화질 추정**: 얼굴 크기/이미지 크기 비율로 자동 판단

#### 3.1.2 매칭 프로세스
1. **Bank 매칭**: Base, Masked, Dynamic Bank와 비교
2. **최고 유사도**: 3개 Bank 중 가장 높은 유사도 선택
3. **2위 유사도**: sim_gap 계산용
4. **임계값 판정**: 화질, 마스크, suspect_ids 모드 고려
5. **결과 분류**: Match / Review / Unknown

### 3.2 임베딩 Bank 시스템

#### 3.2.1 Bank 종류

| Bank 타입 | 용도 | 수집 방법 | 용량 |
|-----------|------|-----------|------|
| **Base Bank** | 기본 인식용 | 수동 등록 (face_enroll.py) | 정적 |
| **Masked Bank** | 마스크 착용자 인식 | 자동 수집 (연속 3프레임) | 동적 |
| **Dynamic Bank** | 다양한 각도 인식 | 자동 수집 (매칭 성공 시) | 동적 |
| **Centroid** | 평균 임베딩 | 자동 계산 (Bank 평균) | 정적 |

#### 3.2.2 Bank 로딩 우선순위
```python
# backend/main.py: load_persons_from_db()
1순위: bank_dynamic.npy  # → gallery_dynamic_cache
2순위: bank_base.npy     # → gallery_base_cache  
3순위: bank_masked.npy   # → gallery_masked_cache
4순위: centroid.npy      # fallback
5순위: DB 임베딩         # final fallback
```

#### 3.2.3 Dynamic Bank 자동 수집
- **수집 조건**:
  - 매칭 성공 시 (`is_match == True`)
  - 중복 체크: `유사도 < 0.95`
  - 각도 다양성: 같은 각도 제한 (front: 1개, left: 3개, right: 3개 등)
  - 수집 완료 전: `is_all_angles_collected() == False`

- **수집 완료 기준**:
  - front: 최소 1개
  - left: 최소 1개
  - right: 최소 1개
  - top: 최소 1개

### 3.3 각도 감지

#### 3.3.1 각도 분류 (5가지)

| 각도 타입 | Yaw 범위 | 설명 |
|-----------|----------|------|
| `front` | -15° ~ +15° | 정면 |
| `left` | -45° ~ -15° | 약간 왼쪽 |
| `right` | +15° ~ +45° | 약간 오른쪽 |
| `left_profile` | < -45° | 왼쪽 프로필 |
| `right_profile` | > +45° | 오른쪽 프로필 |

#### 3.3.2 Yaw 계산 방법
```python
# src/utils/face_angle_detector.py
def estimate_face_angle(face):
    # 랜드마크 기반 Yaw 계산
    left_eye = face.kps[0]   # 좌측 눈
    right_eye = face.kps[1]  # 우측 눈
    nose = face.kps[2]       # 코
    
    # 눈 중심점
    eye_center = (left_eye + right_eye) / 2
    
    # 코와 눈 중심점의 거리 비율로 Yaw 추정
    # ...
    return angle_type, yaw_angle
```

### 3.4 마스크 감지

#### 3.4.1 마스크 확률 추정

| base_sim | mask_prob | 설명 |
|----------|-----------|------|
| < 0.25 | 0.9 | 매우 높은 마스크 가능성 |
| 0.25 ~ 0.28 | 0.7 | 높은 마스크 가능성 |
| 0.28 ~ 0.32 | 0.5 | 중간 마스크 가능성 |
| 0.32 ~ 0.35 | 0.3 | 낮은 마스크 가능성 |
| >= 0.35 | 0.0 | 마스크 아님 |

#### 3.4.2 Masked Bank 수집 조건
```python
# backend/main.py:1376-1387
조건 1: base_sim < main_threshold  # Base Bank로는 매칭 실패
조건 2: base_sim >= 0.25           # 완전 다른 사람은 아님
조건 3: mask_prob >= 0.5           # 마스크 가능성 높음
조건 4: 연속 3프레임 이상 조건 충족
```

### 3.5 화질 감지

#### 3.5.1 화질 분류 (3단계)

```python
# src/utils/mask_detector.py
def estimate_face_quality(bbox, image_size):
    face_width = bbox[2] - bbox[0]
    face_height = bbox[3] - bbox[1]
    img_h, img_w = image_size
    
    # 얼굴 크기 비율
    face_ratio = (face_width * face_height) / (img_w * img_h)
    
    # 분류
    if face_width >= 150 and face_ratio >= 0.05:
        return "high"    # 고화질
    elif face_width >= 100 and face_ratio >= 0.02:
        return "medium"  # 중화질
    else:
        return "low"     # 저화질
```

---

## 4. 처리 플로우

### 4.1 실시간 처리 플로우 (WebSocket)

```
[클라이언트]
  ↓
1. 비디오 프레임 캡처 (requestAnimationFrame)
  ↓
2. Canvas로 Base64 인코딩
  ↓
3. WebSocket 전송
   {
     type: "frame",
     data: {
       image: "data:image/jpeg;base64,...",
       suspect_ids: ["yh", "ja"],
       frame_id: 123
     }
   }
  ↓
[서버: backend/main.py]
  ↓
4. 이미지 디코딩 (base64 → numpy)
  ↓
5. 전처리 (저화질 영상은 업스케일링 + 샤프닝)
  ↓
6. InsightFace 처리
   a. 얼굴 감지 (RetinaFace)
   b. 임베딩 추출 (buffalo_l)
   c. 랜드마크 추출
  ↓
7. 각도 및 화질 추정
  ↓
8. Bank 매칭 (Base / Masked / Dynamic)
  ↓
9. 임계값 판정
   a. 화질 기반 main_threshold 설정
   b. suspect_ids 모드 시 +0.02 증가
   c. sim_gap 검증
  ↓
10. 오탐 필터링
    a. 같은 bbox 다중 매칭 체크
    b. 프레임 간 연속성 체크 (영상만)
    c. Review 후보 분리
  ↓
11. 결과 분류
    - Match: 자동 알림
    - Review: 관제요원 검토 필요
    - Unknown: 표시만
  ↓
12. Dynamic Bank 자동 추가 (Match인 경우)
  ↓
13. WebSocket 응답
    {
      type: "detection",
      data: {
        frame_id: 123,
        detections: [
          {
            bbox: [x1, y1, x2, y2],
            name: "홍길동",
            confidence: 85,
            color: "green",
            angle_type: "front",
            status: "normal"
          }
        ],
        alert: false
      }
    }
  ↓
[클라이언트]
  ↓
14. Canvas API로 박스 렌더링
  ↓
15. 다음 프레임 처리 (loop)
```

### 4.2 매칭 상세 플로우

```
[얼굴 임베딩 입력]
  ↓
┌──────────────────────────────────┐
│ suspect_ids 모드 확인            │
└──────────────────┬───────────────┘
                   │
         ┌─────────┴─────────┐
         │                   │
    [있음]              [없음]
         │                   │
         ▼                   ▼
  선택된 용의자만     전체 갤러리
  갤러리 생성         사용
         │                   │
         └─────────┬─────────┘
                   │
                   ▼
┌──────────────────────────────────┐
│ 3개 Bank와 매칭                  │
│ • Base Bank                      │
│ • Masked Bank                    │
│ • Dynamic Bank                   │
└──────────────────┬───────────────┘
                   │
                   ▼
┌──────────────────────────────────┐
│ 최고 유사도 선택                 │
│ if dynamic_sim >= max(base, masked): │
│     best_sim = dynamic_sim      │
│ elif base_sim > masked_sim:     │
│     best_sim = base_sim         │
│ else:                            │
│     best_sim = masked_sim       │
└──────────────────┬───────────────┘
                   │
                   ▼
┌──────────────────────────────────┐
│ 화질 및 모드 기반 임계값 설정    │
│                                  │
│ 화질:                            │
│ • high: 0.42                     │
│ • medium: 0.40                   │
│ • low: 0.38                      │
│                                  │
│ suspect_ids 모드:                │
│ • main_threshold += 0.02         │
│ • gap_margin += 0.03             │
└──────────────────┬───────────────┘
                   │
                   ▼
┌──────────────────────────────────┐
│ 3가지 조건 검증                  │
│                                  │
│ 1. max_sim >= main_threshold    │
│ 2. sim_gap >= gap_margin        │
│ 3. second_sim < threshold-0.02  │
│                                  │
│ 모두 만족 → Match                │
│ 일부 만족 → Review               │
│ 불만족 → Unknown                 │
└──────────────────┬───────────────┘
                   │
          ┌────────┼────────┐
          │        │        │
       Match    Review   Unknown
```

---

## 5. 임계값 및 파라미터

### 5.1 주요 임계값 테이블

#### 5.1.1 기본 임계값

| 파라미터 | 값 | 위치 | 설명 |
|----------|-----|------|------|
| `BASE_THRESH` | 0.32 | `src/face_match_cctv.py:606` | 기본 임계값 (조정 전) |
| `BANK_DUPLICATE_THRESHOLD` | 0.95 | `backend/main.py:1416` | Bank 중복 체크 |
| `MASKED_BANK_MASK_PROB_THRESHOLD` | 0.5 | `backend/main.py:54` | Masked Bank 분류 기준 |
| `MASKED_CANDIDATE_MIN_SIM` | 0.25 | `backend/main.py:55` | Masked 후보 최소 유사도 |
| `MASKED_CANDIDATE_MIN_FRAMES` | 3 | `backend/main.py:56` | Masked 수집 최소 프레임 |
| `MASKED_TRACKING_IOU_THRESHOLD` | 0.5 | `backend/main.py:57` | Bbox tracking IoU |

#### 5.1.2 화질별 임계값

| 화질 | main_threshold | gap_margin | 조건 |
|------|----------------|------------|------|
| **high** | **0.42** | **0.12** | 얼굴 너비≥150px AND 비율≥5% |
| **medium** | **0.40** | **0.10** | 얼굴 너비≥100px AND 비율≥2% |
| **low** | **0.38** | **0.08** | 그 외 |

**위치**: `backend/main.py:1353-1361`

#### 5.1.3 suspect_ids 모드 강화

```python
# backend/main.py:1363-1366
if suspect_ids:
    main_threshold += 0.02  # 임계값 상향 (더 보수적으로)
    gap_margin += 0.03      # Gap 기준 강화
```

| 화질 | 일반 모드 threshold | suspect_ids 모드 threshold |
|------|---------------------|----------------------------|
| **high** | 0.42 | **0.44** |
| **medium** | 0.40 | **0.42** |
| **low** | 0.38 | **0.40** |

#### 5.1.4 절대 최소값 (suspect_ids 모드)

```python
# backend/main.py:1440
if max_similarity < 0.45:
    is_match = False  # 무조건 Match 포기
```

### 5.2 Gap 기준

#### 5.2.1 sim_gap 정의
```python
sim_gap = max_similarity - second_similarity
```

#### 5.2.2 최소 Gap 요구사항

| 화질 | 일반 모드 | suspect_ids 모드 |
|------|-----------|------------------|
| **high** | 0.12 | **0.15** |
| **medium** | 0.10 | **0.13** |
| **low** | 0.08 | **0.11** |

**이유**: Gap이 작으면 1위와 2위가 비슷 → 오판 가능성

### 5.3 각도별 다양성 제한

```python
# src/utils/face_angle_detector.py
ANGLE_TYPE_LIMITS = {
    "front": 1,          # 정면은 1개만
    "left": 3,           # 왼쪽은 3개까지
    "right": 3,          # 오른쪽은 3개까지
    "left_profile": 10,  # 왼쪽 프로필 10개
    "right_profile": 10, # 오른쪽 프로필 10개
    "top": 5,            # 위쪽 5개
    "bottom": 5          # 아래쪽 5개
}
```

**목적**: Bank가 특정 각도로 편중되는 것 방지

### 5.4 IoU 및 거리 임계값

#### 5.4.1 같은 얼굴 영역 판정

```python
# backend/main.py:515
def is_same_face_region(bbox1, bbox2, 
                        iou_threshold=0.3, 
                        distance_threshold=None):
    # IoU >= 0.3 또는
    # 중심점 거리 <= 대각선*0.6
```

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `iou_threshold` | 0.3 | IoU 최소값 |
| `distance_threshold` | 자동 | bbox 대각선 * 0.6 |

#### 5.4.2 Masked Tracking IoU

```python
# backend/main.py:1459
if iou >= MASKED_TRACKING_IOU_THRESHOLD:  # 0.5
    # 같은 track으로 인식
```

### 5.5 프레임 간 연속성 (영상 전용)

```python
# src/face_match_cctv.py:847
continuity_window = 5  # 최근 5프레임 내 같은 인물 확인

# 연속성 없고 유사도 낮으면 Review 처리
if not has_continuity and similarity < continuity_threshold:
    review_reason = "no_continuity"
```

| 화질 | continuity_threshold |
|------|---------------------|
| **high** | 0.42 |
| **medium** | 0.40 |
| **low** | 0.38 |

---

## 6. 임베딩 시스템

### 6.1 임베딩 생성

#### 6.1.1 모델 정보
- **모델**: InsightFace buffalo_l
- **차원**: 512차원
- **정규화**: L2 정규화 적용
- **범위**: 정규화 후 각 차원 -1 ~ 1

#### 6.1.2 생성 과정
```python
# backend/main.py:1217-1218
embedding = face.embedding.astype("float32")  # (512,)
embedding_normalized = l2_normalize(embedding)
```

### 6.2 Bank 파일 구조

```
outputs/embeddings/<person_id>/
├── bank_base.npy          # Base Bank (N, 512)
├── bank_masked.npy        # Masked Bank (M, 512)
├── bank_dynamic.npy       # Dynamic Bank (K, 512)
├── centroid.npy           # Centroid (512,)
├── centroid_dynamic.npy   # Dynamic Centroid (512,)
├── angles_dynamic.json    # 각도 메타데이터
└── collection_status.json # 수집 완료 상태
```

#### 6.2.1 angles_dynamic.json 구조
```json
{
  "angle_types": ["front", "left", "right", "front"],
  "yaw_angles": [5.2, -25.3, 30.1, -2.5]
}
```

#### 6.2.2 collection_status.json 구조
```json
{
  "is_completed": true,
  "completed_at": "2025-11-26T12:34:56",
  "collected_angles": ["front", "left", "right", "top"],
  "required_angles": ["front", "left", "right", "top"],
  "completion_criteria": {
    "min_front": 1,
    "min_left": 1,
    "min_right": 1,
    "min_top": 1
  }
}
```

### 6.3 Bank 로딩 순서

```python
# backend/main.py:178-194
우선순위:
1. bank_dynamic.npy (인식용)
2. bank_base.npy (기본)
3. bank_masked.npy (마스크)
4. centroid.npy (fallback)
5. DB embedding (final fallback)
```

### 6.4 유사도 계산

```python
# src/utils/gallery_loader.py
def match_with_bank_detailed(face_emb, gallery):
    for person_id, bank in gallery.items():
        if bank.ndim == 2:  # Bank (N, 512)
            # 내적 = 코사인 유사도 (L2 정규화 가정)
            similarities = np.dot(bank, face_emb)  # (N,)
            max_sim = np.max(similarities)
        else:  # Centroid (512,)
            max_sim = np.dot(bank, face_emb)
```

**수식**:
```
코사인 유사도 = dot(emb1, emb2) / (||emb1|| * ||emb2||)
              = dot(emb1, emb2)  (L2 정규화 후)
```

---

## 7. 매칭 및 판정 로직

### 7.1 3단계 결과 분류

#### 7.1.1 Match (매칭 성공)

**조건**:
```python
# backend/main.py:1425-1432
is_match = (
    max_similarity >= main_threshold AND
    sim_gap >= gap_margin AND
    second_similarity < (main_threshold - 0.02)
)

# suspect_ids 모드 추가 조건
if suspect_ids:
    is_match = is_match AND (max_similarity >= 0.45)
```

**처리**:
- ✅ 자동 알림 전송 (범죄자인 경우)
- ✅ DB 로그 저장 (`status="criminal"` 또는 `"normal"`)
- ✅ Dynamic Bank 자동 추가

#### 7.1.2 Review (검토 필요)

**발생 조건** (4가지):

1. **same_face_multiple_persons**: 같은 얼굴 영역에서 여러 인물 매칭
   ```python
   if 같은 bbox에서 2명 이상 매칭 AND sim_gap < 0.10:
       review_reason = "same_face_multiple_persons"
   ```

2. **ambiguous_match**: 1위와 2위 유사도가 너무 비슷
   ```python
   if sim_gap < gap_margin:
       review_reason = "ambiguous_match"
   ```

3. **low_confidence**: 임계값은 넘었지만 확신 부족
   ```python
   if similarity < quality_threshold OR sim_gap < gap_threshold:
       review_reason = "low_confidence"
   ```

4. **no_continuity** (영상만): 연속성 없음
   ```python
   if 최근 5프레임 내 매칭 없음 AND similarity < continuity_threshold:
       review_reason = "no_continuity"
   ```

**처리**:
- ⚠️ 관제요원 대시보드에 표시
- ⚠️ 별도 검토 폴더에 스냅샷 저장
- ⚠️ 자동 알림 없음
- ⚠️ DB 로그에 `review_reason` 기록

#### 7.1.3 Unknown (미확인)

**조건**:
```python
# backend/main.py:1729-1740
is_match == False AND 
not review_reason
```

**처리**:
- ℹ️ 노란색 박스만 표시 (화면에)
- ℹ️ DB 로그 저장 (`status="unknown"`)
- ℹ️ 알림 없음
- ℹ️ Bank 추가 없음

### 7.2 결과 분류 플로우차트

```
[얼굴 감지 및 매칭]
  ↓
best_match == None?
  ├─ Yes → Unknown (갤러리 없음)
  └─ No ↓
     
max_sim >= threshold?
  ├─ No → Unknown (유사도 부족)
  └─ Yes ↓
     
sim_gap >= gap_margin?
  ├─ No → Review (애매함)
  └─ Yes ↓
     
second_sim < threshold-0.02?
  ├─ No → Review (2위도 높음)
  └─ Yes ↓
     
suspect_ids 모드?
  ├─ Yes → max_sim >= 0.45?
  │         ├─ No → Unknown
  │         └─ Yes → Match ✅
  └─ No → Match ✅
```

---

## 8. 오탐 방지 메커니즘

### 8.1 같은 얼굴 영역 다중 매칭 처리

#### 8.1.1 감지 방법
```python
# backend/main.py:1527-1547
for bbox1, bbox2 in combinations(face_results, 2):
    if is_same_face_region(bbox1, bbox2, iou_threshold=0.3):
        # 같은 얼굴 그룹으로 묶음
```

#### 8.1.2 처리 로직
```
같은 얼굴 영역에 N개 매칭
  ↓
유사도 순으로 정렬
  ↓
best_match와 second_match 비교
  ↓
sim_gap >= 0.10?
  ├─ Yes → best_match만 Match
  │         나머지는 Review (same_face_multiple_persons)
  └─ No → 전부 Review (ambiguous_match)
```

### 8.2 프레임 간 연속성 체크 (영상 전용)

#### 8.2.1 히스토리 관리
```python
# src/face_match_cctv.py:846
frame_history = defaultdict(list)
# {person_id: [10, 15, 20, 25, 30]}  # 매칭된 프레임 번호
```

#### 8.2.2 연속성 판정
```python
recent_frames = frame_history[person_id]
last_frame = recent_frames[-1]
frame_gap = current_frame - last_frame

has_continuity = (frame_gap <= 5)  # 5프레임 내
```

#### 8.2.3 연속성 없을 때
```python
if not has_continuity and similarity < continuity_threshold:
    review_reason = "no_continuity"
    is_match = False  # Match → Review로 강등
```

### 8.3 Temporal Filter (시간적 일관성)

- **목적**: 같은 인물이 프레임마다 다르게 인식되는 것 방지
- **구현**:
  ```python
  # web/script.js: temporal filter
  if (person_id == previous_person_id):
      confidence = (confidence + previous_confidence) / 2  # 평균
  ```

### 8.4 bbox 안정화

- **좌표 보간**: 이전 프레임 bbox와 현재 bbox 사이를 선형 보간
- **효과**: 박스가 튀는 현상 감소

---

## 9. API 명세

### 9.1 WebSocket API

#### 9.1.1 연결
```
엔드포인트: ws://localhost:5000/ws/detect
프로토콜: WebSocket
```

#### 9.1.2 클라이언트 → 서버

**메시지 타입 1: 프레임 전송**
```json
{
  "type": "frame",
  "data": {
    "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
    "suspect_ids": ["yh", "ja", "js"],
    "frame_id": 123
  }
}
```

**메시지 타입 2: 설정 변경**
```json
{
  "type": "config",
  "suspect_ids": ["yh", "ja"]
}
```

#### 9.1.3 서버 → 클라이언트

**메시지 타입 1: 감지 결과**
```json
{
  "type": "detection",
  "data": {
    "frame_id": 123,
    "detections": [
      {
        "bbox": [100, 200, 300, 400],
        "name": "홍길동",
        "person_id": "yh",
        "confidence": 85,
        "color": "green",  // "red", "green", "yellow"
        "status": "normal",  // "criminal", "normal", "unknown"
        "angle_type": "front",
        "yaw_angle": 5.2
      }
    ],
    "alert": false,
    "metadata": {
      "name": "홍길동",
      "confidence": 85,
      "status": "normal"
    }
  }
}
```

**메시지 타입 2: 설정 확인**
```json
{
  "type": "config_updated",
  "suspect_ids": ["yh", "ja"]
}
```

**메시지 타입 3: 학습 이벤트**
```json
{
  "type": "learning",
  "data": {
    "person_id": "yh",
    "person_name": "홍길동",
    "angle_type": "left",
    "bank_type": "dynamic"
  }
}
```

### 9.2 HTTP API

#### 9.2.1 POST /api/detect
**요청**:
```json
{
  "image": "data:image/jpeg;base64,...",
  "suspect_id": "yh",  // 선택
  "suspect_ids": ["yh", "ja"]  // 선택
}
```

**응답**:
```json
{
  "detections": [...],
  "alert": false,
  "metadata": {...}
}
```

#### 9.2.2 GET /api/persons
**응답**:
```json
[
  {
    "id": "yh",
    "name": "홍길동",
    "is_criminal": false,
    "info": {}
  }
]
```

#### 9.2.3 GET /api/logs?limit=100
**응답**:
```json
[
  {
    "id": 1,
    "person_id": "yh",
    "person_name": "홍길동",
    "similarity": 0.85,
    "is_criminal": false,
    "status": "normal",
    "timestamp": "2025-11-26T12:34:56",
    "metadata": {...}
  }
]
```

#### 9.2.4 POST /api/upload
**파일 업로드**:
- 엔드포인트: `/api/upload`
- Method: POST
- Content-Type: multipart/form-data
- Field 이름: `file`

**응답**:
```json
{
  "filename": "video_20250127_123456.mp4",
  "path": "/uploads/video_20250127_123456.mp4"
}
```

---

## 10. 데이터 구조

### 10.1 PostgreSQL 스키마

#### 10.1.1 persons 테이블
```sql
CREATE TABLE persons (
    id SERIAL PRIMARY KEY,
    person_id VARCHAR(50) UNIQUE NOT NULL,  -- yh, ja 등
    name VARCHAR(100),
    is_criminal BOOLEAN DEFAULT FALSE,
    embedding BYTEA,  -- NumPy 배열 직렬화
    info JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

#### 10.1.2 detection_logs 테이블
```sql
CREATE TABLE detection_logs (
    id SERIAL PRIMARY KEY,
    person_id VARCHAR(50),
    person_name VARCHAR(100),
    similarity FLOAT,
    is_criminal BOOLEAN,
    status VARCHAR(20),  -- 'criminal', 'normal', 'unknown'
    metadata JSONB,
    timestamp TIMESTAMP DEFAULT NOW()
);
```

### 10.2 메모리 캐시 구조

```python
# backend/main.py:96-100
persons_cache: List[Dict] = [
    {
        "id": "yh",
        "name": "홍길동",
        "is_criminal": False,
        "info": {},
        "embedding": np.ndarray  # (512,)
    }
]

gallery_base_cache: Dict[str, np.ndarray] = {
    "yh": np.ndarray  # (N, 512) or (512,)
}

gallery_masked_cache: Dict[str, np.ndarray] = {
    "yh": np.ndarray  # (M, 512)
}

gallery_dynamic_cache: Dict[str, np.ndarray] = {
    "yh": np.ndarray  # (K, 512)
}
```

### 10.3 face_results 구조

```python
# backend/main.py:1504-1525
face_results = [
    {
        "bbox": [x1, y1, x2, y2],
        "embedding": np.ndarray,  # (512,)
        "angle_type": "front",
        "yaw_angle": 5.2,
        "face_quality": "high",
        "max_similarity": 0.85,
        "base_sim": 0.82,
        "masked_sim": 0.30,
        "second_similarity": 0.45,
        "sim_gap": 0.40,
        "main_threshold": 0.42,
        "gap_margin": 0.12,
        "is_match": True,
        "best_match": {...},  # person 정보
        "best_person_id": "yh",
        "mask_prob": 0.1,
        "bank_type": "dynamic",
        "is_masked_candidate": False,
        "candidate_frames_count": 0,
        "track_id": None
    }
]
```

---

## 11. 성능 지표

### 11.1 처리 속도

| 단계 | 시간 | GPU | CPU |
|------|------|-----|-----|
| 얼굴 검출 | 30-50ms | 10-20ms | 50-100ms |
| 임베딩 추출 | 10-20ms | 5-10ms | 20-40ms |
| Bank 매칭 | 1-5ms | 1-5ms | 1-5ms |
| 오탐 필터링 | 1-2ms | 1-2ms | 1-2ms |
| **총합** | **50-150ms** | **20-40ms** | **100-200ms** |

### 11.2 네트워크 사용량

| 통신 방식 | 프레임당 전송량 | 비고 |
|-----------|----------------|------|
| **WebSocket (현재)** | ~2KB | JSON만 전송 |
| HTTP (레거시) | ~2KB | JSON만 전송 |
| 서버사이드 렌더링 (폐기) | ~500KB | 이미지 전송 |

### 11.3 인식 정확도 (예시)

| 시나리오 | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| 정면 고화질 | 95% | 92% | 0.935 |
| 측면 중화질 | 85% | 78% | 0.814 |
| 프로필 저화질 | 70% | 65% | 0.674 |
| 마스크 착용 | 60% | 55% | 0.574 |

**주의**: 실제 정확도는 Bank 품질, 조명, 각도 등에 따라 달라집니다.

---

## 12. 설정 파일 및 환경 변수

### 12.1 환경 변수 (`backend/.env`)

```ini
# 데이터베이스
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/facewatch

# 서버
HOST=0.0.0.0
PORT=5000

# GPU (선택)
CUDA_VISIBLE_DEVICES=0  # GPU 0번 사용
```

### 12.2 주요 설정 위치

| 설정 | 위치 | 값 |
|------|------|-----|
| BASE_THRESH | `src/face_match_cctv.py:606` | 0.32 |
| EMBEDDINGS_DIR | `backend/main.py:51` | `outputs/embeddings` |
| WebSocket 엔드포인트 | `backend/main.py` | `/ws/detect` |
| HTTP API 엔드포인트 | `backend/main.py` | `/api/*` |
| 프론트엔드 포트 | Live Server | 5500 |
| 백엔드 포트 | uvicorn | 5000 |

---

## 13. 주요 개선 이력

### 13.1 통신 방식 개선

```
v1.0: 서버사이드 렌더링
  ↓ (문제: 네트워크 과부하, 지연 500ms)
v2.0: 클라이언트사이드 렌더링 + HTTP
  ↓ (문제: 프레임 끊김, 박스 튐)
v3.0: WebSocket 양방향 통신
  ✅ 지연 50-150ms, 끊김 해결
```

### 13.2 Bank 시스템 발전

```
v1.0: 단일 임베딩 (centroid)
  ↓
v2.0: Bank (다중 임베딩)
  ↓
v3.0: Bank + 각도 메타데이터
  ↓
v4.0: Base/Masked/Dynamic 분리
  ✅ 정확도 대폭 향상
```

### 13.3 오탐 방지 강화

```
v1.0: 단순 임계값
  ↓
v2.0: 화질 기반 적응형 임계값
  ↓
v3.0: sim_gap 추가
  ↓
v4.0: 프레임 간 연속성 체크
  ↓
v5.0: Review 단계 분리
  ✅ False Positive 대폭 감소
```

---

## 부록

### A. 코드 라인 수

| 파일 | 라인 수 | 역할 |
|------|---------|------|
| `backend/main.py` | 2760 | 메인 서버 |
| `web/script.js` | ~1500 | 클라이언트 로직 |
| `src/face_match_cctv.py` | 1075 | CCTV 분석 |
| `src/face_enroll.py` | ~500 | 인물 등록 |
| `src/utils/gallery_loader.py` | ~300 | Bank 로더 |
| `src/utils/face_angle_detector.py` | ~200 | 각도 감지 |
| `src/utils/mask_detector.py` | ~300 | 마스크 감지 |

**총합**: ~6000+ 라인

### B. 주요 의존성

```
insightface==0.7.3
onnxruntime-gpu==1.16.3
opencv-python==4.8.1.78
numpy==1.24.3
fastapi==0.104.1
uvicorn==0.24.0
websockets==12.0
sqlalchemy==2.0.23
psycopg2-binary==2.9.9
```

### C. 참고 문서

- [README.md](README.md) - 프로젝트 개요
- [SERVICE_IMPROVEMENTS.md](SERVICE_IMPROVEMENTS.md) - 개선 이력
- [EVALUATION_GUIDELINES.md](EVALUATION_GUIDELINES.md) - 평가 가이드
- [MASKED_FACE_IMPLEMENTATION.md](MASKED_FACE_IMPLEMENTATION.md) - 마스크 구현
- [POSTGRESQL_MIGRATION.md](POSTGRESQL_MIGRATION.md) - DB 마이그레이션

---

**문서 버전**: 1.0  
**최종 수정**: 2025-11-27  
**작성자**: FaceWatch Development Team
