# FaceWatch — 실시간 얼굴 식별·추적 시스템

> InsightFace 기반 고성능 얼굴 인식 및 추적 시스템

FaceWatch는 CCTV/영상/이미지 콘텐츠에서 특정 인물을 자동으로 식별하고 추적하는 Python 기반 AI 시스템입니다. InsightFace의 고성능 얼굴 인식 모델(buffalo_l)을 기반으로 임베딩 비교, 트래킹, 신뢰도 누적, 스냅샷 저장, 시각화 도구까지 포함하고 있습니다.

---

## 프로젝트 목표

- 정확한 얼굴 인식 및 인물 식별
- 영상(프레임 단위)에서 지속적으로 동일 인물 추적
- 신뢰도 기반 스냅샷 저장(중복 방지)
- 임베딩 분석(히스토그램, heatmap, PCA 등)
- 범죄자 또는 특정 대상자 등록 및 갤러리 관리

---

## 프로젝트 구조

```
FaceWatch/
├─ images/                    # 이미지 관련 폴더
│  ├─ enroll/                 # 인물 등록 이미지 (인물별 폴더)
│  │   ├─ hani/
│  │   │   └─ hani.jpg        # 등록용 이미지
│  │   ├─ danielle/
│  │   └─ ...
│  ├─ source/                 # 추출할 소스 이미지 (분석 대상)
│  │   └─ test.jpg            # 분석할 이미지 파일
│  └─ extracted_frames/       # 추출된 프레임 (수동 추가용)
│     └─ hani/
│        └─ hani_f00001.jpg
│
├─ videos/                    # 영상 관련 폴더
│  └─ source/                 # 추출할 소스 영상 (분석 대상)
│     └─ yh.MOV               # 분석할 영상 파일
│
├─ notebooks/                 # 주피터 노트북 파일 (테스트/실험용)
│  └─ *.ipynb                 # 각 노트북은 자동으로 프로젝트 루트를 작업 디렉토리로 설정
│
├─ outputs/                   # 출력 폴더
│  ├─ embeddings/             # 인물별 임베딩 (사람별 폴더 구조)
│  │   ├─ hani/
│  │   │   ├─ bank.npy        # Bank 임베딩 (N×512)
│  │   │   └─ centroid.npy   # Centroid 임베딩 (512)
│  │   ├─ danielle/
│  │   │   ├─ bank.npy
│  │   │   └─ centroid.npy
│  │   └─ ...
│  └─ results/                 # 분석 결과 (입력 파일명 기반)
│     └─ <파일명>/            # 입력 파일명 기반 폴더 (자동 생성)
│        ├─ matches/          # 매칭된 스냅샷
│        ├─ logs/              # CSV 로그
│        └─ frames/           # 추출된 프레임 (선택적)
│
├─ src/
│  ├─ face_enroll.py               # 통합 등록 스크립트 (권장)
│  │                                # - 기본 등록: enroll 폴더에서 등록
│  │                                # - 영상에서 자동 수집
│  │                                # - 수동 추가: 이미지 폴더에서 추가
│  ├─ face_match_cctv.py           # 통합 인식 스크립트 (권장)
│  │                                # - 이미지/영상 모두 처리
│  │                                # - 얼굴 각도 감지
│  │                                # - 마스크 감지 및 적응형 임계값
│  │                                # - 오탐 방지 (sim_gap 체크)
│  │                                # - CSV 로그 및 스냅샷 저장
│  ├─ embedding_analysis.py        # 임베딩 분석 도구
│  └─ utils/                       # 유틸리티 모듈
│     ├─ device_config.py          # GPU/CPU 설정
│     ├─ gallery_loader.py        # 갤러리 로더
│     ├─ mask_detector.py          # 마스크 감지 및 적응형 임계값
│     └─ face_angle_detector.py   # 얼굴 각도 감지
│
└─ README.md
```

### 폴더 사용 가이드

**입력 파일 배치:**
- 등록용 이미지: `images/enroll/<인물명>/<파일명>.jpg`
- 분석용 이미지: `images/source/<파일명>.jpg`
- 분석용 영상: `videos/source/<파일명>.mp4`

**출력 파일 위치:**
- 임베딩: `outputs/embeddings/<인물명>/bank.npy`, `centroid.npy` (사람별 폴더 구조)
- 분석 결과: `outputs/results/<입력파일명>/`

---

## 주요 기능

### 1. 사람 등록 (Face Enrollment)

**통합 등록 스크립트 사용 (권장):**

```bash
python src/face_enroll.py
```

스크립트 내에서 모드를 선택할 수 있습니다:

#### 1-1. 기본 등록 (MODE = 1)
- `images/enroll/` 폴더의 모든 인물을 등록
- 각 인물별 폴더에서 이미지를 읽어 bank와 centroid 생성
- **출력:**
  - `outputs/embeddings/<person>/bank.npy` (N×512 배열)
  - `outputs/embeddings/<person>/centroid.npy` (512 벡터)

#### 1-2. 영상에서 자동 수집 (MODE = 2)
- `videos/source/` 폴더의 영상에서 특정 인물 찾기
- 다양한 각도의 얼굴을 자동으로 수집하여 bank에 추가
- 중복 임베딩 자동 제거

#### 1-3. 수동 추가 (MODE = 3)
- `images/extracted_frames/<person>/` 폴더의 이미지들을 bank에 추가
- 중복 체크 후 새로운 얼굴만 추가

**폴더 구조:**
```
images/
├─ enroll/              # 등록용 이미지 (인물별 폴더)
│  ├─ hani/
│  │   └─ hani.jpg
│  └─ danielle/
│      └─ danielle.jpg
└─ extracted_frames/    # 추출된 프레임 (수동 추가용)
   └─ hani/
      └─ hani_f00001.jpg
```

---

### 2. 얼굴 인식 및 매칭 (Face Recognition & Matching)

**통합 인식 스크립트 사용 (권장):**

```bash
python src/face_match_cctv.py
```

스크립트 내에서 입력 파일명만 지정하면 자동으로 처리합니다:

```python
# src/face_match_cctv.py 내부 설정
input_filename = "yh.MOV"  # 파일명만 지정
```

**자동 경로 탐색:**
- 이미지: `images/source/` → `images/` (호환성)
- 영상: `videos/source/` → `videos/` → `images/` (호환성)

#### 2-1. 주요 기능

**얼굴 각도 감지**
- 정면(front), 왼쪽(left), 오른쪽(right), 프로필(profile) 자동 감지
- 각도별 인식 성공률 통계 제공

**마스크 감지 및 적응형 임계값**
- 유사도 기반 마스크 착용 가능성 자동 추정
- 마스크 가능성이 높으면 임계값 자동 조정 (0.30 → 0.22~0.28)
- 기존 등록 이미지(마스크 없음)만으로도 마스크 쓴 얼굴 인식 가능
- 최소 임계값(0.22) 보장으로 오탐 방지

**화질 기반 적응형 임계값 조정** ⭐ (추가 기능)
- 얼굴 크기와 이미지 크기 비율로 화질 자동 추정 (high/medium/low)
- 고화질: 더 높은 임계값 적용 (오탐 방지 강화)
- 저화질: 낮은 임계값 적용 (인식률 유지)
- 화질에 따라 동적으로 임계값 조정 (0.28 ~ 0.40)

**고급 오탐 방지** ⭐ (추가 기능)
- **bbox 기반 다중 매칭 필터링**: 같은 얼굴 영역에서 여러 인물로 매칭된 경우 자동 필터링
  - IoU 및 중심점 거리 기반으로 같은 얼굴 영역 판단
  - sim_gap이 충분히 크면 가장 높은 유사도만 인정
  - 애매한 경우 검토 대상으로 분리
- **프레임 간 연속성 체크**: 이전/다음 프레임에서 같은 인물이 매칭되었는지 확인
  - 연속성이 없고 유사도가 낮은 경우 검토 대상으로 분리
  - 일시적 오탐 자동 제거
- **sim_gap 체크**: 최고 유사도와 두 번째 유사도의 차이 확인 (최소 5% 차이 필요)
- **중복 얼굴 필터링**: 같은 프레임 내 동일 인물 중복 감지 방지

**검토 대상 자동 분리** ⭐ (추가 기능)
- 의심스러운 매칭을 자동으로 `matches/review/` 폴더에 분리
- 검토 사유 자동 분류:
  - `same_face_multiple_persons`: 같은 얼굴에 여러 인물 매칭
  - `ambiguous_match`: sim_gap이 작아 애매한 경우
  - `low_confidence`: 낮은 유사도 또는 작은 sim_gap
  - `no_continuity`: 프레임 간 연속성 없음
- CSV 로그에 검토 사유 기록

**상세한 로깅**
- CSV 로그: 모든 얼굴 감지 기록 저장 (화질 정보, 검토 사유 포함)
- 스냅샷: 매칭된 얼굴만 이미지로 저장
- 검토 대상: 의심스러운 매칭 별도 저장
- 통계: 인물별/각도별 매칭 통계 출력

#### 2-2. 출력 구조

```
outputs/
├─ embeddings/            # 인물별 임베딩 (등록 시 생성)
└─ results/               # 분석 결과 (분석 시 자동 생성)
   └─ <파일명>_<타임스탬프>/  # 입력 파일명 + 실행 시간 기반 폴더 ⭐
      ├─ matches/         # 매칭된 스냅샷 (매칭된 얼굴만)
      │   └─ review/      # 검토 대상 스냅샷 (의심스러운 매칭) ⭐
      ├─ logs/            # CSV 로그 (모든 얼굴 감지 기록)
      └─ frames/          # 추출된 프레임 (영상일 때, 선택적)
```

> **참고:** 
> - 이전 버전의 `outputs/test_results/` 폴더는 더 이상 사용되지 않습니다.
> - 현재는 입력 파일명 + 타임스탬프 기반으로 폴더가 자동 생성됩니다 (예: `ive_iam_20240101_120000/`)
> - 각 실행마다 새로운 폴더가 생성되어 결과가 덮어써지지 않습니다. ⭐

#### 2-3. 사용 예시

**이미지 분석:**
```bash
# images/source/test.jpg 파일 분석
# 스크립트 내부에서 input_filename = "test.jpg" 설정
python src/face_match_cctv.py
```

**영상 분석:**
```bash
# videos/source/yh.MOV 파일 분석
# 스크립트 내부에서 input_filename = "yh.MOV" 설정
python src/face_match_cctv.py
```

**프레임 저장 옵션:**
- 영상 분석 시 프레임 이미지 저장 가능 (기본: 비활성화)
- `SAVE_FRAMES = True`로 설정하면 N프레임마다 저장

---

### 3. 임베딩 분석 도구 (Embedding Analysis)

```bash
python src/embedding_analysis.py
```

임베딩 파일의 통계 및 분포를 분석합니다.

---

## 설정 및 사용법

### 기본 설정

**임계값 (Threshold):**
- 기본 임계값: `BASE_THRESH = 0.32`
- 화질 기반 자동 조정:
  - 고화질: `0.40` (오탐 방지 강화)
  - 중화질: `0.32` (기본값)
  - 저화질: `0.33` (인식률 유지, 최소 0.32)
- 마스크 착용 시 추가 조정: `-0.02 ~ -0.05`
- 최소 임계값: `0.28` (고화질), `0.30` (중화질), `0.28` (저화질)

**입력 파일:**
- 이미지: `images/source/` 폴더에 배치
- 영상: `videos/source/` 폴더에 배치
- 스크립트 내부에서 `input_filename` 변수 수정

**출력 폴더:**
- 입력 파일명 + 타임스탬프 기반으로 자동 생성 (`outputs/results/<파일명>_<YYYYMMDD_HHMMSS>/`)
- 각 실행마다 새로운 폴더 생성 (결과 덮어쓰기 방지)
- 매칭 스냅샷, 검토 대상, CSV 로그, 프레임 이미지 저장

---

## 기술 스택

| 분야 | 기술 |
|------|------|
| Face Recognition | InsightFace (buffalo_l), ONNX Runtime |
| Detection | RetinaFace (InsightFace), YOLOv12n-face |
| Tracking | IoU-based lightweight tracking |
| Embedding Analysis | NumPy, Matplotlib, Seaborn, Scikit-learn |
| Video Handling | OpenCV, imageio |
| Language | Python 3.9+ |

---

## 핵심 알고리즘

### 1. 임베딩 기반 Similarity Matching

- 얼굴 → 512-d vector (InsightFace buffalo_l)
- L2-normalized
- cosine similarity로 비교
- Bank 기반 매칭 (여러 임베딩 중 최고 유사도 사용)

### 2. 마스크 감지 및 적응형 임계값

- 유사도 기반 마스크 착용 가능성 추정 (0.0 ~ 1.0)
- 마스크 가능성이 높으면 임계값 자동 조정
- 기존 등록 이미지만으로도 마스크 쓴 얼굴 인식 가능

### 3. 얼굴 각도 감지

- Yaw 각도 기반 각도 분류
- 정면(front), 측면(left/right), 프로필(profile) 자동 감지
- 각도별 인식 성능 통계 제공

### 4. 고급 오탐 방지 ⭐ (추가 기능)

**bbox 기반 다중 매칭 필터링**
- 같은 얼굴 영역에서 여러 인물로 매칭된 경우 자동 처리
- IoU(Intersection over Union) 및 중심점 거리로 같은 얼굴 영역 판단
- sim_gap이 충분히 크면(≥0.10) 가장 높은 유사도만 인정
- 애매한 경우 검토 대상으로 분리

**프레임 간 연속성 체크**
- 최근 5프레임 내 같은 인물 매칭 여부 확인
- 연속성이 없고 유사도가 낮으면 검토 대상으로 분리
- 일시적 오탐 자동 제거

**기본 오탐 방지**
- sim_gap 체크: 최고 유사도와 두 번째 유사도 차이 확인 (최소 5% 차이 필요)
- 중복 얼굴 필터링: 같은 프레임 내 동일 인물 중복 감지 방지
- 최소 임계값 보장: 너무 낮은 유사도는 제외

### 5. 화질 기반 적응형 임계값 ⭐ (추가 기능)

- 얼굴 크기와 이미지 크기 비율로 화질 자동 추정
- 화질 등급: high (≥150px, ≥5%), medium (≥100px, ≥2%), low (그 외)
- 화질에 따라 동적으로 임계값 조정:
  - 고화질: 기본값 +0.04 (오탐 방지 강화)
  - 중화질: 기본값 유지
  - 저화질: 기본값 -0.03 (인식률 유지)

### 6. 스냅샷 저장 및 검토 시스템 ⭐ (추가 기능)

- 매칭된 얼굴만 이미지로 저장
- 프레임 번호, 인물 ID, 유사도, 화질 정보 포함
- 검토 대상 자동 분리: 의심스러운 매칭을 `matches/review/` 폴더에 별도 저장
- CSV 로그로 모든 감지 기록 저장 (화질, 검토 사유 포함)

---

## 사용 예시

### 1. 인물 등록

```bash
# 1. images/enroll/ 폴더에 인물별 폴더 생성
#    images/enroll/hani/hani.jpg
#    images/enroll/danielle/danielle.jpg

# 2. 등록 실행
python src/face_enroll.py
# MODE = 1로 설정하여 기본 등록 실행
```

### 2. 이미지 분석

```bash
# 1. 분석할 이미지를 images/source/ 폴더에 배치
#    images/source/test.jpg

# 2. 스크립트에서 파일명 설정
#    input_filename = "test.jpg"

# 3. 분석 실행
python src/face_match_cctv.py
```

### 3. 영상 분석

```bash
# 1. 분석할 영상을 videos/source/ 폴더에 배치
#    videos/source/yh.MOV

# 2. 스크립트에서 파일명 설정
#    input_filename = "yh.MOV"

# 3. 분석 실행
python src/face_match_cctv.py
```

### 4. 결과 확인

```
outputs/
├─ embeddings/            # 임베딩 파일
│   └─ hani/
│      ├─ bank.npy
│      └─ centroid.npy
└─ results/               # 분석 결과
   └─ yh_20240101_120000/  # 입력 파일명 + 타임스탬프
      ├─ matches/         # 매칭된 스냅샷
      │   ├─ match_f000123_hani_0.35.jpg
      │   └─ review/      # 검토 대상 (의심스러운 매칭)
      │      └─ review_f000006_danielle_0.27_low_confidence.jpg
      ├─ logs/            # CSV 로그
      │   └─ detection_log.csv
      └─ frames/          # 추출된 프레임 (선택적)
         └─ frame_000030.jpg
```

---

## 향후 개발 로드맵

**완료된 기능:**
- 얼굴 각도 감지
- 마스크 감지 및 적응형 임계값
- 오탐 방지 (sim_gap 체크, 중복 필터링)
- 통합 스크립트 (등록/인식)
- 폴더 구조 정리 (source 폴더 분리)

**추가된 고급 기능** ⭐ (최근 추가):
- **bbox 기반 다중 매칭 필터링**: 같은 얼굴 영역에서 여러 인물 매칭 시 자동 처리
- **프레임 간 연속성 체크**: 이전/다음 프레임에서 같은 인물 매칭 여부 확인
- **화질 기반 적응형 임계값**: 얼굴 크기 기반 화질 추정 및 동적 임계값 조정
- **검토 대상 자동 분리**: 의심스러운 매칭을 별도 폴더에 분리하여 수동 검토 가능
- **타임스탬프 기반 결과 폴더**: 각 실행마다 새로운 폴더 생성 (결과 덮어쓰기 방지)

**계획 중:**
- YOLOv9 기반 fast detector 추가
- Face anti-spoofing (딥페이크 방지)
- Web dashboard (Flask/React)
- 실시간 스트리밍 지원

---

## 개발자

**Jongwoo Shin**

InsightFace + Computer Vision 기반 얼굴 인식 시스템 개발  
Cloud · AI Engineering · MLOps

---

## 추가된 기능 상세 설명 ⭐

> **참고**: InsightFace 모델 자체는 이미 우수한 성능을 제공합니다. 아래 기능들은 모델의 정확도를 더욱 향상시키고 오탐을 줄이기 위해 추가된 고급 필터링 및 결과 관리 기능입니다.

### 1. bbox 기반 다중 매칭 필터링

**문제**: 같은 프레임에서 같은 얼굴 영역이 여러 인물로 매칭되는 경우 (예: yujin, danielle, harin이 모두 같은 위치에서 매칭됨)

**해결**:
- IoU(Intersection over Union) 및 중심점 거리로 같은 얼굴 영역 판단
- 같은 얼굴 영역에서 여러 인물 매칭 시:
  - sim_gap이 충분히 크면(≥0.10) 가장 높은 유사도만 인정
  - sim_gap이 작으면 모두 검토 대상으로 분리
- 다른 얼굴 영역은 각각 독립적으로 평가 (실제로 여러 인물이 있을 수 있음)

### 2. 프레임 간 연속성 체크

**문제**: 일시적으로 잘못된 인물로 매칭되는 경우 (닮은 사람이 잠깐 나타남)

**해결**:
- 각 인물별로 최근 5프레임 내 매칭 기록 저장
- 연속성이 없고 유사도가 낮은 경우(고화질: <0.42, 중화질: <0.40, 저화질: <0.38) 검토 대상으로 분리
- 실제로 등장한 인물은 연속적으로 매칭되므로, 일시적 오탐 자동 제거

### 3. 화질 기반 적응형 임계값

**문제**: 화질이 좋은 경우와 나쁜 경우에 동일한 임계값을 사용하면 오탐이 발생하거나 인식률이 떨어짐

**해결**:
- 얼굴 크기와 이미지 크기 비율로 화질 자동 추정
- 화질에 따라 동적으로 임계값 조정:
  - 고화질: 더 높은 임계값 (오탐 방지)
  - 저화질: 낮은 임계값 (인식률 유지)
- 마스크 가능성과 함께 고려하여 최종 임계값 결정

### 4. 검토 대상 자동 분리

**문제**: 의심스러운 매칭을 수동으로 찾아야 함

**해결**:
- 자동으로 검토 사유 분류 및 `matches/review/` 폴더에 분리
- 검토 사유:
  - `same_face_multiple_persons`: 같은 얼굴에 여러 인물 매칭
  - `ambiguous_match`: sim_gap이 작아 애매한 경우
  - `low_confidence`: 낮은 유사도 또는 작은 sim_gap
  - `no_continuity`: 프레임 간 연속성 없음
- CSV 로그에 검토 사유 기록하여 후처리 가능

### 5. 타임스탬프 기반 결과 폴더

**문제**: 여러 번 테스트할 때마다 결과가 덮어써짐

**해결**:
- 각 실행마다 타임스탬프가 포함된 새 폴더 생성
- 형식: `{파일명}_{YYYYMMDD_HHMMSS}`
- 모든 실행 결과 보존 및 비교 가능

---

## 라이선스

MIT License
