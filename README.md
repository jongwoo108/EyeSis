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

**오탐 방지**
- sim_gap 체크: 최고 유사도와 두 번째 유사도의 차이 확인
- 중복 얼굴 필터링: 같은 프레임 내 동일 인물 중복 감지 방지

**상세한 로깅**
- CSV 로그: 모든 얼굴 감지 기록 저장
- 스냅샷: 매칭된 얼굴만 이미지로 저장
- 통계: 인물별/각도별 매칭 통계 출력

#### 2-2. 출력 구조

```
outputs/
├─ embeddings/            # 인물별 임베딩 (등록 시 생성)
└─ results/               # 분석 결과 (분석 시 자동 생성)
   └─ <파일명>/           # 입력 파일명 기반 폴더
      ├─ matches/         # 매칭된 스냅샷 (매칭된 얼굴만)
      ├─ logs/            # CSV 로그 (모든 얼굴 감지 기록)
      └─ frames/          # 추출된 프레임 (영상일 때, 선택적)
```

> **참고:** 이전 버전의 `outputs/test_results/` 폴더는 더 이상 사용되지 않습니다. 현재는 입력 파일명 기반으로 폴더가 자동 생성됩니다.

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
- 기본 임계값: `BASE_THRESH = 0.30`
- 마스크 착용 시 자동 조정: `0.22 ~ 0.28`
- 최소 임계값: `0.22` (오탐 방지)

**입력 파일:**
- 이미지: `images/source/` 폴더에 배치
- 영상: `videos/source/` 폴더에 배치
- 스크립트 내부에서 `input_filename` 변수 수정

**출력 폴더:**
- 입력 파일명 기반으로 자동 생성 (`outputs/results/<파일명>/`)
- 매칭 스냅샷, CSV 로그, 프레임 이미지 저장

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

### 4. 오탐 방지

- sim_gap 체크: 최고 유사도와 두 번째 유사도 차이 확인
- 중복 얼굴 필터링: 같은 프레임 내 동일 인물 중복 감지 방지
- 최소 임계값 보장: 너무 낮은 유사도는 제외

### 5. 스냅샷 저장

- 매칭된 얼굴만 이미지로 저장
- 프레임 번호, 인물 ID, 유사도 정보 포함
- CSV 로그로 모든 감지 기록 저장

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
   └─ yh/                 # 입력 파일명 기반
      ├─ matches/         # 매칭된 스냅샷
      │   └─ match_f000123_hani_0.35.jpg
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

## 라이선스

MIT License
