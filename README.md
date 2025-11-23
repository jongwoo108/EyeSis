# 🎯 FaceWatch - 실시간 얼굴 식별·추적 시스템

<div align="center">

**InsightFace 기반 고성능 얼굴 인식 및 실시간 추적 AI 시스템**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![InsightFace](https://img.shields.io/badge/InsightFace-buffalo__l-orange.svg)](https://github.com/deepinsight/insightface)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-14+-blue.svg)](https://www.postgresql.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[프로젝트 개요](#-프로젝트-개요) • [주요 기능](#-주요-기능) • [기술 스택](#-기술-스택) • [빠른 시작](#-빠른-시작) • [아키텍처](#-아키텍처)

</div>

---

## 📋 프로젝트 개요

**FaceWatch**는 CCTV 영상, 이미지, 실시간 스트림에서 특정 인물을 자동으로 식별하고 추적하는 엔터프라이즈급 얼굴 인식 시스템입니다. InsightFace의 고성능 모델을 기반으로 하며, WebSocket 기반 실시간 통신, Bank 임베딩 시스템, 옆모습 감지 등 고급 기능을 제공합니다.

### 🎯 핵심 가치

- **실시간 처리**: WebSocket 기반 양방향 통신으로 50-150ms 지연 시간 달성
- **높은 정확도**: Bank 임베딩 시스템과 적응형 임계값으로 오탐 최소화
- **다양한 각도 지원**: 정면부터 프로필까지 모든 각도의 얼굴 인식 가능
- **자동 학습**: 감지된 얼굴 임베딩을 자동으로 Bank에 추가하여 성능 지속 향상
- **엔터프라이즈급**: PostgreSQL 기반 데이터 영구 저장 및 감지 로그 관리

---

## ✨ 주요 기능

### 🔍 1. 실시간 얼굴 감지 및 인식

- **WebSocket 기반 실시간 통신**: 프레임 끊김 없는 부드러운 실시간 감지
- **다중 용의자 동시 추적**: 여러 명의 대상자를 동시에 모니터링
- **자동 재연결**: 연결 끊김 시 지수 백오프 방식으로 자동 재연결
- **HTTP 폴백**: WebSocket 실패 시 자동으로 HTTP API로 전환

### 🎨 2. 고급 박스 렌더링

- **시각적 구분**: 범죄자(빨강), 일반인(초록), 미확인(노랑) 색상 구분
- **모서리 강조**: 박스 네 모서리 강조로 가시성 향상
- **동적 텍스트 배치**: 화면 경계를 고려한 스마트 텍스트 위치 조정
- **각도 정보 표시**: 감지된 얼굴의 각도 정보 실시간 표시

### 🧠 3. Bank 임베딩 시스템

- **다중 각도 임베딩 저장**: 한 인물에 대한 다양한 각도의 얼굴 임베딩 보관
- **자동 학습**: 매칭 성공 시 해당 얼굴의 임베딩을 자동으로 Bank에 추가
- **중복 방지**: 유사도 0.95 이상인 임베딩은 중복으로 간주하여 스킵
- **성능 향상**: 시간이 지날수록 인식 정확도 자동 향상

### 📐 4. 얼굴 각도 감지

- **5가지 각도 분류**: 정면, 왼쪽, 오른쪽, 왼쪽 프로필, 오른쪽 프로필
- **Yaw 각도 계산**: InsightFace 랜드마크 기반 정확한 각도 측정
- **각도별 통계**: 각도별 인식 성공률 통계 제공
- **UI 표시**: 감지된 각도 정보를 박스 레이블에 표시

### 🎭 5. 마스크 감지 및 적응형 임계값

- **자동 마스크 감지**: 유사도 기반으로 마스크 착용 가능성 자동 추정
- **동적 임계값 조정**: 마스크 착용 시 임계값 자동 낮춤 (0.30 → 0.22~0.28)
- **기존 이미지 활용**: 마스크 없는 등록 이미지로도 마스크 쓴 얼굴 인식 가능
- **오탐 방지**: 최소 임계값 보장으로 오탐 방지

### 🖼️ 6. 화질 기반 적응형 임계값

- **자동 화질 추정**: 얼굴 크기와 이미지 크기 비율로 화질 자동 판단
- **동적 임계값 조정**: 
  - 고화질: 높은 임계값 (오탐 방지 강화)
  - 저화질: 낮은 임계값 (인식률 유지)
- **3단계 화질 분류**: High (≥150px, ≥5%), Medium (≥100px, ≥2%), Low (그 외)

### 🛡️ 7. 고급 오탐 방지

- **bbox 기반 다중 매칭 필터링**: 같은 얼굴 영역에서 여러 인물 매칭 시 자동 처리
- **프레임 간 연속성 체크**: 최근 5프레임 내 같은 인물 매칭 여부 확인
- **sim_gap 체크**: 최고 유사도와 두 번째 유사도 차이 확인 (최소 5% 차이 필요)
- **검토 대상 자동 분리**: 의심스러운 매칭을 별도 폴더에 자동 분리

### 💾 8. 데이터베이스 통합

- **PostgreSQL 지원**: 인물 정보 및 감지 로그 영구 저장
- **다중 데이터 소스**: PostgreSQL → Bank 임베딩 → JSON 순서로 자동 fallback
- **감지 로그 저장**: 모든 감지 이벤트를 데이터베이스에 자동 저장
- **API 제공**: 감지 로그 조회 및 통계 API 제공

---

## 🛠️ 기술 스택

### Backend
| 분야 | 기술 |
|------|------|
| **Face Recognition** | InsightFace (buffalo_l), ONNX Runtime |
| **Detection** | RetinaFace (InsightFace 내장) |
| **Backend Framework** | FastAPI, Uvicorn |
| **WebSocket** | FastAPI WebSocket |
| **Database** | PostgreSQL, SQLAlchemy ORM |
| **Image Processing** | OpenCV, NumPy |

### Frontend
| 분야 | 기술 |
|------|------|
| **UI Framework** | HTML5, JavaScript (ES6+) |
| **Styling** | Tailwind CSS |
| **Canvas API** | HTML5 Canvas (박스 렌더링) |
| **Real-time** | WebSocket API |

### DevOps & Tools
| 분야 | 기술 |
|------|------|
| **Language** | Python 3.9+ |
| **Package Manager** | pip |
| **Version Control** | Git |
| **Analysis** | NumPy, Matplotlib, Seaborn, Scikit-learn |

---

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 저장소 클론
git clone <repository-url>
cd FaceWatch

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 인물 등록

```bash
# 1. 등록용 이미지를 images/enroll/<인물명>/ 폴더에 배치
# 예: images/enroll/hani/hani.jpg

# 2. 등록 실행
python src/face_enroll.py
# MODE = 1 선택하여 기본 등록 실행
```

### 3. 데이터베이스 설정 (선택사항, 권장)

```bash
# PostgreSQL 설치 및 데이터베이스 생성
psql -U postgres
CREATE DATABASE facewatch;
\q

# 환경 변수 설정 (backend/.env)
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/facewatch

# 데이터 마이그레이션
python backend/init_db.py
```

### 4. 백엔드 서버 실행

```bash
# 프로젝트 루트에서 실행
uvicorn backend.main:app --reload --host 0.0.0.0 --port 5000
```

서버가 시작되면:
- API 엔드포인트: http://localhost:5000/api/detect
- API 문서: http://localhost:5000/docs (FastAPI 자동 생성)
- WebSocket: ws://localhost:5000/ws/detect

### 5. 프론트엔드 실행

**방법 1: VSCode Live Server (권장)**
1. VSCode에서 `web/index.html` 파일 열기
2. 우클릭 → "Open with Live Server"

**방법 2: Python HTTP 서버**
```bash
cd web
python -m http.server 5500
```

**방법 3: Node.js http-server**
```bash
npx http-server web -p 5500
```

### 6. 웹 UI 사용

1. 브라우저에서 접근: http://localhost:5500/index.html
2. CCTV 영상 파일 업로드 (MP4, AVI, MOV 등)
3. 감지할 용의자 선택 (카드 클릭, 다중 선택 가능)
4. "모니터링 시작" 버튼 클릭
5. 우측 제어판에서 "AI 감지 활성화" 토글 ON
6. 실시간 감지 결과 확인

---

## 🏗️ 아키텍처

### 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                      클라이언트 (웹 브라우저)                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  HTML5 Video │  │ Canvas API   │  │  WebSocket   │     │
│  │   (재생)      │  │ (박스 렌더링) │  │  (실시간 통신) │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└───────────────────────┬─────────────────────────────────────┘
                        │ WebSocket / HTTP
                        │ (Base64 이미지)
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI 백엔드 서버                       │
│  ┌────────────────────────────────────────────────────┐   │
│  │  WebSocket 엔드포인트 (/ws/detect)                │   │
│  │  HTTP API 엔드포인트 (/api/detect)                │   │
│  └────────────────────────────────────────────────────┘   │
│                        │                                    │
│                        ▼                                    │
│  ┌────────────────────────────────────────────────────┐   │
│  │  InsightFace 모델 (buffalo_l)                      │   │
│  │  - 얼굴 감지 (RetinaFace)                          │   │
│  │  - 임베딩 추출 (512차원)                            │   │
│  │  - 얼굴 각도 추정                                   │   │
│  └────────────────────────────────────────────────────┘   │
│                        │                                    │
│                        ▼                                    │
│  ┌────────────────────────────────────────────────────┐   │
│  │  매칭 엔진                                          │   │
│  │  - Bank 임베딩 매칭                                 │   │
│  │  - 적응형 임계값 적용                               │   │
│  │  - 오탐 방지 로직                                   │   │
│  └────────────────────────────────────────────────────┘   │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  PostgreSQL  │ │ Bank 임베딩  │ │ JSON (레거시)│
│  (우선순위 1) │ │ (우선순위 2) │ │ (우선순위 3) │
└──────────────┘ └──────────────┘ └──────────────┘
```

### 데이터 흐름

```
1. 클라이언트: 비디오 프레임 캡처 (Base64 인코딩)
   ↓
2. WebSocket: 실시간 프레임 전송
   ↓
3. 서버: Base64 디코딩 → OpenCV 이미지
   ↓
4. InsightFace: 얼굴 감지 및 임베딩 추출
   ↓
5. 매칭 엔진: Bank 임베딩과 Cosine 유사도 비교
   ↓
6. 결과 처리: 좌표, 메타데이터, 각도 정보 추출
   ↓
7. WebSocket: JSON 응답 (bbox, name, confidence 등)
   ↓
8. 클라이언트: Canvas API로 박스 렌더링
```

### 프로젝트 구조

```
FaceWatch/
├── backend/                 # FastAPI 백엔드
│   ├── main.py              # 메인 서버 (WebSocket + HTTP API)
│   ├── database.py          # PostgreSQL ORM 모델
│   ├── init_db.py           # 데이터베이스 초기화 및 마이그레이션
│   └── database/            # 레거시 JSON 데이터베이스
│
├── web/                     # 프론트엔드
│   ├── index.html           # 메인 페이지
│   ├── script.js            # 클라이언트 로직 (WebSocket, 렌더링)
│   └── style.css            # 커스텀 스타일
│
├── src/                     # 코어 로직
│   ├── face_enroll.py       # 인물 등록 스크립트
│   ├── face_match_cctv.py  # 통합 인식 스크립트
│   └── utils/               # 유틸리티 모듈
│       ├── gallery_loader.py      # Bank 임베딩 로더
│       ├── face_angle_detector.py # 얼굴 각도 감지
│       ├── mask_detector.py      # 마스크 감지
│       └── device_config.py       # GPU/CPU 설정
│
├── outputs/                 # 출력 폴더
│   ├── embeddings/          # Bank 임베딩 (인물별 폴더)
│   │   └── <person>/
│   │       ├── bank.npy     # 다중 각도 임베딩
│   │       ├── centroid.npy # 평균 임베딩
│   │       └── angles.json  # 각도 메타데이터
│   └── results/             # 분석 결과
│
├── images/                  # 이미지 입력
│   ├── enroll/              # 등록용 이미지
│   └── source/              # 분석용 이미지
│
└── videos/                  # 영상 입력
    └── source/              # 분석용 영상
```

---

## 📊 주요 성과 및 개선 사항

### 성능 개선 이력

| 단계 | 네트워크 사용량 | 지연 시간 | 프레임 끊김 | 박스 안정성 |
|------|----------------|----------|------------|------------|
| **서버사이드 렌더링** | 매우 높음 (~500KB/프레임) | 200-500ms | 심함 | - |
| **클라이언트사이드 렌더링** | 낮음 (~2KB/프레임) | 100-300ms | 보통 | 불안정 |
| **WebSocket 기반** | 매우 낮음 (~2KB/프레임) | **50-150ms** | **없음** | **안정적** |

### 주요 개선 사항

#### 1. 실시간 통신 최적화
- ✅ 서버사이드 렌더링 → 클라이언트사이드 렌더링 전환
- ✅ HTTP 요청 → WebSocket 양방향 통신 도입
- ✅ 프레임 끊김 현상 완전 해결
- ✅ 박스 튀는 현상 해결 (프레임 ID 추적, 순서 보장)

#### 2. 인식 정확도 향상
- ✅ Bank 임베딩 시스템 도입 (다중 각도 임베딩 저장)
- ✅ 자동 학습 기능 (감지된 얼굴 임베딩 자동 추가)
- ✅ 옆모습 감지 기능 추가
- ✅ 마스크 감지 및 적응형 임계값

#### 3. 오탐 방지 강화
- ✅ bbox 기반 다중 매칭 필터링
- ✅ 프레임 간 연속성 체크
- ✅ 화질 기반 적응형 임계값
- ✅ 검토 대상 자동 분리

#### 4. 사용자 경험 개선
- ✅ 박스 렌더링 최적화 (모서리 강조, 동적 텍스트 배치)
- ✅ 다중 용의자 동시 추적
- ✅ 자동 재연결 기능
- ✅ HTTP 폴백 메커니즘

자세한 개선 이력은 [SERVICE_IMPROVEMENTS.md](SERVICE_IMPROVEMENTS.md)를 참조하세요.

---

## 📖 사용 예시

### 이미지 분석

```bash
# 1. 분석할 이미지를 images/source/ 폴더에 배치
# 2. 스크립트에서 파일명 설정
#    input_filename = "test.jpg"
# 3. 분석 실행
python src/face_match_cctv.py
```

### 영상 분석

```bash
# 1. 분석할 영상을 videos/source/ 폴더에 배치
# 2. 스크립트에서 파일명 설정
#    input_filename = "yh.MOV"
# 3. 분석 실행
python src/face_match_cctv.py
```

### 결과 확인

```
outputs/
├── embeddings/            # 임베딩 파일
│   └── hani/
│       ├── bank.npy       # 다중 각도 임베딩
│       ├── centroid.npy   # 평균 임베딩
│       └── angles.json   # 각도 메타데이터
└── results/               # 분석 결과
    └── yh_20240101_120000/
        ├── matches/       # 매칭된 스냅샷
        │   └── review/    # 검토 대상
        ├── logs/          # CSV 로그
        └── frames/         # 추출된 프레임
```

---

## 🔧 설정 및 커스터마이징

### 임계값 설정

기본 임계값: `BASE_THRESH = 0.32`

- **화질 기반 자동 조정**:
  - 고화질: `0.40` (오탐 방지 강화)
  - 중화질: `0.32` (기본값)
  - 저화질: `0.33` (인식률 유지)
- **마스크 착용 시**: `-0.02 ~ -0.05` 추가 조정
- **최소 임계값**: `0.28` (고화질), `0.30` (중화질), `0.28` (저화질)

### 데이터 소스 우선순위

백엔드는 다음 순서로 얼굴 데이터를 로드합니다:

1. **PostgreSQL 데이터베이스** (권장)
2. `outputs/embeddings/<person>/bank.npy` 또는 `centroid.npy` (fallback)
3. `backend/database/*.json` (레거시 지원)

---

## 📚 API 문서

### WebSocket 엔드포인트

**연결**: `ws://localhost:5000/ws/detect`

**클라이언트 → 서버**:
```json
{
  "type": "frame",
  "data": {
    "image": "data:image/jpeg;base64,...",
    "suspect_ids": ["hani", "danielle"],
    "frame_id": 123
  }
}
```

**서버 → 클라이언트**:
```json
{
  "type": "detection",
  "data": {
    "frame_id": 123,
    "detections": [
      {
        "bbox": [100, 200, 300, 400],
        "name": "하니",
        "confidence": 85.5,
        "color": "green",
        "angle_type": "front",
        "status": "normal"
      }
    ],
    "alert": false
  }
}
```

### HTTP API 엔드포인트

**POST `/api/detect`**
- 얼굴 감지 및 인식 (WebSocket 폴백용)

**GET `/api/persons`**
- 등록된 모든 인물 목록 조회

**GET `/api/logs?limit=100`**
- 감지 로그 조회 (최근 N개)

자세한 API 문서는 http://localhost:5000/docs 에서 확인할 수 있습니다.

---

## 🗺️ 향후 개발 로드맵

### 완료된 기능 ✅
- [x] WebSocket 기반 실시간 통신
- [x] Bank 임베딩 시스템
- [x] 옆모습 감지
- [x] 마스크 감지 및 적응형 임계값
- [x] 화질 기반 적응형 임계값
- [x] 고급 오탐 방지
- [x] PostgreSQL 데이터베이스 통합
- [x] 박스 렌더링 최적화

### 계획 중 🚧
- [ ] 얼굴 ID 추적 (동일 인물 연속 감지)
- [ ] 박스 위치 보간 (더 부드러운 이동)
- [ ] 실시간 FPS 및 성능 모니터링
- [ ] 감지 히스토리 타임라인
- [ ] 알림 설정 및 필터링
- [ ] Face anti-spoofing (딥페이크 방지)

---

## 📄 라이선스

MIT License

---

## 👨‍💻 개발자

**Jongwoo Shin**

InsightFace + Computer Vision 기반 얼굴 인식 시스템 개발  
Cloud · AI Engineering · MLOps

---

## 📝 참고 문서

- [서비스 개선 이력](SERVICE_IMPROVEMENTS.md) - 기술적 진화 과정 상세 기록
- [백엔드 설정 가이드](backend/SETUP.md) - PostgreSQL 설정 및 데이터 마이그레이션
- [프로젝트 계획서](PROJECT_PLAN.md) - 초기 프로젝트 계획 및 설계

---

<div align="center">

**Made with ❤️ using InsightFace and FastAPI**

[⬆ 맨 위로 이동](#-facewatch---실시간-얼굴-식별추적-시스템)

</div>
