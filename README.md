# FaceWatch - 실시간 얼굴 식별·추적 시스템

<div align="center">

![FaceWatch Logo](https://img.shields.io/badge/🎯-FaceWatch-4F46E5?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?style=flat-square&logo=fastapi&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-336791?style=flat-square&logo=postgresql&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

**InsightFace 기반 고성능 실시간 얼굴 인식 및 추적 시스템**

[시작하기](#-빠른-시작) • [기능](#-주요-기능) • [아키텍처](#-시스템-아키텍처) • [API](#-api-reference) • [기술 스택](#-기술-스택)

</div>

---

## 프로젝트 개요

FaceWatch는 CCTV, 영상, 이미지에서 **특정 인물을 자동으로 식별하고 추적**하는 AI 기반 얼굴 인식 시스템입니다.

### 핵심 가치

| 특징 | 설명 |
|------|------|
| 🎯 **높은 정확도** | InsightFace buffalo_l 모델 기반 SOTA 성능 (정확도 >95%) |
| ⚡ **실시간 처리** | WebSocket 기반 저지연 스트리밍 (50-150ms) |
| 🛡️ **오탐 최소화** | 다층 필터링 시스템으로 오탐률 <5% |
| 🔄 **자동 학습** | 감지된 얼굴 임베딩 자동 수집으로 인식률 지속 향상 |

---

## 빠른 시작

### 1. 환경 설정

```bash
# 저장소 클론
git clone https://github.com/yourusername/FaceWatch.git
cd FaceWatch

# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정
cp backend/.env.example backend/.env
# DATABASE_URL, INSIGHTFACE_CTX_ID 등 설정
```

### 2. 데이터베이스 초기화

```bash
# PostgreSQL 데이터베이스 생성
psql -U postgres -c "CREATE DATABASE facewatch;"

# 데이터 마이그레이션
python backend/init_db.py
```

### 3. 서버 실행

```bash
# 백엔드 서버 시작
uvicorn backend.main:app --reload --host 0.0.0.0 --port 5000

# 프론트엔드 서버 시작 (별도 터미널)
cd web && python -m http.server 5500
```

### 4. 접속

- **웹 UI**: http://localhost:5500
- **API 문서**: http://localhost:5000/docs

---

## 주요 기능

### 1. 인물 등록 (Face Enrollment)

```
images/enroll/{person_id}/
    └── face.jpg
         ↓
    얼굴 감지 → 임베딩 추출 → Bank 생성
         ↓
outputs/embeddings/{person_id}/
    ├── bank_base.npy      # Multi-angle embeddings (N×512)
    ├── centroid_base.npy  # Average embedding (512)
    └── bank_dynamic.npy   # Auto-collected embeddings
```

### 2. 실시간 얼굴 인식

- **WebSocket 기반** 실시간 프레임 처리
- **HTTP 폴백** 메커니즘으로 안정적 연결
- **인물별 타임라인** 시각화
- **감지 로그** CSV 내보내기

### 3. 고급 오탐 방지 시스템&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;4. 적응형 임계값 시스템
<img width="2901" height="1604" alt="Untitled" src="https://github.com/user-attachments/assets/a2ed646b-fb66-4dd9-ac2c-fc4695efbc5b" />

---

## 시스템 아키텍처

### 전체 구조

```mermaid
graph TB
    subgraph FaceWatch["FaceWatch System"]
        Frontend["Frontend<br/>(ES Modules)"]
        Backend["Backend<br/>(FastAPI)"]
        Database["Database<br/>(PostgreSQL)"]
        InsightFace["InsightFace<br/>(buffalo_l)"]
        
        Frontend <-->|WebSocket/HTTP| Backend
        Backend <-->|SQL| Database
        Backend -->|Model| InsightFace
    end
    
    style FaceWatch fill:#4F46E5,stroke:#312E81,stroke-width:2px,color:#fff
    style Frontend fill:#10B981,stroke:#059669,stroke-width:2px,color:#fff
    style Backend fill:#3B82F6,stroke:#1E40AF,stroke-width:2px,color:#fff
    style Database fill:#8B5CF6,stroke:#6D28D9,stroke-width:2px,color:#fff
    style InsightFace fill:#F59E0B,stroke:#D97706,stroke-width:2px,color:#fff
```

### 프론트엔드 모듈 구조

```mermaid
graph TD
    Entry["script.js<br/>(Entry Point)<br/>~2,100 lines"]
    
    subgraph Core["Core Modules"]
        Config["config.js<br/>설정 및 URL"]
        State["state.js<br/>전역 상태 관리"]
        UI["ui.js<br/>DOM 요소 참조"]
        Utils["utils.js<br/>유틸리티 함수"]
    end
    
    subgraph Features["Feature Modules"]
        API["api.js<br/>API 호출"]
        Handlers["handlers.js<br/>이벤트 핸들러"]
        Timeline["timeline.js<br/>타임라인 렌더링"]
        Persons["persons.js<br/>인물 관리 UI"]
        Clips["clips.js<br/>클립 기능"]
        Snapshots["snapshots.js<br/>스냅샷 기능"]
        Log["log.js<br/>감지 로그"]
        Detection["detection.js<br/>박스 렌더링"]
        Enroll["enroll.js<br/>등록 폼"]
    end
    
    Entry --> Core
    Entry --> Features
    Core --> Features
    
    style Entry fill:#4F46E5,stroke:#312E81,stroke-width:3px,color:#fff
    style Core fill:#10B981,stroke:#059669,stroke-width:2px,color:#fff
    style Features fill:#3B82F6,stroke:#1E40AF,stroke-width:2px,color:#fff
```

### 백엔드 구조

```mermaid
graph TD
    Main["main.py<br/>FastAPI 앱 진입점"]
    Config["config.py<br/>설정 관리"]
    DB["database.py<br/>SQLAlchemy 모델"]
    
    subgraph API["API Layer"]
        DetectionAPI["detection.py<br/>감지 API<br/>(HTTP + WebSocket)"]
        PersonsAPI["persons.py<br/>인물 CRUD"]
        VideoAPI["video.py<br/>비디오 처리"]
    end
    
    subgraph Services["Service Layer"]
        FaceDetection["face_detection.py<br/>얼굴 감지"]
        FaceEnroll["face_enroll.py<br/>인물 등록"]
        DataLoader["data_loader.py<br/>데이터 로딩"]
        BankManager["bank_manager.py<br/>Bank 관리"]
        TemporalFilter["temporal_filter.py<br/>시간 필터"]
    end
    
    subgraph Utils["Utils Layer"]
        DeviceConfig["device_config.py<br/>GPU/CPU 설정"]
        ImageUtils["image_utils.py<br/>이미지 처리"]
        AngleDetector["face_angle_detector.py<br/>각도 감지"]
        MaskDetector["mask_detector.py<br/>마스크 감지"]
    end
    
    Main --> Config
    Main --> DB
    Main --> API
    API --> Services
    Services --> Utils
    
    style Main fill:#4F46E5,stroke:#312E81,stroke-width:3px,color:#fff
    style API fill:#10B981,stroke:#059669,stroke-width:2px,color:#fff
    style Services fill:#3B82F6,stroke:#1E40AF,stroke-width:2px,color:#fff
    style Utils fill:#F59E0B,stroke:#D97706,stroke-width:2px,color:#fff
```

---

## API Reference

### WebSocket `/ws/detect`

실시간 프레임 감지 스트리밍

```json
// Request
{
  "type": "frame",
  "data": {
    "image": "base64_string",
    "suspect_ids": ["person_001"],
    "frame_id": 123,
    "video_time": 12.5
  }
}

// Response
{
  "type": "detection",
  "data": {
    "frame_id": 123,
    "detections": [{
      "bbox": [100, 50, 200, 180],
      "name": "홍길동",
      "confidence": 87,
      "status": "criminal",
      "angle_type": "front"
    }],
    "alert": true,
    "snapshot_base64": "..."
  }
}
```

### REST API

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/api/detect` | POST | 단일 프레임 감지 |
| `/api/persons` | GET | 등록 인물 목록 |
| `/api/persons/{id}` | PUT/DELETE | 인물 수정/삭제 |
| `/api/enroll` | POST | 인물 등록 |
| `/api/logs` | GET | 감지 로그 조회 |
| `/api/extract_clip` | POST | 비디오 클립 추출 |
| `/api/health` | GET | 서버 상태 확인 |

---

## 기술 스택

### Core

| 분야 | 기술 | 버전 |
|------|------|------|
| **Face Recognition** | InsightFace (buffalo_l) | 0.7.3 |
| **Runtime** | ONNX Runtime GPU | 1.18.0 |
| **Backend** | FastAPI + Uvicorn | 0.104+ |
| **Database** | PostgreSQL + SQLAlchemy | 15+ |
| **Frontend** | Vanilla JS (ES Modules) | ES2020+ |
| **Styling** | Tailwind CSS | 3.4 |

### AI/ML

- **Detection**: RetinaFace (InsightFace 내장)
- **Embedding**: 512-d L2-normalized vectors
- **Matching**: Cosine Similarity
- **Tracking**: IoU-based + Temporal Filter

---

## 📊 성능 지표

| 지표 | 목표 | 실제 |
|------|------|------|
| 정확도 (Accuracy) | >95% | ✅ 달성 |
| 오탐률 (FPR) | <5% | ✅ 달성 |
| 미탐률 (FNR) | <10% | ✅ 달성 |
| 처리 속도 (GPU) | >10 FPS | ✅ 15+ FPS |
| 지연 시간 (Latency) | <200ms | ✅ 50-150ms |

---

## 📁 프로젝트 구조

```
FaceWatch/
├── backend/              # FastAPI 백엔드
├── web/                  # 프론트엔드
│   ├── modules/          # ES Modules (13개)
│   └── index.html
├── outputs/              # 출력 폴더
│   ├── embeddings/       # 인물별 임베딩
│   └── results/          # 분석 결과
├── scripts/              # 유틸리티 스크립트
├── requirements.txt
└── README.md
```

---

## 설정

### 환경 변수 (`backend/.env`)

```env
DATABASE_URL=postgresql://postgres:password@localhost:5432/facewatch
HOST=0.0.0.0
PORT=5000
INSIGHTFACE_MODEL=buffalo_l
INSIGHTFACE_CTX_ID=0  # GPU: 0, CPU: -1
```

### 임계값 설정 (`backend/config.py`)

```python
MAIN_THRESHOLD = 0.45          # 기본 임계값
SUSPECT_THRESHOLD = 0.48       # 용의자 모드
DYNAMIC_BANK_THRESHOLD = 0.9   # 중복 체크 임계값
```

---

## 로드맵

### 완료

- [x] 실시간 WebSocket 감지
- [x] Multi-Bank 임베딩 시스템
- [x] 적응형 임계값 시스템
- [x] 다층 오탐 방지 필터링
- [x] ES Modules 프론트엔드 리팩토링
- [x] 인물별 타임라인 시각화

### 진행 중

- [ ] Face Anti-Spoofing (딥페이크 방지)
- [ ] 다중 카메라 지원

### 계획

- [ ] 분산 처리 (멀티 GPU)
- [ ] 클라우드 배포 (AWS/GCP)
- [ ] 모바일 앱 지원

---

## 라이선스
 
MIT License - 자유롭게 사용, 수정, 배포 가능

---

<div align="center">

**Built with ❤️ by FaceWatch Team**

[⬆ 맨 위로](#facewatch---실시간-얼굴-식별추적-시스템)

</div>
