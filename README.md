# EyeSis - 실시간 얼굴 식별·추적 시스템

> **EyeSis** = **Eye** + **Analysis** (눈 + 분석)

<div align="center">

![EyeSis Logo](https://img.shields.io/badge/🎯-EyeSis-4F46E5?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?style=flat-square&logo=fastapi&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-336791?style=flat-square&logo=postgresql&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

**InsightFace 기반 고성능 실시간 얼굴 인식 및 추적 시스템**

[시작하기](#-빠른-시작) • [기능](#-주요-기능) • [아키텍처](#-시스템-아키텍처) • [API](#-api-reference) • [기술 스택](#-기술-스택)

</div>

---

## 프로젝트 개요

EyeSis는 CCTV, 영상, 이미지에서 **특정 인물을 자동으로 식별하고 추적**하는 AI 기반 얼굴 인식 시스템입니다.

> **🎯 이 프로젝트의 핵심 혁신**: **한 장의 정면 사진만으로도** CCTV 환경에서 다양한 각도, 조명, 마스크 착용 상황에서 안정적으로 인식할 수 있는 **Dynamic Bank 시스템**을 구현했습니다. InsightFace의 최고 성능 모델인 **buffalo_l**을 활용하여 초기 등록의 한계를 극복하고, 자동 학습을 통해 인식률을 지속적으로 향상시킵니다.

### 🎯 핵심 가치: 한 장의 정면 사진으로 CCTV 인식

**이 프로젝트의 가장 중요한 기술적 도전과 해결책:**

| 핵심 기술 | 설명 |
|----------|------|
| 🦬 **Buffalo L 모델** | InsightFace의 최고 성능 모델(buffalo_l)을 적용하여 SOTA 얼굴 인식 성능 달성 (정확도 >95%) |
| 🔄 **Dynamic Bank 시스템** | **한 장의 정면 사진만으로도** CCTV에서 다양한 각도, 조명, 마스크 착용 상황에서 인식 가능하도록 자동으로 다양한 얼굴 임베딩을 수집하고 관리하는 핵심 시스템 |
| 📸 **초기 등록의 한계 극복** | 초기 등록 시 정면 사진 1장만 있어도, 영상 분석 중 감지된 얼굴의 임베딩을 자동으로 Dynamic Bank에 추가하여 인식률을 지속적으로 향상 |
| 🎥 **CCTV 환경 최적화** | 실제 CCTV 환경에서 발생하는 다양한 각도, 조명 변화, 마스크 착용 등 어려운 조건에서도 안정적인 인식 성능 제공 |

### 기술적 특징

| 특징 | 설명 |
|------|------|
| ⚡ **실시간 처리** | WebSocket 기반 저지연 스트리밍 (50-150ms) |
| 🛡️ **오탐 최소화** | 다층 필터링 시스템으로 오탐률 <5% |
| 🔄 **자동 학습** | 감지된 얼굴 임베딩 자동 수집으로 인식률 지속 향상 |
| 📊 **Multi-Bank 아키텍처** | Base Bank(정면), Dynamic Bank(자동 수집), Masked Bank(마스크)로 구성된 지능형 임베딩 관리 시스템 |

---

## 빠른 시작

### 1. 환경 설정

```bash
# 저장소 클론
git clone https://github.com/yourusername/EyeSis.git
cd EyeSis

# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정
cp backend/.env.example backend/.env
# DATABASE_URL, INSIGHTFACE_CTX_ID 등 설정
```

### 2. 데이터베이스 초기화

```bash
# PostgreSQL 데이터베이스 생성
psql -U postgres -c "CREATE DATABASE eyesis;"

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

**핵심: 한 장의 정면 사진으로 시작**

```
images/enroll/{person_id}/
    └── face.jpg  (정면 사진 1장만 필요!)
         ↓
    얼굴 감지 → 임베딩 추출 → Base Bank 생성
         ↓
outputs/embeddings/{person_id}/
    ├── bank_base.npy      # 초기 등록: 정면 사진 기반 (1×512)
    ├── centroid_base.npy  # 평균 임베딩 (512)
    └── bank_dynamic.npy   # ⭐ 자동 수집: CCTV 분석 중 다양한 각도/조건 임베딩 추가
```

**Dynamic Bank의 작동 원리:**
- 초기 등록 시 정면 사진 1장만으로 Base Bank 생성
- CCTV 영상 분석 중 감지된 얼굴의 임베딩을 자동으로 Dynamic Bank에 추가
- 각도별 다양성 체크 (정면, 측면, 프로필 등) 및 중복 방지 (유사도 0.9 이상 스킵)
- 시간이 지날수록 인식 성능이 자동으로 향상되는 자가 학습 시스템

### 2. 실시간 얼굴 인식 (CCTV 환경 최적화)

**Dynamic Bank를 활용한 지능형 인식:**

- **Multi-Bank 매칭**: Base Bank → Dynamic Bank → Masked Bank 순서로 최적 매칭 탐색
- **자동 학습**: 매칭 성공 시 해당 얼굴의 임베딩을 Dynamic Bank에 자동 추가
- **각도별 다양성**: 정면 사진 1장으로 시작하지만, 다양한 각도의 얼굴도 인식 가능
- **WebSocket 기반** 실시간 프레임 처리 (50-150ms 지연)
- **HTTP 폴백** 메커니즘으로 안정적 연결
- **인물별 타임라인** 시각화
- **감지 로그** CSV 내보내기

**인식 프로세스:**
```
CCTV 프레임 → 얼굴 감지 → 임베딩 추출
    ↓
Base Bank 매칭 (정면 사진 기반)
    ↓ (매칭 실패 시)
Dynamic Bank 매칭 (자동 수집된 다양한 각도)
    ↓ (매칭 성공 시)
임베딩을 Dynamic Bank에 자동 추가 (학습)
```

### 3. 고급 오탐 방지 시스템

```mermaid
flowchart TD
    Input["Input Detection"]
    
    L1["L1: sim_gap 체크<br/>1위-2위 유사도 차이 ≥5%"]
    L2["L2: bbox 필터링<br/>동일 영역 다중 매칭 제거"]
    L3["L3: 프레임 연속성<br/>최근 5프레임 내 동일 인물"]
    L4["L4: 화질 적응형 임계값<br/>환경별 동적 조정"]
    
    Output["Final Match ✓"]
    
    Input --> L1
    L1 -->|Pass| L2
    L2 -->|Pass| L3
    L3 -->|Pass| L4
    L4 -->|Pass| Output
    
    L1 -->|Fail| Reject["Reject"]
    L2 -->|Fail| Reject
    L3 -->|Fail| Reject
    L4 -->|Fail| Reject
    
    style Input fill:#3B82F6,stroke:#1E40AF,stroke-width:2px,color:#fff
    style L1 fill:#10B981,stroke:#059669,stroke-width:2px,color:#fff
    style L2 fill:#10B981,stroke:#059669,stroke-width:2px,color:#fff
    style L3 fill:#10B981,stroke:#059669,stroke-width:2px,color:#fff
    style L4 fill:#10B981,stroke:#059669,stroke-width:2px,color:#fff
    style Output fill:#4F46E5,stroke:#312E81,stroke-width:3px,color:#fff
    style Reject fill:#EF4444,stroke:#DC2626,stroke-width:2px,color:#fff
```

### 4. 적응형 임계값 시스템

```mermaid
flowchart TD
    Base["기본 임계값<br/>0.45"]
    
    Quality["화질 조정<br/>±0.04"]
    Mask["마스크 조정<br/>-0.02 ~ -0.05"]
    
    High["고화질<br/>+0.04"]
    Medium["중화질<br/>±0"]
    Low["저화질<br/>-0.03"]
    
    Final["최종 임계값<br/>0.28 ~ 0.50"]
    
    Base --> Quality
    Base --> Mask
    
    Quality --> High
    Quality --> Medium
    Quality --> Low
    
    High --> Final
    Medium --> Final
    Low --> Final
    Mask --> Final
    
    style Base fill:#4F46E5,stroke:#312E81,stroke-width:3px,color:#fff
    style Quality fill:#3B82F6,stroke:#1E40AF,stroke-width:2px,color:#fff
    style Mask fill:#F59E0B,stroke:#D97706,stroke-width:2px,color:#fff
    style Final fill:#10B981,stroke:#059669,stroke-width:3px,color:#fff
```

### 3-4. 고급 오탐 방지 및 적응형 임계값 시스템 (시각화)

<img width="2901" height="1604" alt="Untitled" src="https://github.com/user-attachments/assets/a2ed646b-fb66-4dd9-ac2c-fc4695efbc5b" />

---

## 시스템 아키텍처

### 전체 구조
<img width="2148" height="885" alt="system architecture" src="https://github.com/user-attachments/assets/b847ee1c-729b-4a38-843f-4886341af5b5" />

```mermaid
graph TB
    subgraph EyeSis["EyeSis System"]
        Frontend["Frontend<br/>(ES Modules)"]
        Backend["Backend<br/>(FastAPI)"]
        Database["Database<br/>(PostgreSQL)"]
        InsightFace["InsightFace<br/>(buffalo_l)"]
        
        Frontend <-->|WebSocket/HTTP| Backend
        Backend <-->|SQL| Database
        Backend -->|Model| InsightFace
    end
    
    style EyeSis fill:#4F46E5,stroke:#312E81,stroke-width:2px,color:#fff
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

| 분야 | 기술 | 버전 | 역할 |
|------|------|------|------|
| **Face Recognition** | InsightFace (buffalo_l) | 0.7.3 | ⭐ **핵심**: SOTA 성능의 얼굴 인식 모델 |
| **Runtime** | ONNX Runtime GPU | 1.18.0 | 고속 추론 엔진 |
| **Backend** | FastAPI + Uvicorn | 0.104+ | 비동기 웹 서버 |
| **Database** | PostgreSQL + SQLAlchemy | 15+ | 인물 정보 및 로그 저장 |
| **Frontend** | Vanilla JS (ES Modules) | ES2020+ | 경량 웹 인터페이스 |
| **Styling** | Tailwind CSS | 3.4 | 모던 UI 스타일링 |

### AI/ML

- **Detection**: RetinaFace (InsightFace 내장) - 얼굴 탐지
- **Embedding**: 512-d L2-normalized vectors (buffalo_l 모델 출력)
- **Matching**: Cosine Similarity - Multi-Bank 기반 최적 매칭
- **Tracking**: IoU-based + Temporal Filter - 프레임 간 일관성 유지
- **⭐ Dynamic Bank**: 자동 임베딩 수집 및 관리 시스템 (핵심 혁신)

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
EyeSis/
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
DATABASE_URL=postgresql://postgres:password@localhost:5432/eyesis
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

**Built with ❤️ by EyeSis Team**

[⬆ 맨 위로](#eyesis---실시간-얼굴-식별추적-시스템)

</div>
