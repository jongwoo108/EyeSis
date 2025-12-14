# EyeSis 프로젝트 기획서

<div align="center">

**실시간 얼굴 식별·추적 시스템 아키텍처 및 기술 설계 문서**

![Version](https://img.shields.io/badge/Version-2.0-blue?style=flat-square)
![Status](https://img.shields.io/badge/Status-Production-success?style=flat-square)
![Last Updated](https://img.shields.io/badge/Updated-2024.12-lightgrey?style=flat-square)

</div>

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [시스템 아키텍처](#2-시스템-아키텍처)
3. [기술 스택 및 Trade-off](#3-기술-스택-및-trade-off)
4. [핵심 알고리즘](#4-핵심-알고리즘)
5. [데이터 플로우](#5-데이터-플로우)
6. [모듈 설계](#6-모듈-설계)
7. [성능 최적화](#7-성능-최적화)
8. [보안 및 확장성](#8-보안-및-확장성)

---

## 1. 프로젝트 개요

### 1.1 핵심 목표

| 목표 | 설명 | KPI |
|------|------|-----|
| **정확한 인식** | InsightFace 기반 SOTA 얼굴 인식 | 정확도 >95% |
| **실시간 처리** | WebSocket 기반 저지연 스트리밍 | 지연시간 <150ms |
| **오탐 최소화** | 다층 필터링 시스템 | 오탐률 <5% |
| **확장 가능** | 모듈화된 아키텍처 | 인물 수 무제한 |

### 1.2 핵심 가치 제안

```mermaid
mindmap
  root((EyeSis<br/>Value Proposition))
    Accuracy
      >95% 정확도
      Multi-Bank 시스템
    Real-time
      50-150ms 지연
      WebSocket 기반
    Security
      <5% 오탐률
      다층 필터링
    Auto-Learning
      자동 임베딩 수집
      동적 Bank 관리
    Analytics
      타임라인 시각화
      CSV 내보내기
    Extensible
      모듈화 아키텍처
      ES Modules 기반
```

---

## 2. 시스템 아키텍처

### 2.1 전체 시스템 구조

<img width="2148" height="885" alt="system architecture" src="https://github.com/user-attachments/assets/b922c33b-8fb8-4b13-bd2c-1a5723da6947" />


### 2.2 프론트엔드 모듈 아키텍처
<img width="1524" height="1172" alt="FE_modules" src="https://github.com/user-attachments/assets/9583e763-61ea-4e75-8c5d-a56bdfe9a1c3" />


### 2.3 데이터 흐름

```mermaid
sequenceDiagram
    participant Client
    participant Server
    participant RetinaFace
    participant BuffaloL
    participant Bank
    participant Filter
    
    Client->>Server: 1. WebSocket Connect
    Client->>Server: 2. Frame (Base64)
    
    Server->>RetinaFace: 3. Face Detection
    RetinaFace-->>Server: Bounding Boxes
    
    Server->>BuffaloL: 4. Embedding Extraction
    BuffaloL-->>Server: 512-d Vector
    
    Server->>Bank: 5. Bank Matching
    Bank-->>Server: Best Match
    
    Server->>Filter: 6. Filtering
    Filter-->>Server: Filtered Result
    
    Server->>Client: 7. Detection Result (JSON)
    Client->>Client: 8. Canvas Render
```

---

## 3. 기술 스택 및 Trade-off

### 3.1 핵심 기술 선택

| 분야 | 선택 | 대안 | 선택 이유 |
|------|------|------|----------|
| **Face Model** | InsightFace buffalo_l | ArcFace, FaceNet | SOTA 성능, ONNX 지원 |
| **Detection** | RetinaFace | MTCNN, YOLOv5-face | InsightFace 통합, 높은 정확도 |
| **Backend** | FastAPI | Flask, Django | 비동기 처리, WebSocket 네이티브 |
| **Database** | PostgreSQL | MySQL, MongoDB | 복잡 쿼리, JSONB 지원 |
| **Frontend** | Vanilla JS + ES Modules | React, Vue | 경량화, 빠른 로딩 |

### 3.2 Trade-off 분석

#### 3.2.1 정확도 vs 속도

<img width="2900" height="660" alt="tradeoff" src="https://github.com/user-attachments/assets/d3640315-451b-44d0-bb80-e7797edada25" />

#### 3.2.2 임베딩 저장 방식

| 방식 | 정확도 | 속도 | 메모리 | 사용 시점 |
|------|--------|------|--------|----------|
| **Bank** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | 기본 |
| **Centroid** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Fallback |

---

## 4. 핵심 알고리즘

### 4.1 Multi-Bank 임베딩 시스템

<img width="1572" height="1061" alt="multi-bank" src="https://github.com/user-attachments/assets/4e25b700-a96f-44f4-aa62-f9e0b98334a1" />


### 4.2 다층 오탐 방지 시스템/4.3 적응형 임계값 시스템

<img width="2901" height="1604" alt="Untitled" src="https://github.com/user-attachments/assets/1dc0de4e-ba85-48cb-abd1-ce7a760d88e1" />

### 4.4 적응형 임계값 계산 로직

```python
def calculate_threshold(quality, mask_prob):
    base = 0.45
    
    # 화질 조정
    quality_adj = {
        'high': +0.04,
        'medium': 0,
        'low': -0.03
    }[quality]
    
    # 마스크 조정
    mask_adj = -0.05 * mask_prob
    
    # 최종 임계값 (0.28 ~ 0.50 범위)
    return clamp(base + quality_adj + mask_adj, 0.28, 0.50)
```

---

## 5. 데이터 플로우

### 5.1 인물 등록 플로우

<img width="2917" height="396" alt="enroll_flow" src="https://github.com/user-attachments/assets/0e7da950-3f1d-42ba-b7bb-fb9bb4839f9f" />

### 5.2 실시간 감지 플로우

```mermaid
flowchart TD
    Input["프레임 입력"]
    Skip{"프레임 스킵?"}
    Detect["얼굴 감지<br/>RetinaFace"]
    Process["임베딩 추출<br/>Bank 매칭<br/>임계값 계산"]
    Filter["오탐 방지 필터링<br/>L1-L4"]
    Classify["결과 분류"]
    Send["클라이언트 전송"]
    
    Input --> Skip
    Skip -->|No| Detect
    Skip -->|Yes| Send
    Detect --> Process
    Process --> Filter
    Filter --> Classify
    Classify --> Send
    
    style Input fill:#60A5FA,stroke:#3B82F6,stroke-width:2px,color:#fff
    style Skip fill:#FB923C,stroke:#F97316,stroke-width:2px,color:#fff
    style Detect fill:#34D399,stroke:#10B981,stroke-width:2px,color:#fff
    style Process fill:#A78BFA,stroke:#8B5CF6,stroke-width:2px,color:#fff
    style Filter fill:#F87171,stroke:#EF4444,stroke-width:2px,color:#fff
    style Classify fill:#818CF8,stroke:#6366F1,stroke-width:2px,color:#fff
    style Send fill:#34D399,stroke:#10B981,stroke-width:3px,color:#fff
```

---

## 6. 모듈 설계

### 6.1 프론트엔드 모듈 책임

| 모듈 | 책임 | 주요 함수 |
|------|------|----------|
| `config.js` | 설정 관리 | `API_BASE_URL`, `WS_URL` |
| `state.js` | 상태 관리 | `state` 객체 |
| `ui.js` | DOM 참조 | `initUI()` |
| `utils.js` | 유틸리티 | `formatTime()`, `getCategoryStyle()` |
| `api.js` | API 호출 | `loadPersons()`, `checkServerHealth()` |
| `handlers.js` | 이벤트 처리 | 15+ 핸들러 함수 |
| `timeline.js` | 타임라인 | `renderTimelineWithMerging()` |
| `persons.js` | 인물 관리 | `createSuspectCard()` |
| `clips.js` | 클립 기능 | `downloadVideoClip()` |
| `snapshots.js` | 스냅샷 | `renderSnapshotCard()` |
| `log.js` | 로그 관리 | `addDetectionLogItem()` |
| `detection.js` | 박스 렌더링 | `drawDetections()` |
| `enroll.js` | 등록 폼 | `checkFormValidity()` |

### 6.2 백엔드 서비스 책임

| 서비스 | 책임 | 주요 메서드 |
|--------|------|------------|
| `FaceDetection` | 얼굴 감지/인식 | `detect()`, `match()` |
| `BankManager` | Bank CRUD | `add_embedding()`, `get_best_match()` |
| `TemporalFilter` | 시간적 일관성 | `check_continuity()` |
| `DataLoader` | 데이터 로딩 | `load_gallery()` |

---

## 7. 성능 최적화

### 7.1 최적화 전략

| 영역 | 전략 | 효과 |
|------|------|------|
| **네트워크** | WebSocket 지속 연결 | 핸드셰이크 오버헤드 제거 |
| **프레임** | 동적 스킵 | 서버 부하 분산 |
| **렌더링** | Canvas 캐싱 | 불필요한 렌더링 방지 |
| **모델** | ONNX Runtime | 최적화된 추론 |
| **메모리** | Bank 크기 제한 | 메모리 사용량 관리 |

### 7.2 성능 지표

```mermaid
graph LR
    subgraph Latency["Latency"]
        WS["WebSocket<br/>50-150ms"]
        HTTP["HTTP<br/>100-300ms"]
    end
    
    subgraph Throughput["Throughput"]
        GPU["GPU<br/>15+ FPS"]
        CPU["CPU<br/>5-8 FPS"]
    end
    
    subgraph Memory["Memory"]
        Model["Model<br/>~500MB"]
        Bank["Bank<br/>~10KB/person"]
    end
    
    subgraph Accuracy["Accuracy"]
        Precision["Precision<br/>>95%"]
        FPR["FPR<br/><5%"]
    end
    
    style Latency fill:#60A5FA,stroke:#3B82F6,stroke-width:2px,color:#fff
    style Throughput fill:#34D399,stroke:#10B981,stroke-width:2px,color:#fff
    style Memory fill:#FB923C,stroke:#F97316,stroke-width:2px,color:#fff
    style Accuracy fill:#A78BFA,stroke:#8B5CF6,stroke-width:2px,color:#fff
```

---

## 8. 보안 및 확장성

### 8.1 보안 고려사항

| 영역 | 구현 | 상태 |
|------|------|------|
| **CORS** | 화이트리스트 방식 | 🔄 개발: 전체 허용 |
| **인증** | JWT 토큰 | 📋 계획 |
| **데이터 암호화** | HTTPS + WSS | ✅ 지원 |
| **입력 검증** | Pydantic 스키마 | ✅ 적용 |


---

## 9. 결론

EyeSis는 **정확도, 실시간성, 확장성**을 균형있게 달성한 얼굴 인식 시스템입니다.

### 핵심 성과

| 지표 | 목표 | 달성 |
|------|------|------|
| 정확도 | >95% | ✅ |
| 오탐률 | <5% | ✅ |
| 지연시간 | <200ms | ✅ 50-150ms |
| 코드 모듈화 | - | ✅ 13개 모듈 |

### 주요 Trade-off 결정

1. **정확도 vs 속도**: Bank 방식으로 정확도 우선
2. **복잡도 vs 정확도**: 다층 필터링으로 정확도 향상
3. **메모리 vs 정확도**: Dynamic Bank로 자동 학습

---

<div align="center">

**Last Updated: 2024.12**

</div>
