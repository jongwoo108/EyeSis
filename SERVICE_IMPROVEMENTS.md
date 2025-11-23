# FaceWatch 서비스 개선 이력

본 문서는 FaceWatch 실시간 얼굴 감지 서비스의 주요 개선 사항과 기술적 진화 과정을 기록합니다.

---

## 개요

FaceWatch는 초기 서버사이드 렌더링 방식에서 시작하여, 실시간성과 사용자 경험을 개선하기 위해 여러 단계의 최적화를 거쳐 현재의 WebSocket 기반 실시간 감지 시스템으로 발전했습니다.

---

## 1단계: 서버사이드 렌더링 (초기 구현)

### 구현 방식
- 클라이언트에서 비디오 프레임을 캡처하여 Base64로 인코딩
- HTTP POST 요청으로 서버에 전송
- 서버에서 얼굴 감지 및 인식 수행 후 **렌더링된 이미지(박스가 그려진 이미지)**를 Base64로 반환
- 클라이언트는 반환된 이미지를 화면에 표시

### 문제점
- **프레임 끊김 현상**: 매 프레임마다 이미지 인코딩/디코딩 및 네트워크 전송으로 인한 지연
- **높은 네트워크 대역폭 사용**: 렌더링된 이미지 전체를 전송해야 함
- **서버 부하 증가**: 이미지 렌더링 작업이 서버에서 수행됨
- **지연 시간 증가**: 요청-응답 사이클로 인한 실시간성 저하

### 기술 스택
- HTTP POST 요청
- Base64 이미지 인코딩/디코딩
- 서버사이드 OpenCV 이미지 렌더링

---

## 2단계: 클라이언트사이드 렌더링 (좌표 전송 방식)

### 구현 방식
- 서버는 얼굴 감지 결과를 **좌표 정보(bbox)와 메타데이터**만 JSON으로 반환
- 클라이언트에서 Canvas API를 사용하여 박스를 직접 렌더링
- 이미지 데이터는 전송하지 않고 좌표만 전송하여 네트워크 부하 감소

### 개선 사항
- 네트워크 대역폭 사용량 대폭 감소 (이미지 → 좌표 데이터)
- 서버 부하 감소 (렌더링 작업 제거)
- 응답 데이터 크기 감소로 전송 속도 향상

### 문제점
- **여전히 프레임 끊김 현상**: HTTP 요청-응답 사이클의 지연
- **박스 튀는 현상**: 
  - 비동기 요청 처리로 인한 프레임 순서 불일치
  - 네트워크 지연으로 인한 박스 위치 업데이트 지연
  - 프레임 스킵 시 이전 박스가 남아있는 현상
- **동기화 문제**: 비디오 재생 속도와 감지 결과 업데이트 속도 불일치

### 기술 스택
- HTTP POST 요청 (JSON 응답)
- Canvas API (클라이언트사이드 렌더링)
- 비동기 프레임 처리 (`setInterval`)

---

## 3단계: WebSocket 기반 실시간 통신

### 구현 방식
- WebSocket 연결을 통한 양방향 실시간 통신
- 클라이언트에서 프레임을 캡처하여 WebSocket으로 전송
- 서버에서 실시간으로 감지 결과를 스트리밍
- HTTP 폴백 메커니즘 구현 (WebSocket 실패 시 자동 전환)

### 개선 사항
- **프레임 끊김 현상 해결**: 
  - 지속적인 연결로 핸드셰이크 오버헤드 제거
  - 양방향 통신으로 실시간 데이터 스트리밍
- **박스 튀는 현상 완화**:
  - 프레임 ID 추적으로 순서 보장
  - 낮은 지연 시간으로 부드러운 업데이트
  - 마지막 감지 결과 캐싱으로 프레임 스킵 시에도 안정적 표시
- **자동 재연결**: 연결 끊김 시 지수 백오프(exponential backoff) 방식으로 재연결 시도
- **중복 요청 방지**: `isProcessing` 플래그로 동시 요청 방지

### 기술 스택
- WebSocket (FastAPI WebSocket 엔드포인트)
- HTTP 폴백 메커니즘
- 프레임 ID 추적 및 순서 보장
- 자동 재연결 로직

### 코드 구조
```javascript
// WebSocket 연결 및 프레임 전송
function connectWebSocket() { ... }
function sendWebSocketFrame(frameData, suspectIds) { ... }
function handleWebSocketMessage(message) { ... }

// HTTP 폴백
async function detectFrameToServerHTTP(frameData) { ... }
```

---

## 4단계: Bank 임베딩 시스템 도입

### 구현 방식
- **Bank 임베딩**: 한 인물에 대한 여러 각도의 얼굴 임베딩을 저장
- **Centroid 임베딩**: Bank의 평균값으로 계산된 대표 임베딩
- 매칭 시 Bank의 모든 임베딩과 비교하여 최고 유사도 선택
- 매칭된 얼굴의 임베딩을 자동으로 Bank에 추가 (학습)

### 개선 사항
- **정확도 향상**: 
  - 단일 임베딩 대신 여러 각도의 임베딩과 비교
  - 다양한 조명, 각도, 표정에 대한 내성 향상
- **자동 학습**: 
  - 감지된 얼굴의 임베딩을 자동으로 Bank에 추가
  - 시간이 지날수록 인식 정확도 향상
- **중복 방지**: 
  - 유사도 0.95 이상인 임베딩은 중복으로 간주하여 스킵
  - Bank 크기 최적화

### 기술 구현
- NumPy 배열 기반 Bank 저장 (`bank.npy`)
- 각도 정보 메타데이터 저장 (`angles.json`)
- Cosine 유사도 기반 매칭
- L2 정규화를 통한 임베딩 정규화

### 파일 구조
```
outputs/embeddings/<person_id>/
├── bank.npy          # N×512 차원 임베딩 배열
├── centroid.npy      # 512 차원 평균 임베딩
└── angles.json       # 각도 정보 메타데이터
```

---

## 5단계: 옆모습 감지 기능 추가

### 구현 방식
- **얼굴 각도 추정**: InsightFace의 얼굴 랜드마크를 이용한 yaw 각도 계산
- **각도 분류**: 
  - `front`: 정면 (-15° ~ 15°)
  - `left`: 왼쪽으로 약간 회전 (-45° ~ -15°)
  - `right`: 오른쪽으로 약간 회전 (15° ~ 45°)
  - `left_profile`: 왼쪽 프로필 (-90° ~ -45°)
  - `right_profile`: 오른쪽 프로필 (45° ~ 90°)
- **각도별 임베딩 저장**: Bank에 추가 시 각도 정보도 함께 저장
- **UI 표시**: 감지된 얼굴의 각도 정보를 박스 레이블에 표시

### 개선 사항
- **다양한 각도 인식**: 정면뿐만 아니라 옆모습, 프로필까지 인식 가능
- **각도 정보 표시**: 사용자가 얼굴 각도를 시각적으로 확인 가능
- **각도별 학습**: 다양한 각도의 얼굴을 Bank에 저장하여 인식 범위 확대

### 기술 구현
```python
# 얼굴 각도 추정
def estimate_face_angle(face) -> Tuple[str, float]:
    # 랜드마크 기반 yaw 각도 계산
    # 각도 범위에 따라 분류
    return angle_type, yaw_angle
```

### UI 표시
- 박스 레이블에 각도 정보 표시: `[왼쪽 프로필]`, `[오른쪽]` 등
- 정면이 아닌 경우에만 각도 정보 표시

---

## 6단계: 박스 렌더링 최적화 (최근 개선)

### 개선 사항

#### 1. 박스 스타일 개선
- **선 두께 증가**: 3px → 4px로 변경하여 가시성 향상
- **모서리 강조**: 박스의 네 모서리에 강조선 추가로 시각적 구분 향상
- **색상 구분**: 
  - 빨간색: 범죄자 감지
  - 초록색: 일반인 확인
  - 노란색: 미확인 얼굴

#### 2. 텍스트 레이아웃 최적화
- **중복 제거**: 각도 정보가 중복 표시되던 문제 해결
- **계층적 표시**:
  - 첫 번째 줄: 경고 메시지 (범죄자인 경우)
  - 두 번째 줄: 이름 및 신뢰도, 각도 정보
- **텍스트 배경**: 반투명 배경(80% 투명도)으로 가독성 향상

#### 3. 동적 위치 조정
- **화면 경계 체크**: 텍스트가 화면 밖으로 나가지 않도록 자동 조정
- **스마트 배치**: 
  - 박스 위쪽에 공간이 없으면 박스 아래쪽에 표시
  - 좌우 경계를 넘지 않도록 위치 조정

#### 4. 렌더링 성능 최적화
- **캔버스 크기 동기화**: 비디오 표시 크기와 캔버스 크기를 정확히 일치
- **좌표 변환 정확도 향상**: `object-contain` 스타일을 고려한 정확한 좌표 변환
- **불필요한 렌더링 제거**: 박스가 없을 때 즉시 캔버스 클리어

### 코드 개선 내용

```javascript
// 개선된 박스 렌더링 함수
function drawDetections(detections, videoWidth, videoHeight) {
    // 1. 비디오 표시 영역 정확히 계산
    const videoAspect = videoWidth / videoHeight;
    const containerAspect = containerRect.width / containerRect.height;
    
    // 2. object-contain 스타일 고려한 좌표 변환
    let displayWidth, displayHeight, offsetX, offsetY;
    if (videoAspect > containerAspect) {
        // 비디오가 더 넓음 - 컨테이너 높이에 맞춤
        displayHeight = containerRect.height;
        displayWidth = videoWidth * (containerRect.height / videoHeight);
        offsetX = (containerRect.width - displayWidth) / 2;
        offsetY = 0;
    } else {
        // 비디오가 더 높음 - 컨테이너 너비에 맞춤
        displayWidth = containerRect.width;
        displayHeight = videoHeight * (containerRect.width / videoWidth);
        offsetX = 0;
        offsetY = (containerRect.height - displayHeight) / 2;
    }
    
    // 3. 박스 그리기 (모서리 강조 포함)
    ctx.strokeStyle = color;
    ctx.lineWidth = 4;
    ctx.strokeRect(scaledX1, scaledY1, scaledX2 - scaledX1, scaledY2 - scaledY1);
    
    // 4. 텍스트 위치 동적 조정
    let textY = scaledY1 - textBoxHeight - 4;
    if (textY < 0) {
        textY = scaledY2 + 4; // 박스 아래에 배치
    }
    
    // 5. 반투명 배경 및 계층적 텍스트 표시
    ctx.fillStyle = color + 'CC'; // 80% 투명도
    ctx.fillRect(textX, textY, textBoxWidth, textBoxHeight);
}
```

### 주요 변경 사항
1. **박스 모서리 강조**: 네 모서리에 8px 크기의 강조선 추가
2. **텍스트 계층 구조**: 경고 → 이름/신뢰도 → 각도 정보 순서로 표시
3. **반투명 배경**: `color + 'CC'` 형식으로 80% 투명도 적용
4. **동적 위치 조정**: 화면 경계를 넘지 않도록 자동 조정
5. **좌표 변환 정확도**: `object-contain` CSS 스타일을 정확히 반영

---

## 성능 비교

| 단계 | 네트워크 사용량 | 지연 시간 | 프레임 끊김 | 박스 안정성 |
|------|----------------|----------|------------|------------|
| 서버사이드 렌더링 | 매우 높음 (~500KB/프레임) | 200-500ms | 심함 | - |
| 클라이언트사이드 렌더링 | 낮음 (~2KB/프레임) | 100-300ms | 보통 | 불안정 |
| WebSocket | 매우 낮음 (~2KB/프레임) | 50-150ms | 없음 | 안정적 |

---

## 현재 시스템 아키텍처

```
[클라이언트]                    [서버]
    │                              │
    │  WebSocket 연결               │
    ├─────────────────────────────>│
    │                              │
    │  프레임 전송 (Base64)         │
    ├─────────────────────────────>│
    │                              ├─> 얼굴 감지 (InsightFace)
    │                              ├─> Bank 임베딩 매칭
    │                              ├─> 각도 추정
    │                              ├─> Bank 자동 추가 (학습)
    │                              │
    │  감지 결과 (JSON)            │
    │<─────────────────────────────┤
    │                              │
    │  Canvas 렌더링                │
    └─> 박스 및 텍스트 표시         │
```

---

## 향후 개선 계획

1. **프레임 스킵 최적화**: 
   - 서버 부하가 높을 때 프레임 스킵 로직 개선
   - 우선순위 기반 프레임 처리

2. **다중 얼굴 추적**:
   - 얼굴 ID 추적으로 동일 인물의 연속 감지
   - 박스 위치 보간(interpolation)으로 더 부드러운 이동

3. **성능 모니터링**:
   - 실시간 FPS 표시
   - 네트워크 지연 시간 모니터링
   - 서버 처리 시간 통계

4. **UI/UX 개선**:
   - 감지 히스토리 타임라인
   - 감지 통계 대시보드
   - 알림 설정 및 필터링

---

## 참고 자료

- [WebSocket 구현 코드](web/script.js)
- [백엔드 WebSocket 엔드포인트](backend/main.py)
- [Bank 임베딩 시스템](src/utils/gallery_loader.py)
- [얼굴 각도 감지](src/utils/face_angle_detector.py)
- [박스 렌더링 함수](web/script.js#L248-L370)

---

**작성일**: 2024년  
**최종 업데이트**: 박스 렌더링 최적화 완료

