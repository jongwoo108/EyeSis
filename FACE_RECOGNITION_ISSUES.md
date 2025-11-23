# 얼굴 인식 시스템 문제 및 개선 이력

## 📋 목차
1. [초기 문제 상황](#초기-문제-상황)
2. [주요 문제 및 개선 시도](#주요-문제-및-개선-시도)
3. [지속적인 문제](#지속적인-문제)
4. [해결 방안](#해결-방안)

---

## 초기 문제 상황

### 1. 스냅샷 누락 문제
**문제**: 범죄자 감지 시 스냅샷이 생성되지 않거나 전달되지 않는 문제

**원인 분석**:
- 백엔드에서 스냅샷 생성 로직은 구현되어 있었으나, 프론트엔드에서 수신하지 못하는 경우 발생
- WebSocket과 HTTP API 모두에서 스냅샷 전달 로직이 있었지만, 일부 케이스에서 누락

**해결**: 스냅샷 생성 및 전달 로직 확인 및 수정 완료

---

## 주요 문제 및 개선 시도

### 2. 마스크 쓴 사람 인식 문제

#### 2.1 문제 상황
- 마스크를 쓴 사람을 인식하지 못함
- GitHub의 원래 코드는 정상 작동했으나, 현재 코드에서는 마스크 쓴 사람을 놓침

#### 2.2 원인 분석
**설정 값 차이**:
- `estimate_mask_from_similarity`: 유사도 범위가 0.25~0.40으로 확장됨 (원래: 0.25~0.35)
- `get_adjusted_threshold`: 마스크 가능성 기준이 0.5로 상향됨 (원래: 0.3)
- 임계값 조정 폭이 너무 큼 (원래: -0.06/-0.04/-0.02, 변경: -0.15/-0.12/-0.10)
- 최소 임계값이 너무 낮음 (원래: 0.28/0.22, 변경: 0.20/0.18)

#### 2.3 개선 시도
1. **GitHub 원본 설정으로 복원** ✅
   - `estimate_mask_from_similarity`: 유사도 범위 0.25~0.35로 축소
   - `get_adjusted_threshold`: 마스크 가능성 기준 0.3으로 복원
   - 임계값 조정 폭 축소 (-0.06/-0.04/-0.02)
   - 최소 임계값 상향 (0.28/0.22)

2. **추가 개선** ✅
   - 마스크 가능성이 높을 때 임계값 조정 폭 축소 (오인식 방지)
   - 최소 임계값 상향 (고화질: 0.30, 중화질: 0.25, 저화질: 0.22)

#### 2.4 결과
- 마스크 쓴 사람 인식은 개선되었으나, 오인식률이 증가하는 부작용 발생

---

### 3. 오인식률 문제

#### 3.1 문제 상황
- 여러 사람이 있는 영상에서 오인식 발생
- 같은 얼굴이 여러 인물로 매칭되는 경우 발생
- GitHub의 원래 코드보다 오인식률이 높음

#### 3.2 원인 분석
**누락된 오인식 방지 로직**:
- `src/face_match_cctv.py`에는 있으나 `backend/main.py`에는 없는 로직:
  1. 같은 얼굴 영역에서 여러 인물로 매칭되는 경우 필터링
  2. 화질 기반 추가 신뢰도 체크
  3. 동적 gap_threshold (화질에 따라 0.06/0.08/0.10)

#### 3.3 개선 시도
1. **헬퍼 함수 추가** ✅
   - `calculate_bbox_iou()`: 두 bbox 간 IoU 계산
   - `calculate_bbox_center_distance()`: 중심점 간 거리 계산
   - `is_same_face_region()`: 같은 얼굴 영역 판단

2. **다중 인물 인식 로직 추가** ✅
   - 모든 얼굴에 대해 매칭 결과 수집
   - 같은 얼굴 영역 그룹화 및 필터링
   - 화질 기반 동적 gap_threshold 적용 (0.06/0.08/0.10)

3. **추가 신뢰도 체크** ✅
   - 화질에 따른 sim_threshold와 gap_threshold 체크
   - 고화질: sim >= 0.38 AND gap >= 0.10
   - 중화질: sim >= 0.35 AND gap >= 0.08
   - 저화질: sim >= 0.32 AND gap >= 0.06

4. **더 엄격한 기준 적용** ✅
   - 같은 얼굴 영역에서 여러 인물 매칭 시:
     - 화질에 따른 `min_gap_for_confidence` 상향 (0.08/0.10/0.12)
     - 최고 유사도 기준 추가 (0.35/0.38/0.40)
     - 마스크 가능성 체크 추가
   - 마스크 가능성이 높으면 더 엄격한 기준 적용

#### 3.4 결과
- 일부 개선되었으나 여전히 오인식률이 높음
- 기준을 엄격하게 하면 인식률이 떨어지고, 관대하게 하면 오인식이 증가하는 딜레마

---

### 4. Bank 데이터 로드 문제

#### 4.1 문제 상황
- 페이지 새로고침 후 마스크 쓴 사람을 인식하지 못함
- 영상을 두 번째 돌려야 그때부터 마스크 쓴 사람을 인식하기 시작함
- 이미 수집되어 저장된 Bank 데이터가 로드되지 않음

#### 4.2 원인 분석
**PostgreSQL에서 Bank 데이터 미로드**:
- `load_persons_from_db()` 함수가 PostgreSQL의 단일 임베딩만 로드
- `outputs/embeddings/<person_id>/bank.npy` 파일을 확인하지 않음
- Bank 데이터가 있어도 사용하지 않고 Centroid만 사용

**코드 분석**:
```python
# 기존 코드 (문제)
gallery_cache[person.person_id] = person.get_embedding().reshape(1, -1)
# → 단일 임베딩만 사용, Bank 데이터 무시
```

#### 4.3 개선 시도
1. **Bank 데이터 로드 로직 추가** ✅
   - PostgreSQL에서 인물 정보 로드 후, `outputs/embeddings/<person_id>/bank.npy` 확인
   - Bank 파일이 있으면 Bank 데이터 로드 (여러 임베딩 포함)
   - Bank가 없으면 Centroid 파일 확인
   - 둘 다 없으면 PostgreSQL의 임베딩 사용

2. **로드 순서**:
   ```
   1. outputs/embeddings/<person_id>/bank.npy 확인
   2. outputs/embeddings/<person_id>/centroid.npy 확인
   3. PostgreSQL의 임베딩 사용 (fallback)
   ```

#### 4.4 결과
- Bank 데이터 로드는 해결되었으나, 여전히 일부 케이스에서 문제 발생 가능성

---

## 지속적인 문제

### 1. WebSocket 통신 딜레이로 인한 초기 마스크 감지 실패

#### 1.1 문제 상황
- 영상 재생 시 처음 몇 프레임에서 마스크 쓴 사람을 감지하지 못함
- 영상을 두 번째 돌려야 그때부터 마스크 쓴 사람을 인식하기 시작함
- WebSocket 연결 딜레이와 suspect_ids 설정 타이밍 문제

#### 1.2 원인 분석

**타이밍 문제**:
1. **비디오 재생 시점** (`web/script.js` line 226-244):
   ```javascript
   UI.video.addEventListener('play', () => {
       // 비디오가 재생되면 즉시 첫 프레임 처리
       processRealtimeDetection();  // 즉시 호출
       state.detectionInterval = setInterval(processRealtimeDetection, 100);
       
       // WebSocket 연결은 비동기로 시작 (완료를 기다리지 않음)
       if (state.useWebSocket && !state.isWsConnected) {
           connectWebSocket();  // 비동기 연결 시작
       }
   });
   ```

2. **WebSocket 연결 완료 시점** (`web/script.js` line 677-697):
   ```javascript
   ws.onopen = () => {
       // 연결 완료 후 suspect_ids 설정 전송
       if (state.selectedSuspects.length > 0) {
           sendWebSocketConfig(suspectIds);  // 설정 전송
       }
       
       // 50ms 대기 후 첫 프레임 전송
       setTimeout(() => {
           processRealtimeDetection();
       }, 50);
   };
   ```

3. **문제점**:
   - 비디오 재생 시 **즉시** 첫 프레임이 전송됨
   - WebSocket 연결은 **비동기**로 시작되어 완료를 기다리지 않음
   - 첫 프레임이 WebSocket 연결 **완료 전**에 전송될 수 있음
   - 첫 프레임이 `suspect_ids` 설정 **전**에 전송될 수 있음

4. **백엔드 처리** (`backend/main.py` line 1046-1058):
   ```python
   suspect_ids = frame_data.get("suspect_ids")  # 프레임 메시지에서 가져옴
   
   if suspect_ids is None:
       # 연결 상태에서 suspect_ids 사용
       suspect_ids = connection_states[websocket].get("suspect_ids", [])
       # → 빈 배열이면 전체 DB 검색
   ```

5. **결과**:
   - 첫 프레임이 `suspect_ids` 없이 전송됨
   - 백엔드에서 전체 DB 검색 수행
   - Bank 데이터가 제대로 로드되지 않았을 수 있음
   - 마스크 쓴 사람 인식 실패

#### 1.3 개선 시도
- Bank 데이터 로드 로직 개선 (이미 완료)
- 하지만 WebSocket 통신 딜레이 문제는 여전히 존재

#### 1.4 해결 방안

**단기 해결책**:
1. WebSocket 연결 완료를 기다린 후 첫 프레임 전송
2. `suspect_ids` 설정이 완료된 후 프레임 전송
3. 프레임 전송 시 항상 `suspect_ids` 포함

**코드 수정 예시**:
```javascript
// WebSocket 연결 완료 후 설정 전송 및 확인
ws.onopen = () => {
    // suspect_ids 설정 전송
    if (state.selectedSuspects.length > 0) {
        const suspectIds = state.selectedSuspects.map(s => s.id);
        sendWebSocketConfig(suspectIds);
        
        // 설정 전송 후 응답 대기
        // config_updated 메시지를 받은 후 첫 프레임 전송
    }
};

// config_updated 메시지 수신 시
if (msgType === "config_updated") {
    // 설정이 완료되었으므로 첫 프레임 전송 가능
    if (state.isDetectionActive && !state.firstFrameSent) {
        processRealtimeDetection();
        state.firstFrameSent = true;
    }
}
```

**또는 더 간단한 방법**:
```javascript
// 프레임 전송 시 항상 suspect_ids 포함
function sendWebSocketFrame(frameData, suspectIds) {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
        // suspectIds가 null이어도 빈 배열로 전송
        const ids = suspectIds || [];
        state.ws.send(JSON.stringify({
            type: "frame",
            data: {
                image: frameData,
                suspect_ids: ids,  // 항상 포함
                frame_id: state.frameId
            }
        }));
    }
}
```

---

### 2. 오인식률이 여전히 높음

**현재 상태**:
- 여러 사람이 있는 영상에서 오인식 발생
- 같은 얼굴이 여러 인물로 매칭되는 경우가 있음
- 기준을 엄격하게 하면 정상 인식도 놓치고, 관대하게 하면 오인식 증가

**근본 원인**:
- 유사도 기반 매칭의 한계
- Bank 데이터의 품질과 다양성 부족
- 임계값과 gap_threshold의 균형점 찾기 어려움

### 2. 마스크 쓴 사람 인식의 정확도 문제

**현재 상태**:
- 마스크 쓴 사람을 인식하지만, 오인식도 함께 증가
- 마스크 가능성 판단 기준이 애매함
- 유사도가 0.25~0.35 사이일 때 마스크로 판단하지만, 일반 얼굴도 이 범위에 포함될 수 있음

**근본 원인**:
- 마스크 착용 여부를 유사도만으로 판단하는 한계
- 실제 마스크 감지 기능 부재 (현재는 유사도 기반 추정만 사용)

### 3. 페이지 새로고침 후 인식 문제

**현재 상태**:
- Bank 데이터 로드는 해결되었으나, 여전히 일부 케이스에서 문제 발생 가능
- 서버 재시작 시 Bank 데이터가 제대로 로드되는지 확인 필요

**근본 원인**:
- Bank 데이터 저장과 로드의 일관성 문제
- 실시간으로 추가된 Bank 데이터가 파일에 저장되지 않는 경우 가능성

### 4. 다중 인물 환경에서의 성능 문제

**현재 상태**:
- 여러 사람이 있을 때 처리 시간 증가
- 같은 얼굴 영역 필터링 로직이 복잡하여 성능 저하 가능

**근본 원인**:
- 모든 얼굴에 대해 매칭 후 필터링하는 구조
- 실시간 처리에 부적합할 수 있음

---

## 해결 방안

### 1. 오인식률 개선

**단기 해결책**:
- 임계값과 gap_threshold를 더 세밀하게 조정
- 화질별, 마스크 여부별로 다른 기준 적용
- Bank 데이터의 품질 향상 (다양한 각도, 조명 조건의 임베딩 수집)

**장기 해결책**:
- 실제 얼굴 랜드마크 기반 마스크 감지 기능 추가
- 얼굴 각도, 표정 등 추가 메타데이터 활용
- 머신러닝 기반 오인식 방지 모델 도입

### 2. Bank 데이터 관리 개선

**단기 해결책**:
- Bank 데이터 저장과 로드의 일관성 보장
- 실시간 추가된 Bank 데이터를 즉시 파일에 저장
- 서버 시작 시 Bank 데이터 로드 확인 로그 추가

**장기 해결책**:
- PostgreSQL에 Bank 데이터 저장 기능 추가
- Bank 데이터 버전 관리 및 롤백 기능
- Bank 데이터 품질 검증 기능

### 3. 성능 최적화

**단기 해결책**:
- 같은 얼굴 영역 필터링 로직 최적화
- 불필요한 계산 제거
- 캐싱 전략 개선

**장기 해결책**:
- 비동기 처리 및 배치 처리 도입
- GPU 가속 활용
- 분산 처리 시스템 구축

### 4. 테스트 및 검증

**필요한 테스트**:
- 다양한 시나리오 테스트 (단일 인물, 다중 인물, 마스크 착용 등)
- 오인식률 측정 및 모니터링
- 성능 벤치마크

**검증 방법**:
- 정확도(Accuracy), 정밀도(Precision), 재현율(Recall) 측정
- F1-Score 계산
- 혼동 행렬(Confusion Matrix) 분석

---

## 참고 사항

### 현재 설정 값

**마스크 감지**:
- 유사도 범위: 0.25~0.35
- 마스크 가능성 기준: 0.3
- 임계값 조정: -0.05/-0.03/-0.01
- 최소 임계값: 0.30/0.25/0.22 (고/중/저화질)

**오인식 방지**:
- 기본 gap_threshold: 0.06/0.08/0.10 (저/중/고화질)
- 같은 얼굴 영역 필터링: 0.08/0.10/0.12
- 추가 신뢰도 체크: sim >= 0.32/0.35/0.38 AND gap >= 0.06/0.08/0.10

### 관련 파일

- `backend/main.py`: 메인 백엔드 로직
- `src/utils/mask_detector.py`: 마스크 감지 유틸리티
- `src/utils/gallery_loader.py`: 갤러리 로더
- `src/face_match_cctv.py`: 참고용 CCTV 매칭 로직

---

**작성일**: 2024년
**최종 수정일**: 2024년
**상태**: 진행 중

