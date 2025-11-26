# 마스크 얼굴 인식 파이프라인 구현 완료

## 📌 구현 개요

마스크를 쓴 얼굴도 threshold를 낮추지 않고 인식할 수 있도록 "masked-face aware" 매칭 파이프라인을 구현했습니다.

## 🔑 핵심 전략

1. **threshold는 절대 낮추지 않음** (오탐 방지)
2. **base bank와 masked bank 분리 관리**
3. **bbox tracking 기반 multi-frame 확인**
4. **보수적인 masked bank 자동 추가**

## 📝 주요 변경 사항

### 1. 설정 상수 추가

```python
MASKED_BANK_MASK_PROB_THRESHOLD = 0.7  # mask_prob >= 0.7이면 masked bank로 분류
MASKED_CANDIDATE_MIN_SIM = 0.30  # base_sim >= 0.30 이상이어야 masked candidate로 판단
MASKED_CANDIDATE_MIN_FRAMES = 5  # 연속 N 프레임 이상 조건 충족 시 masked bank에 추가
MASKED_TRACKING_IOU_THRESHOLD = 0.5  # bbox tracking을 위한 IoU 임계값
```

### 2. 매칭 로직 개선

#### Base Bank와 Masked Bank 분리 매칭

- `base_sim`: base bank에서의 최고 유사도
- `masked_sim`: masked bank에서의 최고 유사도
- 두 값 중 더 높은 값을 `best_sim`으로 선택
- `best_sim >= threshold AND sim_gap >= gap_margin`일 때만 match 성공

#### Masked Candidate Frame 판단

다음 조건을 모두 만족하면 "masked candidate frame"으로 판단:

1. `base_sim < main_threshold` (threshold 미만)
2. `base_sim >= 0.30` (최소 유사도 이상)
3. `mask_prob >= 0.7` (마스크 가능성 높음)

### 3. Bbox Tracking 기반 Multi-Frame 확인

- 같은 track_id 또는 IoU > 0.5 기반으로 동일 인물 추적
- 연속 5~10 프레임 동안 masked candidate 조건을 만족하면:
  - 해당 embedding을 masked bank에 자동 추가
  - 중복 체크 포함 (BANK_DUPLICATE_THRESHOLD = 0.95)

### 4. Bank 자동 추가 규칙

#### Base Bank
- **절대 자동 추가하지 않음** (오염 방지)
- enrollment 시에만 수동으로 추가

#### Masked Bank
- bbox tracking 기반 multi-frame 확인 후 조건 충족 시 자동 추가
- 측면/프로파일 각도 + 고화질 + 고유사도 조건 충족 시에도 추가 (기존 로직 유지)

### 5. 디버깅 출력 개선

매칭 디버깅 로그에 다음 정보 포함:

```
🎯 [매칭 디버깅] bank=masked, base_sim=0.320, masked_sim=0.410, best_sim=0.410
   - main_threshold=0.400, sim_gap=0.120, gap_margin=0.100, 매칭=True
   - mask_prob=0.700, masked_candidate=True, candidate_frames=5
   - 유사도 >= main_threshold: 0.410 >= 0.400 = True
   - sim_gap >= gap_margin: 0.120 >= 0.100 = True
```

### 6. 결과에 bank_type 포함

매칭 성공 시 detection 결과에 `bank_type` 필드 추가:

```json
{
  "bbox": [x1, y1, x2, y2],
  "status": "normal",
  "name": "홍길동",
  "person_id": "hong",
  "confidence": 85,
  "bank_type": "masked"  // "base" 또는 "masked"
}
```

## 🔧 수정된 파일

### backend/main.py

1. **설정 상수 추가** (라인 50-54)
2. **process_detection 함수 시그니처 변경** (tracking_state 파라미터 추가)
3. **Base/Masked Bank 분리 매칭 로직** (라인 910-1008)
4. **Masked candidate 판단 로직** (라인 1053-1058)
5. **Bbox tracking 기반 multi-frame 확인** (라인 1099-1144)
6. **디버깅 출력 개선** (라인 1200-1206)
7. **WebSocket tracking_state 관리** (라인 1526-1530, 1560-1572)

## 📊 동작 흐름

```
1. 얼굴 감지 및 임베딩 추출
   ↓
2. Base Bank 매칭 → base_sim 계산
   ↓
3. Masked Bank 매칭 → masked_sim 계산
   ↓
4. best_sim = max(base_sim, masked_sim) 선택
   ↓
5. Masked candidate 판단
   - base_sim < threshold AND base_sim >= 0.30 AND mask_prob >= 0.7?
   ↓
6. Bbox tracking (IoU 기반)
   - 기존 track 찾기 또는 새 track 생성
   - 연속 프레임 카운트
   ↓
7. 조건 충족 시 masked bank에 자동 추가
   - 연속 5프레임 이상 조건 만족
   - 중복 체크 포함
   ↓
8. 최종 매칭 판단
   - best_sim >= threshold AND sim_gap >= gap_margin
   - bank_type 정보 포함하여 결과 반환
```

## ⚠️ 주의 사항

1. **threshold는 절대 내리지 않음** - 오탐 방지를 위해 유지
2. **base bank는 절대 자동 업데이트하지 않음** - 오염 방지
3. **masked bank는 보수적으로만 추가** - 연속 프레임 확인 필수
4. **base bank와 masked bank는 절대 섞지 않음** - 완전 분리 관리

## 🧪 테스트 방법

### 1. 마스크 쓴 얼굴 인식 테스트

1. 마스크를 쓰지 않은 상태로 얼굴 등록 (base bank에 저장)
2. 마스크를 쓴 상태로 동일 인물이 연속 5프레임 이상 나타남
3. base_sim이 0.30 ~ threshold 사이이고 mask_prob >= 0.7인지 확인
4. masked bank에 자동 추가되는지 확인
5. 이후 프레임에서 masked bank로 인식되는지 확인

### 2. 디버깅 로그 확인

다음 로그가 출력되는지 확인:

```
🎯 [매칭 디버깅] bank=masked, base_sim=0.320, masked_sim=0.410, best_sim=0.410
   - mask_prob=0.700, masked_candidate=True, candidate_frames=5
✅ Masked Bank 자동 추가: hong (연속 5프레임, base_sim=0.320, mask_prob=0.700)
```

### 3. 결과 확인

매칭 성공 시 `bank_type` 필드가 올바르게 포함되는지 확인:

```json
{
  "detections": [
    {
      "name": "홍길동",
      "bank_type": "masked",
      "confidence": 85
    }
  ]
}
```

## 📈 성능 최적화

- bbox tracking은 IoU 기반으로 빠르게 처리
- 중복 체크는 BANK_DUPLICATE_THRESHOLD = 0.95로 엄격하게 관리
- tracking_state는 WebSocket 연결별로 관리하여 메모리 효율적

## 🔄 향후 개선 사항

1. 랜드마크 기반 occlusion 판단 (현재는 유사도 기반 추정)
2. tracking 알고리즘 고도화 (Kalman filter 등)
3. masked bank 크기 제한 및 오래된 임베딩 제거 로직





