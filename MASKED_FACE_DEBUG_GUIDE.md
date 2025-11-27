# 마스크 얼굴 감지 디버깅 가이드

## 🔍 문제 진단 체크리스트

### 1. 파일명 일치 확인 ✅

**저장하는 파일명:**
- `outputs/embeddings/{person_id}/bank_masked.npy`
- `outputs/embeddings/{person_id}/angles_masked.json`

**로딩하는 파일명:**
- `outputs/embeddings/{person_id}/bank_masked.npy`

**결론:** 파일명이 일치합니다. ✅

### 2. Masked Candidate 로그 확인

서버 실행 시 다음 로그가 출력되는지 확인하세요:

#### ✅ Masked Candidate 감지됨
```
🎭 [MASKED CAND] 감지됨! person_id=hong, base_sim=0.320, mask_prob=0.700, threshold=0.400
🆕 [MASKED CAND] 새 track 생성: hong (track_id=track_0, base_sim=0.320)
📊 [MASKED CAND] 추적 중: hong (1/5프레임, base_sim=0.320)
📊 [MASKED CAND] 추적 중: hong (2/5프레임, base_sim=0.315)
...
📊 [MASKED CAND] 추적 중: hong (5/5프레임, base_sim=0.325)
✅ [MASKED BANK] 자동 추가 성공: hong (연속 5프레임, base_sim=0.325, mask_prob=0.700)
✅ [Masked BANK] 파일 저장: outputs/embeddings/hong/bank_masked.npy (총 1개 임베딩, angle: front)
```

#### ⚠️ 조건 미충족
```
🎭 [MASKED CAND] 조건 미충족: person_id=hong, base_sim=0.250 (min=0.300), mask_prob=0.500 (min=0.700)
```

#### ❌ 로그가 전혀 안 뜨는 경우
- `base_sim`이 너무 낮아서 매칭 자체가 안 되는 경우
- `best_person_id == "unknown"`인 경우

### 3. 현재 조건 (기본값)

```python
MASKED_CANDIDATE_MIN_SIM = 0.30  # base_sim >= 0.30 이상
MASKED_CANDIDATE_MIN_FRAMES = 5   # 연속 5프레임 이상
MASKED_BANK_MASK_PROB_THRESHOLD = 0.7  # mask_prob >= 0.7
MASKED_TRACKING_IOU_THRESHOLD = 0.5  # IoU >= 0.5
```

**조건 요약:**
1. `base_sim < main_threshold` (예: 0.40 미만)
2. `base_sim >= 0.30` (최소 유사도)
3. `mask_prob >= 0.7` (마스크 가능성)
4. `best_person_id != "unknown"` (매칭된 인물이 있어야 함)
5. 연속 5프레임 이상 조건 충족

## 🛠️ 디버깅 방법

### 방법 1: 로그 확인

서버 실행 후 마스크 쓴 얼굴이 나타날 때:

1. **로그가 전혀 안 뜨는 경우:**
   - `base_sim`이 너무 낮아서 매칭이 안 되는 것
   - `best_person_id == "unknown"`인 상태
   - 해결: 조건 완화 (아래 참고)

2. **조건 미충족 로그가 뜨는 경우:**
   - 어떤 조건이 부족한지 로그에서 확인
   - 예: `base_sim=0.250 (min=0.300)` → base_sim이 너무 낮음

3. **추적 중 로그는 뜨는데 파일이 안 생기는 경우:**
   - 연속 프레임 수가 부족한 것
   - 예: `(3/5프레임)` → 2프레임 더 필요

### 방법 2: 조건 완화 (디버깅용)

`backend/main.py`에서 다음 값들을 조정:

```python
# 더 완화된 조건 (디버깅용)
MASKED_CANDIDATE_MIN_SIM = 0.25  # 0.30 → 0.25로 낮춤
MASKED_CANDIDATE_MIN_FRAMES = 3  # 5 → 3으로 낮춤
MASKED_BANK_MASK_PROB_THRESHOLD = 0.5  # 0.7 → 0.5로 낮춤
```

### 방법 3: 수동으로 임베딩 저장 (빠른 테스트)

```python
import numpy as np
from pathlib import Path

# 1. 마스크 쓴 얼굴의 embedding 추출 (서버에서)
# face.embedding을 복사

# 2. 직접 저장
person_id = "hong"  # 실제 person_id로 변경
embedding = np.array([...])  # 실제 embedding으로 변경

embeddings_dir = Path("outputs/embeddings") / person_id
embeddings_dir.mkdir(parents=True, exist_ok=True)

masked_bank_path = embeddings_dir / "bank_masked.npy"
np.save(masked_bank_path, embedding.reshape(1, -1))

print(f"✅ 저장 완료: {masked_bank_path}")
```

서버 재시작 후 masked bank가 로딩되는지 확인:

```
✅ Bank 로드: 홍길동 (ID: hong, base: 5개, masked: 1개) [masked 파일: outputs/embeddings/hong/bank_masked.npy]
```

### 방법 4: 파일 존재 확인

```bash
# 프로젝트 루트에서 실행
find outputs/embeddings -name "bank_masked.npy"
```

파일이 있으면:
- 서버 재시작 시 로딩되는지 확인
- 로딩 로그에서 개수 확인

파일이 없으면:
- masked candidate 조건이 한 번도 충족되지 않은 것
- 조건 완화 또는 수동 저장 필요

## 📊 예상 시나리오별 대응

### 시나리오 1: 로그가 전혀 안 뜸

**원인:** `base_sim`이 너무 낮아서 매칭이 안 됨

**해결:**
1. 조건 완화: `MASKED_CANDIDATE_MIN_SIM = 0.25`
2. 또는 수동으로 임베딩 저장 후 테스트

### 시나리오 2: 조건 미충족 로그만 계속 뜸

**원인:** 조건이 너무 빡빡함

**해결:**
1. 부족한 조건 확인 (로그에서)
2. 해당 조건 완화
3. 예: `mask_prob`가 낮으면 `MASKED_BANK_MASK_PROB_THRESHOLD` 낮추기

### 시나리오 3: 추적 중 로그는 뜨는데 파일이 안 생김

**원인:** 연속 프레임 수 부족

**해결:**
1. `MASKED_CANDIDATE_MIN_FRAMES = 3`으로 낮추기
2. 또는 동일 인물이 더 오래 나타나도록 테스트

### 시나리오 4: 파일은 생기는데 매칭이 안 됨

**원인:** 파일 로딩 문제 또는 매칭 로직 문제

**해결:**
1. 서버 재시작 후 로딩 로그 확인
2. `gallery_masked_cache`에 제대로 로딩되었는지 확인
3. 매칭 디버깅 로그에서 `masked_sim` 값 확인

## 🔧 빠른 테스트 스크립트

```python
# test_masked_bank.py
import numpy as np
from pathlib import Path

# 테스트용 임베딩 생성 (실제로는 서버에서 추출)
person_id = "test_person"
test_embedding = np.random.rand(512).astype(np.float32)
test_embedding = test_embedding / np.linalg.norm(test_embedding)

# 저장
embeddings_dir = Path("outputs/embeddings") / person_id
embeddings_dir.mkdir(parents=True, exist_ok=True)
masked_bank_path = embeddings_dir / "bank_masked.npy"
np.save(masked_bank_path, test_embedding.reshape(1, -1))

print(f"✅ 테스트 파일 생성: {masked_bank_path}")
print("서버 재시작 후 로딩되는지 확인하세요.")
```

## 📝 체크리스트

- [ ] 서버 시작 시 `bank_masked.npy` 로딩 로그 확인
- [ ] 마스크 쓴 얼굴 나타날 때 `[MASKED CAND]` 로그 확인
- [ ] 조건 미충족 로그에서 부족한 조건 확인
- [ ] 추적 중 로그에서 프레임 수 확인
- [ ] `bank_masked.npy` 파일 생성 여부 확인
- [ ] 파일 생성 후 서버 재시작하여 로딩 확인
- [ ] 매칭 디버깅 로그에서 `masked_sim` 값 확인

## 🎯 최종 확인

모든 것이 정상 작동하면:

1. 서버 시작 시:
   ```
   ✅ Bank 로드: 홍길동 (ID: hong, base: 5개, masked: 3개) [masked 파일: outputs/embeddings/hong/bank_masked.npy]
   ```

2. 마스크 쓴 얼굴 감지 시:
   ```
   🎯 [매칭 디버깅] bank=masked, base_sim=0.320, masked_sim=0.410, best_sim=0.410
   ```

3. 매칭 성공:
   ```json
   {
     "name": "홍길동",
     "bank_type": "masked",
     "confidence": 85
   }
   ```






