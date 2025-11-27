# Bank Base 재생성 및 Threshold/Gap 튜닝 가이드

## 목표

1. 기준 사진만으로 깨끗한 `bank_base.npy` 재생성
2. 새로운 base/dynamic 구조에서 threshold/gap 값 튜닝

---

## 1. Bank Base 재생성 전략

### 1-1. 현재 상황 파악

```bash
# outputs/embeddings 구조 확인
ls -la outputs/embeddings/*/
```

각 person_id 폴더에:
- `bank.npy` - 기존 Bank (오염 가능성 있음)
- `centroid.npy` - 기존 Centroid
- `bank_dynamic.npy` - 자동 학습 Bank (있을 수도 없을 수도)

### 1-2. 방법 A: 기존 bank.npy를 bank_base.npy로 변환

**가장 빠른 방법** - 기존 bank.npy를 기준 Bank로 사용

```bash
# 전체 인물 변환 (백업 포함)
python scripts/rebuild_base_bank.py --backup

# 특정 인물만 변환
python scripts/rebuild_base_bank.py --person-id person_id --backup
```

**결과:**
- `bank.npy` → `bank_base.npy` (복사)
- `centroid.npy` → `centroid_base.npy` (복사)
- `bank.npy.backup` 생성 (백업)

### 1-3. 방법 B: Enroll 폴더에서 기준 사진으로 재생성

**가장 깨끗한 방법** - 기준 사진만으로 새로 생성

```bash
# 전체 인물 재생성
python scripts/rebuild_base_bank.py --from-enroll --backup

# 특정 인물만 재생성
python scripts/rebuild_base_bank.py --person-id person_id --from-enroll --backup
```

**전제 조건:**
- `enroll/{person_id}/` 폴더에 기준 사진이 있어야 함
- 정면/좋은 품질의 사진만 포함

**결과:**
- `bank_base.npy` - 기준 사진에서 추출한 임베딩
- `centroid_base.npy` - bank_base 기반 centroid
- 기존 파일은 백업됨

### 1-4. Dynamic Bank 정리 (선택 사항)

자동 학습으로 쌓인 noisy 데이터를 삭제하고 싶다면:

```bash
# bank_dynamic.npy 삭제
python scripts/rebuild_base_bank.py --delete-dynamic
```

**주의:** 이 작업은 되돌릴 수 없으므로 신중하게 결정하세요.

---

## 2. Threshold/Gap 튜닝 전략

### 2-1. 현재 설정값

**화질별 Threshold:**
- High:   0.42
- Medium: 0.40
- Low:    0.38

**화질별 Gap Margin:**
- High:   0.12
- Medium: 0.10
- Low:    0.08

**Suspect IDs 모드 추가 조건:**
- Threshold +0.02 (예: Medium → 0.42)
- Gap +0.03 (예: Medium → 0.13)
- 절대값 최소 0.45

### 2-2. 튜닝 가이드 출력

```bash
python scripts/tune_threshold_gap.py --guide
```

### 2-3. 튜닝 전략

#### A. False Positive가 많을 때 (오탐 증가)

**증상:**
- 전혀 다른 사람을 범죄자로 오인식
- Unknown이 너무 적게 나옴

**해결책:**
```python
# backend/main.py의 process_detection 함수에서
if face_quality == "high":
    main_threshold = 0.44  # +0.02 상향
    gap_margin = 0.14      # +0.02 상향
elif face_quality == "medium":
    main_threshold = 0.42  # +0.02 상향
    gap_margin = 0.12      # +0.02 상향
else:  # low
    main_threshold = 0.40  # +0.02 상향
    gap_margin = 0.10      # +0.02 상향
```

#### B. True Positive가 적을 때 (누락 증가)

**증상:**
- 실제 범죄자가 감지되지 않음
- Unknown이 너무 많이 나옴

**해결책:**
```python
if face_quality == "high":
    main_threshold = 0.40  # -0.02 하향
    gap_margin = 0.10      # -0.02 하향
elif face_quality == "medium":
    main_threshold = 0.38  # -0.02 하향
    gap_margin = 0.08      # -0.02 하향
else:  # low
    main_threshold = 0.36  # -0.02 하향
    gap_margin = 0.06      # -0.02 하향
```

#### C. 특정 화질에서만 문제가 있을 때

해당 화질의 threshold/gap만 조정:

```python
# 예: Medium 화질만 조정
elif face_quality == "medium":
    main_threshold = 0.41  # +0.01만 조정
    gap_margin = 0.11      # +0.01만 조정
```

### 2-4. 튜닝 프로세스

1. **기준 Bank 재생성**
   ```bash
   python scripts/rebuild_base_bank.py --from-enroll --backup
   ```

2. **테스트 영상으로 여러 설정값 테스트**
   - 각 설정값으로 서버 재시작
   - 테스트 영상 처리
   - 결과 기록 (매칭률, 오탐률 등)

3. **결과 분석**
   - Precision (정확도) 높은 설정 선택
   - Recall (재현율)도 충분한지 확인
   - False Positive 최소화

4. **최종 설정 적용**
   - `backend/main.py`의 `process_detection` 함수 수정
   - 서버 재시작 및 검증

---

## 3. 새 구조 검증 체크리스트

### 3-1. 파일 구조 확인

```bash
# 각 person_id 폴더에 다음 파일들이 있어야 함:
outputs/embeddings/{person_id}/
  ├── bank_base.npy          # ✅ 기준 Bank (read-only)
  ├── centroid_base.npy      # ✅ 기준 Centroid (read-only)
  ├── bank_dynamic.npy       # ✅ 동적 Bank (자동 학습용, 없을 수도 있음)
  └── angles_dynamic.json    # ✅ 동적 Bank 각도 정보 (있을 수도 있음)
```

### 3-2. 서버 시작 시 로그 확인

서버 시작 시 다음과 같은 로그가 나와야 함:

```
✅ Base Bank 로드: {name} (ID: {person_id}, {N}개 임베딩)
✅ Dynamic Bank 로드: {name} (ID: {person_id}, {M}개 임베딩)
```

또는

```
⚠️ Legacy Bank를 Base로 사용: {name} (ID: {person_id}, {N}개 임베딩)
```

### 3-3. 자동 학습 동작 확인

매칭 성공 시:

```
✅ Dynamic Bank 업데이트: {person_id} (총 {M}개 임베딩, angle: {angle_type})
```

**중요:** `bank_base.npy`는 절대 수정되지 않아야 함!

---

## 4. 문제 해결

### Q1: bank_base.npy가 없어서 서버가 시작되지 않음

**해결책:**
```bash
# 기존 bank.npy를 bank_base.npy로 변환
python scripts/rebuild_base_bank.py --backup
```

### Q2: Dynamic Bank가 너무 커져서 성능 저하

**해결책:**
```bash
# Dynamic Bank 삭제 (기준 Bank는 유지)
python scripts/rebuild_base_bank.py --delete-dynamic
```

### Q3: Threshold/Gap 값을 실시간으로 테스트하고 싶음

**해결책:**
- `backend/main.py`의 `process_detection` 함수에서 값 수정
- 서버 재시작 (--reload 옵션 사용 시 자동 재시작)
- 테스트 후 원래 값으로 복구

---

## 5. 참고 사항

### 5-1. Bank 오염 방지 원칙

1. **bank_base.npy는 절대 수정하지 않음**
   - Enrollment 시에만 생성
   - 런타임에서는 read-only

2. **bank_dynamic.npy만 자동 학습**
   - 고화질 + 고유사도 + 프로파일 각도만 추가
   - 중복 체크는 base + dynamic 전체 대상

3. **기준 Bank 유지**
   - 시간이 지나도 기준 Bank는 변하지 않음
   - 오염되지 않은 기준으로 계속 매칭 가능

### 5-2. Threshold/Gap 튜닝 원칙

1. **보수적으로 시작**
   - False Positive를 줄이는 것이 우선
   - Unknown이 많아도 오인식보다 낫다

2. **화질별로 다르게 설정**
   - 고화질: 높은 threshold/gap
   - 저화질: 낮은 threshold/gap

3. **Suspect IDs 모드는 더 보수적으로**
   - 추가 threshold/gap 보너스 적용
   - 절대값 최소 0.45 체크

---

**작성일:** 2024년
**최종 수정일:** 2024년







