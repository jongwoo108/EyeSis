# FaceWatch 시스템 개선 계획서

> 오인식 확산 및 임베딩 오염 문제 해결을 위한 단계별 개선 계획

---

## 목차

1. [핵심 문제점 분석](#1-핵심-문제점-분석)
2. [개선 실행 계획](#2-개선-실행-계획)
3. [완료된 작업](#3-완료된-작업)
4. [예정된 작업](#4-예정된-작업)
5. [개선 효과 및 예상 결과](#5-개선-효과-및-예상-결과)

---

## 1. 핵심 문제점 분석

### 1.1 문제 1: 방어력 없는 Dynamic Bank 업데이트 로직

**위치**: `backend/main.py:1657-1672` (임베딩 자동 수집 로직)

**현재 로직**:
```python
# 매칭 성공(is_match=True)하고 각도가 다양하면
if AUTO_ADD_TO_DYNAMIC_BANK:
    # 중복 체크(>=0.95)와 각도 다양성만 확인
    learning_events.append({
        "person_id": person_id,
        "embedding": embedding_normalized.tolist(),
        "bank_type": "dynamic"
    })
```

**문제점**:
1. **오인식 확산**: 한 번이라도 오인식(False Positive)이 발생하여 잘못된 사람과 매칭 성공 판정을 내리면, 그 잘못된 임베딩이 Dynamic Bank에 영구적으로 저장됩니다.
2. **오염된 임베딩의 영구적 영향**: 이후에는 이 '오염된 임베딩' 때문에 계속해서 오인식이 발생하는 악순환이 반복됩니다.
3. **검증 부재**: 
   - Base Bank와의 최소 유사도 검증 없음
   - 화질 검증이 약함 (현재는 얼굴 크기만 확인)
   - Occlusion(가림) 검증 없음 (마스크나 손으로 가린 상태도 저장됨)

**영향**:
- Base Bank(정면)와는 전혀 닮지 않았는데, 오염된 Dynamic Bank(측면) 데이터만 믿고 79% 유사도가 나오는 상황 발생
- 시스템이 한 번 오인식을 하면 그 오인식이 계속 확산됨

**실제 사례**:
- 이미지 속 상황처럼 Base Bank와는 안 닮았는데 오염된 Dynamic Bank 데이터만 믿고 높은 유사도가 나오는 경우
- 과거에 잘못 수집된 'yh'의 측면 임베딩이 원인일 확률이 매우 높음

---

### 1.2 문제 2: "승자 독식" 방식의 Bank 매칭 로직

**위치**: `backend/main.py:1273-1285` (Bank 매칭 상세 로직)

**현재 로직**:
```python
# Base, Masked, Dynamic Bank 각각 매칭 후
max_similarity = max(base_sim, masked_sim, dynamic_sim)
best_person_id = best_base_person_id  # 우선순위: Base > Masked > Dynamic
if dynamic_sim > max(base_sim, masked_sim):
    best_person_id = best_dynamic_person_id
```

**문제점**:
1. **신뢰도 무시**: Base Bank(등록된 정면 사진)와는 전혀 닮지 않았더라도(예: 유사도 0.2), 오염된 Dynamic Bank에 우연히 닮은 임베딩이 있어 높은 점수(예: 0.79)가 나오면, 시스템은 최종적으로 0.79라고 판단해버립니다.
2. **데이터 소스 동등 취급**: 신뢰도가 다른 데이터 소스를 동등하게 취급하는 것이 문제입니다.
   - Base Bank: 가장 신뢰 (등록된 정면 사진)
   - Dynamic Bank: 중간 신뢰 (오염 가능성 있음)
   - Masked Bank: 낮은 신뢰 (불확실성 높음)

**영향**:
- Base Bank와는 안 닮았는데 오염된 Dynamic Bank 데이터만 믿고 높은 유사도가 나오는 상황
- 오인식 확산의 직접적 원인

---

### 1.3 문제 3: 오염된 Bank 정화 메커니즘 부재

**현재 상태**:
- Dynamic Bank에 이미 오염된 데이터가 들어간 경우, 이를 정리하는 메커니즘이 없음
- Bank 크기 관리도 없어서 오래된 오염 데이터가 계속 남아있음

**영향**:
- 이미 오염된 현재의 Dynamic Bank 데이터가 계속 사용됨
- 오염 데이터의 재발 방지 불가

---

## 2. 개선 실행 계획

### 2.1 1단계: 가장 시급한 조치 - Dynamic Bank 입력 필터 강화 (Hygiene Check)

**목표**: "쓰레기가 들어오면 쓰레기가 나간다(Garbage In, Garbage Out)" 문제 해결

**수정 대상**: `backend/main.py:1657-1672` (임베딩 자동 수집 로직)

**적용 로직**:
1. **Base Bank와의 최소 유사도 검증 (>= 0.6)**
   - 새로 들어온 임베딩이 현재 매칭된 인물의 Base Bank와 비교했을 때 최소한의 유사도(0.6 이상)는 넘어야 Dynamic Bank에 추가 허용
   - 목적: 정면 얼굴(Base)과 너무 다르게 생긴 임베딩이 들어오는 것을 막음

2. **고화질 검증 강화 (얼굴 크기 >= 200px)**
   - 현재 화질 추정(`estimate_face_quality`)에서 확실한 "high" 등급일 때만 수집
   - 얼굴 크기 최소 200px 이상 등 기준 상향 필요

3. **Occlusion 없는 상태 검증 (랜드마크 기반)**
   - 현재의 유사도 기반 마스크 추정은 부정확함
   - 랜드마크 기반 Occlusion 판단 도입: 주요 랜드마크(눈, 코, 입)가 모두 선명하게 보일 때만 Dynamic Bank에 추가
   - 마스크나 손으로 가린 상태의 임베딩은 실시간 매칭에만 쓰고 저장하지 않음

**예상 소요 시간**: 2-3시간  
**위험도**: 낮음 (기존 로직에 검증만 추가)

---

### 2.2 2단계: 핵심 로직 변경 - 가중치 기반 매칭 (Weighted Voting)

**목표**: "승자 독식" 방식 개선

**수정 대상**: `backend/main.py:1273-1285` (Bank 매칭 상세 로직)

**적용 로직**:
1. **가중치 기반 매칭 (Weighted Voting)**
   - `max()` 함수를 신뢰도 기반 가중치 합산으로 변경
   - 각 Bank의 신뢰도 가중치 설정:
     - `W_BASE = 1.0` (가장 신뢰)
     - `W_DYNAMIC = 0.8` (중간 신뢰, 오염 가능성 있음)
     - `W_MASKED = 0.6` (낮은 신뢰, 불확실성 높음)

2. **Base 점수 기준 보정**
   ```python
   # Base 점수가 너무 낮으면 다른 Bank 점수가 아무리 높아도 신뢰하지 않음
   if base_sim < 0.4:
       # Base와 너무 다르면 Dynamic/Masked 점수를 크게 깎아서 반영
       confident_dynamic = dynamic_sim * 0.7
       confident_masked = masked_sim * 0.6
       final_sim = max(base_sim, confident_dynamic, confident_masked)
   else:
       # Base와 어느 정도 닮았다면, 다른 Bank의 높은 점수를 인정해주되 가중치 적용
       final_sim = max(
           base_sim * W_BASE,
           dynamic_sim * W_DYNAMIC,
           masked_sim * W_MASKED
       )
   final_sim = min(final_sim, 1.0)  # 최종 유사도는 1.0을 넘을 수 없음
   ```

**예상 소요 시간**: 3-4시간  
**위험도**: 중간 (핵심 로직 변경이므로 테스트 필요)

---

### 2.3 3단계: 유지보수 - 오염된 Bank 정화 (Cleanup)

**목표**: 이미 오염된 현재의 Dynamic Bank 데이터 정리

**수정 대상**: 별도 유지보수 스크립트 생성

**적용 로직**:
1. **전수 검사 스크립트 실행**
   - 현재 저장된 모든 `bank_dynamic.npy` 파일을 로드
   - Dynamic Bank에 있는 각 임베딩을 해당 인물의 Base Bank와 다시 비교
   - Base Bank와의 유사도가 특정 기준(예: 0.5) 이하인 임베딩은 오염된 것으로 간주하고 삭제

2. **각도별 최대 개수 제한**
   - 특정 각도(예: 오른쪽 프로필)에 너무 많은 임베딩이 쌓이지 않도록, 각도별로 가장 품질이 좋은 상위 N개(예: 5개)만 남기고 나머지는 삭제하는 로직 추가

3. **정기적 정화 스케줄**
   - 주기적으로(예: 매주) Dynamic Bank를 검사하여 오염 데이터 제거

**예상 소요 시간**: 4-5시간  
**위험도**: 낮음 (별도 스크립트이므로 기존 로직에 영향 없음)

---

## 3. 완료된 작업

### ✅ 1단계: Dynamic Bank 입력 필터 강화 (완료)

**완료 일자**: 2024년 (진행 중)

**구현 내용**:

1. **랜드마크 기반 Occlusion 검증 함수 추가**
   - 파일: `src/utils/face_angle_detector.py`
   - 함수: `check_face_occlusion(face, bbox)`
   - 기능:
     - 주요 랜드마크(눈, 코, 입) 유효성 검증
     - bbox 내 랜드마크 위치 확인
     - 눈 간격, 코 위치, 입 위치, 입 너비 검증

2. **Dynamic Bank 추가 로직에 3가지 검증 추가**
   - 파일: `backend/main.py:1661-1710`
   - 검증 1: Base Bank와의 최소 유사도 검증 (>= 0.6)
   - 검증 2: 고화질 검증 강화 (얼굴 크기 >= 200px)
   - 검증 3: Occlusion 검증 (랜드마크 기반)

**코드 변경 사항**:
```python
# 검증 1: Base Bank와의 최소 유사도 검증
MIN_BASE_SIMILARITY_FOR_DYNAMIC_BANK = 0.6
if base_sim_result >= MIN_BASE_SIMILARITY_FOR_DYNAMIC_BANK:
    # 검증 2: 고화질 검증 강화
    MIN_FACE_SIZE_FOR_DYNAMIC_BANK = 200
    if face_size >= MIN_FACE_SIZE_FOR_DYNAMIC_BANK:
        # 검증 3: Occlusion 검증
        if check_face_occlusion(face_obj, box):
            should_add_to_dynamic_bank = True
```

**개선 효과**:
- ✅ 오인식으로 인한 임베딩 오염 방지
- ✅ Base Bank와 유사도가 낮은 임베딩 필터링
- ✅ 고화질 얼굴만 수집
- ✅ 마스크/가림 상태의 얼굴 제외

---

## 4. 예정된 작업

### 🔄 2단계: 가중치 기반 매칭 로직 변경 (예정)

**예상 시작 일자**: 1단계 테스트 완료 후

**구현 계획**:

1. **가중치 상수 정의**
   ```python
   W_BASE = 1.0    # 가장 신뢰
   W_DYNAMIC = 0.8 # 중간 신뢰 (오염 가능성 있음)
   W_MASKED = 0.6  # 낮은 신뢰 (불확실성 높음)
   ```

2. **Base 점수 기준 보정 로직 구현**
   - `backend/main.py:1273-1285` 수정
   - `max()` 함수 대신 가중치 기반 계산으로 변경

3. **테스트 및 검증**
   - 실제 데이터로 False Positive/False Negative 측정
   - Base Bank와 Dynamic Bank 점수 분포 분석

**예상 효과**:
- Base Bank와는 안 닮았는데 오염된 Dynamic Bank 데이터만 믿고 높은 점수가 나오는 상황 방지
- Base 점수가 낮으면 최종 점수도 강제로 낮아짐

---

### 🔄 3단계: 오염된 Bank 정화 스크립트 (예정)

**예상 시작 일자**: 2단계 완료 후

**구현 계획**:

1. **정화 스크립트 생성**
   - 파일: `scripts/cleanup_dynamic_bank.py`
   - 기능:
     - 모든 `bank_dynamic.npy` 파일 로드
     - Base Bank와 역검증
     - 오염 데이터 삭제
     - 각도별 최대 개수 제한

2. **정기적 정화 스케줄 설정**
   - 주기적 실행 스크립트 또는 cron job 설정

**예상 효과**:
- 이미 오염된 현재의 Dynamic Bank 데이터 정리
- 오염 데이터의 재발 방지

---

## 5. 개선 효과 및 예상 결과

### 5.1 즉시 효과 (1단계 완료 후)

1. **오인식 확산 방지**
   - 잘못된 임베딩이 Dynamic Bank에 저장되는 것을 막음
   - "쓰레기가 들어오면 쓰레기가 나간다" 문제 해결

2. **데이터 품질 향상**
   - 고화질 얼굴만 수집
   - Occlusion 없는 얼굴만 저장

3. **Base Bank 일관성 유지**
   - Base Bank와 유사도가 낮은 임베딩 필터링
   - 정면 얼굴과 일관성 있는 데이터만 수집

### 5.2 중기 효과 (2단계 완료 후)

1. **매칭 정확도 향상**
   - Base Bank 신뢰도 기반 가중치 적용
   - 오염된 Dynamic Bank의 영향 감소

2. **False Positive 감소**
   - Base 점수가 낮으면 최종 점수도 낮아짐
   - 오인식 확산 방지

### 5.3 장기 효과 (3단계 완료 후)

1. **시스템 안정성 향상**
   - 오염된 데이터 정기적 정화
   - Bank 크기 관리로 성능 유지

2. **유지보수 용이성**
   - 자동화된 정화 프로세스
   - 데이터 품질 모니터링

---

## 6. 참고 자료

- **시스템 로직 문서**: `SYSTEM_LOGIC.md` (7.5절, 7.6절)
- **관련 코드**:
  - `backend/main.py:1657-1710` (Dynamic Bank 추가 로직)
  - `backend/main.py:1273-1285` (Bank 매칭 로직)
  - `src/utils/face_angle_detector.py` (Occlusion 검증 함수)

---

**작성일**: 2024년  
**버전**: 1.0  
**작성자**: FaceWatch 개발팀

