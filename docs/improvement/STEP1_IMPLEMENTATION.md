# 1단계 개선: Dynamic Bank 입력 필터 강화

> 오인식으로 인한 임베딩 오염 방지를 위한 검증 로직 추가

---

## 개요

Dynamic Bank에 임베딩을 추가하기 전에 3가지 검증을 수행하여 오인식으로 인한 임베딩 오염을 방지합니다.

---

## 구현 내용

### 1. 랜드마크 기반 Occlusion 검증 함수

**파일**: `src/utils/face_angle_detector.py`

**함수**: `check_face_occlusion(face, bbox)`

**기능**:
- 주요 랜드마크(눈, 코, 입) 유효성 검증
- bbox 내 랜드마크 위치 확인
- 눈 간격, 코 위치, 입 위치, 입 너비 검증

**검증 항목**:
1. 랜드마크 존재 여부 확인
2. 랜드마크가 bbox 내에 있는지 확인
3. 눈 간격 검증 (너무 작으면 occlusion)
4. 코 위치 검증 (눈 중심에서 너무 멀면 occlusion)
5. 입 위치 검증 (코보다 위에 있으면 비정상)
6. 입 너비 검증 (너무 좁으면 occlusion)
7. 랜드마크 포인트 유효성 검증 (NaN, Inf 체크)

**반환값**:
- `True`: occlusion 없음 (모든 랜드마크가 선명함)
- `False`: occlusion 있음 (마스크나 손으로 가린 상태)

---

### 2. Dynamic Bank 추가 로직 수정

**파일**: `backend/main.py:1661-1710`

**검증 1: Base Bank와의 최소 유사도 검증**

```python
MIN_BASE_SIMILARITY_FOR_DYNAMIC_BANK = 0.6

if base_sim_result >= MIN_BASE_SIMILARITY_FOR_DYNAMIC_BANK:
    # 다음 검증 진행
else:
    validation_failures.append(f"base_sim={base_sim_result:.3f} < {MIN_BASE_SIMILARITY_FOR_DYNAMIC_BANK}")
```

**목적**: 정면 얼굴(Base)과 너무 다르게 생긴 임베딩이 들어오는 것을 막음

---

**검증 2: 고화질 검증 강화**

```python
MIN_FACE_SIZE_FOR_DYNAMIC_BANK = 200

face_width = box[2] - box[0]
face_height = box[3] - box[1]
face_size = max(face_width, face_height)

if face_size >= MIN_FACE_SIZE_FOR_DYNAMIC_BANK:
    # 다음 검증 진행
else:
    validation_failures.append(f"face_size={face_size}px < {MIN_FACE_SIZE_FOR_DYNAMIC_BANK}px")
```

**목적**: 고화질 얼굴만 수집하여 데이터 품질 향상

---

**검증 3: Occlusion 검증 (랜드마크 기반)**

```python
face_index = result.get("face_index", -1)
if face_index >= 0 and face_index < len(face_objects):
    face_obj = face_objects[face_index]
    if check_face_occlusion(face_obj, box):
        should_add_to_dynamic_bank = True
    else:
        validation_failures.append(f"occlusion detected (랜드마크 검증 실패)")
```

**목적**: 마스크나 손으로 가린 상태의 임베딩은 실시간 매칭에만 쓰고 저장하지 않음

---

## 코드 변경 사항

### 1. Import 추가

```python
from src.utils.face_angle_detector import estimate_face_angle, is_diverse_angle, is_all_angles_collected, check_face_occlusion
```

### 2. face_objects 리스트 추가

```python
face_results = []
face_objects = []  # face 객체를 인덱스로 매핑하여 저장 (Dynamic Bank 검증용)
for face in faces:
    # ... 기존 로직 ...
    face_results.append({
        # ... 기존 필드 ...
        "face_index": face_index  # face 객체 인덱스 저장
    })
    face_objects.append(face)  # face 객체 저장
```

### 3. 검증 로직 추가

```python
# 1단계 개선: Dynamic Bank 입력 필터 강화 (Hygiene Check)
should_add_to_dynamic_bank = False
validation_failures = []

if AUTO_ADD_TO_DYNAMIC_BANK:
    # 검증 1: Base Bank와의 최소 유사도 검증
    MIN_BASE_SIMILARITY_FOR_DYNAMIC_BANK = 0.6
    base_sim_result = result.get("base_sim", 0.0)
    
    if base_sim_result >= MIN_BASE_SIMILARITY_FOR_DYNAMIC_BANK:
        # 검증 2: 고화질 검증 강화
        face_width = box[2] - box[0]
        face_height = box[3] - box[1]
        face_size = max(face_width, face_height)
        MIN_FACE_SIZE_FOR_DYNAMIC_BANK = 200
        
        if face_size >= MIN_FACE_SIZE_FOR_DYNAMIC_BANK:
            # 검증 3: Occlusion 검증
            face_index = result.get("face_index", -1)
            if face_index >= 0 and face_index < len(face_objects):
                face_obj = face_objects[face_index]
                if check_face_occlusion(face_obj, box):
                    should_add_to_dynamic_bank = True
                else:
                    validation_failures.append(f"occlusion detected (랜드마크 검증 실패)")
            else:
                validation_failures.append(f"face object not found (face_index={face_index})")
        else:
            validation_failures.append(f"face_size={face_size}px < {MIN_FACE_SIZE_FOR_DYNAMIC_BANK}px (고화질 검증 실패)")
    else:
        validation_failures.append(f"base_sim={base_sim_result:.3f} < {MIN_BASE_SIMILARITY_FOR_DYNAMIC_BANK} (Base Bank 유사도 검증 실패)")
    
    if should_add_to_dynamic_bank:
        # 모든 검증 통과: 동적 bank에 추가
        learning_events.append({
            "person_id": person_id,
            "person_name": name,
            "angle_type": angle_type,
            "yaw_angle": yaw_angle,
            "embedding": embedding_normalized.tolist(),
            "bank_type": "dynamic"
        })
        print(f"  ✅ [DYNAMIC BANK] 검증 통과: {person_id} (base_sim={base_sim_result:.3f}, face_size={face_size}px, angle={angle_type})")
    else:
        # 검증 실패: Dynamic Bank에 추가하지 않음
        print(f"  ⏭ [DYNAMIC BANK] 검증 실패: {person_id} | 이유: {', '.join(validation_failures)}")
```

---

## 테스트 방법

### 1. 검증 로그 확인

서버 실행 시 콘솔에서 다음 로그를 확인할 수 있습니다:

```
✅ [DYNAMIC BANK] 검증 통과: person_id (base_sim=0.75, face_size=250px, angle=left)
⏭ [DYNAMIC BANK] 검증 실패: person_id | 이유: base_sim=0.45 < 0.6 (Base Bank 유사도 검증 실패)
⏭ [DYNAMIC BANK] 검증 실패: person_id | 이유: face_size=150px < 200px (고화질 검증 실패)
⏭ [DYNAMIC BANK] 검증 실패: person_id | 이유: occlusion detected (랜드마크 검증 실패)
```

### 2. Dynamic Bank 파일 확인

`outputs/embeddings/{person_id}/bank_dynamic.npy` 파일을 확인하여:
- 검증을 통과한 임베딩만 저장되었는지 확인
- 이전에 저장된 오염된 임베딩이 있는지 확인

### 3. 매칭 정확도 측정

- 실제 데이터로 False Positive/False Negative 측정
- Base Bank와 Dynamic Bank 점수 분포 분석

---

## 개선 효과

### 즉시 효과

1. **오인식 확산 방지**
   - 잘못된 임베딩이 Dynamic Bank에 저장되는 것을 막음
   - "쓰레기가 들어오면 쓰레기가 나간다" 문제 해결

2. **데이터 품질 향상**
   - 고화질 얼굴만 수집
   - Occlusion 없는 얼굴만 저장

3. **Base Bank 일관성 유지**
   - Base Bank와 유사도가 낮은 임베딩 필터링
   - 정면 얼굴과 일관성 있는 데이터만 수집

---

## 다음 단계

1. **테스트 및 검증**
   - 실제 데이터로 검증 로직 테스트
   - False Positive/False Negative 측정

2. **2단계 개선 준비**
   - 가중치 기반 매칭 로직 변경
   - Base 점수 기준 보정 로직 구현

---

**작성일**: 2024년  
**버전**: 1.0  
**작성자**: FaceWatch 개발팀

