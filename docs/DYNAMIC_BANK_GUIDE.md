# 동적 Bank 시스템 가이드

## 📋 목차
1. [개요](#개요)
2. [파일 구조 및 용도](#파일-구조-및-용도)
3. [동작 원리](#동작-원리)
4. [사용 방법](#사용-방법)
5. [주의사항](#주의사항)

---

## 개요

### 목적
CCTV 영상에서 인물을 식별한 후, 다양한 각도의 임베딩을 자동으로 수집하여 **인식률을 향상**시키는 시스템입니다.

### 핵심 개념
- **동적 Bank**: CCTV 영상에서 실시간으로 수집되는 다양한 각도 임베딩
- **자동 학습**: 매칭 성공 시 자동으로 각도별 임베딩 수집
- **실시간 반영**: 수집된 임베딩이 즉시 인식에 활용됨

---

## 파일 구조 및 용도

### 1. 인식용 파일 (실시간 인식에 사용)

#### `bank_dynamic.npy`
- **용도**: 실시간 인식에 사용되는 통합 임베딩 파일
- **내용**: 모든 각도의 임베딩이 하나의 배열로 통합됨
- **형태**: `(N, 512)` numpy 배열
- **예시**: front 1개 + left 2개 + right 2개 + top 1개 = 6개 임베딩

**사용 위치**:
- 서버 시작 시 `gallery_dynamic_cache`에 로드
- 매칭 시 Base/Masked/Dynamic bank 중 최고 유사도 선택
- 동적 bank 추가 시 메모리 캐시 즉시 갱신

### 2. 평가용 파일 (정답 데이터와 비교)

#### 각도별 분리 파일
- `bank_left.npy`: 왼쪽 각도 임베딩만
- `bank_right.npy`: 오른쪽 각도 임베딩만
- `bank_top.npy`: 위쪽 각도 임베딩만
- `bank_front.npy`: 정면 각도 임베딩만
- `embedding_{angle_type}.npy`: 각도별 centroid (평균) 임베딩

**용도**:
- 정답 데이터(`embeddings_manual`)와 비교하여 정확도 평가
- 각도별 정확도 분석
- 평가 스크립트(`src/evaluate_dynamic_bank.py`)에서 사용

**주의**: 이 파일들은 **인식에는 사용되지 않습니다**. 평가 목적으로만 사용됩니다.

### 3. 메타데이터 파일

#### `angles_dynamic.json`
```json
{
  "angle_types": ["front", "left", "right", "top"],
  "yaw_angles": [5.2, -25.3, 28.1, 18.7]
}
```

#### `collection_status.json`
```json
{
  "is_completed": true,
  "completed_at": "2024-01-15T10:30:00",
  "collected_angles": ["front", "left", "right", "top"],
  "required_angles": ["front", "left", "right", "top"]
}
```

---

## 동작 원리

### 1. 임베딩 수집 프로세스

```
CCTV 영상 처리
  ↓
얼굴 감지 및 각도 추정
  ↓
매칭 성공 (Base/Masked/Dynamic bank 중 하나)
  ↓
각도별 다양성 체크
  - 이미 수집된 각도인지 확인
  - 각도별 제한 확인 (예: left 최대 3개)
  ↓
중복 체크
  - Base + Dynamic bank와 유사도 비교
  - 0.95 이상이면 중복으로 간주하여 스킵
  ↓
동적 Bank에 추가
  - bank_dynamic.npy 업데이트
  - 각도별 분리 파일 생성 (bank_left.npy 등)
  - 메모리 캐시 즉시 갱신
  ↓
실시간 인식에 즉시 반영 ✅
```

### 2. 인식 프로세스

```
CCTV 영상 처리
  ↓
얼굴 감지 및 임베딩 추출
  ↓
Base/Masked/Dynamic Bank 각각 매칭
  - Base Bank: base_sim
  - Masked Bank: masked_sim
  - Dynamic Bank: dynamic_sim (새로 추가!)
  ↓
세 결과 중 최고 유사도 선택
  - Dynamic bank 우선순위 높음 (다양한 각도 포함)
  - max(base_sim, masked_sim, dynamic_sim)
  ↓
매칭 결과 반환
```

### 3. 각도별 다양성 체크

동일한 각도의 임베딩이 너무 많이 수집되는 것을 방지합니다:

- **front**: 최대 2개
- **left**: 최대 3개
- **right**: 최대 3개
- **top**: 최대 2개
- **left_profile**, **right_profile**: 제한 없음 (드물기 때문)

### 4. 수집 완료 조건

다음 각도가 모두 수집되면 자동 수집이 중단됩니다:
- `front`: 최소 1개
- `left`: 최소 1개
- `right`: 최소 1개
- `top`: 최소 1개

---

## 사용 방법

### 1. 기본 사용 (자동)

동적 bank는 **자동으로 작동**합니다. 별도 설정이 필요 없습니다.

**동작 조건**:
- CCTV 영상에서 인물 매칭 성공 시
- 각도별 다양성 체크 통과 시
- 중복 체크 통과 시

**결과**:
- `bank_dynamic.npy`에 자동 추가
- 각도별 분리 파일 자동 생성
- 메모리 캐시 즉시 갱신
- 다음 프레임부터 인식에 즉시 활용

### 2. 수집 상태 확인

#### 파일 확인
```bash
# 동적 bank 확인
ls outputs/embeddings/{person_id}/bank_dynamic.npy

# 각도별 파일 확인 (평가용)
ls outputs/embeddings/{person_id}/bank_*.npy

# 수집 상태 확인
cat outputs/embeddings/{person_id}/collection_status.json
```

#### 서버 로그 확인
```
✅ Dynamic Bank 추가: {person_id} [left] (동적: 3개, 기준: 1개)
🔄 메모리 캐시 갱신 완료 (실시간 인식에 즉시 반영)
🎉 모든 필수 각도 수집 완료: {person_id} (front, left, right, top 모두 수집됨)
```

### 3. 수동 제어

#### 수집 완료 후 재개
`collection_status.json` 파일을 삭제하거나 `is_completed`를 `false`로 변경하면 다시 수집을 시작합니다.

```bash
# 수집 완료 상태 해제
rm outputs/embeddings/{person_id}/collection_status.json
```

#### 특정 각도만 수집
현재는 모든 각도를 수집하도록 설정되어 있습니다. 특정 각도만 수집하려면 코드 수정이 필요합니다.

---

## 주의사항

### 1. 파일 용도 구분

**인식용**:
- `bank_dynamic.npy` ✅
- 실시간 인식에 사용됨

**평가용**:
- `bank_left.npy`, `bank_right.npy` 등 ❌
- 인식에는 사용되지 않음
- 평가 스크립트에서만 사용

### 2. 메모리 사용량

- 동적 bank는 메모리에 로드되어 사용됩니다
- 인물 수가 많고 각도별 임베딩이 많으면 메모리 사용량이 증가할 수 있습니다
- 일반적으로 인물당 10개 이하의 임베딩이면 문제없습니다

### 3. 성능

- 동적 bank 추가 시 메모리 캐시가 즉시 갱신됩니다
- 다음 프레임부터 인식에 반영됩니다
- 서버 재시작 없이 실시간으로 작동합니다

### 4. 데이터 일관성

- `bank_dynamic.npy`: 인식용 (통합 파일)
- `bank_{angle}.npy`: 평가용 (각도별 분리)
- 두 파일은 동일한 데이터를 다르게 저장한 것입니다
- 평가 시에는 각도별 분리 파일을 사용합니다

---

## 관련 문서

- [각도별 임베딩 정확도 평가 가이드](ANGLE_BASED_EVALUATION_GUIDE.md)
- [Bank 재구축 가이드](BANK_REBUILD_GUIDE.md)
- [빠른 시작 가이드](QUICK_START_AFTER_REBUILD.md)

---

**마지막 업데이트**: 2024-01-15













