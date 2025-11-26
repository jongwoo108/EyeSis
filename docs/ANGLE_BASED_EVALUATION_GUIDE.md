# 각도별 임베딩 정확도 평가 가이드

## 📋 목차
1. [개요](#개요)
2. [전제 조건](#전제-조건)
3. [데이터 준비](#데이터-준비)
4. [평가 실행](#평가-실행)
5. [결과 해석](#결과-해석)
6. [문제 해결](#문제-해결)

---

## 개요

### 목적
CCTV 영상에서 수집된 동적 임베딩의 정확도를 평가하기 위한 가이드입니다. 정답 데이터(수동 추출)와 CCTV에서 자동 수집된 임베딩을 각도별로 정확히 일치하는 것만 비교하여 평가합니다.

### 파일 용도 구분
- **인식용**: `bank_dynamic.npy` - 실시간 인식에 사용되는 통합 파일
- **평가용**: `bank_left.npy`, `bank_right.npy` 등 - 정답 데이터와 비교하기 위한 각도별 분리 파일

**중요**: 각도별 분리 파일(`bank_left.npy` 등)은 **인식에는 사용되지 않습니다**. 평가 목적으로만 사용됩니다.

### 평가 방식
- **정답 데이터**: `outputs/embeddings_manual/{person_id}/bank_{angle}.npy`
- **평가 대상**: `outputs/embeddings/{person_id}/bank_{angle}.npy` (평가용 각도별 분리 파일)
- **비교 원칙**: 같은 각도끼리만 비교 (예: `bank_left.npy` vs `bank_left.npy`)

### 평가 스크립트
- **파일 위치**: `src/evaluate_dynamic_bank.py`
- **실행 방법**: `python src/evaluate_dynamic_bank.py`

---

## 전제 조건

### 1. 필수 데이터 구조

#### 정답 데이터 (`outputs/embeddings_manual/`)
```
outputs/embeddings_manual/
├── {person_id}/
│   ├── bank_left.npy      # 왼쪽 각도 임베딩 배열
│   ├── bank_right.npy     # 오른쪽 각도 임베딩 배열
│   ├── bank_top.npy       # 위쪽 각도 임베딩 배열
│   ├── bank_front.npy     # 정면 각도 임베딩 배열
│   ├── embedding_left.npy # 왼쪽 각도 centroid
│   ├── embedding_right.npy
│   ├── embedding_top.npy
│   └── embedding_front.npy
```

#### CCTV 데이터 (`outputs/embeddings/`)
```
outputs/embeddings/
├── {person_id}/
│   ├── bank_dynamic.npy   # 인식용: 모든 각도 통합 (실시간 인식에 사용)
│   ├── bank_left.npy      # 평가용: 왼쪽 각도 임베딩 배열
│   ├── bank_right.npy     # 평가용: 오른쪽 각도 임베딩 배열
│   ├── bank_top.npy       # 평가용: 위쪽 각도 임베딩 배열
│   ├── bank_front.npy     # 평가용: 정면 각도 임베딩 배열
│   ├── embedding_left.npy # 평가용: 왼쪽 각도 centroid
│   ├── embedding_right.npy # 평가용: 오른쪽 각도 centroid
│   ├── embedding_top.npy  # 평가용: 위쪽 각도 centroid
│   ├── embedding_front.npy # 평가용: 정면 각도 centroid
│   ├── angles_dynamic.json # 각도 메타데이터
│   └── collection_status.json # 수집 완료 상태
```

**주의**: 
- `bank_dynamic.npy`: 인식용 (실시간 인식에 사용)
- `bank_{angle}.npy`: 평가용 (인식에는 사용되지 않음, 평가 스크립트에서만 사용)

### 2. 각도 분류 기준

#### 정답 데이터 (파일명 기반)
- 파일명에 `left`, `_l`, `_L` 포함 → `left`
- 파일명에 `right`, `_r`, `_R` 포함 → `right`
- 파일명에 `top`, `_t`, `_T`, `up`, `_u`, `_U` 포함 → `top`
- 그 외 → `front` (기본값)

#### CCTV 데이터 (랜드마크 기반)
- **left**: yaw 각도 -45° ~ -15°
- **right**: yaw 각도 15° ~ 45°
- **top**: pitch 각도 > 15°
- **front**: yaw 각도 -15° ~ 15°
- **left_profile**: yaw 각도 < -45°
- **right_profile**: yaw 각도 > 45°

### 3. 필수 각도
평가를 위해서는 최소한 다음 각도가 필요합니다:
- `front`: 최소 1개
- `left`: 최소 1개
- `right`: 최소 1개
- `top`: 최소 1개

---

## 데이터 준비

### 1단계: 정답 데이터 생성

#### 1-1. 각도별 사진 준비
`images/enroll/{person_id}/` 폴더에 각도별 사진을 준비합니다:

```
images/enroll/
├── {person_id}/
│   ├── {person_id}.jpg        # 정면 (front)
│   ├── {person_id}_left.jpg   # 왼쪽 (left)
│   ├── {person_id}_right.jpg  # 오른쪽 (right)
│   └── {person_id}_top.jpg    # 위쪽 (top)
```

**주의사항**:
- 파일명에 각도 정보가 포함되어야 합니다 (`_left`, `_right`, `_top` 등)
- 사진은 해당 각도로 정확하게 촬영되어야 합니다

#### 1-2. 정답 임베딩 추출
```bash
python scripts/extract_angle_embeddings.py
```

이 스크립트는:
- `images/enroll/` 폴더의 각도별 사진을 읽어서
- 파일명에서 각도를 추출하고
- 각도별 임베딩을 추출하여
- `outputs/embeddings_manual/{person_id}/` 폴더에 저장합니다

**출력 예시**:
```
📐 LEFT 각도 (1개 파일):
  ▶ jw_left.jpeg
    ✅ 임베딩 추출 완료
  💾 저장 완료: embedding_left.npy (1개 임베딩)
```

### 2단계: CCTV 데이터 수집

#### 2-1. CCTV 영상 처리
CCTV 영상이나 웹캠을 통해 인물을 식별하고 각도별 임베딩을 수집합니다.

**방법 1: 웹캠 사용**
```bash
python src/face_match_webcam.py
```

**방법 2: CCTV 영상 처리**
```bash
python src/face_match_cctv.py --input videos/source/{video_file}
```

**방법 3: 백엔드 서버 사용**
```bash
python backend/main.py
# 웹 인터페이스에서 실시간 처리
```

#### 2-2. 각도별 임베딩 자동 저장
CCTV에서 인물을 식별하면:
1. 얼굴 각도가 자동으로 감지됩니다 (랜드마크 기반)
2. 각도별로 다양성 체크를 수행합니다
3. 중복이 아니고 각도가 다양하면 `bank_dynamic.npy`에 추가됩니다
4. **자동으로 각도별 파일로 분리 저장됩니다**:
   - `bank_left.npy`
   - `bank_right.npy`
   - `bank_top.npy`
   - `bank_front.npy`

#### 2-3. 수집 완료 확인
각 인물별로 다음 파일을 확인합니다:
- `outputs/embeddings/{person_id}/collection_status.json`

**예시**:
```json
{
  "is_completed": true,
  "completed_at": "2024-01-15T10:30:00",
  "collected_angles": ["front", "left", "right", "top"],
  "required_angles": ["front", "left", "right", "top"]
}
```

**수집이 완료되지 않은 경우**:
- CCTV 영상에서 해당 각도가 감지될 때까지 대기하거나
- 추가 영상을 처리하여 각도를 수집합니다

---

## 평가 실행

### 1단계: 데이터 확인

평가를 실행하기 전에 다음을 확인합니다:

```bash
# 정답 데이터 확인
ls outputs/embeddings_manual/{person_id}/

# CCTV 데이터 확인
ls outputs/embeddings/{person_id}/
```

**확인 사항**:
- ✅ 정답 데이터에 `bank_left.npy`, `bank_right.npy`, `bank_top.npy`, `bank_front.npy`가 있는지
- ✅ CCTV 데이터에 동일한 각도 파일들이 있는지
- ✅ 각 파일에 임베딩이 포함되어 있는지

### 2단계: 평가 스크립트 실행

```bash
python src/evaluate_dynamic_bank.py
```

### 3단계: 결과 확인

스크립트 실행 시 다음과 같은 출력이 표시됩니다:

```
======================================================================
📊 동적 Bank 정확도 평가 (각도별 정확히 일치하는 것만 비교)
======================================================================
   정답 데이터: outputs/embeddings_manual
   평가 대상: outputs/embeddings
   비교 방식: 정답 데이터의 각도와 CCTV 데이터의 각도가 정확히 일치하는 경우만 비교
   예: bank_left.npy (정답) vs bank_left.npy (CCTV)

👥 평가 대상 인물: 4명
  - js
  - jw
  - yh
  - ja

======================================================================
📈 평가 결과
======================================================================
   평가된 인물 수: 4/4
   정답 임베딩 수: 12개
   동적 임베딩 수: 15개

📊 전체 유사도 통계:
   평균: 0.8234
   최소: 0.7123
   최대: 0.9456
   표준편차: 0.0456

📊 임계값별 정확도:
   0.3 이상: 100.0% (12/12)
   0.4 이상: 100.0% (12/12)
   0.5 이상: 100.0% (12/12)
   0.6 이상: 100.0% (12/12)
   0.7 이상: 91.7% (11/12)
   0.8 이상: 75.0% (9/12)
   0.9 이상: 25.0% (3/12)

👤 인물별 상세 결과:

   js:
     정답: 3개, 동적: 4개
     평균 유사도: 0.8456 (최소: 0.7890, 최대: 0.9012)
     각도별 통계:
       front          :   1개, 평균: 0.9012, 범위: [0.9012, 0.9012]
       left           :   1개, 평균: 0.8234, 범위: [0.8234, 0.8234]
       right          :   1개, 평균: 0.7890, 범위: [0.7890, 0.7890]

📊 각도별 전체 통계:
   front         :    4개, 평균: 0.9012, 범위: [0.8567, 0.9456]
   left          :    4개, 평균: 0.8234, 범위: [0.7890, 0.8567]
   right         :    4개, 평균: 0.7890, 범위: [0.7567, 0.8234]
   top           :    0개, 평균: 0.0000, 범위: [0.0000, 0.0000]

======================================================================
✅ 평가 완료
======================================================================
```

---

## 결과 해석

### 1. 전체 유사도 통계

- **평균 유사도**: 전체 비교 결과의 평균값
  - 0.8 이상: 매우 좋음
  - 0.7 ~ 0.8: 양호
  - 0.7 미만: 개선 필요

- **최소/최대 유사도**: 가장 낮은/높은 유사도 값
  - 최소값이 너무 낮으면 (예: 0.5 미만) 해당 각도나 인물의 데이터 품질을 확인해야 합니다

- **표준편차**: 유사도 값의 분산
  - 낮을수록 일관성 있는 결과
  - 높으면 (예: 0.1 이상) 일부 각도나 인물에서 문제가 있을 수 있습니다

### 2. 임계값별 정확도

각 임계값 이상의 유사도를 가진 비교 비율을 보여줍니다.

- **0.7 이상**: 일반적인 얼굴 인식 임계값
- **0.8 이상**: 높은 신뢰도
- **0.9 이상**: 매우 높은 신뢰도

### 3. 인물별 상세 결과

각 인물에 대한 상세 정보:
- **정답/동적 개수**: 비교에 사용된 임베딩 개수
- **평균/최소/최대 유사도**: 해당 인물의 유사도 통계
- **각도별 통계**: 각도별 상세 통계

### 4. 각도별 전체 통계

모든 인물을 합친 각도별 통계:
- **front**: 일반적으로 가장 높은 유사도 (정면이 가장 정확)
- **left/right**: 측면 각도는 정면보다 낮을 수 있음
- **top**: 위쪽 각도는 가장 낮을 수 있음 (얼굴 특징이 덜 보임)

### 5. 해석 예시

**좋은 결과**:
```
평균: 0.8234
각도별:
  front: 0.9012
  left: 0.8234
  right: 0.7890
```
→ 모든 각도에서 양호한 유사도, 일관성 있는 결과

**개선이 필요한 결과**:
```
평균: 0.6234
각도별:
  front: 0.9012
  left: 0.4567  ← 문제!
  right: 0.5123  ← 문제!
```
→ 측면 각도의 유사도가 낮음, CCTV에서 수집한 측면 각도 데이터 품질 확인 필요

---

## 문제 해결

### 문제 1: "평가 불가 (데이터 없음)" 메시지

**원인**:
- 정답 데이터 또는 CCTV 데이터가 없음
- 각도별 파일이 없음

**해결 방법**:
1. 데이터 경로 확인:
   ```bash
   ls outputs/embeddings_manual/{person_id}/
   ls outputs/embeddings/{person_id}/
   ```

2. 각도별 파일 확인:
   - 정답: `bank_left.npy`, `bank_right.npy` 등이 있는지 확인
   - CCTV: 동일한 각도 파일이 있는지 확인

3. 데이터 재생성:
   - 정답 데이터: `python scripts/extract_angle_embeddings.py`
   - CCTV 데이터: CCTV 영상 처리 후 각도별 파일 자동 생성 확인

### 문제 2: 특정 각도만 비교되지 않음

**원인**:
- 정답 데이터에 해당 각도가 없음
- CCTV 데이터에 해당 각도가 없음

**해결 방법**:
1. 정답 데이터 확인:
   ```bash
   ls outputs/embeddings_manual/{person_id}/bank_*.npy
   ```

2. CCTV 데이터 확인:
   ```bash
   ls outputs/embeddings/{person_id}/bank_*.npy
   ```

3. 누락된 각도 수집:
   - 정답: 해당 각도 사진 추가 후 `extract_angle_embeddings.py` 재실행
   - CCTV: 해당 각도가 감지될 때까지 CCTV 영상 처리

### 문제 3: 유사도가 너무 낮음

**원인**:
- CCTV에서 수집한 각도가 정답 데이터의 각도와 실제로 다름
- 임베딩 품질 문제
- 각도 감지 오류

**해결 방법**:
1. 각도별 통계 확인:
   - 어떤 각도에서 낮은지 확인
   - 특정 인물에서만 낮은지 확인

2. CCTV 데이터 확인:
   ```bash
   # 각도 정보 확인
   cat outputs/embeddings/{person_id}/angles_dynamic.json
   ```

3. 수집 완료 상태 확인:
   ```bash
   cat outputs/embeddings/{person_id}/collection_status.json
   ```

4. 데이터 재수집:
   - 문제가 있는 각도의 데이터를 삭제하고 재수집
   - 더 많은 프레임에서 해당 각도를 수집

### 문제 4: 특정 인물만 평가되지 않음

**원인**:
- 해당 인물의 정답 데이터 또는 CCTV 데이터가 없음
- 각도별 파일이 불완전함

**해결 방법**:
1. 인물별 데이터 확인:
   ```bash
   # 정답 데이터
   ls outputs/embeddings_manual/{person_id}/
   
   # CCTV 데이터
   ls outputs/embeddings/{person_id}/
   ```

2. 최소 요구사항 확인:
   - `front`, `left`, `right`, `top` 각도가 모두 있어야 함
   - 각도별로 최소 1개 이상의 임베딩이 있어야 함

---

## 추가 팁

### 1. 평가 전 데이터 검증

평가 전에 다음 스크립트로 데이터를 검증할 수 있습니다:

```python
import numpy as np
from pathlib import Path

# 정답 데이터 확인
manual_dir = Path("outputs/embeddings_manual")
for person_dir in manual_dir.iterdir():
    if person_dir.is_dir():
        print(f"\n{person_dir.name}:")
        for angle_file in person_dir.glob("bank_*.npy"):
            bank = np.load(angle_file)
            print(f"  {angle_file.name}: {bank.shape[0]}개 임베딩")
```

### 2. 부분 평가

특정 인물만 평가하려면 스크립트를 수정하거나 필터링을 추가할 수 있습니다.

### 3. 결과 저장

평가 결과를 파일로 저장하려면 스크립트를 수정하여 JSON 형식으로 저장할 수 있습니다.

---

## 관련 문서

- [동적 Bank 시스템 가이드](DYNAMIC_BANK_GUIDE.md) - 동적 bank의 인식용/평가용 구분 및 사용 방법
- [Bank 재구축 가이드](BANK_REBUILD_GUIDE.md)
- [빠른 시작 가이드](QUICK_START_AFTER_REBUILD.md)
- [각도별 임베딩 추출 스크립트](../scripts/extract_angle_embeddings.py)
- [동적 Bank 평가 스크립트](../src/evaluate_dynamic_bank.py)

---

**마지막 업데이트**: 2024-01-15

