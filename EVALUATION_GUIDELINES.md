# FaceWatch 얼굴 인식 시스템 평가 지표 가이드라인

## 📋 목차

1. [하이퍼파라미터 목록](#하이퍼파라미터-목록)
2. [프레임 정보 및 메타데이터](#프레임-정보-및-메타데이터)
3. [평가 지표 정의](#평가-지표-정의)
4. [Confusion Matrix 구성](#confusion-matrix-구성)
5. [평가 프로세스](#평가-프로세스)
6. [데이터 수집 방법](#데이터-수집-방법)

---

## 1. 하이퍼파라미터 목록

### 1.1 매칭 임계값 (Matching Thresholds)

#### 화질별 기본 임계값
| 화질 | `main_threshold` | `gap_margin` | 설명 |
|------|-----------------|--------------|------|
| **High** | 0.42 | 0.12 | 고화질 얼굴 (큰 얼굴, 선명함) |
| **Medium** | 0.40 | 0.10 | 중화질 얼굴 (중간 크기) |
| **Low** | 0.38 | 0.08 | 저화질 얼굴 (작은 얼굴, 흐릿함) |

**튜닝 가이드:**
- False Positive가 많으면: `+0.01 ~ +0.02` 증가
- True Positive가 적으면: `-0.01 ~ -0.02` 감소
- 특정 화질에서만 문제가 있으면 해당 화질만 조정

#### Suspect IDs 모드 추가 조건
```python
if suspect_ids:
    main_threshold += 0.02  # threshold 상향
    gap_margin += 0.03      # gap 기준 더 엄격하게
    min_absolute = 0.45     # 절대값 최소 0.45
```

### 1.2 마스크 얼굴 인식 파라미터

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `MASKED_BANK_MASK_PROB_THRESHOLD` | 0.5 | mask_prob >= 0.5이면 masked bank로 분류 |
| `MASKED_CANDIDATE_MIN_SIM` | 0.25 | base_sim >= 0.25 이상이어야 masked candidate로 판단 |
| `MASKED_CANDIDATE_MIN_FRAMES` | 3 | 연속 N 프레임 이상 조건 충족 시 masked bank에 추가 |
| `MASKED_TRACKING_IOU_THRESHOLD` | 0.5 | bbox tracking을 위한 IoU 임계값 |

### 1.3 Bank 관리 파라미터

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `BANK_DUPLICATE_THRESHOLD` | 0.95 | Bank에 추가 시 중복 체크 임계값 (유사도 >= 0.95면 스킵) |
| `EMBEDDINGS_DIR` | `outputs/embeddings` | 임베딩 저장 경로 |

### 1.4 얼굴 감지 파라미터

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `det_size` | (640, 640) | RetinaFace 감지 크기 |
| `model_name` | "buffalo_l" | InsightFace 모델 이름 |

### 1.5 얼굴 필터링 파라미터

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `is_same_face_region` IoU | 0.3 | 같은 얼굴 영역 판단 IoU 임계값 |
| `distance_threshold` | `face_diag * 0.6` | 중심점 거리 임계값 (대각선의 60%) |

### 1.6 매칭 조건

**매칭 성공 조건 (모두 만족해야 함):**
1. `max_similarity >= main_threshold` (화질별 임계값)
2. `sim_gap >= gap_margin` (최고 유사도와 두 번째 유사도의 차이)
3. `second_similarity < main_threshold - 0.02` (두 번째 후보가 너무 높으면 미매칭)

**Suspect IDs 모드 추가 조건:**
- `max_similarity >= 0.45` (절대값 최소 기준)

---

## 2. 프레임 정보 및 메타데이터

### 2.1 프레임별 수집 정보

각 프레임에서 수집해야 하는 정보:

```python
{
    "frame_idx": int,              # 프레임 번호
    "timestamp": float,             # 타임스탬프 (초)
    "video_path": str,              # 비디오 파일 경로
    
    # 얼굴 감지 정보
    "face_count": int,              # 감지된 얼굴 개수
    "faces": [
        {
            "bbox": [x1, y1, x2, y2],  # 바운딩 박스 좌표
            "face_id": int,             # 얼굴 ID (프레임 내 인덱스)
            
            # 임베딩 정보
            "embedding": np.ndarray,     # 512차원 임베딩 벡터 (L2 정규화됨)
            
            # 매칭 결과
            "best_person_id": str,      # 매칭된 person_id ("unknown" 또는 실제 ID)
            "max_similarity": float,    # 최고 유사도 (0.0 ~ 1.0)
            "base_sim": float,          # base bank 유사도
            "masked_sim": float,        # masked bank 유사도
            "second_similarity": float,  # 두 번째 유사도
            "sim_gap": float,           # sim_gap = max_sim - second_sim
            
            # 임계값 정보
            "main_threshold": float,    # 적용된 임계값
            "gap_margin": float,        # 적용된 gap_margin
            "face_quality": str,        # "high" | "medium" | "low"
            
            # 매칭 판단
            "is_match": bool,           # 매칭 성공 여부
            "bank_type": str,           # "base" | "masked" | null
            
            # 얼굴 특성
            "angle_type": str,          # "front" | "left" | "right" | "left_profile" | "right_profile"
            "yaw_angle": float,         # Yaw 각도 (도)
            "mask_prob": float,         # 마스크 가능성 (0.0 ~ 1.0)
            "is_masked_candidate": bool, # masked candidate 여부
            
            # Ground Truth (평가용)
            "ground_truth_person_id": str,  # 실제 person_id (평가 시 필요)
            "ground_truth_is_present": bool # 실제로 해당 인물이 있는지 여부
        }
    ]
}
```

### 2.2 프레임 처리 정보

```python
{
    # 비디오 정보
    "video_fps": float,              # FPS
    "video_duration": float,          # 총 길이 (초)
    "total_frames": int,             # 총 프레임 수
    
    # 처리 설정
    "suspect_ids": List[str],        # 선택된 용의자 ID 리스트 (None이면 전체 검색)
    "use_webcam": bool,              # 웹캠 사용 여부
    
    # Bank 정보
    "bank_info": {
        "person_id": {
            "base_count": int,       # base bank 임베딩 개수
            "masked_count": int      # masked bank 임베딩 개수
        }
    }
}
```

---

## 3. 평가 지표 정의

### 3.1 기본 평가 지표

#### True Positive (TP)
- **정의**: 실제로 해당 인물이 있고, 시스템이 올바르게 매칭한 경우
- **조건**: 
  - `ground_truth_person_id == best_person_id`
  - `is_match == True`
  - `ground_truth_is_present == True`

#### False Positive (FP)
- **정의**: 실제로는 다른 인물이거나 없는 경우인데, 시스템이 매칭한 경우
- **조건**:
  - `is_match == True`
  - `ground_truth_person_id != best_person_id` 또는 `ground_truth_is_present == False`

#### False Negative (FN)
- **정의**: 실제로 해당 인물이 있는데, 시스템이 매칭하지 못한 경우
- **조건**:
  - `ground_truth_is_present == True`
  - `is_match == False` 또는 `best_person_id == "unknown"`

#### True Negative (TN)
- **정의**: 실제로 해당 인물이 없고, 시스템도 매칭하지 않은 경우
- **조건**:
  - `ground_truth_is_present == False`
  - `is_match == False` 또는 `best_person_id == "unknown"`

### 3.2 계산 지표

#### Precision (정밀도)
```
Precision = TP / (TP + FP)
```
- **의미**: 매칭한 것 중에서 실제로 맞는 비율
- **목표**: 높을수록 좋음 (오탐 방지)

#### Recall (재현율)
```
Recall = TP / (TP + FN)
```
- **의미**: 실제로 있는 인물 중에서 찾아낸 비율
- **목표**: 높을수록 좋음 (미탐 방지)

#### F1-Score
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```
- **의미**: Precision과 Recall의 조화 평균
- **목표**: 높을수록 좋음 (균형잡힌 성능)

#### Accuracy (정확도)
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
- **의미**: 전체 중 맞게 판단한 비율
- **주의**: TN이 많으면 Accuracy가 높아질 수 있음 (불균형 데이터셋)

### 3.3 화질별 평가 지표

각 화질(high/medium/low)별로 별도로 계산:
- `precision_high`, `recall_high`, `f1_high`
- `precision_medium`, `recall_medium`, `f1_medium`
- `precision_low`, `recall_low`, `f1_low`

### 3.4 마스크 여부별 평가 지표

- `precision_masked`: 마스크 쓴 얼굴의 정밀도
- `recall_masked`: 마스크 쓴 얼굴의 재현율
- `precision_no_mask`: 마스크 없는 얼굴의 정밀도
- `recall_no_mask`: 마스크 없는 얼굴의 재현율

### 3.5 Bank 타입별 평가 지표

- `precision_base`: base bank로 매칭한 경우의 정밀도
- `recall_base`: base bank로 매칭한 경우의 재현율
- `precision_masked_bank`: masked bank로 매칭한 경우의 정밀도
- `recall_masked_bank`: masked bank로 매칭한 경우의 재현율

---

## 4. Confusion Matrix 구성

### 4.1 인물별 Confusion Matrix

각 인물(person_id)별로 Confusion Matrix 생성:

```
                    예측
                매칭  미매칭
실제  매칭     TP    FN
      미매칭   FP    TN
```

**예시:**
```
person_id: "hani"

                    예측
                매칭  미매칭
실제  매칭     85    15    (TP=85, FN=15)
      미매칭   3     97     (FP=3, TN=97)

Precision = 85 / (85 + 3) = 0.966
Recall = 85 / (85 + 15) = 0.850
F1 = 2 * (0.966 * 0.850) / (0.966 + 0.850) = 0.904
```

### 4.2 전체 Confusion Matrix (다중 클래스)

모든 인물을 포함한 다중 클래스 Confusion Matrix:

```
                    예측
            hani  yh  js  jw  unknown
실제  hani   85   2   1   0    12
      yh      1  92   0   1     6
      js      0   0  78   2    20
      jw      2   1   1  88     8
      없음    3   2   1   1    N/A
```

### 4.3 매칭 여부 Confusion Matrix (이진 분류)

매칭/미매칭만 구분하는 이진 분류:

```
                    예측
                매칭  미매칭
실제  매칭     TP    FN
      미매칭   FP    TN
```

---

## 5. 평가 프로세스

### 5.1 데이터 준비

1. **테스트 비디오 준비**
   - Ground Truth가 있는 비디오 파일
   - 각 프레임별로 실제 인물 정보가 표시된 데이터

2. **Ground Truth 데이터 형식**
   ```json
   {
       "video_path": "videos/test_video.mp4",
       "ground_truth": [
           {
               "frame_idx": 0,
               "timestamp": 0.0,
               "faces": [
                   {
                       "bbox": [100, 200, 300, 400],
                       "person_id": "hani",
                       "is_present": true
                   }
               ]
           },
           ...
       ]
   }
   ```

### 5.2 평가 실행

1. **비디오 처리**
   ```bash
   python src/face_match_cctv.py --video-path videos/test_video.mp4 --output-dir outputs/evaluation
   ```

2. **결과 수집**
   - 각 프레임의 매칭 결과를 CSV 또는 JSON으로 저장
   - Ground Truth와 비교

3. **지표 계산**
   ```python
   # 평가 스크립트 예시
   from sklearn.metrics import confusion_matrix, classification_report
   import pandas as pd
   
   # 결과 로드
   results = pd.read_csv("outputs/evaluation/results.csv")
   ground_truth = pd.read_csv("ground_truth.csv")
   
   # Confusion Matrix 생성
   cm = confusion_matrix(ground_truth["person_id"], results["best_person_id"])
   
   # 지표 계산
   report = classification_report(ground_truth["person_id"], results["best_person_id"])
   ```

### 5.3 평가 리포트 생성

평가 리포트에 포함할 내용:

1. **전체 요약**
   - 총 프레임 수
   - 총 얼굴 감지 수
   - 매칭 성공 수
   - Precision, Recall, F1-Score

2. **인물별 성능**
   - 각 person_id별 TP, FP, FN, TN
   - Precision, Recall, F1-Score
   - Confusion Matrix

3. **화질별 성능**
   - High/Medium/Low 화질별 지표

4. **마스크 여부별 성능**
   - 마스크 쓴 얼굴 vs 마스크 없는 얼굴

5. **Bank 타입별 성능**
   - Base bank vs Masked bank

6. **임계값 분석**
   - 각 임계값에서의 성능 변화
   - 최적 임계값 추천

---

## 6. 데이터 수집 방법

### 6.1 로그 수집

`backend/main.py`의 `process_detection` 함수에서 다음 정보를 로그로 저장:

```python
# 평가용 로그 저장
evaluation_log = {
    "frame_idx": frame_idx,
    "timestamp": timestamp,
    "face_id": face_id,
    "bbox": bbox.tolist(),
    "best_person_id": best_person_id,
    "max_similarity": float(max_similarity),
    "base_sim": float(base_sim),
    "masked_sim": float(masked_sim),
    "second_similarity": float(second_similarity),
    "sim_gap": float(sim_gap),
    "main_threshold": float(main_threshold),
    "gap_margin": float(gap_margin),
    "face_quality": face_quality,
    "is_match": is_match,
    "bank_type": bank_type,
    "angle_type": angle_type,
    "yaw_angle": float(yaw_angle),
    "mask_prob": float(mask_prob),
    "is_masked_candidate": is_masked_candidate
}
```

### 6.2 CSV 로그 형식

```csv
frame_idx,timestamp,face_id,bbox_x1,bbox_y1,bbox_x2,bbox_y2,best_person_id,max_similarity,base_sim,masked_sim,second_similarity,sim_gap,main_threshold,gap_margin,face_quality,is_match,bank_type,angle_type,yaw_angle,mask_prob,is_masked_candidate
0,0.0,0,100,200,300,400,hani,0.85,0.82,0.85,0.35,0.50,0.42,0.12,high,True,base,front,5.2,0.1,False
```

### 6.3 Ground Truth 수집

Ground Truth는 수동으로 수집하거나, 자동화된 도구를 사용:

1. **수동 수집**: 비디오를 프레임별로 확인하며 실제 인물 정보 기록
2. **자동화 도구**: 비디오 플레이어에 어노테이션 기능 추가

---

## 7. 하이퍼파라미터 튜닝 가이드

### 7.1 현재 설정값 요약

```python
# 화질별 임계값
THRESHOLDS = {
    "high": {"main_threshold": 0.42, "gap_margin": 0.12},
    "medium": {"main_threshold": 0.40, "gap_margin": 0.10},
    "low": {"main_threshold": 0.38, "gap_margin": 0.08}
}

# 마스크 관련
MASKED_BANK_MASK_PROB_THRESHOLD = 0.5
MASKED_CANDIDATE_MIN_SIM = 0.25
MASKED_CANDIDATE_MIN_FRAMES = 3
MASKED_TRACKING_IOU_THRESHOLD = 0.5

# Bank 관리
BANK_DUPLICATE_THRESHOLD = 0.95

# Suspect IDs 모드
SUSPECT_IDS_THRESHOLD_BONUS = 0.02
SUSPECT_IDS_GAP_BONUS = 0.03
SUSPECT_IDS_MIN_ABSOLUTE = 0.45
```

### 7.2 튜닝 전략

1. **False Positive가 많을 때**
   - `main_threshold` 증가: `+0.01 ~ +0.02`
   - `gap_margin` 증가: `+0.01 ~ +0.02`

2. **True Positive가 적을 때**
   - `main_threshold` 감소: `-0.01 ~ -0.02`
   - `gap_margin` 감소: `-0.01 ~ -0.02`

3. **특정 화질에서 문제가 있을 때**
   - 해당 화질의 파라미터만 조정

4. **마스크 쓴 얼굴 인식이 안 될 때**
   - `MASKED_CANDIDATE_MIN_SIM` 감소: `0.25 → 0.20`
   - `MASKED_BANK_MASK_PROB_THRESHOLD` 감소: `0.5 → 0.3`
   - `MASKED_CANDIDATE_MIN_FRAMES` 감소: `3 → 2`

---

## 8. 평가 스크립트 예시

### 8.1 기본 평가 스크립트 구조

```python
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_results(results_path, ground_truth_path):
    # 데이터 로드
    results = pd.read_csv(results_path)
    ground_truth = pd.read_csv(ground_truth_path)
    
    # 병합
    merged = pd.merge(
        results,
        ground_truth,
        on=["frame_idx", "face_id"],
        how="inner"
    )
    
    # TP, FP, FN, TN 계산
    merged["TP"] = (merged["best_person_id"] == merged["ground_truth_person_id"]) & (merged["is_match"] == True)
    merged["FP"] = (merged["best_person_id"] != merged["ground_truth_person_id"]) & (merged["is_match"] == True)
    merged["FN"] = (merged["ground_truth_is_present"] == True) & (merged["is_match"] == False)
    merged["TN"] = (merged["ground_truth_is_present"] == False) & (merged["is_match"] == False)
    
    # 지표 계산
    TP = merged["TP"].sum()
    FP = merged["FP"].sum()
    FN = merged["FN"].sum()
    TN = merged["TN"].sum()
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Confusion Matrix
    cm = confusion_matrix(
        merged["ground_truth_person_id"],
        merged["best_person_id"],
        labels=list(set(merged["ground_truth_person_id"].unique()) | set(merged["best_person_id"].unique()))
    )
    
    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm
    }

def plot_confusion_matrix(cm, labels, title="Confusion Matrix"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{title.replace(' ', '_')}.png")
    plt.close()
```

---

## 9. 참고 자료

- **튜닝 스크립트**: `scripts/tune_threshold_gap.py`
- **Bank 재구성**: `scripts/rebuild_base_bank.py`
- **매칭 로직**: `backend/main.py`의 `process_detection` 함수
- **마스크 감지**: `src/utils/mask_detector.py`
- **얼굴 각도**: `src/utils/face_angle_detector.py`

---

## 10. 체크리스트

평가 전 확인 사항:

- [ ] Ground Truth 데이터 준비 완료
- [ ] 테스트 비디오 준비 완료
- [ ] Bank 데이터 로드 확인 (`outputs/embeddings/`)
- [ ] 하이퍼파라미터 값 기록
- [ ] 로그 수집 설정 확인
- [ ] 평가 스크립트 준비 완료
- [ ] 결과 저장 경로 설정

평가 후 확인 사항:

- [ ] Confusion Matrix 생성 완료
- [ ] Precision, Recall, F1-Score 계산 완료
- [ ] 인물별 성능 분석 완료
- [ ] 화질별 성능 분석 완료
- [ ] 마스크 여부별 성능 분석 완료
- [ ] Bank 타입별 성능 분석 완료
- [ ] 평가 리포트 작성 완료

