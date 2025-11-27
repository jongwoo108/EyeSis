# FaceWatch 모델 평가 전략

## 📋 목차

1. [개요](#1-개요)
2. [평가 데이터 준비](#2-평가-데이터-준비)
3. [평가 지표](#3-평가-지표)
4. [평가 구현](#4-평가-구현)
5. [임계값 최적화](#5-임계값-최적화)
6. [결과 해석](#6-결과-해석)
7. [실전 예시](#7-실전-예시)

---

## 1. 개요

### 1.1 평가 목적

FaceWatch 모델이 예측한 결과의 **신뢰도**를 정량적으로 평가합니다.

**핵심 질문**:
- ✅ 모델이 "96%"라고 예측하면 실제로 96% 정확한가?
- ✅ "50%"로 예측한 것은 신뢰할 수 있는가?
- ✅ 어떤 임계값(threshold)을 사용해야 최적인가?

### 1.2 왜 크로스엔트로피가 아닌가?

| 크로스엔트로피 | FaceWatch (코사인 유사도) |
|---------------|--------------------------|
| **Closed-set**: 고정된 N개 클래스 | **Open-set**: "unknown" 포함 |
| **확률 분포**: 합=1.0 필요 | **유사도**: 합≠1.0 OK |
| **학습용**: 모델 가중치 업데이트 | **추론용**: 이미 학습 완료 |

**결론**: Open-set 얼굴 인식에는 **다른 평가 지표** 필요

---

## 2. 평가 데이터 준비

### 2.1 Ground Truth 라벨링

#### 라벨링 형식

```json
{
  "frame_001.jpg": "yh",
  "frame_002.jpg": "ja",
  "frame_003.jpg": "unknown",
  "frame_004.jpg": "yh",
  "frame_005.jpg": "js"
}
```

#### 라벨링 가이드

| 라벨 | 설명 | 예시 |
|------|------|------|
| `"yh"` | 명확히 yh로 식별 가능 | 정면, 선명한 얼굴 |
| `"ja"` | 명확히 ja로 식별 가능 | 측면이지만 식별 가능 |
| `"unknown"` | 식별 불가능 또는 등록되지 않은 인물 | 뒷모습, 흐림, 다른 사람 |
| `"ambiguous"` (선택) | 애매한 경우 | 경계 케이스 |

#### 라벨링 도구 (간단한 스크립트)

```python
# scripts/label_ground_truth.py
import cv2
import json
from pathlib import Path

def label_frames(frames_dir, output_json):
    """
    프레임 이미지를 보여주고 사람이 직접 라벨링
    """
    frames = sorted(Path(frames_dir).glob("*.jpg"))
    labels = {}
    
    print("라벨링 가이드:")
    print("  yh, ja, js, jw: 해당 인물")
    print("  unknown: 식별 불가")
    print("  skip: 이 프레임 건너뛰기")
    print("  quit: 종료")
    
    for frame_path in frames:
        img = cv2.imread(str(frame_path))
        cv2.imshow("Frame", img)
        cv2.waitKey(1)
        
        label = input(f"{frame_path.name}: ").strip()
        
        if label == "quit":
            break
        elif label == "skip":
            continue
        else:
            labels[frame_path.name] = label
    
    cv2.destroyAllWindows()
    
    # JSON 저장
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 라벨링 완료: {len(labels)}개 프레임")
    print(f"저장 위치: {output_json}")

if __name__ == "__main__":
    label_frames(
        frames_dir="outputs/results/test_video/frames",
        output_json="outputs/evaluation/ground_truth.json"
    )
```

### 2.2 모델 예측 수집

#### 예측 형식

```json
{
  "frame_001.jpg": {
    "predicted_id": "yh",
    "confidence": 0.96,
    "all_scores": {
      "yh": 0.96,
      "ja": 0.25,
      "js": 0.18
    }
  },
  "frame_002.jpg": {
    "predicted_id": "yh",
    "confidence": 0.50,
    "all_scores": {
      "yh": 0.50,
      "ja": 0.48,
      "js": 0.20
    }
  }
}
```

#### 예측 수집 스크립트

```python
# scripts/collect_predictions.py
import json
import cv2
from pathlib import Path
from insightface.app import FaceAnalysis
from src.utils.gallery_loader import load_gallery, match_with_bank_detailed
from src.face_enroll import l2_normalize

def collect_predictions(frames_dir, gallery_dir, output_json):
    """
    프레임들에 대한 모델 예측 수집
    """
    # 모델 로드
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    # 갤러리 로드
    gallery = load_gallery(gallery_dir, use_bank=True)
    
    predictions = {}
    frames = sorted(Path(frames_dir).glob("*.jpg"))
    
    for frame_path in frames:
        img = cv2.imread(str(frame_path))
        faces = app.get(img)
        
        if len(faces) == 0:
            predictions[frame_path.name] = {
                "predicted_id": "no_face",
                "confidence": 0.0,
                "all_scores": {}
            }
            continue
        
        # 첫 번째 얼굴만 사용 (주인공 가정)
        face = faces[0]
        embedding = l2_normalize(face.embedding.astype("float32"))
        
        # 모든 인물과 유사도 계산
        all_scores = {}
        for person_id, bank in gallery.items():
            if bank.ndim == 2:
                sims = bank @ embedding
                max_sim = float(sims.max())
            else:
                max_sim = float(bank @ embedding)
            all_scores[person_id] = max_sim
        
        # 최고 유사도
        best_id = max(all_scores, key=all_scores.get)
        best_score = all_scores[best_id]
        
        predictions[frame_path.name] = {
            "predicted_id": best_id,
            "confidence": best_score,
            "all_scores": all_scores
        }
        
        print(f"{frame_path.name}: {best_id} ({best_score:.2f})")
    
    # JSON 저장
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 예측 수집 완료: {len(predictions)}개 프레임")

if __name__ == "__main__":
    collect_predictions(
        frames_dir="outputs/results/test_video/frames",
        gallery_dir="outputs/embeddings",
        output_json="outputs/evaluation/predictions.json"
    )
```

---

## 3. 평가 지표

### 3.1 기본 분류 지표

#### 3.1.1 Confusion Matrix

```
                Predicted
              yh   ja   js   unknown
Actual  yh   [10   1    0     2    ]
        ja   [ 2   8    1     1    ]
        js   [ 0   1    7     0    ]
     unknown [ 1   2    0    15    ]
```

**Python 구현**:
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('outputs/evaluation/confusion_matrix.png', dpi=300)
    plt.show()
```

#### 3.1.2 Precision, Recall, F1-Score

```python
from sklearn.metrics import classification_report

report = classification_report(y_true, y_pred, 
                                target_names=['yh', 'ja', 'js', 'unknown'])
print(report)
```

**출력 예시**:
```
              precision    recall  f1-score   support

          yh       0.77      0.77      0.77        13
          ja       0.67      0.67      0.67        12
          js       0.88      0.88      0.88         8
     unknown       0.83      0.83      0.83        18

    accuracy                           0.78        51
   macro avg       0.79      0.79      0.79        51
weighted avg       0.78      0.78      0.78        51
```

### 3.2 신뢰도 보정 지표 (핵심!)

#### 3.2.1 ECE (Expected Calibration Error)

**목적**: "96%"라고 예측한 것 중 실제로 96%가 맞는지 확인

**수식**:
```
ECE = Σ (|avg_confidence - avg_accuracy| × n_samples_in_bin) / n_total
```

**Python 구현**:
```python
import numpy as np

def compute_ece(confidences, correctness, n_bins=10):
    """
    Expected Calibration Error
    
    Args:
        confidences: np.array([0.96, 0.50, 0.85, ...])
        correctness: np.array([True, False, True, ...])
        n_bins: 구간 개수
    
    Returns:
        ece: float (0에 가까울수록 좋음)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0
    
    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]
        
        # 이 구간에 속하는 예측
        in_bin = (confidences >= lower) & (confidences < upper)
        
        if np.sum(in_bin) > 0:
            avg_confidence = np.mean(confidences[in_bin])
            avg_accuracy = np.mean(correctness[in_bin])
            weight = np.sum(in_bin) / len(confidences)
            
            ece += np.abs(avg_confidence - avg_accuracy) * weight
    
    return ece

# 사용 예시
confidences = np.array([0.96, 0.50, 0.85, 0.42, 0.78])
correctness = np.array([1, 0, 1, 1, 0])  # 1=맞음, 0=틀림

ece = compute_ece(confidences, correctness)
print(f"ECE: {ece:.4f}")
# ECE < 0.05: 매우 잘 보정됨
# ECE < 0.10: 잘 보정됨
# ECE > 0.20: 과신 또는 과소신
```

#### 3.2.2 Reliability Diagram

```python
def plot_reliability_diagram(confidences, correctness, n_bins=10):
    """
    신뢰도 다이어그램
    
    y=x 직선에 가까울수록 잘 보정됨
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_accs = []
    bin_confs = []
    
    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]
        
        in_bin = (confidences >= lower) & (confidences < upper)
        
        if np.sum(in_bin) > 0:
            bin_centers.append((lower + upper) / 2)
            bin_accs.append(np.mean(correctness[in_bin]))
            bin_confs.append(np.mean(confidences[in_bin]))
    
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    plt.plot(bin_confs, bin_accs, 'o-', label='Model')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Reliability Diagram')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('outputs/evaluation/reliability_diagram.png', dpi=300)
    plt.show()
```

### 3.3 임계값 무관 지표

#### 3.3.1 ROC-AUC (Binary Classification)

**특정 인물 검출 평가** (예: yh vs not yh)

```python
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def evaluate_person_detection(y_true, y_scores, person_id):
    """
    특정 인물 검출 성능 평가
    
    Args:
        y_true: ["yh", "ja", "unknown", "yh", ...]
        y_scores: [0.96, 0.50, 0.30, 0.85, ...]
        person_id: "yh"
    """
    # Binary로 변환
    y_binary = [1 if label == person_id else 0 for label in y_true]
    
    # ROC 계산
    fpr, tpr, thresholds = roc_curve(y_binary, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # 플롯
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{person_id} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {person_id}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'outputs/evaluation/roc_{person_id}.png', dpi=300)
    plt.show()
    
    # 최적 임계값 (Youden's J)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"{person_id} Detection:")
    print(f"  AUC: {roc_auc:.3f}")
    print(f"  Optimal Threshold: {optimal_threshold:.3f}")
    print(f"  TPR at optimal: {tpr[optimal_idx]:.3f}")
    print(f"  FPR at optimal: {fpr[optimal_idx]:.3f}")
    
    return roc_auc, optimal_threshold
```

#### 3.3.2 Precision-Recall Curve

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

def plot_pr_curve(y_binary, y_scores, person_id):
    """
    Precision-Recall Curve
    """
    precision, recall, thresholds = precision_recall_curve(y_binary, y_scores)
    ap = average_precision_score(y_binary, y_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'{person_id} (AP = {ap:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {person_id}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'outputs/evaluation/pr_{person_id}.png', dpi=300)
    plt.show()
    
    # F1 최대화 임계값
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    
    print(f"  Best F1 Threshold: {best_threshold:.3f}")
    print(f"  F1 at best: {f1_scores[best_idx]:.3f}")
    
    return ap, best_threshold
```

### 3.4 확신도 분포 분석

```python
def analyze_confidence_distribution(confidences, correctness):
    """
    확신도 구간별 정확도 분석
    """
    bins = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    labels = ['0-30%', '30-40%', '40-50%', '50-60%', 
              '60-70%', '70-80%', '80-100%']
    
    df = pd.DataFrame({
        'confidence': confidences,
        'correct': correctness
    })
    
    df['bin'] = pd.cut(df['confidence'], bins=bins, labels=labels)
    
    # 구간별 통계
    stats = df.groupby('bin').agg({
        'correct': ['count', 'sum', 'mean']
    })
    
    stats.columns = ['Total', 'Correct', 'Accuracy']
    stats['Wrong'] = stats['Total'] - stats['Correct']
    
    print("Confidence Distribution Analysis:")
    print(stats)
    
    # 시각화
    stats['Accuracy'].plot(kind='bar', figsize=(10, 6))
    plt.ylabel('Accuracy')
    plt.xlabel('Confidence Range')
    plt.title('Accuracy by Confidence Range')
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('outputs/evaluation/confidence_distribution.png', dpi=300)
    plt.show()
```

---

## 4. 평가 구현

### 4.1 전체 평가 파이프라인

```python
# scripts/evaluate_model.py
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class FaceWatchEvaluator:
    """
    FaceWatch 모델 평가 클래스
    """
    
    def __init__(self, ground_truth_path, predictions_path, threshold=0.42):
        """
        Args:
            ground_truth_path: Ground truth JSON 경로
            predictions_path: Predictions JSON 경로
            threshold: 유사도 임계값
        """
        # 데이터 로드
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            self.ground_truth = json.load(f)
        
        with open(predictions_path, 'r', encoding='utf-8') as f:
            self.predictions = json.load(f)
        
        self.threshold = threshold
        
        # 공통 프레임만 선택
        common_frames = set(self.ground_truth.keys()) & set(self.predictions.keys())
        self.frames = sorted(common_frames)
        
        # 데이터 정리
        self._prepare_data()
    
    def _prepare_data(self):
        """데이터 정리"""
        self.y_true = []
        self.y_pred = []
        self.confidences = []
        self.correctness = []
        
        for frame in self.frames:
            gt = self.ground_truth[frame]
            pred_info = self.predictions[frame]
            
            pred_id = pred_info['predicted_id']
            confidence = pred_info['confidence']
            
            # 임계값 적용
            if confidence < self.threshold:
                pred_id = 'unknown'
            
            self.y_true.append(gt)
            self.y_pred.append(pred_id)
            self.confidences.append(confidence)
            self.correctness.append(gt == pred_id)
        
        self.confidences = np.array(self.confidences)
        self.correctness = np.array(self.correctness)
    
    def evaluate_all(self, output_dir='outputs/evaluation'):
        """전체 평가 실행"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*70)
        print("FaceWatch 모델 평가")
        print("="*70)
        print(f"프레임 수: {len(self.frames)}")
        print(f"Threshold: {self.threshold}")
        print()
        
        # 1. 기본 분류 지표
        self.evaluate_classification()
        
        # 2. 신뢰도 보정
        self.evaluate_calibration()
        
        # 3. 인물별 ROC/PR
        self.evaluate_per_person()
        
        # 4. 확신도 분포
        self.evaluate_confidence_distribution()
        
        print("="*70)
        print("평가 완료!")
        print(f"결과 저장 위치: {output_dir}")
        print("="*70)
    
    def evaluate_classification(self):
        """분류 성능 평가"""
        print("\n[1] Classification Metrics")
        print("-"*70)
        
        # Classification report
        labels = sorted(set(self.y_true) | set(self.y_pred))
        report = classification_report(self.y_true, self.y_pred, 
                                        target_names=labels, 
                                        zero_division=0)
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(self.y_true, self.y_pred, labels=labels)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title(f'Confusion Matrix (Threshold = {self.threshold})')
        plt.tight_layout()
        plt.savefig('outputs/evaluation/confusion_matrix.png', dpi=300)
        plt.close()
        
        print("✅ Confusion matrix saved")
    
    def evaluate_calibration(self):
        """신뢰도 보정 평가"""
        print("\n[2] Calibration Metrics")
        print("-"*70)
        
        # ECE
        ece = self.compute_ece(self.confidences, self.correctness)
        print(f"Expected Calibration Error (ECE): {ece:.4f}")
        
        if ece < 0.05:
            print("  → 매우 잘 보정됨 ✅")
        elif ece < 0.10:
            print("  → 잘 보정됨 ✅")
        elif ece < 0.20:
            print("  → 보통 ⚠️")
        else:
            print("  → 과신 또는 과소신 경향 ❌")
        
        # Reliability diagram
        self.plot_reliability_diagram()
        print("✅ Reliability diagram saved")
    
    def compute_ece(self, confidences, correctness, n_bins=10):
        """ECE 계산"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0
        
        for i in range(n_bins):
            lower = bin_boundaries[i]
            upper = bin_boundaries[i + 1]
            
            in_bin = (confidences >= lower) & (confidences < upper)
            
            if np.sum(in_bin) > 0:
                avg_confidence = np.mean(confidences[in_bin])
                avg_accuracy = np.mean(correctness[in_bin])
                weight = np.sum(in_bin) / len(confidences)
                
                ece += np.abs(avg_confidence - avg_accuracy) * weight
        
        return ece
    
    def plot_reliability_diagram(self, n_bins=10):
        """Reliability Diagram"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_centers = []
        bin_accs = []
        bin_confs = []
        bin_counts = []
        
        for i in range(n_bins):
            lower = bin_boundaries[i]
            upper = bin_boundaries[i + 1]
            
            in_bin = (self.confidences >= lower) & (self.confidences < upper)
            
            if np.sum(in_bin) > 0:
                bin_centers.append((lower + upper) / 2)
                bin_accs.append(np.mean(self.correctness[in_bin]))
                bin_confs.append(np.mean(self.confidences[in_bin]))
                bin_counts.append(np.sum(in_bin))
        
        plt.figure(figsize=(10, 8))
        
        # Gap 표시
        for i in range(len(bin_centers)):
            plt.plot([bin_confs[i], bin_confs[i]], 
                     [bin_confs[i], bin_accs[i]], 
                     'r-', alpha=0.3, linewidth=2)
        
        # 이상적인 선
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
        
        # 모델 성능
        plt.scatter(bin_confs, bin_accs, s=np.array(bin_counts)*10, 
                    alpha=0.7, label='Model', color='blue')
        
        plt.xlabel('Confidence', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Reliability Diagram', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig('outputs/evaluation/reliability_diagram.png', dpi=300)
        plt.close()
    
    def evaluate_per_person(self):
        """인물별 ROC/PR 평가"""
        print("\n[3] Per-Person Evaluation")
        print("-"*70)
        
        # 등록된 인물 목록
        persons = sorted(set(self.y_true) - {'unknown'})
        
        for person_id in persons:
            # Binary 변환
            y_binary = [1 if label == person_id else 0 for label in self.y_true]
            
            # 해당 인물의 유사도 추출
            y_scores = []
            for frame in self.frames:
                pred_info = self.predictions[frame]
                score = pred_info['all_scores'].get(person_id, 0.0)
                y_scores.append(score)
            
            # ROC-AUC
            from sklearn.metrics import roc_curve, auc, average_precision_score
            
            fpr, tpr, _ = roc_curve(y_binary, y_scores)
            roc_auc = auc(fpr, tpr)
            
            # PR
            ap = average_precision_score(y_binary, y_scores)
            
            print(f"{person_id}:")
            print(f"  ROC-AUC: {roc_auc:.3f}")
            print(f"  Average Precision: {ap:.3f}")
    
    def evaluate_confidence_distribution(self):
        """확신도 분포 분석"""
        print("\n[4] Confidence Distribution")
        print("-"*70)
        
        bins = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
        labels = ['0-30%', '30-40%', '40-50%', '50-60%', 
                  '60-70%', '70-80%', '80-100%']
        
        df = pd.DataFrame({
            'confidence': self.confidences,
            'correct': self.correctness
        })
        
        df['bin'] = pd.cut(df['confidence'], bins=bins, labels=labels)
        
        stats = df.groupby('bin').agg({
            'correct': ['count', 'sum', 'mean']
        })
        
        stats.columns = ['Total', 'Correct', 'Accuracy']
        
        print(stats)
        
        # 시각화
        plt.figure(figsize=(12, 6))
        
        # 서브플롯 1: 샘플 수
        plt.subplot(1, 2, 1)
        stats['Total'].plot(kind='bar', color='skyblue')
        plt.ylabel('Count')
        plt.xlabel('Confidence Range')
        plt.title('Sample Distribution')
        plt.xticks(rotation=45)
        
        # 서브플롯 2: 정확도
        plt.subplot(1, 2, 2)
        stats['Accuracy'].plot(kind='bar', color='coral')
        plt.ylabel('Accuracy')
        plt.xlabel('Confidence Range')
        plt.title('Accuracy by Confidence Range')
        plt.ylim(0, 1.0)
        plt.axhline(y=self.threshold, color='r', linestyle='--', 
                    label=f'Threshold ({self.threshold})')
        plt.xticks(rotation=45)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('outputs/evaluation/confidence_distribution.png', dpi=300)
        plt.close()
        
        print("✅ Confidence distribution saved")

# 사용 예시
if __name__ == "__main__":
    evaluator = FaceWatchEvaluator(
        ground_truth_path="outputs/evaluation/ground_truth.json",
        predictions_path="outputs/evaluation/predictions.json",
        threshold=0.42
    )
    
    evaluator.evaluate_all()
```

---

## 5. 임계값 최적화

### 5.1 여러 임계값 비교

```python
def find_optimal_threshold(ground_truth_path, predictions_path):
    """
    여러 임계값에서 F1-Score 계산하여 최적값 찾기
    """
    thresholds = np.arange(0.30, 0.70, 0.02)
    results = []
    
    for thresh in thresholds:
        evaluator = FaceWatchEvaluator(
            ground_truth_path, predictions_path, threshold=thresh
        )
        
        # F1 계산
        from sklearn.metrics import f1_score
        f1_macro = f1_score(evaluator.y_true, evaluator.y_pred, 
                            average='macro', zero_division=0)
        f1_weighted = f1_score(evaluator.y_true, evaluator.y_pred, 
                               average='weighted', zero_division=0)
        
        # ECE 계산
        ece = evaluator.compute_ece(evaluator.confidences, evaluator.correctness)
        
        results.append({
            'threshold': thresh,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'ece': ece
        })
        
        print(f"Threshold {thresh:.2f}: F1={f1_macro:.3f}, ECE={ece:.4f}")
    
    # 결과 플롯
    df = pd.DataFrame(results)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(df['threshold'], df['f1_macro'], 'o-', label='F1 (Macro)')
    plt.plot(df['threshold'], df['f1_weighted'], 's-', label='F1 (Weighted)')
    plt.xlabel('Threshold')
    plt.ylabel('F1-Score')
    plt.title('F1-Score vs Threshold')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(df['threshold'], df['ece'], 'o-', color='red')
    plt.xlabel('Threshold')
    plt.ylabel('ECE')
    plt.title('Calibration Error vs Threshold')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('outputs/evaluation/threshold_optimization.png', dpi=300)
    plt.show()
    
    # 최적 임계값
    best_f1_idx = df['f1_macro'].idxmax()
    best_ece_idx = df['ece'].idxmin()
    
    print("\n최적 임계값:")
    print(f"  F1 최대: {df.loc[best_f1_idx, 'threshold']:.2f} "
          f"(F1={df.loc[best_f1_idx, 'f1_macro']:.3f})")
    print(f"  ECE 최소: {df.loc[best_ece_idx, 'threshold']:.2f} "
          f"(ECE={df.loc[best_ece_idx, 'ece']:.4f})")
    
    return df
```

---

## 6. 결과 해석

### 6.1 지표 해석 가이드

#### ECE (Expected Calibration Error)

| ECE 값 | 해석 | 조치 |
|--------|------|------|
| **< 0.05** | 매우 잘 보정됨 | ✅ 현재 설정 유지 |
| **0.05~0.10** | 잘 보정됨 | ✅ 양호 |
| **0.10~0.20** | 적당한 보정 | ⚠️ 임계값 조정 고려 |
| **> 0.20** | 과신/과소신 | ❌ 보정 필요 |

#### ROC-AUC

| AUC 값 | 해석 |
|--------|------|
| **0.9~1.0** | 탁월 |
| **0.8~0.9** | 우수 |
| **0.7~0.8** | 양호 |
| **0.6~0.7** | 보통 |
| **< 0.6** | 개선 필요 |

### 6.2 일반적인 문제와 해결책

#### 문제 1: ECE가 높음 (과신)

**증상**: 모델이 "90%"라고 예측했는데 실제로는 60%만 맞음

**원인**:
- 임계값이 너무 낮음
- Bank가 편향됨 (특정 각도만 많음)

**해결책**:
1. 임계값 상향 (0.42 → 0.45)
2. Gap margin 증가 (0.12 → 0.15)
3. Dynamic Bank 재수집 (더 다양한 각도)

#### 문제 2: Recall이 낮음 (미탐)

**증상**: 실제로 yh인데 "unknown"으로 분류

**원인**:
- 임계값이 너무 높음
- Bank에 해당 각도 임베딩 부족

**해결책**:
1. 임계값 하향 (0.42 → 0.38)
2. 특정 각도 임베딩 추가 수집

#### 문제 3: Precision이 낮음 (오탐)

**증상**: 실제로 ja인데 "yh"로 분류

**원인**:
- yh와 ja의 임베딩이 너무 비슷함
- Gap margin이 너무 작음

**해결책**:
1. Gap margin 증가
2. Base Bank 품질 개선 (더 나은 등록 사진)

---

## 7. 실전 예시

### 7.1 전체 워크플로우

```bash
# 1. 프레임 추출
python src/face_match_cctv.py
# → outputs/results/test_video/frames/

# 2. Ground Truth 라벨링
python scripts/label_ground_truth.py
# → outputs/evaluation/ground_truth.json

# 3. 모델 예측 수집
python scripts/collect_predictions.py
# → outputs/evaluation/predictions.json

# 4. 평가 실행
python scripts/evaluate_model.py
# → outputs/evaluation/*.png

# 5. 임계값 최적화
python scripts/find_optimal_threshold.py
```

### 7.2 예시 출력

```
======================================================================
FaceWatch 모델 평가
======================================================================
프레임 수: 150
Threshold: 0.42

[1] Classification Metrics
----------------------------------------------------------------------
              precision    recall  f1-score   support

          ja       0.85      0.88      0.86        32
          js       0.78      0.75      0.76        28
          jw       0.82      0.80      0.81        25
          yh       0.91      0.93      0.92        45
     unknown       0.88      0.85      0.86        20

    accuracy                           0.86       150
   macro avg       0.85      0.84      0.84       150
weighted avg       0.86      0.86      0.86       150

✅ Confusion matrix saved

[2] Calibration Metrics
----------------------------------------------------------------------
Expected Calibration Error (ECE): 0.0723
  → 잘 보정됨 ✅
✅ Reliability diagram saved

[3] Per-Person Evaluation
----------------------------------------------------------------------
ja:
  ROC-AUC: 0.921
  Average Precision: 0.875
js:
  ROC-AUC: 0.902
  Average Precision: 0.843
jw:
  ROC-AUC: 0.915
  Average Precision: 0.868
yh:
  ROC-AUC: 0.948
  Average Precision: 0.923

[4] Confidence Distribution
----------------------------------------------------------------------
                Total  Correct  Accuracy
bin                                     
0-30%              15       10  0.666667
30-40%             20       16  0.800000
40-50%             18       15  0.833333
50-60%             22       20  0.909091
60-70%             28       26  0.928571
70-80%             25       24  0.960000
80-100%            22       22  1.000000
✅ Confidence distribution saved

======================================================================
평가 완료!
결과 저장 위치: outputs/evaluation
======================================================================
```

---

## 8. 체크리스트

### 평가 전 준비

- [ ] Ground Truth 라벨링 완료
- [ ] 모델 예측 수집 완료
- [ ] 평가 스크립트 설치 (`pip install scikit-learn matplotlib seaborn pandas`)

### 필수 평가 항목

- [ ] Confusion Matrix 확인
- [ ] Precision/Recall/F1 계산
- [ ] ECE 계산 (신뢰도 보정)
- [ ] Reliability Diagram 시각화
- [ ] 인물별 ROC-AUC 계산
- [ ] 확신도 분포 분석

### 임계값 최적화

- [ ] 여러 임계값에서 F1 비교
- [ ] ECE vs Threshold 플롯
- [ ] 최적 임계값 선정
- [ ] 실서비스에 적용

---

## 부록

### A. 평가 지표 요약

| 지표 | 목적 | 임계값 영향 | 추천도 |
|------|------|------------|--------|
| **ECE** | 신뢰도 보정 | 있음 | ⭐⭐⭐⭐⭐ |
| **Precision/Recall/F1** | 전체 성능 | 있음 | ⭐⭐⭐⭐ |
| **ROC-AUC** | 임계값 무관 성능 | 없음 | ⭐⭐⭐⭐ |
| **Confusion Matrix** | 오류 패턴 분석 | 있음 | ⭐⭐⭐⭐ |
| **Reliability Diagram** | 보정 시각화 | 있음 | ⭐⭐⭐ |
| **크로스엔트로피** | (부적합) | - | ❌ |

### B. 참고 문헌

- Guo et al., "On Calibration of Modern Neural Networks", ICML 2017
- Naeini et al., "Obtaining Well Calibrated Probabilities Using Bayesian Binning", AAAI 2015

---

**문서 버전**: 1.0  
**최종 수정**: 2025-11-27  
**작성자**: FaceWatch Development Team
