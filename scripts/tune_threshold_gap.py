#!/usr/bin/env python3
"""
Threshold/Gap 튜닝을 위한 테스트 스크립트

새로운 base/dynamic 구조에서 threshold와 gap 값을 튜닝하기 위한 도구입니다.

사용법:
    python scripts/tune_threshold_gap.py --video-path VIDEO_PATH [--suspect-ids ID1,ID2] [--config CONFIG_JSON]

기능:
1. 다양한 threshold/gap 조합으로 테스트
2. 매칭 결과 통계 수집
3. 최적의 threshold/gap 값 추천
"""
import sys
from pathlib import Path
import json
import argparse
from typing import Dict, List, Tuple
import numpy as np

# 프로젝트 루트를 Python 경로에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 현재 threshold/gap 설정값 (backend/main.py에서 가져옴)
DEFAULT_CONFIG = {
    "high": {
        "main_threshold": 0.42,
        "gap_margin": 0.12
    },
    "medium": {
        "main_threshold": 0.40,
        "gap_margin": 0.10
    },
    "low": {
        "main_threshold": 0.38,
        "gap_margin": 0.08
    },
    "suspect_ids_bonus": {
        "threshold_add": 0.02,
        "gap_add": 0.03,
        "min_absolute": 0.45
    }
}


def generate_test_configs(base_config: Dict) -> List[Dict]:
    """
    다양한 threshold/gap 조합 생성
    
    Args:
        base_config: 기본 설정
    
    Returns:
        테스트할 설정 리스트
    """
    configs = []
    
    # 기본값
    configs.append(base_config)
    
    # Threshold 조정 (-0.02, -0.01, +0.01, +0.02)
    for th_delta in [-0.02, -0.01, 0.01, 0.02]:
        config = base_config.copy()
        for quality in ["high", "medium", "low"]:
            config[quality]["main_threshold"] += th_delta
        config["name"] = f"threshold_{th_delta:+.2f}"
        configs.append(config)
    
    # Gap 조정 (-0.02, -0.01, +0.01, +0.02)
    for gap_delta in [-0.02, -0.01, 0.01, 0.02]:
        config = base_config.copy()
        for quality in ["high", "medium", "low"]:
            config[quality]["gap_margin"] += gap_delta
        config["name"] = f"gap_{gap_delta:+.2f}"
        configs.append(config)
    
    return configs


def analyze_results(results: List[Dict], expected_matches: List[str]) -> Dict:
    """
    테스트 결과 분석
    
    Args:
        results: 매칭 결과 리스트
        expected_matches: 예상 매칭 person_id 리스트
    
    Returns:
        분석 결과 딕셔너리
    """
    total_frames = len(results)
    matched_frames = sum(1 for r in results if r.get("matched", False))
    correct_matches = sum(1 for r in results 
                         if r.get("matched", False) and r.get("person_id") in expected_matches)
    false_positives = sum(1 for r in results 
                         if r.get("matched", False) and r.get("person_id") not in expected_matches)
    
    return {
        "total_frames": total_frames,
        "matched_frames": matched_frames,
        "match_rate": matched_frames / total_frames if total_frames > 0 else 0.0,
        "correct_matches": correct_matches,
        "false_positives": false_positives,
        "precision": correct_matches / matched_frames if matched_frames > 0 else 0.0,
        "recall": correct_matches / len(expected_matches) if expected_matches else 0.0
    }


def print_tuning_guide():
    """튜닝 가이드 출력"""
    guide = """
═══════════════════════════════════════════════════════════════
📊 Threshold/Gap 튜닝 가이드
═══════════════════════════════════════════════════════════════

1. 현재 설정값 (backend/main.py)

   화질별 Threshold:
   - High:   0.42
   - Medium: 0.40
   - Low:    0.38
   
   화질별 Gap Margin:
   - High:   0.12
   - Medium: 0.10
   - Low:    0.08
   
   Suspect IDs 모드 추가 조건:
   - Threshold +0.02
   - Gap +0.03
   - 절대값 최소 0.45

2. 튜닝 전략

   A. False Positive가 많을 때:
      → Threshold를 높이기 (+0.01 ~ +0.02)
      → Gap을 높이기 (+0.01 ~ +0.02)
   
   B. True Positive가 적을 때:
      → Threshold를 낮추기 (-0.01 ~ -0.02)
      → Gap을 낮추기 (-0.01 ~ -0.02)
   
   C. 특정 화질에서 문제가 있을 때:
      → 해당 화질의 threshold/gap만 조정

3. 테스트 방법

   Step 1: 기존 bank.npy를 bank_base.npy로 변환
      python scripts/rebuild_base_bank.py --backup
   
   Step 2: 테스트 영상으로 여러 설정값 테스트
      python scripts/tune_threshold_gap.py --video-path VIDEO_PATH
   
   Step 3: 결과 분석 후 최적값 선택
      - Precision이 높고 Recall도 충분한 설정 선택
      - False Positive가 최소화된 설정 선택

4. 설정 변경 방법

   backend/main.py의 process_detection 함수에서:
   
   # 화질 기반 절대 임계값 설정
   if face_quality == "high":
       main_threshold = 0.42  # ← 여기 수정
       gap_margin = 0.12      # ← 여기 수정
   elif face_quality == "medium":
       main_threshold = 0.40  # ← 여기 수정
       gap_margin = 0.10      # ← 여기 수정
   else:  # low
       main_threshold = 0.38  # ← 여기 수정
       gap_margin = 0.08      # ← 여기 수정

═══════════════════════════════════════════════════════════════
"""
    print(guide)


def main():
    parser = argparse.ArgumentParser(description="Threshold/Gap 튜닝 도구")
    parser.add_argument("--guide", action="store_true", help="튜닝 가이드 출력")
    parser.add_argument("--config", type=str, help="설정 JSON 파일 경로")
    parser.add_argument("--list-configs", action="store_true", help="테스트할 설정 목록 출력")
    
    args = parser.parse_args()
    
    if args.guide:
        print_tuning_guide()
        return
    
    if args.list_configs:
        configs = generate_test_configs(DEFAULT_CONFIG)
        print("테스트할 설정 조합:")
        for i, config in enumerate(configs):
            print(f"\n{i+1}. {config.get('name', 'default')}")
            for quality in ["high", "medium", "low"]:
                print(f"   {quality}: threshold={config[quality]['main_threshold']:.3f}, "
                      f"gap={config[quality]['gap_margin']:.3f}")
        return
    
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        config = DEFAULT_CONFIG
    
    print("=" * 70)
    print("📊 Threshold/Gap 튜닝 도구")
    print("=" * 70)
    print("\n현재 설정:")
    for quality in ["high", "medium", "low"]:
        print(f"  {quality}: threshold={config[quality]['main_threshold']:.3f}, "
              f"gap={config[quality]['gap_margin']:.3f}")
    print("\n사용법:")
    print("  python scripts/tune_threshold_gap.py --guide  # 튜닝 가이드")
    print("  python scripts/tune_threshold_gap.py --list-configs  # 테스트 설정 목록")


if __name__ == "__main__":
    main()
















