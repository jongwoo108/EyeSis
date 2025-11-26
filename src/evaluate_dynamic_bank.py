# src/evaluate_dynamic_bank.py
"""
동적 Bank 정확도 평가 스크립트

목적: CCTV 영상에서 수집된 동적 임베딩의 정확도를 평가
- 정답 데이터: outputs/embeddings_manual 아래의 수동 추출 임베딩
- 평가 대상: outputs/embeddings 아래의 각도별 분리 파일 (bank_left.npy, bank_right.npy 등)

평가 방법:
1. 각 인물별로 정답 데이터의 각도별 파일을 찾음 (bank_left.npy, bank_right.npy, bank_top.npy 등)
2. CCTV 데이터에서 동일한 각도 파일을 찾음
3. 같은 각도끼리만 비교하여 정확도 계산
4. 각도별 정확도 분석
"""
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# CUDA 경로를 먼저 설정
from src.utils.device_config import _ensure_cuda_in_path
_ensure_cuda_in_path()

import numpy as np
import json
from collections import defaultdict


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """벡터를 L2 정규화"""
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def load_angle_separated_banks(person_dir: Path):
    """
    사람별 폴더에서 각도별로 분리된 bank 파일들을 로드
    
    Returns:
        각도별 bank 딕셔너리: {angle_type: bank_array}
        - angle_type: "left", "right", "top", "front" 등
        - bank_array: (N, 512) numpy 배열 (L2 정규화됨)
    """
    angle_banks = {}
    
    # 각도별 파일 패턴
    angle_types = ["left", "right", "top", "front"]
    
    for angle_type in angle_types:
        bank_path = person_dir / f"bank_{angle_type}.npy"
        if bank_path.exists():
            bank = np.load(bank_path)
            if bank.ndim == 1:
                bank = bank.reshape(1, -1)
            # L2 정규화
            bank = bank / (np.linalg.norm(bank, axis=1, keepdims=True) + 1e-6)
            angle_banks[angle_type] = bank
    
    return angle_banks


def evaluate_person(person_id: str, manual_dir: Path, dynamic_dir: Path):
    """
    한 인물에 대한 동적 bank 정확도 평가 (각도별 정확히 일치하는 것만 비교)
    
    Returns:
        평가 결과 딕셔너리
    """
    manual_person_dir = manual_dir / person_id
    dynamic_person_dir = dynamic_dir / person_id
    
    # 정답 데이터의 각도별 bank 로드
    manual_angle_banks = load_angle_separated_banks(manual_person_dir)
    
    if not manual_angle_banks:
        return None  # 정답 데이터 없음
    
    # CCTV 데이터의 각도별 bank 로드
    dynamic_angle_banks = load_angle_separated_banks(dynamic_person_dir)
    
    if not dynamic_angle_banks:
        return None  # CCTV 데이터 없음
    
    # 각도별로 정확히 일치하는 것만 비교
    similarities = []
    angle_stats = {}
    all_similarities = []
    
    # 정답 데이터에 있는 각도만 비교
    for angle_type in manual_angle_banks.keys():
        if angle_type not in dynamic_angle_banks:
            continue  # CCTV에 해당 각도가 없으면 스킵
        
        manual_bank = manual_angle_banks[angle_type]
        dynamic_bank = dynamic_angle_banks[angle_type]
        
        # 각 정답 임베딩과 CCTV 임베딩 간 유사도 계산
        angle_similarities = []
        for i, manual_emb in enumerate(manual_bank):
            # 해당 각도의 CCTV 임베딩과만 비교
            sims = np.dot(dynamic_bank, manual_emb)  # (N_dynamic,)
            max_sim = float(np.max(sims))
            best_idx = int(np.argmax(sims))
            
            angle_similarities.append(max_sim)
            all_similarities.append(max_sim)
            
            similarities.append({
                "angle_type": angle_type,
                "manual_idx": i,
                "max_similarity": max_sim,
                "best_dynamic_idx": best_idx,
                "manual_count": manual_bank.shape[0],
                "dynamic_count": dynamic_bank.shape[0]
            })
        
        # 각도별 통계 계산
        if angle_similarities:
            angle_stats[angle_type] = {
                "count": len(angle_similarities),
                "avg_similarity": float(np.mean(angle_similarities)),
                "min_similarity": float(np.min(angle_similarities)),
                "max_similarity": float(np.max(angle_similarities))
            }
    
    # 전체 통계 계산
    if not all_similarities:
        return None
    
    total_manual_count = sum(bank.shape[0] for bank in manual_angle_banks.values())
    total_dynamic_count = sum(bank.shape[0] for bank in dynamic_angle_banks.values())
    
    return {
        "person_id": person_id,
        "manual_count": total_manual_count,
        "dynamic_count": total_dynamic_count,
        "avg_similarity": float(np.mean(all_similarities)),
        "min_similarity": float(np.min(all_similarities)),
        "max_similarity": float(np.max(all_similarities)),
        "angle_stats": angle_stats,
        "detailed_matches": similarities
    }


def main():
    """메인 평가 함수"""
    manual_dir = Path("outputs") / "embeddings_manual"
    dynamic_dir = Path("outputs") / "embeddings"
    
    print(f"{'='*70}")
    print(f"📊 동적 Bank 정확도 평가 (각도별 정확히 일치하는 것만 비교)")
    print(f"{'='*70}")
    print(f"   정답 데이터: {manual_dir}")
    print(f"   평가 대상: {dynamic_dir}")
    print(f"   비교 방식: 정답 데이터의 각도와 CCTV 데이터의 각도가 정확히 일치하는 경우만 비교")
    print(f"   예: bank_left.npy (정답) vs bank_left.npy (CCTV)")
    print()
    
    if not manual_dir.exists():
        print(f"❌ 정답 데이터 폴더를 찾을 수 없음: {manual_dir}")
        return
    
    if not dynamic_dir.exists():
        print(f"❌ 평가 대상 폴더를 찾을 수 없음: {dynamic_dir}")
        return
    
    # 평가할 인물 목록 (manual에 있는 인물들)
    manual_person_dirs = [d for d in manual_dir.iterdir() if d.is_dir()]
    
    if not manual_person_dirs:
        print(f"⚠️ {manual_dir} 안에 인물 폴더가 없습니다.")
        return
    
    print(f"👥 평가 대상 인물: {len(manual_person_dirs)}명")
    for d in manual_person_dirs:
        print(f"  - {d.name}")
    print()
    
    # 각 인물별 평가
    results = []
    overall_stats = {
        "total_persons": 0,
        "evaluated_persons": 0,
        "total_manual_embeddings": 0,
        "total_dynamic_embeddings": 0,
        "all_similarities": [],
        "angle_stats": defaultdict(lambda: {"count": 0, "sims": []})
    }
    
    for person_dir in manual_person_dirs:
        person_id = person_dir.name
        overall_stats["total_persons"] += 1
        
        result = evaluate_person(person_id, manual_dir, dynamic_dir)
        
        if result is None:
            print(f"⚠️ {person_id}: 평가 불가 (데이터 없음)")
            continue
        
        results.append(result)
        overall_stats["evaluated_persons"] += 1
        overall_stats["total_manual_embeddings"] += result["manual_count"]
        overall_stats["total_dynamic_embeddings"] += result["dynamic_count"]
        overall_stats["all_similarities"].extend([s["max_similarity"] for s in result["detailed_matches"]])
        
        # 각도별 통계 누적
        for angle_type, stats in result["angle_stats"].items():
            overall_stats["angle_stats"][angle_type]["count"] += stats["count"]
            overall_stats["angle_stats"][angle_type]["sims"].extend(
                [s["max_similarity"] for s in result["detailed_matches"] 
                 if s["angle_type"] == angle_type]
            )
    
    # 결과 출력
    print(f"\n{'='*70}")
    print(f"📈 평가 결과")
    print(f"{'='*70}")
    print(f"   평가된 인물 수: {overall_stats['evaluated_persons']}/{overall_stats['total_persons']}")
    print(f"   정답 임베딩 수: {overall_stats['total_manual_embeddings']}개")
    print(f"   동적 임베딩 수: {overall_stats['total_dynamic_embeddings']}개")
    print()
    
    if overall_stats["all_similarities"]:
        all_sims = np.array(overall_stats["all_similarities"])
        print(f"📊 전체 유사도 통계:")
        print(f"   평균: {np.mean(all_sims):.4f}")
        print(f"   최소: {np.min(all_sims):.4f}")
        print(f"   최대: {np.max(all_sims):.4f}")
        print(f"   표준편차: {np.std(all_sims):.4f}")
        print()
        
        # 임계값별 정확도
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        print(f"📊 임계값별 정확도:")
        for thresh in thresholds:
            accuracy = np.mean(all_sims >= thresh) * 100
            print(f"   {thresh:.1f} 이상: {accuracy:.1f}% ({np.sum(all_sims >= thresh)}/{len(all_sims)})")
        print()
    
    # 인물별 상세 결과
    print(f"👤 인물별 상세 결과:")
    for result in sorted(results, key=lambda x: x["avg_similarity"], reverse=True):
        print(f"\n   {result['person_id']}:")
        print(f"     정답: {result['manual_count']}개, 동적: {result['dynamic_count']}개")
        print(f"     평균 유사도: {result['avg_similarity']:.4f} "
              f"(최소: {result['min_similarity']:.4f}, 최대: {result['max_similarity']:.4f})")
        
        if result["angle_stats"]:
            print(f"     각도별 통계:")
            for angle_type, stats in sorted(result["angle_stats"].items()):
                print(f"       {angle_type:15s}: {stats['count']:3d}개, "
                      f"평균: {stats['avg_similarity']:.4f}, "
                      f"범위: [{stats['min_similarity']:.4f}, {stats['max_similarity']:.4f}]")
    
    # 각도별 전체 통계
    if overall_stats["angle_stats"]:
        print(f"\n📊 각도별 전체 통계:")
        for angle_type in sorted(overall_stats["angle_stats"].keys()):
            stats = overall_stats["angle_stats"][angle_type]
            if stats["sims"]:
                sims = np.array(stats["sims"])
                print(f"   {angle_type:15s}: {stats['count']:4d}개, "
                      f"평균: {np.mean(sims):.4f}, "
                      f"범위: [{np.min(sims):.4f}, {np.max(sims):.4f}]")
    
    print(f"\n{'='*70}")
    print(f"✅ 평가 완료")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()


