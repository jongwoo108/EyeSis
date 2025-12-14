"""
jw 인물에 대한 동적 Bank 정확도 평가 스크립트

목적: CCTV 영상에서 수집된 동적 임베딩의 정확도를 평가 (jw만)
- 정답 데이터: outputs/embeddings_manual/jw 아래의 수동 추출 임베딩
- 평가 대상: outputs/embeddings/jw 아래의 각도별 분리 파일 (bank_left.npy, bank_right.npy 등)

평가 방법:
1. 정답 데이터의 각도별 파일을 찾음 (bank_left.npy, bank_right.npy, bank_top.npy, bank_front.npy)
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


def load_angle_info(person_dir: Path, is_manual: bool = False):
    """
    각도 정보 파일 로드
    
    Args:
        person_dir: 사람별 폴더 경로
        is_manual: True면 정답 데이터 (angles_manual.json), False면 CCTV 데이터 (angles_dynamic.json)
    
    Returns:
        각도 정보 딕셔너리 또는 None
    """
    if is_manual:
        angles_file = person_dir / "angles_manual.json"
    else:
        angles_file = person_dir / "angles_dynamic.json"
    
    if not angles_file.exists():
        return None
    
    with open(angles_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def evaluate_person(person_id: str, manual_dir: Path, dynamic_dir: Path, angle_tolerance: float = 5.0):
    """
    한 인물에 대한 동적 bank 정확도 평가 (각도 기반 매칭)
    
    Args:
        person_id: 인물 ID
        manual_dir: 정답 데이터 디렉토리
        dynamic_dir: CCTV 데이터 디렉토리
        angle_tolerance: 각도 허용 범위 (도 단위, 기본 5°)
    
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
    
    # 각도 정보 로드
    manual_angles_info = load_angle_info(manual_person_dir, is_manual=True)
    dynamic_angles_info = load_angle_info(dynamic_person_dir, is_manual=False)
    
    # CCTV 각도 정보를 인덱스별로 매핑 (각도별 bank의 인덱스와 매칭)
    # dynamic_angles_info의 각도 정보를 각도별로 그룹화
    # 각도별 bank 파일의 인덱스는 해당 각도 내에서의 인덱스이므로,
    # 원본 bank_dynamic.npy의 인덱스와 매핑 필요
    dynamic_angle_indices = {}  # {angle_type: [(angle_bank_idx, original_idx, yaw), ...]}
    
    if dynamic_angles_info:
        angle_types_list = dynamic_angles_info.get("angle_types", [])
        yaw_angles_list = dynamic_angles_info.get("yaw_angles", [])
        
        # 각도별로 원본 인덱스 그룹화
        angle_original_indices = {}  # {angle_type: [original_idx1, original_idx2, ...]}
        for angle_type in ["left", "right", "top", "front"]:
            angle_original_indices[angle_type] = []
            for i, (detected_angle, yaw) in enumerate(zip(angle_types_list, yaw_angles_list)):
                if detected_angle == angle_type:
                    angle_original_indices[angle_type].append((i, yaw))
        
        # 각도별 bank 파일의 인덱스와 원본 인덱스 매핑
        for angle_type in ["left", "right", "top", "front"]:
            dynamic_angle_indices[angle_type] = []
            if angle_type in dynamic_angle_banks and angle_type in angle_original_indices:
                # 각도별 bank의 인덱스는 원본에서 해당 각도만 추출한 순서
                for angle_bank_idx, (original_idx, yaw) in enumerate(angle_original_indices[angle_type]):
                    if angle_bank_idx < dynamic_angle_banks[angle_type].shape[0]:
                        dynamic_angle_indices[angle_type].append((angle_bank_idx, original_idx, yaw))
    
    # 각도 기반 매칭
    similarities = []
    angle_stats = {}
    all_similarities = []
    
    # 정답 데이터에 있는 각도만 비교
    for angle_type in manual_angle_banks.keys():
        if angle_type not in dynamic_angle_banks:
            continue  # CCTV에 해당 각도가 없으면 스킵
        
        manual_bank = manual_angle_banks[angle_type]
        dynamic_bank = dynamic_angle_banks[angle_type]
        
        # 정답 데이터의 각도 정보 가져오기
        manual_yaws = []
        if manual_angles_info:
            file_mapping = manual_angles_info.get("file_mapping", [])
            for mapping in file_mapping:
                if mapping["angle_type"] == angle_type:
                    manual_yaws.append(mapping["yaw"])
        
        # 각도 정보가 부족하면 bank 파일 개수만큼 None으로 채움
        while len(manual_yaws) < manual_bank.shape[0]:
            manual_yaws.append(None)
        
        # 정답 임베딩이 여러 개인 경우 각각 처리
        angle_similarities = []
        for i, manual_emb in enumerate(manual_bank):
            # 정답 이미지의 yaw 각도 (있으면 사용, 없으면 None)
            manual_yaw = manual_yaws[i] if i < len(manual_yaws) else None
            
            if manual_yaw is not None and angle_type in dynamic_angle_indices:
                # 각도 기반 매칭: 가장 가까운 각도의 CCTV 임베딩 찾기
                best_sim = -1.0
                best_idx = -1
                best_yaw_diff = 999.0
                best_cctv_yaw = None
                
                for angle_bank_idx, original_idx, cctv_yaw in dynamic_angle_indices[angle_type]:
                    if angle_bank_idx >= dynamic_bank.shape[0]:
                        continue
                    
                    # 각도 차이 계산 (180도를 넘어가는 경우 처리)
                    yaw_diff = abs(manual_yaw - cctv_yaw)
                    # 180도를 넘어가는 경우 반대 방향으로 계산
                    if yaw_diff > 180:
                        yaw_diff = 360 - yaw_diff
                    
                    # 각도 차이가 허용 범위 내인 경우만 비교
                    if yaw_diff <= angle_tolerance:
                        sim = float(np.dot(dynamic_bank[angle_bank_idx], manual_emb))
                        if sim > best_sim:
                            best_sim = sim
                            best_idx = angle_bank_idx
                            best_yaw_diff = yaw_diff
                            best_cctv_yaw = cctv_yaw
                
                if best_idx >= 0:
                    angle_similarities.append(best_sim)
                    if best_sim is not None:
                        all_similarities.append(best_sim)
                    
                    similarities.append({
                        "angle_type": angle_type,
                        "manual_idx": i,
                        "max_similarity": best_sim,
                        "best_dynamic_idx": best_idx,
                        "manual_yaw": manual_yaw,
                        "cctv_yaw": best_cctv_yaw,
                        "yaw_diff": best_yaw_diff,
                        "angle_based": True,
                        "manual_count": manual_bank.shape[0],
                        "dynamic_count": dynamic_bank.shape[0]
                    })
                else:
                    # 각도 범위 내에 매칭되는 것이 없음
                    similarities.append({
                        "angle_type": angle_type,
                        "manual_idx": i,
                        "max_similarity": None,
                        "best_dynamic_idx": None,
                        "manual_yaw": manual_yaw,
                        "cctv_yaw": None,
                        "yaw_diff": None,
                        "angle_based": True,
                        "skipped": True,
                        "reason": f"각도 범위 내 매칭 없음 (±{angle_tolerance}°)"
                    })
            else:
                # 각도 정보가 없으면 기존 방식 사용 (최대 유사도)
                sims = np.dot(dynamic_bank, manual_emb)  # (N_dynamic,)
                max_sim = float(np.max(sims))
                best_idx = int(np.argmax(sims))
                
                angle_similarities.append(max_sim)
                all_similarities.append(max_sim)
                
                # 각도 정보가 있으면 추가
                cctv_yaw_info = None
                if angle_type in dynamic_angle_indices and best_idx < len(dynamic_angle_indices[angle_type]):
                    cctv_yaw_info = dynamic_angle_indices[angle_type][best_idx][2]  # yaw 값
                
                similarities.append({
                    "angle_type": angle_type,
                    "manual_idx": i,
                    "max_similarity": max_sim,
                    "best_dynamic_idx": best_idx,
                    "manual_yaw": manual_yaw,
                    "cctv_yaw": cctv_yaw_info,
                    "yaw_diff": None,
                    "angle_based": False,
                    "manual_count": manual_bank.shape[0],
                    "dynamic_count": dynamic_bank.shape[0]
                })
        
        # 각도별 통계 계산 (유효한 매칭만)
        valid_similarities = [s for s in angle_similarities if s is not None]
        if valid_similarities:
            angle_stats[angle_type] = {
                "count": len(valid_similarities),
                "avg_similarity": float(np.mean(valid_similarities)),
                "min_similarity": float(np.min(valid_similarities)),
                "max_similarity": float(np.max(valid_similarities))
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
        "detailed_matches": similarities,
        "angle_tolerance": angle_tolerance
    }


def main():
    """메인 평가 함수 (jw만 평가)"""
    person_id = "jw"  # 평가 대상 인물
    manual_dir = Path("outputs") / "embeddings_manual"
    dynamic_dir = Path("outputs") / "embeddings"
    
    print(f"{'='*70}")
    print(f"📊 동적 Bank 정확도 평가 - {person_id} (각도별 정확히 일치하는 것만 비교)")
    print(f"{'='*70}")
    print(f"   평가 대상: {person_id}")
    print(f"   정답 데이터: {manual_dir / person_id}")
    print(f"   CCTV 데이터: {dynamic_dir / person_id}")
    print(f"   비교 방식: 정답 데이터의 각도와 CCTV 데이터의 각도가 정확히 일치하는 경우만 비교")
    print(f"   예: bank_left.npy (정답) vs bank_left.npy (CCTV)")
    print()
    
    if not manual_dir.exists():
        print(f"❌ 정답 데이터 폴더를 찾을 수 없음: {manual_dir}")
        return
    
    if not dynamic_dir.exists():
        print(f"❌ 평가 대상 폴더를 찾을 수 없음: {dynamic_dir}")
        return
    
    manual_person_dir = manual_dir / person_id
    dynamic_person_dir = dynamic_dir / person_id
    
    if not manual_person_dir.exists():
        print(f"❌ 정답 데이터 폴더를 찾을 수 없음: {manual_person_dir}")
        return
    
    if not dynamic_person_dir.exists():
        print(f"❌ CCTV 데이터 폴더를 찾을 수 없음: {dynamic_person_dir}")
        return
    
    # 각도별 파일 확인
    print(f"📁 정답 데이터 파일 확인:")
    manual_angles = []
    for angle_type in ["left", "right", "top", "front"]:
        bank_path = manual_person_dir / f"bank_{angle_type}.npy"
        if bank_path.exists():
            bank = np.load(bank_path)
            count = bank.shape[0] if bank.ndim == 2 else 1
            print(f"   ✅ bank_{angle_type}.npy: {count}개 임베딩")
            manual_angles.append(angle_type)
        else:
            print(f"   ❌ bank_{angle_type}.npy: 없음")
    
    print(f"\n📁 CCTV 데이터 파일 확인:")
    dynamic_angles = []
    for angle_type in ["left", "right", "top", "front"]:
        bank_path = dynamic_person_dir / f"bank_{angle_type}.npy"
        if bank_path.exists():
            bank = np.load(bank_path)
            count = bank.shape[0] if bank.ndim == 2 else 1
            print(f"   ✅ bank_{angle_type}.npy: {count}개 임베딩")
            dynamic_angles.append(angle_type)
        else:
            print(f"   ❌ bank_{angle_type}.npy: 없음")
    
    # 비교 가능한 각도 확인
    common_angles = set(manual_angles) & set(dynamic_angles)
    print(f"\n📊 비교 가능한 각도: {sorted(common_angles) if common_angles else '없음'}")
    
    if not common_angles:
        print(f"❌ 비교 가능한 각도가 없습니다.")
        return
    
    print()
    
    # 각도 정보 확인
    manual_angles_file = manual_person_dir / "angles_manual.json"
    if manual_angles_file.exists():
        print(f"\n정답 데이터 각도 정보:")
        with open(manual_angles_file, 'r', encoding='utf-8') as f:
            manual_angles = json.load(f)
            for mapping in manual_angles.get("file_mapping", []):
                print(f"  {mapping['file']}: {mapping['angle_type']} (yaw: {mapping['yaw']:.1f}°, pitch: {mapping['pitch']:.1f}°)")
    else:
        print(f"\n⚠️ 정답 데이터 각도 정보 없음: {manual_angles_file}")
        print(f"   각도 기반 매칭을 위해 extract_angle_embeddings.py를 먼저 실행하세요.")
    
    # 평가 실행 (각도 허용 범위 ±5°)
    angle_tolerance = 5.0
    print(f"\n각도 기반 매칭 시작 (허용 범위: ±{angle_tolerance}°)")
    result = evaluate_person(person_id, manual_dir, dynamic_dir, angle_tolerance=angle_tolerance)
    
    if result is None:
        print(f"❌ 평가 불가 (데이터 없음)")
        return
    
    # 결과 출력
    print(f"\n{'='*70}")
    print(f"📈 평가 결과 - {person_id}")
    print(f"{'='*70}")
    print(f"   정답 임베딩 수: {result['manual_count']}개")
    print(f"   동적 임베딩 수: {result['dynamic_count']}개")
    print(f"   비교된 임베딩 수: {len(result['detailed_matches'])}개")
    print()
    
    # 전체 유사도 통계 (유효한 매칭만)
    valid_sims = [s["max_similarity"] for s in result["detailed_matches"] 
                  if s.get("max_similarity") is not None and not s.get("skipped", False)]
    
    if not valid_sims:
        print(f"❌ 유효한 매칭이 없습니다.")
        return
    
    all_sims = np.array(valid_sims)
    skipped_count = len([s for s in result["detailed_matches"] if s.get("skipped", False)])
    
    print(f"📊 전체 유사도 통계:")
    print(f"   유효한 매칭: {len(all_sims)}개")
    if skipped_count > 0:
        print(f"   스킵된 매칭: {skipped_count}개")
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
        count = np.sum(all_sims >= thresh)
        print(f"   {thresh:.1f} 이상: {accuracy:.1f}% ({count}/{len(all_sims)})")
    print()
    
    # 각도별 상세 통계
    print(f"📊 각도별 상세 통계:")
    for angle_type in sorted(result["angle_stats"].keys()):
        stats = result["angle_stats"][angle_type]
        print(f"\n   {angle_type.upper()}:")
        print(f"     비교 개수: {stats['count']}개")
        print(f"     평균 유사도: {stats['avg_similarity']:.4f}")
        print(f"     최소 유사도: {stats['min_similarity']:.4f}")
        print(f"     최대 유사도: {stats['max_similarity']:.4f}")
        
        # 해당 각도의 상세 매칭 정보
        angle_matches = [s for s in result["detailed_matches"] if s["angle_type"] == angle_type]
        print(f"     상세 매칭:")
        for match in angle_matches:
            if match.get("skipped"):
                print(f"       정답 #{match['manual_idx']}: 스킵 ({match.get('reason', '')})")
            else:
                angle_info = ""
                if match.get("angle_based") and match.get("manual_yaw") is not None:
                    angle_info = f" (각도: {match['manual_yaw']:.1f}° vs {match.get('cctv_yaw', 0):.1f}°, 차이: {match.get('yaw_diff', 0):.1f}°)"
                print(f"       정답 #{match['manual_idx']} → CCTV #{match['best_dynamic_idx']}: "
                      f"{match['max_similarity']:.4f}{angle_info}")
    
    # 결과 요약
    print(f"\n{'='*70}")
    print(f"✅ 평가 완료 - {person_id}")
    print(f"{'='*70}")
    print(f"   평가된 각도: {sorted(result['angle_stats'].keys())}")
    print(f"   전체 평균 유사도: {result['avg_similarity']:.4f}")
    print(f"   평가 품질: ", end="")
    if result['avg_similarity'] >= 0.8:
        print("✅ 매우 좋음")
    elif result['avg_similarity'] >= 0.7:
        print("✅ 양호")
    elif result['avg_similarity'] >= 0.6:
        print("⚠️ 보통")
    else:
        print("❌ 개선 필요")
    print()


if __name__ == "__main__":
    main()

