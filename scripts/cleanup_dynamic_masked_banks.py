"""
Dynamic Bank와 Masked Bank만 삭제하는 스크립트
Base Bank는 유지하여 새로운 검증 로직 테스트를 위한 깨끗한 환경을 만듭니다.
"""
import argparse
import shutil
from pathlib import Path
import sys

# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EMBEDDINGS_DIR = PROJECT_ROOT / "outputs" / "embeddings"


def cleanup_person_banks(person_id: str, backup: bool = True) -> bool:
    """
    특정 인물의 Dynamic Bank와 Masked Bank만 삭제 (Base Bank는 유지)
    
    Args:
        person_id: 인물 ID
        backup: 기존 파일 백업 여부
    
    Returns:
        성공 여부
    """
    person_dir = EMBEDDINGS_DIR / person_id
    
    if not person_dir.exists():
        print(f"  ⚠️ 인물 폴더 없음: {person_dir}")
        return False
    
    # 백업 폴더 생성
    if backup:
        backup_dir = person_dir / "backup_before_cleanup"
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        backup_dir.mkdir(exist_ok=True)
        print(f"  💾 백업 폴더 생성: {backup_dir}")
    
    # 삭제할 파일 목록 (Dynamic Bank와 Masked Bank 관련 파일만)
    files_to_delete = [
        # Dynamic Bank 관련
        "bank_dynamic.npy",
        "angles_dynamic.json",
        "collection_status.json",
        # 각도별 Dynamic Bank 파일들
        "bank_front.npy",
        "bank_left.npy",
        "bank_right.npy",
        "bank_top.npy",
        "bank_left_profile.npy",
        "bank_right_profile.npy",
        "embedding_front.npy",
        "embedding_left.npy",
        "embedding_right.npy",
        "embedding_top.npy",
        "embedding_left_profile.npy",
        "embedding_right_profile.npy",
        # Masked Bank 관련
        "bank_masked.npy",
        "angles_masked.json",
        # 레거시 파일 (Dynamic/Masked 관련)
        "bank.npy",  # 레거시 (Base가 아닌 경우)
        "centroid.npy",  # 레거시 (Dynamic/Masked 관련일 수 있음)
        "angles.json",  # 레거시
    ]
    
    # Base Bank 파일은 유지 (삭제하지 않음)
    files_to_keep = [
        "bank_base.npy",
        "centroid_base.npy",
        "angles_base.json",
    ]
    
    deleted_files = []
    for filename in files_to_delete:
        file_path = person_dir / filename
        if file_path.exists():
            if backup:
                # 백업
                backup_path = backup_dir / filename
                shutil.copy2(file_path, backup_path)
            # 삭제
            file_path.unlink()
            deleted_files.append(filename)
    
    # Base Bank 파일 존재 여부 확인
    base_bank_path = person_dir / "bank_base.npy"
    has_base_bank = base_bank_path.exists()
    
    if deleted_files:
        print(f"  🗑️ 삭제된 파일 ({len(deleted_files)}개):")
        for f in deleted_files:
            print(f"     - {f}")
        if backup:
            print(f"  ✅ 백업 완료: {backup_dir}")
        
        if has_base_bank:
            print(f"  ✅ Base Bank 유지: bank_base.npy")
        else:
            print(f"  ⚠️ Base Bank 없음: bank_base.npy (새로 등록 필요)")
        
        return True
    else:
        print(f"  ℹ️ 삭제할 파일이 없습니다.")
        if has_base_bank:
            print(f"  ✅ Base Bank 유지: bank_base.npy")
        return False


def cleanup_all_banks(backup: bool = True) -> int:
    """
    모든 인물의 Dynamic Bank와 Masked Bank 삭제 (Base Bank는 유지)
    
    Args:
        backup: 기존 파일 백업 여부
    
    Returns:
        처리된 인물 수
    """
    if not EMBEDDINGS_DIR.exists():
        print(f"❌ Embeddings 디렉토리 없음: {EMBEDDINGS_DIR}")
        return 0
    
    person_dirs = [d for d in EMBEDDINGS_DIR.iterdir() if d.is_dir()]
    
    if not person_dirs:
        print(f"ℹ️ 처리할 인물이 없습니다.")
        return 0
    
    success_count = 0
    for person_dir in person_dirs:
        person_id = person_dir.name
        print(f"\n👤 {person_id}:")
        if cleanup_person_banks(person_id, backup=backup):
            success_count += 1
    
    return success_count


def cleanup_contaminated_embeddings(person_id: str, min_base_sim: float = 0.5, backup: bool = True) -> dict:
    """
    오염된 임베딩만 선별 삭제 (Base Bank와 유사도가 낮은 것만)
    
    Args:
        person_id: 인물 ID
        min_base_sim: Base Bank와의 최소 유사도 (이 값 미만이면 오염으로 간주)
        backup: 기존 파일 백업 여부
    
    Returns:
        {"removed": int, "kept": int, "total": int} 통계
    """
    import numpy as np
    import json
    
    person_dir = EMBEDDINGS_DIR / person_id
    
    if not person_dir.exists():
        print(f"  ⚠️ 인물 폴더 없음: {person_dir}")
        return {"removed": 0, "kept": 0, "total": 0}
    
    base_bank_path = person_dir / "bank_base.npy"
    dynamic_bank_path = person_dir / "bank_dynamic.npy"
    angles_path = person_dir / "angles_dynamic.json"
    
    # Base Bank 로드
    if not base_bank_path.exists():
        print(f"  ⚠️ Base Bank 없음: {base_bank_path}")
        return {"removed": 0, "kept": 0, "total": 0}
    
    base_bank = np.load(base_bank_path)
    if base_bank.ndim == 1:
        base_bank = base_bank.reshape(1, -1)
    
    # Dynamic Bank 로드
    if not dynamic_bank_path.exists():
        print(f"  ℹ️ Dynamic Bank 없음 (정리할 것이 없음)")
        return {"removed": 0, "kept": 0, "total": 0}
    
    dynamic_bank = np.load(dynamic_bank_path)
    if dynamic_bank.ndim == 1:
        dynamic_bank = dynamic_bank.reshape(1, -1)
    
    # 각도 정보 로드
    angles_info = {}
    if angles_path.exists():
        with open(angles_path, 'r', encoding='utf-8') as f:
            angles_info = json.load(f)
    
    # 백업
    if backup:
        backup_dir = person_dir / "backup_before_cleanup"
        backup_dir.mkdir(exist_ok=True)
        shutil.copy2(dynamic_bank_path, backup_dir / "bank_dynamic.npy")
        if angles_path.exists():
            shutil.copy2(angles_path, backup_dir / "angles_dynamic.json")
    
    # Base Bank 정규화
    base_bank_normalized = base_bank / (np.linalg.norm(base_bank, axis=1, keepdims=True) + 1e-6)
    
    # 각 Dynamic 임베딩을 Base Bank와 비교
    keep_indices = []
    remove_indices = []
    
    for i, emb in enumerate(dynamic_bank):
        # 정규화
        emb_normalized = emb / (np.linalg.norm(emb) + 1e-6)
        
        # Base Bank와의 최대 유사도 계산
        similarities = np.dot(base_bank_normalized, emb_normalized)
        max_sim = float(np.max(similarities))
        
        if max_sim >= min_base_sim:
            keep_indices.append(i)
        else:
            remove_indices.append(i)
    
    if not remove_indices:
        print(f"  ✅ 오염된 임베딩 없음 (모두 Base와 유사도 {min_base_sim} 이상)")
        return {"removed": 0, "kept": len(keep_indices), "total": dynamic_bank.shape[0]}
    
    # 유지할 임베딩만 저장
    cleaned_dynamic_bank = dynamic_bank[keep_indices]
    
    # 각도 정보도 업데이트
    if angles_info and "angle_types" in angles_info:
        cleaned_angles = {
            "angle_types": [angles_info["angle_types"][i] for i in keep_indices],
            "yaw_angles": [angles_info["yaw_angles"][i] for i in keep_indices] if "yaw_angles" in angles_info else []
        }
    else:
        cleaned_angles = {"angle_types": [], "yaw_angles": []}
    
    # 저장
    np.save(dynamic_bank_path, cleaned_dynamic_bank)
    with open(angles_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_angles, f, indent=2, ensure_ascii=False)
    
    print(f"  🗑️ 오염된 임베딩 삭제: {len(remove_indices)}개")
    print(f"  ✅ 유지된 임베딩: {len(keep_indices)}개")
    
    return {"removed": len(remove_indices), "kept": len(keep_indices), "total": dynamic_bank.shape[0]}


def limit_embeddings_per_angle(person_id: str, max_per_angle: int = 5, backup: bool = True) -> dict:
    """
    각 각도별로 품질 좋은 상위 N개만 유지
    
    Args:
        person_id: 인물 ID
        max_per_angle: 각도당 최대 임베딩 개수
        backup: 기존 파일 백업 여부
    
    Returns:
        {"removed": int, "kept": int, "total": int} 통계
    """
    import numpy as np
    import json
    from collections import defaultdict
    
    person_dir = EMBEDDINGS_DIR / person_id
    
    if not person_dir.exists():
        print(f"  ⚠️ 인물 폴더 없음: {person_dir}")
        return {"removed": 0, "kept": 0, "total": 0}
    
    base_bank_path = person_dir / "bank_base.npy"
    dynamic_bank_path = person_dir / "bank_dynamic.npy"
    angles_path = person_dir / "angles_dynamic.json"
    
    # Base Bank 로드
    if not base_bank_path.exists():
        print(f"  ⚠️ Base Bank 없음: {base_bank_path}")
        return {"removed": 0, "kept": 0, "total": 0}
    
    base_bank = np.load(base_bank_path)
    if base_bank.ndim == 1:
        base_bank = base_bank.reshape(1, -1)
    
    # Dynamic Bank 로드
    if not dynamic_bank_path.exists():
        print(f"  ℹ️ Dynamic Bank 없음 (정리할 것이 없음)")
        return {"removed": 0, "kept": 0, "total": 0}
    
    dynamic_bank = np.load(dynamic_bank_path)
    if dynamic_bank.ndim == 1:
        dynamic_bank = dynamic_bank.reshape(1, -1)
    
    # 각도 정보 로드
    if not angles_path.exists():
        print(f"  ⚠️ 각도 정보 없음: {angles_path}")
        return {"removed": 0, "kept": 0, "total": 0}
    
    with open(angles_path, 'r', encoding='utf-8') as f:
        angles_info = json.load(f)
    
    angle_types = angles_info.get("angle_types", [])
    
    if not angle_types:
        print(f"  ℹ️ 각도 정보 비어 있음")
        return {"removed": 0, "kept": 0, "total": 0}
    
    # 백업
    if backup:
        backup_dir = person_dir / "backup_before_cleanup"
        backup_dir.mkdir(exist_ok=True)
        shutil.copy2(dynamic_bank_path, backup_dir / "bank_dynamic.npy")
        shutil.copy2(angles_path, backup_dir / "angles_dynamic.json")
    
    # Base Bank 정규화
    base_bank_normalized = base_bank / (np.linalg.norm(base_bank, axis=1, keepdims=True) + 1e-6)
    
    # 각도별로 그룹화
    angle_groups = defaultdict(list)
    for i, angle_type in enumerate(angle_types):
        angle_groups[angle_type].append(i)
    
    # 각 각도별로 Base와 유사도 높은 상위 N개만 선택
    keep_indices = []
    
    for angle_type, indices in angle_groups.items():
        if len(indices) <= max_per_angle:
            # 이미 개수 제한 이내
            keep_indices.extend(indices)
        else:
            # Base와 유사도 계산
            similarities = []
            for idx in indices:
                emb = dynamic_bank[idx]
                emb_normalized = emb / (np.linalg.norm(emb) + 1e-6)
                max_sim = float(np.max(np.dot(base_bank_normalized, emb_normalized)))
                similarities.append((idx, max_sim))
            
            # 유사도 높은 순으로 정렬
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # 상위 N개만 선택
            top_n = similarities[:max_per_angle]
            keep_indices.extend([idx for idx, _ in top_n])
            
            print(f"  📊 [{angle_type}] {len(indices)}개 → {max_per_angle}개 (상위 {max_per_angle}개 유지)")
    
    # 인덱스 정렬 (순서 유지)
    keep_indices.sort()
    
    original_count = dynamic_bank.shape[0]
    
    if len(keep_indices) == original_count:
        print(f"  ✅ 모든 각도가 제한 이내 (변경 없음)")
        return {"removed": 0, "kept": original_count, "total": original_count}
    
    # 유지할 임베딩만 저장
    cleaned_dynamic_bank = dynamic_bank[keep_indices]
    
    # 각도 정보도 업데이트
    cleaned_angles = {
        "angle_types": [angle_types[i] for i in keep_indices],
        "yaw_angles": [angles_info["yaw_angles"][i] for i in keep_indices] if "yaw_angles" in angles_info else []
    }
    
    # 저장
    np.save(dynamic_bank_path, cleaned_dynamic_bank)
    with open(angles_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_angles, f, indent=2, ensure_ascii=False)
    
    removed_count = original_count - len(keep_indices)
    print(f"  🗑️ 제거된 임베딩: {removed_count}개")
    print(f"  ✅ 유지된 임베딩: {len(keep_indices)}개")
    
    return {"removed": removed_count, "kept": len(keep_indices), "total": original_count}


def main():
    parser = argparse.ArgumentParser(
        description="Dynamic Bank와 Masked Bank만 삭제하는 스크립트 - Base Bank는 유지합니다."
    )
    parser.add_argument(
        "--person-id",
        type=str,
        help="특정 인물 ID만 처리 (없으면 전체)"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="백업 없이 삭제 (주의: 되돌릴 수 없음)"
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="확인 없이 실행"
    )
    # v2 신규: --mode 옵션 추가
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "clean", "limit", "all"],
        default="full",
        help="정리 모드: full(전체 삭제), clean(오염 선별 삭제), limit(각도별 제한), all(clean+limit 순차 실행)"
    )
    parser.add_argument(
        "--min-base-sim",
        type=float,
        default=0.5,
        help="clean 모드에서 Base Bank와의 최소 유사도 (기본: 0.5)"
    )
    parser.add_argument(
        "--max-per-angle",
        type=int,
        default=5,
        help="limit 모드에서 각도별 최대 임베딩 개수 (기본: 5)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🧹 Dynamic/Masked Bank 정리 스크립트")
    print("=" * 70)
    print("\n📋 작업 내용:")
    print("   ✅ 유지: bank_base.npy (Base Bank)")
    print("   🗑️  삭제: bank_dynamic.npy, bank_masked.npy")
    print("   🗑️  삭제: angles_dynamic.json, angles_masked.json")
    print("   🗑️  삭제: collection_status.json")
    print("\n⚠️  주의사항:")
    print("   - Dynamic Bank와 Masked Bank 파일만 삭제됩니다.")
    print("   - Base Bank는 유지되어 새로운 검증 로직 테스트가 가능합니다.")
    print("   - 삭제된 파일은 복구할 수 없습니다 (백업 옵션 사용 시 제외).")
    print()
    
    # 확인
    if not args.confirm:
        if args.person_id:
            target = f"인물 '{args.person_id}'"
        else:
            target = "모든 인물"
        
        response = input(f"정말 {target}의 Dynamic/Masked Bank를 삭제하시겠습니까? (yes/no): ")
        if response.lower() != "yes":
            print("❌ 취소되었습니다.")
            return
    
    backup = not args.no_backup
    
    if backup:
        print("✅ 백업 모드: 기존 파일을 백업 폴더에 저장합니다.")
    else:
        print("⚠️  백업 없음: 파일이 영구적으로 삭제됩니다!")
    
    print()
    
    # 실행
    if args.person_id:
        print(f"📂 처리 대상: {args.person_id}\n")
        success = cleanup_person_banks(args.person_id, backup=backup)
        if success:
            print(f"\n✅ 완료: {args.person_id} 정리 완료")
        else:
            print(f"\n⚠️  완료: {args.person_id} 정리 실패 또는 파일 없음")
    else:
        print(f"📂 처리 대상: 모든 인물\n")
        success_count = cleanup_all_banks(backup=backup)
        print(f"\n✅ 완료: {success_count}명 처리 완료")
    
    print("\n" + "=" * 70)
    print("📝 다음 단계:")
    print("   1. 서버를 재시작하여 변경사항 반영")
    print("   2. 웹 인터페이스에서 영상 업로드 및 감지 테스트")
    print("   3. 콘솔 로그에서 다음 메시지 확인:")
    print("      ✅ [DYNAMIC BANK] 검증 통과: ...")
    print("      ⏭ [DYNAMIC BANK] 검증 실패: ...")
    print("   4. 새로운 검증 로직이 제대로 작동하는지 확인")
    print("=" * 70)


if __name__ == "__main__":
    main()













