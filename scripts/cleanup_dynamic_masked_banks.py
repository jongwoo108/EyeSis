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





