"""
기존 인물들의 레거시 파일(bank.npy, centroid.npy) 재생성 스크립트

bank_base.npy와 centroid_base.npy가 있으면 동일한 내용으로 레거시 파일을 생성합니다.
이 스크립트는 기존 인물들에 대해 레거시 파일이 없을 때 실행하면 됩니다.
"""
import argparse
import shutil
from pathlib import Path
import sys
import numpy as np

# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EMBEDDINGS_DIR = PROJECT_ROOT / "outputs" / "embeddings"


def regenerate_legacy_files(person_id: str, backup: bool = True, force: bool = False) -> bool:
    """
    특정 인물의 레거시 파일 재생성
    
    Args:
        person_id: 인물 ID
        backup: 기존 파일 백업 여부
        force: 기존 레거시 파일이 있어도 덮어쓰기 여부
    
    Returns:
        성공 여부
    """
    person_dir = EMBEDDINGS_DIR / person_id
    
    if not person_dir.exists():
        print(f"  ⚠️ 인물 폴더 없음: {person_dir}")
        return False
    
    bank_base_path = person_dir / "bank_base.npy"
    centroid_base_path = person_dir / "centroid_base.npy"
    legacy_bank_path = person_dir / "bank.npy"
    legacy_centroid_path = person_dir / "centroid.npy"
    
    regenerated = False
    
    # bank_base.npy → bank.npy 복사
    if bank_base_path.exists():
        if legacy_bank_path.exists() and not force:
            print(f"  ℹ️ Legacy Bank 이미 존재: {legacy_bank_path} (스킵, --force로 덮어쓰기 가능)")
        else:
            if backup and legacy_bank_path.exists():
                backup_path = person_dir / "bank.npy.backup"
                shutil.copy2(legacy_bank_path, backup_path)
                print(f"  💾 백업 생성: {backup_path}")
            
            bank_data = np.load(bank_base_path)
            np.save(legacy_bank_path, bank_data)
            print(f"  ✅ Legacy Bank 생성: {legacy_bank_path} (shape: {bank_data.shape})")
            regenerated = True
    else:
        print(f"  ⚠️ bank_base.npy 없음: {bank_base_path}")
    
    # centroid_base.npy → centroid.npy 복사
    if centroid_base_path.exists():
        if legacy_centroid_path.exists() and not force:
            print(f"  ℹ️ Legacy Centroid 이미 존재: {legacy_centroid_path} (스킵, --force로 덮어쓰기 가능)")
        else:
            if backup and legacy_centroid_path.exists():
                backup_path = person_dir / "centroid.npy.backup"
                shutil.copy2(legacy_centroid_path, backup_path)
                print(f"  💾 백업 생성: {backup_path}")
            
            centroid_data = np.load(centroid_base_path)
            np.save(legacy_centroid_path, centroid_data)
            print(f"  ✅ Legacy Centroid 생성: {legacy_centroid_path} (shape: {centroid_data.shape})")
            regenerated = True
    else:
        print(f"  ⚠️ centroid_base.npy 없음: {centroid_base_path}")
    
    return regenerated


def regenerate_all_legacy_files(backup: bool = True, force: bool = False) -> int:
    """
    모든 인물의 레거시 파일 재생성
    
    Args:
        backup: 기존 파일 백업 여부
        force: 기존 레거시 파일이 있어도 덮어쓰기 여부
    
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
        if regenerate_legacy_files(person_id, backup=backup, force=force):
            success_count += 1
    
    return success_count


def main():
    parser = argparse.ArgumentParser(
        description="기존 인물들의 레거시 파일(bank.npy, centroid.npy) 재생성 스크립트"
    )
    parser.add_argument(
        "--person-id",
        type=str,
        help="특정 인물 ID만 처리 (없으면 전체)"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="백업 없이 생성 (주의: 기존 파일이 덮어써짐)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="기존 레거시 파일이 있어도 덮어쓰기"
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="확인 없이 실행"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🔄 레거시 파일 재생성 스크립트")
    print("=" * 70)
    print("\n📋 작업 내용:")
    print("   - bank_base.npy → bank.npy 복사")
    print("   - centroid_base.npy → centroid.npy 복사")
    print("\n⚠️  주의사항:")
    print("   - 기존 레거시 파일이 있으면 스킵됩니다 (--force로 덮어쓰기 가능)")
    print("   - 이 스크립트는 하위 호환성을 위해 레거시 파일을 생성합니다")
    print()
    
    # 확인
    if not args.confirm:
        if args.person_id:
            target = f"인물 '{args.person_id}'"
        else:
            target = "모든 인물"
        
        response = input(f"정말 {target}의 레거시 파일을 재생성하시겠습니까? (yes/no): ")
        if response.lower() != "yes":
            print("❌ 취소되었습니다.")
            return
    
    backup = not args.no_backup
    
    if backup:
        print("✅ 백업 모드: 기존 파일을 백업합니다.")
    else:
        print("⚠️  백업 없음: 기존 파일이 덮어써집니다!")
    
    if args.force:
        print("⚠️  강제 모드: 기존 레거시 파일도 덮어씁니다!")
    
    print()
    
    # 실행
    if args.person_id:
        print(f"📂 처리 대상: {args.person_id}\n")
        success = regenerate_legacy_files(args.person_id, backup=backup, force=args.force)
        if success:
            print(f"\n✅ 완료: {args.person_id} 레거시 파일 재생성 완료")
        else:
            print(f"\n⚠️  완료: {args.person_id} 레거시 파일 재생성 실패 또는 이미 존재")
    else:
        print(f"📂 처리 대상: 모든 인물\n")
        success_count = regenerate_all_legacy_files(backup=backup, force=args.force)
        print(f"\n✅ 완료: {success_count}명 처리 완료")
    
    print("\n" + "=" * 70)
    print("📝 참고:")
    print("   - 레거시 파일은 gallery_loader.py에서 fallback으로 사용될 수 있습니다")
    print("   - 새로 등록하는 인물은 자동으로 레거시 파일이 생성됩니다")
    print("=" * 70)


if __name__ == "__main__":
    main()









