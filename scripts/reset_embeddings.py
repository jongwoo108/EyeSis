"""
임베딩 초기화 스크립트
모델 평가를 위해 임베딩을 초기화하고 새로 등록할 수 있도록 준비합니다.
"""
import argparse
import shutil
from pathlib import Path
import sys

# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EMBEDDINGS_DIR = PROJECT_ROOT / "outputs" / "embeddings"


def reset_person_embeddings(person_id: str, backup: bool = True) -> bool:
    """
    특정 인물의 임베딩 파일 초기화
    
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
        backup_dir = person_dir / "backup_before_reset"
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        backup_dir.mkdir(exist_ok=True)
        print(f"  💾 백업 폴더 생성: {backup_dir}")
    
    # 삭제할 파일 목록
    files_to_delete = [
        "bank_base.npy",
        "bank_masked.npy",
        "bank_dynamic.npy",
        "bank.npy",  # 레거시
        "centroid_base.npy",
        "centroid_masked.npy",
        "centroid.npy",  # 레거시
        "angles_base.json",
        "angles_masked.json",
        "angles.json"  # 레거시
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
    
    if deleted_files:
        print(f"  🗑️ 삭제된 파일 ({len(deleted_files)}개):")
        for f in deleted_files:
            print(f"     - {f}")
        if backup:
            print(f"  ✅ 백업 완료: {backup_dir}")
        return True
    else:
        print(f"  ℹ️ 삭제할 파일이 없습니다.")
        return False


def reset_all_embeddings(backup: bool = True) -> int:
    """
    모든 인물의 임베딩 파일 초기화
    
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
        if reset_person_embeddings(person_id, backup=backup):
            success_count += 1
    
    return success_count


def main():
    parser = argparse.ArgumentParser(
        description="임베딩 초기화 스크립트 - 모델 평가를 위해 임베딩을 초기화합니다."
    )
    parser.add_argument(
        "--person-id",
        type=str,
        help="특정 인물 ID만 초기화 (없으면 전체)"
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
    print("🔄 임베딩 초기화 스크립트")
    print("=" * 70)
    print("\n⚠️  주의사항:")
    print("   - 이 스크립트는 임베딩 파일을 삭제합니다.")
    print("   - 삭제된 임베딩은 복구할 수 없습니다 (백업 옵션 사용 시 제외).")
    print("   - 데이터베이스의 Person 정보는 유지됩니다.")
    print("   - 초기화 후 새로 등록(enroll)해야 합니다.")
    print()
    
    # 확인
    if not args.confirm:
        if args.person_id:
            target = f"인물 '{args.person_id}'"
        else:
            target = "모든 인물"
        
        response = input(f"정말 {target}의 임베딩을 초기화하시겠습니까? (yes/no): ")
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
        success = reset_person_embeddings(args.person_id, backup=backup)
        if success:
            print(f"\n✅ 완료: {args.person_id} 초기화 완료")
        else:
            print(f"\n⚠️  완료: {args.person_id} 초기화 실패 또는 파일 없음")
    else:
        print(f"📂 처리 대상: 모든 인물\n")
        success_count = reset_all_embeddings(backup=backup)
        print(f"\n✅ 완료: {success_count}명 처리 완료")
    
    print("\n" + "=" * 70)
    print("📝 다음 단계:")
    print("   1. 웹 인터페이스에서 '용의자 추가' 버튼 클릭")
    print("   2. 정면 사진 업로드하여 새로 등록")
    print("   3. 또는 face_enroll.py 스크립트 사용")
    print("=" * 70)


if __name__ == "__main__":
    main()














