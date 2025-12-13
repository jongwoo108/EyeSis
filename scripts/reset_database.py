"""
데이터베이스 초기화 스크립트
PostgreSQL 데이터베이스의 모든 테이블을 삭제하고 처음부터 다시 시작할 수 있도록 합니다.
"""
import argparse
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.database import Base, engine, SessionLocal, Person, DetectionLog


def reset_database(confirm: bool = False):
    """
    데이터베이스의 모든 테이블 삭제 및 재생성
    
    Args:
        confirm: 확인 없이 실행 여부
    """
    print("=" * 70)
    print("🗑️  데이터베이스 초기화 스크립트")
    print("=" * 70)
    print("\n⚠️  주의사항:")
    print("   - 이 작업은 모든 데이터를 삭제합니다.")
    print("   - 삭제된 데이터는 복구할 수 없습니다.")
    print("   - 다음 테이블이 삭제됩니다:")
    print("     - persons (인물 정보)")
    print("     - detection_logs (감지 로그)")
    print()
    
    # 확인
    if not confirm:
        response = input("정말 데이터베이스를 초기화하시겠습니까? (yes/no): ")
        if response.lower() != "yes":
            print("❌ 취소되었습니다.")
            return False
    
    try:
        # 데이터베이스 연결 확인
        print("\n1️⃣ 데이터베이스 연결 확인 중...")
        db = SessionLocal()
        db.close()
        print("   ✅ 데이터베이스 연결 성공")
        
        # 테이블 삭제
        print("\n2️⃣ 기존 테이블 삭제 중...")
        Base.metadata.drop_all(bind=engine)
        print("   ✅ 모든 테이블 삭제 완료")
        
        # 테이블 재생성
        print("\n3️⃣ 테이블 재생성 중...")
        Base.metadata.create_all(bind=engine)
        print("   ✅ 테이블 재생성 완료")
        
        # 확인
        print("\n4️⃣ 확인 중...")
        db = SessionLocal()
        try:
            person_count = db.query(Person).count()
            log_count = db.query(DetectionLog).count()
            print(f"   ✅ 확인 완료:")
            print(f"      - persons 테이블: {person_count}개 레코드")
            print(f"      - detection_logs 테이블: {log_count}개 레코드")
        finally:
            db.close()
        
        print("\n" + "=" * 70)
        print("✅ 데이터베이스 초기화 완료!")
        print("=" * 70)
        print("\n📝 다음 단계:")
        print("   1. 임베딩 파일도 초기화하려면:")
        print("      python scripts/reset_embeddings.py")
        print()
        print("   2. 새로 인물을 등록하려면:")
        print("      - 웹 인터페이스에서 '용의자 추가' 버튼 클릭")
        print("      - 또는 python src/face_enroll.py 실행")
        print()
        print("   3. 기존 데이터를 마이그레이션하려면:")
        print("      python backend/init_db.py")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 데이터베이스 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="데이터베이스 초기화 스크립트 - 모든 테이블을 삭제하고 재생성합니다."
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="확인 없이 실행"
    )
    
    args = parser.parse_args()
    
    success = reset_database(confirm=args.confirm)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()















