"""
DB에 저장된 인물의 범죄자 여부 업데이트 스크립트
"""
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.database import SessionLocal, Person

def update_criminal_status():
    """범죄자 여부 업데이트"""
    db = SessionLocal()
    
    try:
        # yh (황윤하)를 범죄자로 설정
        person = db.query(Person).filter(Person.person_id == "yh").first()
        if person:
            person.is_criminal = True
            db.commit()
            print(f"✅ {person.name} (ID: {person.person_id})를 범죄자로 설정했습니다.")
        else:
            print(f"⚠️ person_id='yh'인 인물을 찾을 수 없습니다.")
        
        # 다른 인물들은 일반인으로 유지 (필요시 여기서 수정 가능)
        # 예: js, jw, ja는 일반인으로 유지
        
        print("\n📋 현재 범죄자 목록:")
        criminals = db.query(Person).filter(Person.is_criminal == True).all()
        for p in criminals:
            print(f"  - {p.name} (ID: {p.person_id})")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    print("=" * 70)
    print("🔧 범죄자 여부 업데이트")
    print("=" * 70)
    update_criminal_status()




















