"""
데이터베이스 초기화 스크립트
기존 JSON 파일 또는 outputs/embeddings에서 데이터를 PostgreSQL로 마이그레이션
"""
import os
import sys
import json
import glob
import numpy as np
from pathlib import Path

# 프로젝트 루트를 경로에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.database import init_db, SessionLocal, Person, create_person
from backend.utils.gallery_loader import load_gallery


def load_json_database():
    """기존 JSON 파일에서 데이터 로드"""
    db_path = PROJECT_ROOT / "backend" / "database"
    if not db_path.exists():
        print(f"⚠️ database 폴더를 찾을 수 없습니다: {db_path}")
        return []
    
    json_files = glob.glob(str(db_path / "*.json"))
    print(f"📂 JSON 파일 검색 중... ({len(json_files)}개 파일)")
    
    persons_data = []
    for filepath in json_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                person_id = data.get("person_id")
                name = data.get("name", person_id)
                is_criminal = person_id == "criminal" or data.get("is_criminal", False)
                info = data.get("info", {})
                
                # 임베딩 벡터 추출
                embedding = None
                if "mean_embedding" in data:
                    embedding = np.array(data["mean_embedding"], dtype=np.float32)
                elif "embedding" in data:
                    embedding = np.array(data["embedding"], dtype=np.float32)
                
                if embedding is not None:
                    persons_data.append({
                        "person_id": person_id,
                        "name": name,
                        "is_criminal": is_criminal,
                        "info": info,
                        "embedding": embedding
                    })
                    print(f"  ✅ 로드 완료: {name} (ID: {person_id}, 범죄자: {is_criminal})")
                else:
                    print(f"  ⚠️ 임베딩 없음: {filepath}")
        except Exception as e:
            print(f"  ❌ 로드 실패 ({filepath}): {e}")
    
    return persons_data


def load_embeddings_database():
    """outputs/embeddings 폴더에서 데이터 로드"""
    embeddings_dir = PROJECT_ROOT / "outputs" / "embeddings"
    if not embeddings_dir.exists():
        print(f"⚠️ embeddings 폴더를 찾을 수 없습니다: {embeddings_dir}")
        return []
    
    try:
        gallery = load_gallery(embeddings_dir, use_bank=True)
        print(f"📂 Embeddings 폴더에서 데이터 로드 중... ({len(gallery)}명)")
        
        persons_data = []
        for person_id, emb_data in gallery.items():
            # emb_data가 2D 배열(bank)이면 첫 번째 임베딩 사용, 아니면 centroid 사용
            if emb_data.ndim == 2:
                embedding = emb_data[0]  # bank의 첫 번째 임베딩
            else:
                embedding = emb_data  # centroid
            
            # person_id에서 이름 추출 (폴더명이 person_id)
            name = person_id
            # 범죄자 여부 설정 (yh는 황윤하로 범죄자)
            is_criminal = person_id == "criminal" or person_id == "yh"
            
            persons_data.append({
                "person_id": person_id,
                "name": name,
                "is_criminal": is_criminal,
                "info": {},
                "embedding": embedding.astype(np.float32)
            })
            print(f"  ✅ 로드 완료: {name} (ID: {person_id}, 범죄자: {is_criminal})")
        
        return persons_data
    except Exception as e:
        print(f"  ❌ 로드 실패: {e}")
        return []


def migrate_to_postgresql():
    """데이터를 PostgreSQL로 마이그레이션"""
    print("=" * 70)
    print("🗄️  데이터베이스 초기화 시작")
    print("=" * 70)
    
    # 1. 데이터베이스 테이블 생성
    print("\n1️⃣ 데이터베이스 테이블 생성 중...")
    try:
        init_db()
    except Exception as e:
        print(f"❌ 데이터베이스 초기화 실패: {e}")
        print("   PostgreSQL이 실행 중인지 확인해주세요.")
        return
    
    # 2. 데이터 소스에서 로드 (우선순위: embeddings > JSON)
    print("\n2️⃣ 데이터 소스에서 로드 중...")
    
    # 우선순위 1: outputs/embeddings
    persons_data = load_embeddings_database()
    
    # 우선순위 2: backend/database/*.json (embeddings가 없을 때만)
    if not persons_data:
        print("\n   outputs/embeddings에 데이터가 없어 JSON 파일을 확인합니다...")
        persons_data = load_json_database()
    
    if not persons_data:
        print("\n⚠️ 마이그레이션할 데이터가 없습니다.")
        print("   face_enroll.py를 실행하여 인물을 등록하거나,")
        print("   backend/database/*.json 파일을 생성해주세요.")
        return
    
    # 3. PostgreSQL에 데이터 삽입
    print(f"\n3️⃣ PostgreSQL에 데이터 삽입 중... ({len(persons_data)}개)")
    db = SessionLocal()
    
    added_count = 0
    skipped_count = 0
    
    try:
        for person_data in persons_data:
            # 기존 데이터 확인
            existing = db.query(Person).filter(
                Person.person_id == person_data["person_id"]
            ).first()
            
            if existing:
                print(f"  ⏭ 스킵 (이미 존재): {person_data['name']}")
                skipped_count += 1
                continue
            
            # 새 데이터 삽입
            create_person(
                db=db,
                person_id=person_data["person_id"],
                name=person_data["name"],
                embedding=person_data["embedding"],
                is_criminal=person_data["is_criminal"],
                info=person_data["info"]
            )
            print(f"  ✅ 추가 완료: {person_data['name']}")
            added_count += 1
        
        print(f"\n🎉 마이그레이션 완료!")
        print(f"   추가됨: {added_count}개")
        print(f"   스킵됨: {skipped_count}개 (이미 존재)")
        print(f"   총 인물 수: {len(persons_data)}개")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    migrate_to_postgresql()
