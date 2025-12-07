# backend/services/data_loader.py
"""
데이터 로딩 및 캐싱 서비스
"""

from pathlib import Path
from typing import Optional, List, Dict
import numpy as np
from sqlalchemy.orm import Session

from backend.database import get_all_persons
from backend.utils.image_utils import l2_normalize


# 프로젝트 루트를 Python 경로에 추가
PROJECT_ROOT = Path(__file__).parent.parent.parent
EMBEDDINGS_DIR = PROJECT_ROOT / "outputs" / "embeddings"

# 메모리 캐시 (성능 향상을 위해)
persons_cache: List[Dict] = []
gallery_base_cache: Dict[str, np.ndarray] = {}  # base bank (정면, 측면, 마스크 없는 얼굴)
gallery_masked_cache: Dict[str, np.ndarray] = {}  # masked bank (마스크 쓴 얼굴)
gallery_dynamic_cache: Dict[str, np.ndarray] = {}  # dynamic bank (CCTV에서 수집한 다양한 각도 임베딩 - 인식용)

def load_persons_from_db(db: Session):
    """PostgreSQL에서 인물 정보 로드 및 캐시 (Bank 데이터 포함 - base/masked/dynamic 분리)"""
    global persons_cache, gallery_base_cache, gallery_masked_cache, gallery_dynamic_cache
    
    persons = get_all_persons(db)
    
    persons_cache = []
    gallery_base_cache = {}
    gallery_masked_cache = {}
    gallery_dynamic_cache = {}
    
    for person in persons:
        person_id = person.person_id
        
        # outputs/embeddings 폴더에서 Bank 데이터 확인
        person_dir = EMBEDDINGS_DIR / person_id
        base_bank_path = person_dir / "bank_base.npy"
        masked_bank_path = person_dir / "bank_masked.npy"
        dynamic_bank_path = person_dir / "bank_dynamic.npy"  # 동적 bank (인식용)
        centroid_path = person_dir / "centroid.npy"
        
        # 레거시 파일 경로 (참고용, 사용하지 않음)
        # legacy_bank_path = person_dir / "bank.npy"
        # legacy_centroid_path = person_dir / "centroid.npy"
        
        base_bank = None
        masked_bank = None
        dynamic_bank = None
        
        # ===== Base Bank 로딩 (새 구조만 사용, 레거시 파일 사용 안 함) =====
        # 1. bank_base.npy (새 구조) - 필수
        if base_bank_path.exists():
            try:
                base_bank = np.load(base_bank_path)
                if base_bank.ndim == 1:
                    base_bank = base_bank.reshape(1, -1)
                # L2 정규화
                base_bank = base_bank / (np.linalg.norm(base_bank, axis=1, keepdims=True) + 1e-6)
            except Exception as e:
                print(f"  ⚠️ Base Bank 로드 실패 ({person_id}): {e}")
                base_bank = None
        
        # 2. DB 임베딩 사용 (fallback)
        if base_bank is None:
            try:
                db_embedding = person.get_embedding()
                db_embedding = l2_normalize(db_embedding)
                base_bank = db_embedding.reshape(1, -1)
                print(f"  ℹ️ DB 임베딩을 Base Bank로 사용: {person_id}")
            except Exception as e:
                print(f"  ⚠️ DB 임베딩 로드 실패 ({person_id}): {e}")
                base_bank = None
        
        # Base가 없으면 스킵
        if base_bank is None:
            print(f"  ❌ Base Bank를 찾을 수 없음: {person.name} (ID: {person_id}), 스킵")
            continue
        
        # ===== Masked Bank 로딩 =====
        if masked_bank_path.exists():
            try:
                masked_bank = np.load(masked_bank_path)
                if masked_bank.ndim == 1:
                    masked_bank = masked_bank.reshape(1, -1)
                if masked_bank.shape[0] > 0:
                    # L2 정규화
                    masked_bank = masked_bank / (np.linalg.norm(masked_bank, axis=1, keepdims=True) + 1e-6)
                else:
                    masked_bank = None
            except Exception as e:
                print(f"  ⚠️ Masked Bank 로드 실패 ({person_id}): {e}")
                masked_bank = None
        else:
            # Masked Bank가 없으면 None (빈 상태)
            masked_bank = None
        
        # ===== Dynamic Bank 로딩 (인식용) =====
        if dynamic_bank_path.exists():
            try:
                dynamic_bank = np.load(dynamic_bank_path)
                if dynamic_bank.ndim == 1:
                    dynamic_bank = dynamic_bank.reshape(1, -1)
                if dynamic_bank.shape[0] > 0:
                    # L2 정규화
                    dynamic_bank = dynamic_bank / (np.linalg.norm(dynamic_bank, axis=1, keepdims=True) + 1e-6)
                else:
                    dynamic_bank = None
            except Exception as e:
                print(f"  ⚠️ Dynamic Bank 로드 실패 ({person_id}): {e}")
                dynamic_bank = None
        else:
            # Dynamic Bank가 없으면 None (빈 상태)
            dynamic_bank = None
        
        # gallery_base_cache, gallery_masked_cache, gallery_dynamic_cache에 저장
        gallery_base_cache[person_id] = base_bank
        if masked_bank is not None:
            gallery_masked_cache[person_id] = masked_bank
        if dynamic_bank is not None:
            gallery_dynamic_cache[person_id] = dynamic_bank
        
        # persons_cache에는 base의 첫 번째 임베딩 사용 (표시용)
        first_embedding = base_bank[0] if base_bank.ndim == 2 else base_bank.flatten()
        
        person_data = {
            "id": person_id,
            "name": person.name,
            "is_criminal": person.is_criminal,
            "info": person.info or {},
            "embedding": first_embedding
        }
        persons_cache.append(person_data)
        
        # 로드 결과 출력
        masked_count = masked_bank.shape[0] if masked_bank is not None else 0
        dynamic_count = dynamic_bank.shape[0] if dynamic_bank is not None else 0
        masked_file_path = str(masked_bank_path.relative_to(PROJECT_ROOT)) if masked_bank_path.exists() else "없음"
        dynamic_file_path = str(dynamic_bank_path.relative_to(PROJECT_ROOT)) if dynamic_bank_path.exists() else "없음"
        print(f"  ✅ Bank 로드: {person.name} (ID: {person_id}, base: {base_bank.shape[0]}개, masked: {masked_count}개, dynamic: {dynamic_count}개)")
    
    print(f"📂 데이터베이스 로딩 완료 ({len(persons_cache)}명, Base/Masked/Dynamic Bank 분리 구조)\n")

def load_persons_from_embeddings():
    """outputs/embeddings에서 gallery 로드 (fallback - base/masked/dynamic 분리 구조)"""
    global gallery_base_cache, gallery_masked_cache, gallery_dynamic_cache, persons_cache
    
    if not EMBEDDINGS_DIR.exists():
        print(f"⚠️ embeddings 폴더를 찾을 수 없습니다: {EMBEDDINGS_DIR}")
        return
    
    try:
        gallery_base_cache = {}
        gallery_masked_cache = {}
        gallery_dynamic_cache = {}
        persons_cache = []
        
        # 사람별 폴더 구조 확인
        person_dirs = [d for d in EMBEDDINGS_DIR.iterdir() if d.is_dir()]
        
        for person_dir in person_dirs:
            person_id = person_dir.name
            
            base_bank_path = person_dir / "bank_base.npy"
            masked_bank_path = person_dir / "bank_masked.npy"
            dynamic_bank_path = person_dir / "bank_dynamic.npy"  # 동적 bank (인식용)
            # 레거시 파일 경로 (참고용, 사용하지 않음)
            # legacy_bank_path = person_dir / "bank.npy"
            # legacy_centroid_path = person_dir / "centroid.npy"
            
            base_bank = None
            masked_bank = None
            dynamic_bank = None
            
            # Base Bank 로딩 (새 구조만 사용, 레거시 파일 사용 안 함)
            if base_bank_path.exists():
                try:
                    base_bank = np.load(base_bank_path)
                    if base_bank.ndim == 1:
                        base_bank = base_bank.reshape(1, -1)
                    base_bank = base_bank / (np.linalg.norm(base_bank, axis=1, keepdims=True) + 1e-6)
                except Exception as e:
                    print(f"  ⚠️ Base Bank 로드 실패 ({person_id}): {e}")
                    base_bank = None
            
            if base_bank is None:
                continue  # Base가 없으면 스킵
            
            # Masked Bank 로딩
            if masked_bank_path.exists():
                try:
                    masked_bank = np.load(masked_bank_path)
                    if masked_bank.ndim == 1:
                        masked_bank = masked_bank.reshape(1, -1)
                    if masked_bank.shape[0] > 0:
                        masked_bank = masked_bank / (np.linalg.norm(masked_bank, axis=1, keepdims=True) + 1e-6)
                    else:
                        masked_bank = None
                except Exception as e:
                    print(f"  ⚠️ Masked Bank 로드 실패 ({person_id}): {e}")
                    masked_bank = None
            
            # Dynamic Bank 로딩 (인식용)
            if dynamic_bank_path.exists():
                try:
                    dynamic_bank = np.load(dynamic_bank_path)
                    if dynamic_bank.ndim == 1:
                        dynamic_bank = dynamic_bank.reshape(1, -1)
                    if dynamic_bank.shape[0] > 0:
                        dynamic_bank = dynamic_bank / (np.linalg.norm(dynamic_bank, axis=1, keepdims=True) + 1e-6)
                    else:
                        dynamic_bank = None
                except Exception as e:
                    print(f"  ⚠️ Dynamic Bank 로드 실패 ({person_id}): {e}")
                    dynamic_bank = None
            
            # gallery_base_cache, gallery_masked_cache, gallery_dynamic_cache에 저장
            gallery_base_cache[person_id] = base_bank
            if masked_bank is not None:
                gallery_masked_cache[person_id] = masked_bank
            if dynamic_bank is not None:
                gallery_dynamic_cache[person_id] = dynamic_bank
            
            # persons_cache에 추가
            first_embedding = base_bank[0] if base_bank.ndim == 2 else base_bank.flatten()
            persons_cache.append({
                "id": person_id,
                "name": person_id,  # 이름이 없으면 ID 사용
                "is_criminal": person_id == "criminal",
                "info": {},
                "embedding": first_embedding
            })
            masked_count = masked_bank.shape[0] if masked_bank is not None else 0
            dynamic_count = dynamic_bank.shape[0] if dynamic_bank is not None else 0
            print(f"  - {person_id} (base: {base_bank.shape[0]}개, masked: {masked_count}개, dynamic: {dynamic_count}개)")
        
        print(f"📂 Gallery 로딩 완료 ({len(gallery_base_cache)}명, Base/Masked/Dynamic Bank 분리 구조)\n")
    except Exception as e:
        print(f"⚠️ Gallery 로딩 실패: {e}\n")
        import traceback
        traceback.print_exc()

# ==========================================
# 레거시 파일 전용 로딩 함수 (독립적으로 사용 가능)
# ==========================================

def load_persons_from_legacy_files():
    """
    레거시 파일(bank.npy, centroid.npy)만 사용하여 갤러리 로드
    새 구조 파일(bank_base.npy, bank_masked.npy)은 사용하지 않음 (독립적인 레거시 모드)
    
    사용 예시:
        # 레거시 모드로 전환하려면 이 함수를 호출
        load_persons_from_legacy_files()
    """
    global gallery_base_cache, gallery_masked_cache, persons_cache
    
    if not EMBEDDINGS_DIR.exists():
        print(f"⚠️ embeddings 폴더를 찾을 수 없습니다: {EMBEDDINGS_DIR}")
        return
    
    try:
        gallery_base_cache = {}
        gallery_masked_cache = {}
        persons_cache = []
        
        person_dirs = [d for d in EMBEDDINGS_DIR.iterdir() if d.is_dir()]
        
        for person_dir in person_dirs:
            person_id = person_dir.name
            
            legacy_bank_path = person_dir / "bank.npy"
            legacy_centroid_path = person_dir / "centroid.npy"
            
            base_bank = None
            
            # 레거시 bank.npy 로딩
            if legacy_bank_path.exists():
                try:
                    base_bank = np.load(legacy_bank_path)
                    if base_bank.ndim == 1:
                        base_bank = base_bank.reshape(1, -1)
                    base_bank = base_bank / (np.linalg.norm(base_bank, axis=1, keepdims=True) + 1e-6)
                    print(f"  ✅ Legacy Bank 로드: {person_id} ({base_bank.shape[0]}개 임베딩)")
                except Exception as e:
                    print(f"  ⚠️ Legacy Bank 로드 실패 ({person_id}): {e}")
                    base_bank = None
            
            # 레거시 centroid.npy 로딩 (bank.npy가 없을 때만)
            if base_bank is None and legacy_centroid_path.exists():
                try:
                    centroid_data = np.load(legacy_centroid_path)
                    centroid_data = l2_normalize(centroid_data)
                    base_bank = centroid_data.reshape(1, -1)
                    print(f"  ✅ Legacy Centroid 로드: {person_id}")
                except Exception as e:
                    print(f"  ⚠️ Legacy Centroid 로드 실패 ({person_id}): {e}")
                    base_bank = None
            
            if base_bank is None:
                continue  # 레거시 파일이 없으면 스킵
            
            # gallery_base_cache에 저장 (레거시 파일을 base로 사용)
            gallery_base_cache[person_id] = base_bank
            
            # persons_cache에 추가
            first_embedding = base_bank[0] if base_bank.ndim == 2 else base_bank.flatten()
            person_data = {
                "id": person_id,
                "name": person_id,  # 레거시 모드에서는 이름 정보 없음
                "is_criminal": False,
                "info": {},
                "embedding": first_embedding
            }
            persons_cache.append(person_data)
        
        print(f"📂 레거시 파일 로딩 완료 ({len(persons_cache)}명, Legacy 모드)\n")
        
    except Exception as e:
        print(f"❌ 레거시 파일 로딩 실패: {e}")
        import traceback
        traceback.print_exc()

    
def find_person_info(person_id: str) -> Optional[Dict]:
    """person_id로 인물 정보 찾기"""
    for person in persons_cache:
        if person["id"] == person_id:
            return person
    return None