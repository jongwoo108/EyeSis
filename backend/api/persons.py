# backend/api/persons.py
"""
인물 관리 API 엔드포인트
"""
import shutil
import numpy as np
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from pathlib import Path

from backend.database import get_db, get_all_persons, get_person_by_id, create_person
from backend.services import data_loader
from backend.services.data_loader import load_persons_from_db

# 프로젝트 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent.parent
EMBEDDINGS_DIR = PROJECT_ROOT / "outputs" / "embeddings"

# model과 face_enroll 함수는 main.py에서 초기화 후 injection 필요
# 일단 지연 import로 처리
_model = None

def set_model(model):
    """main.py에서 모델 injection"""
    global _model
    _model = model

def get_model():
    """모델 가져오기 (지연 로딩)"""
    global _model
    if _model is None:
        # Fallback: main.py에서 injection 안됐으면 직접 import
        from backend.services.face_detection import get_model as fd_get_model
        _model = fd_get_model()
    return _model

# face_enroll 함수들 import
from src.face_enroll import get_main_face_embedding, save_embeddings, l2_normalize

router = APIRouter()

@router.get("/api/persons")
async def get_persons(db: Session = Depends(get_db)):
    """등록된 모든 인물 목록 조회"""

    
    print(f"🔍 [API /persons] 요청 받음 - data_loader.persons_cache 길이: {len(data_loader.persons_cache) if data_loader.persons_cache else 0}")
    
    # 이미지 경로 찾기 헬퍼 함수
    def find_person_image(person_id: str) -> Optional[str]:
        """인물의 등록 이미지 경로 찾기"""
        enroll_dir = PROJECT_ROOT / "images" / "enroll" / person_id
        if enroll_dir.exists():
            # 지원하는 이미지 확장자
            image_exts = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
            # person_id로 시작하는 파일 찾기
            for ext in image_exts:
                img_file = enroll_dir / f"{person_id}{ext}"
                if img_file.exists():
                    return f"/api/images/enroll/{person_id}/{img_file.name}"
            # 또는 첫 번째 이미지 파일 찾기
            for ext in image_exts:
                for img_file in enroll_dir.glob(f"*{ext}"):
                    if img_file.exists():
                        return f"/api/images/enroll/{person_id}/{img_file.name}"
        return None
    
    # ⭐ 버그 수정: 쪼시를 사용하지 않고 항상 DB에서 직접 조회
    # 이렇게 해야 삭제/수정된 인물 정보가 즉시 반영됨
    # 캐시에서 반환 (성능 향상)
    # if data_loader.persons_cache and len(data_loader.persons_cache) > 0:
    #     print(f"📋 [API] data_loader.persons_cache에서 반환: {len(data_loader.persons_cache)}명")
    #     result = {
    #         "success": True,
    #         "count": len(data_loader.persons_cache),
    #         "persons": [
    #             {
    #                 "id": p["id"],
    #                 "name": p["name"],
    #                 "is_criminal": p["is_criminal"],
    #                 "person_type": p.get("info", {}).get("person_type", "criminal" if p["is_criminal"] else "unknown"),
    #                 "info": p.get("info", {}),
    #                 "image_url": find_person_image(p["id"])  # 이미지 URL 추가
    #             }
    #             for p in data_loader.persons_cache
    #         ]
    #     }
    #     print(f"✅ [API] 응답 전송: success={result['success']}, count={result['count']}")
    #     return result
    
    # 쪼시가 없으면 DB에서 직접 조회
    print(f"⚠️ [API] data_loader.persons_cache가 비어있음, DB에서 직접 조회 시도")
    try:
        persons = get_all_persons(db)
        print(f"📋 [API] DB에서 조회: {len(persons)}명")
        
        # DB에서 조회한 데이터로 캐시 갱신 (다음 요청을 위해)
        if persons:
            # 캐시 갱신을 위해 load_persons_from_db 호출
            try:
                load_persons_from_db(db)
                print(f"✅ [API] 캐시 갱신 완료: {len(data_loader.persons_cache)}명")
            except Exception as cache_error:
                print(f"⚠️ [API] 캐시 갱신 실패: {cache_error}")
                import traceback
                traceback.print_exc()
        
        # 이미지 경로 찾기 헬퍼 함수 (중복 정의 방지)
        def find_person_image_db(person_id: str) -> Optional[str]:
            """인물의 등록 이미지 경로 찾기"""
            enroll_dir = PROJECT_ROOT / "images" / "enroll" / person_id
            if enroll_dir.exists():
                image_exts = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
                for ext in image_exts:
                    img_file = enroll_dir / f"{person_id}{ext}"
                    if img_file.exists():
                        return f"/api/images/enroll/{person_id}/{img_file.name}"
                for ext in image_exts:
                    for img_file in enroll_dir.glob(f"*{ext}"):
                        if img_file.exists():
                            return f"/api/images/enroll/{person_id}/{img_file.name}"
            return None
        
        result = {
            "success": True,
            "count": len(persons),
            "persons": [
                {
                    "id": p.person_id,
                    "name": p.name,
                    "is_criminal": p.is_criminal,
                    "person_type": (p.info or {}).get("person_type", "criminal" if p.is_criminal else "unknown"),
                    "info": p.info or {},
                    "image_url": find_person_image_db(p.person_id)  # 이미지 URL 추가
                }
                for p in persons
            ]
        }
        print(f"✅ [API] 응답 전송: success={result['success']}, count={result['count']}")
        return result
    except Exception as e:
        print(f"❌ [API] DB 조회 실패: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "count": 0,
            "persons": []
        }

@router.delete("/api/persons/{person_id}")
async def delete_person(person_id: str, db: Session = Depends(get_db)):
    """
    인물 삭제 API - 인물 데이터와 관련된 모든 파일 및 DB 레코드 삭제
    
    Args:
        person_id: 삭제할 인물의 고유 ID
        db: 데이터베이스 세션
    
    Returns:
        {
            "status": "success",
            "message": "Deleted successfully"
        }
    """
    
    try:
        print(f"🗑️ [DELETE] 인물 삭제 요청: person_id={person_id}")
        
        # 1. DB에서 인물 정보 조회
        from backend.database import get_person_by_id
        person = get_person_by_id(db, person_id)
        
        if not person:
            raise HTTPException(status_code=404, detail=f"인물을 찾을 수 없습니다: {person_id}")
        
        person_name = person.name
        print(f"  📋 삭제 대상: {person_name} ({person_id})")
        
        # 2. 안전성 검사: person_id가 안전한 문자열인지 확인 (경로 조작 방지)
        if not person_id or not person_id.replace('_', '').replace('-', '').isalnum():
            raise HTTPException(status_code=400, detail="잘못된 person_id 형식입니다.")
        
        # 3. 파일 시스템 정리 (DB 삭제 전에 먼저 수행)
        deleted_files = []
        
        # 3-1. images/enroll/{person_id}/ 폴더 삭제
        enroll_dir = PROJECT_ROOT / "images" / "enroll" / person_id
        if enroll_dir.exists() and enroll_dir.is_dir():
            # 안전성 검사: 경로가 올바른지 확인
            if str(enroll_dir).startswith(str(PROJECT_ROOT / "images" / "enroll")):
                try:
                    shutil.rmtree(enroll_dir)
                    deleted_files.append(f"images/enroll/{person_id}/")
                    print(f"  ✅ 이미지 폴더 삭제: {enroll_dir}")
                except Exception as e:
                    print(f"  ⚠️ 이미지 폴더 삭제 실패: {e}")
            else:
                print(f"  ⚠️ 안전성 검사 실패: 잘못된 경로 {enroll_dir}")
        
        # 3-2. outputs/embeddings/{person_id}/ 폴더 삭제
        embedding_dir = EMBEDDINGS_DIR / person_id
        if embedding_dir.exists() and embedding_dir.is_dir():
            # 안전성 검사: 경로가 올바른지 확인
            if str(embedding_dir).startswith(str(EMBEDDINGS_DIR)):
                try:
                    shutil.rmtree(embedding_dir)
                    deleted_files.append(f"outputs/embeddings/{person_id}/")
                    print(f"  ✅ 임베딩 폴더 삭제: {embedding_dir}")
                except Exception as e:
                    print(f"  ⚠️ 임베딩 폴더 삭제 실패: {e}")
            else:
                print(f"  ⚠️ 안전성 검사 실패: 잘못된 경로 {embedding_dir}")
        
        # 4. 데이터베이스에서 레코드 삭제
        try:
            db.delete(person)
            db.commit()
            print(f"  ✅ DB 레코드 삭제 완료: {person_id}")
        except Exception as e:
            db.rollback()
            print(f"  ❌ DB 레코드 삭제 실패: {e}")
            raise HTTPException(status_code=500, detail=f"데이터베이스 삭제 중 오류 발생: {str(e)}")
        
        # 5. 캐시 갱신
        try:
            # 전역 함수 직접 호출
            load_persons_from_db(db)
            print(f"  ✅ 캐시 갱신 완료")
        except Exception as cache_error:
            print(f"  ⚠️ 캐시 갱신 실패: {cache_error}")
            # 캐시 갱신 실패 시 수동으로 제거
            persons_cache
            if data_loader.persons_cache:
                data_loader.persons_cache = [p for p in data_loader.persons_cache if p.get('id') != person_id]
        
        # 6. 갤러리 캐시에서도 제거
        if person_id in data_loader.gallery_base_cache:
            del data_loader.gallery_base_cache[person_id]
        if person_id in data_loader.gallery_masked_cache:
            del data_loader.gallery_masked_cache[person_id]
        
        print(f"  ✅ 인물 삭제 완료: {person_name} ({person_id})")
        print(f"  📁 삭제된 파일: {', '.join(deleted_files) if deleted_files else '없음'}")
        
        return {
            "status": "success",
            "message": f"인물 '{person_name}' 삭제 완료",
            "deleted_files": deleted_files
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ [DELETE] 인물 삭제 실패: {e}")
        raise HTTPException(status_code=500, detail=f"삭제 중 오류 발생: {str(e)}")

@router.put("/api/persons/{person_id}")
async def update_person(person_id: str, db: Session = Depends(get_db),
                       name: str = Form(None),
                       person_type: str = Form(None)):
    """
    인물 정보 수정 API - 이름 및 카테고리 수정
    
    Args:
        person_id: 수정할 인물의 고유 ID
        name: 새로운 이름 (선택)
        person_type: 새로운 카테고리 (선택)
        db: 데이터베이스 세션
    
    Returns:
        {
            "status": "success",
            "person": {...}  # 수정된 인물 정보
        }
    """
    
    try:
        print(f"✏️ [UPDATE] 인물 수정 요청: person_id={person_id}")
        
        # 1. DB에서 인물 정보 조회
        from backend.database import get_person_by_id
        person = get_person_by_id(db, person_id)
        
        if not person:
            raise HTTPException(status_code=404, detail=f"인물을 찾을 수 없습니다: {person_id}")
        
        # 2. 수정할 필드 업데이트
        updated = False
        
        if name is not None and name.strip():
            old_name = person.name
            person.name = name.strip()
            print(f"  📝 이름 변경: {old_name} → {person.name}")
            updated = True
        
        if person_type is not None:
            # info 필드가 None일 경우 빈 딕셔너리로 초기화
            if person.info is None:
                person.info = {}
            
            # 기존 info 복사 (SQLAlchemy 감지용)
            new_info = dict(person.info)
            old_type = new_info.get('person_type', 'unknown')
            
            # person_type 저장
            new_info['person_type'] = person_type
            person.info = new_info
            
            # is_criminal 업데이트 (범죄자, 수배자만 True)
            person.is_criminal = (person_type in ["criminal", "wanted"])
            
            print(f"  📝 타입 변경: {old_type} → {person_type}")
            updated = True
        
        if not updated:
            raise HTTPException(status_code=400, detail="수정할 정보가 없습니다")
        
        # 3. DB 커밋
        db.commit()
        db.refresh(person)
        print(f"  ✅ DB 업데이트 완료")
        
        # 4. 캐시 갱신
        try:
            load_persons_from_db(db)
            print(f"  ✅ 캐시 갱신 완료")
        except Exception as cache_error:
            print(f"  ⚠️ 캐시 갱신 실패: {cache_error}")
        
        # 5. 응답 반환
        return {
            "status": "success",
            "message": f"인물 정보가 수정되었습니다",
            "person": {
                "id": person.person_id,
                "name": person.name,
                "person_type": person.info.get('person_type', 'unknown') if person.info else 'unknown',
                "is_criminal": person.is_criminal
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ [UPDATE] 인물 수정 실패: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"수정 중 오류 발생: {str(e)}")

@router.post("/api/enroll")
async def enroll_person(
    person_id: str = Form(...),
    name: str = Form(...),
    person_type: str = Form("criminal"),  # "criminal", "missing", "dementia", "child", "wanted"
    image: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    인물 등록 API - 정면 사진에서 얼굴 임베딩 추출 및 저장
    
    Args:
        person_id: 인물 고유 ID (자동 생성됨)
        name: 인물 이름
        person_type: 인물 타입 ("criminal", "missing", "dementia", "child", "wanted")
        image: 정면 사진 파일 (JPEG, PNG 등)
        db: 데이터베이스 세션
    
    Returns:
        {
            "success": bool,
            "message": str,
            "person_id": str,
            "name": str,
            "embedding_count": int
        }
    """
    
    try:
        # is_criminal 결정 (criminal, wanted=True, 나머지=False)
        # 강력 범죄자와 지명 수배자는 범죄자로 분류
        is_criminal = (person_type in ["criminal", "wanted"])
        print(f"📝 [ENROLL] 인물 등록 요청: person_id={person_id}, name={name}, type={person_type}, is_criminal={is_criminal}")
        
        # 이미지 파일 읽기
        image_bytes = await image.read()
        
        # 등록 이미지 저장 경로 (images/enroll/{person_id}/)
        enroll_dir = PROJECT_ROOT / "images" / "enroll" / person_id
        enroll_dir.mkdir(parents=True, exist_ok=True)
        
        # 이미지 파일 확장자 결정
        file_extension = Path(image.filename).suffix if image.filename else ".jpg"
        if not file_extension or file_extension not in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
            file_extension = ".jpg"
        
        # 이미지 파일 저장 (person_id를 파일명으로 사용)
        saved_image_path = enroll_dir / f"{person_id}{file_extension}"
        with open(saved_image_path, "wb") as f:
            f.write(image_bytes)
        
        print(f"  💾 이미지 저장: {saved_image_path}")
        
        # face_enroll.py의 함수를 사용하여 임베딩 추출
        embedding_normalized = get_main_face_embedding(get_model(), saved_image_path)
        
        if embedding_normalized is None:
            # 이미지 파일 삭제 (얼굴 감지 실패 시)
            if saved_image_path.exists():
                saved_image_path.unlink()
            raise HTTPException(status_code=400, detail="이미지에서 얼굴을 감지할 수 없습니다. 정면 사진을 업로드해주세요.")
        
        # Bank 저장 경로
        person_dir = EMBEDDINGS_DIR / person_id
        person_dir.mkdir(parents=True, exist_ok=True)
        bank_base_path = person_dir / "bank_base.npy"
        
        # 기존 bank_base.npy 로드 (중복 체크용)
        existing_bank = None
        if bank_base_path.exists():
            existing_bank = np.load(bank_base_path)
            if existing_bank.ndim == 1:
                existing_bank = existing_bank.reshape(1, -1)
            
            # 중복 체크 (유사도 0.95 이상이면 스킵)
            BANK_DUPLICATE_THRESHOLD = 0.95
            max_sim = float(np.max(existing_bank @ embedding_normalized))
            if max_sim >= BANK_DUPLICATE_THRESHOLD:
                return {
                    "success": False,
                    "message": f"이미 등록된 얼굴과 유사도가 너무 높습니다 (유사도: {max_sim:.3f}). 새로운 사진을 업로드해주세요.",
                    "person_id": person_id,
                    "name": name,
                    "embedding_count": existing_bank.shape[0]
                }
        
        # 기존 person 확인
        existing_person = get_person_by_id(db, person_id)
        
        if existing_person:
            # 기존 인물 업데이트
            print(f"  🔄 기존 인물 업데이트: {person_id}")
            
            # Bank에 추가 (기존 bank가 있으면 추가, 없으면 새로 생성)
            if existing_bank is not None:
                updated_bank = np.vstack([existing_bank, embedding_normalized.reshape(1, -1)])
            else:
                updated_bank = embedding_normalized.reshape(1, -1)
            
            # bank_base.npy 저장
            np.save(bank_base_path, updated_bank)
            
            # Centroid 재계산 및 저장
            centroid = updated_bank.mean(axis=0)
            centroid = l2_normalize(centroid)
            centroid_base_path = person_dir / "centroid_base.npy"
            np.save(centroid_base_path, centroid)
            
            # Backward compatibility: centroid.npy도 업데이트
            # 레거시 파일은 gallery_loader.py에서 fallback으로 사용될 수 있음
            legacy_centroid_path = person_dir / "centroid.npy"
            np.save(legacy_centroid_path, centroid)
            
            # 데이터베이스 업데이트 (person_type을 info에 저장)
            existing_person.name = name
            existing_person.is_criminal = is_criminal
            if not existing_person.info:
                existing_person.info = {}
            existing_person.info["person_type"] = person_type
            existing_person.info["category"] = person_type
            existing_person.set_embedding(centroid)  # centroid를 대표 임베딩으로 사용
            db.commit()
            db.refresh(existing_person)
            
            embedding_count = updated_bank.shape[0]
            print(f"  ✅ Bank 업데이트 완료: {person_id} (총 {embedding_count}개 임베딩)")
        else:
            # 새 인물 등록 - face_enroll.py의 save_embeddings 함수 사용
            print(f"  ✨ 새 인물 등록: {person_id}")
            
            # face_enroll.py의 save_embeddings 함수 사용 (bank_base.npy와 centroid_base.npy 저장)
            save_embeddings(person_id, [embedding_normalized], EMBEDDINGS_DIR, save_bank=True, save_centroid=True)
            
            # Centroid는 save_embeddings에서 이미 저장됨
            centroid = embedding_normalized  # 단일 임베딩이므로 그대로 사용
            
            # 데이터베이스에 저장 (person_type을 info에 저장)
            from backend.database import create_person
            info = {"person_type": person_type, "category": person_type}
            create_person(db, person_id, name, centroid, is_criminal=is_criminal, info=info)
            
            embedding_count = 1
            print(f"  ✅ 새 인물 등록 완료: {person_id}")
        
        # 캐시 갱신
        try:
            load_persons_from_db(db)
            print(f"  ✅ 캐시 갱신 완료")
        except Exception as cache_error:
            print(f"  ⚠️ 캐시 갱신 실패: {cache_error}")
        
        return {
            "success": True,
            "message": f"{'업데이트' if existing_person else '등록'} 완료: {name} ({person_id})",
            "person_id": person_id,
            "name": name,
            "embedding_count": embedding_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ [ENROLL] 등록 실패: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"등록 중 오류 발생: {str(e)}")
