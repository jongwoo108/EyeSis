"""
FaceWatch FastAPI 백엔드 서버
웹 프론트엔드와 연동하여 실시간 얼굴 인식 서비스 제공
PostgreSQL 데이터베이스 사용
"""
import base64
import cv2
import numpy as np

import shutil
from typing import Optional, List, Dict, Set
from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
from sqlalchemy.orm import Session

import json
import asyncio
import subprocess
import tempfile
import os
import time

import sys

from backend.utils.bbox_utils import (
    calculate_bbox_iou,
    calculate_bbox_center_distance,
    is_same_face_region
)
from backend.utils.image_utils import (
    l2_normalize,
    compute_cosine_similarity,
    preprocess_image_for_detection,
    base64_to_image,
    image_to_base64
)
from backend.utils.websocket_manager import(
    active_connections,
    connection_states,
    register_connection,
    unregister_connection
)
from backend.services import data_loader
from backend.services.data_loader import (
    load_persons_from_db,
    load_persons_from_embeddings,
    load_persons_from_legacy_files,
    find_person_info
)
from backend.services.bank_manager import(
    save_angle_separated_banks,
    add_embedding_to_bank_async,
    add_embedding_to_dynamic_bank_async,
    update_gallery_cache_in_memory
)
from backend.services.temporal_filter import apply_temporal_filter
from backend.services.face_detection import process_detection
from backend.models.schemas import DetectionRequest


# 프로젝트 루트를 Python 경로에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# CUDA 경로를 먼저 설정 (가장 먼저 import)
from src.utils.device_config import _ensure_cuda_in_path
_ensure_cuda_in_path()

from insightface.app import FaceAnalysis
from src.utils.device_config import get_device_id, safe_prepare_insightface
from src.utils.gallery_loader import load_gallery, match_with_bank, match_with_bank_detailed
from src.utils.face_angle_detector import estimate_face_angle, is_diverse_angle, is_all_angles_collected, check_face_occlusion
from src.utils.mask_detector import estimate_mask_from_similarity, get_adjusted_threshold, estimate_face_quality
from src.face_enroll import get_main_face_embedding, save_embeddings, l2_normalize

# PostgreSQL 데이터베이스 모듈
from backend.database import (
    get_db, get_all_persons, get_person_by_id,
    log_detection, init_db as db_init, Person
)

# ==========================================
# 1. 설정 및 경로
# ==========================================


# Masked Bank 관련 설정
MASKED_BANK_MASK_PROB_THRESHOLD = 0.5  # mask_prob >= 0.5이면 masked bank로 분류 (완화: 0.7 → 0.5)
MASKED_CANDIDATE_MIN_SIM = 0.25  # base_sim >= 0.25 이상이어야 masked candidate로 판단 (완화: 0.30 → 0.25)
MASKED_CANDIDATE_MIN_FRAMES = 3  # 연속 N 프레임 이상 조건 충족 시 masked bank에 추가 (완화: 5 → 3)
MASKED_TRACKING_IOU_THRESHOLD = 0.5  # bbox tracking을 위한 IoU 임계값

# ==========================================
# 2. FastAPI 앱 초기화
# ==========================================

app = FastAPI(title="FaceWatch API", version="1.0.0")

# CORS 허용 (프론트엔드 접근 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 라우터 등록
from backend.api import detection
app.include_router(detection.router, tags=["detection"])

# ==========================================
# 3. InsightFace 모델 초기화 (device_config 사용)
# ==========================================

print("=" * 70)
print("🔧 InsightFace 모델 초기화 중...")
print("=" * 70)

device_id = get_device_id()
device_type = "GPU" if device_id >= 0 else "CPU"
print(f"디바이스: {device_type} (ctx_id={device_id})")

model = FaceAnalysis(name="buffalo_l")
actual_device_id = safe_prepare_insightface(model, device_id, det_size=(640, 640))
if actual_device_id != device_id:
    print(f"   (실제 사용: {'GPU' if actual_device_id >= 0 else 'CPU'})")
print()

# Inject model into face_detection module
from backend.services import face_detection
face_detection.set_model(model)



@app.on_event("startup")
async def startup_event():
    """서버 시작 시 데이터베이스 초기화 및 데이터 로드"""
    print("=" * 70)
    print("🚀 FaceWatch 서버 시작")
    print("=" * 70)
    print("📡 WebSocket 엔드포인트:")
    print("   - /ws/detect (메인 감지 엔드포인트)")
    print("   - /ws/test (테스트 엔드포인트)")
    print("=" * 70)
    
    # 1. 데이터베이스 테이블 생성 (없으면 생성)
    try:
        db_init()
    except Exception as e:
        print(f"⚠️ 데이터베이스 초기화 오류: {e}")
        print("   outputs/embeddings를 사용합니다.")
    
    # 2. PostgreSQL에서 데이터 로드 시도
    try:
        db = next(get_db())
        try:
            load_persons_from_db(db)
        finally:
            db.close()
    except Exception as e:
        print(f"⚠️ PostgreSQL 연결 실패: {e}")
        print("   outputs/embeddings를 사용합니다.")
        # Fallback: outputs/embeddings 사용
        load_persons_from_embeddings()
    
    # 3. 데이터가 없으면 경고
    if not data_loader.gallery_base_cache and not data_loader.persons_cache:
        print("⚠️ 경고: 등록된 얼굴 데이터가 없습니다!")
        print("   face_enroll.py를 실행하여 인물을 등록하거나,")
        print("   python backend/init_db.py를 실행하여 데이터를 마이그레이션해주세요.\n")




# ==========================================
# 7. 공통 감지 로직 함수
# ==========================================


# ==========================================
# 8. API 엔드포인트
# ==========================================





@app.get("/api/persons")
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

@app.delete("/api/persons/{person_id}")
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

@app.put("/api/persons/{person_id}")
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
    persons_cache
    
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


@app.get("/api/logs")
async def get_logs(limit: int = 100, db: Session = Depends(get_db)):
    """감지 로그 조회"""
    from backend.database import DetectionLog
    try:
        logs = db.query(DetectionLog).order_by(DetectionLog.detected_at.desc()).limit(limit).all()
        return {
            "success": True,
            "count": len(logs),
            "logs": [
                {
                    "id": log.id,
                    "person_id": log.person_id,
                    "person_name": log.person_name,
                    "similarity": log.similarity,
                    "is_criminal": log.is_criminal,
                    "status": log.status,
                    "detected_at": log.detected_at.isoformat(),
                    "metadata": log.detection_metadata
                }
                for log in logs
            ]
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "count": 0,
            "logs": []
        }

@app.post("/api/enroll")
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
    persons_cache, data_loader.gallery_base_cache, data_loader.gallery_masked_cache
    
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
        embedding_normalized = get_main_face_embedding(model, saved_image_path)
        
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

@app.get("/api/images/enroll/{person_id}/{filename}")
async def get_person_image(person_id: str, filename: str):
    """등록된 인물의 이미지 제공"""
    image_path = PROJECT_ROOT / "images" / "enroll" / person_id / filename
    
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="이미지를 찾을 수 없습니다.")
    
    # 보안 체크: person_id와 filename이 일치하는지 확인
    if image_path.parent.name != person_id:
        raise HTTPException(status_code=403, detail="접근 권한이 없습니다.")
    
    return FileResponse(image_path)

@app.post("/api/extract_frames")
async def extract_frames(
    video: UploadFile = File(...)
):
    """
    비디오 파일에서 모든 프레임을 추출하여 저장 (라벨링용)
    
    Args:
        video: 비디오 파일
    
    Returns:
        {
            "success": bool,
            "message": str,
            "total_frames": int,
            "output_dir": str
        }
    """
    try:
        print(f"📹 [EXTRACT FRAMES] 프레임 추출 요청: {video.filename}")
        
        # 임시 파일로 비디오 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as input_file:
            input_path = input_file.name
            content = await video.read()
            input_file.write(content)
        
        # 출력 디렉토리 생성 (비디오 파일명 기반)
        video_name = Path(video.filename).stem if video.filename else f"video_{int(time.time())}"
        output_dir = PROJECT_ROOT / "outputs" / "extracted_frames" / video_name
        annotations_dir = output_dir / "annotations"  # JSON 파일 저장 폴더
        output_dir.mkdir(parents=True, exist_ok=True)
        annotations_dir.mkdir(parents=True, exist_ok=True)
        
        # OpenCV로 비디오 열기
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="비디오 파일을 열 수 없습니다.")
        
        # 비디오 정보
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"  📊 비디오 정보:")
        print(f"     - 총 프레임: {total_frames}")
        print(f"     - FPS: {fps:.2f}")
        print(f"     - 해상도: {width}x{height}")
        print(f"     - 출력 디렉토리: {output_dir}")
        print(f"  🔍 얼굴 감지 및 매칭 결과 박스 그리기 활성화")
        
        # DB 세션 생성 (매칭을 위해 필요)
        from backend.database import SessionLocal
        db = SessionLocal()
        
        try:
            # 모든 프레임 추출 (매칭 결과 포함 박스 그리기)
            frame_idx = 0
            saved_count = 0
            total_faces_detected = 0
            total_matches = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 매칭 로직 실행 (브라우저에서 보는 것과 동일한 로직)
                # tracking_state 초기화 (tracks 키 필요)
                tracking_state = {"tracks": {}}
                
                detection_result = process_detection(
                    frame=frame,
                    suspect_ids=None,  # 전체 갤러리 검색
                    db=db,
                    tracking_state=tracking_state  # 프레임별로 독립적으로 처리
                )
                
                # 박스가 그려진 프레임 복사
                frame_with_boxes = frame.copy()
                
                # 매칭 결과에 따라 박스 그리기 및 JSON 데이터 수집
                detections = detection_result.get("detections", [])
                frame_annotations = {
                    "frame_idx": frame_idx,
                    "timestamp": frame_idx / fps if fps > 0 else 0.0,
                    "faces": []
                }
                
                for detection in detections:
                    bbox = detection["bbox"]
                    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                    
                    # 색상 결정 (브라우저와 동일한 로직)
                    status = detection.get("status", "unknown")
                    if status == "criminal":
                        color = (0, 0, 255)  # 빨간색 (BGR)
                        label_color = (0, 0, 255)
                    elif status == "normal":
                        color = (0, 255, 0)  # 초록색 (BGR)
                        label_color = (0, 255, 0)
                    else:  # unknown
                        color = (0, 255, 255)  # 노란색 (BGR)
                        label_color = (0, 255, 255)
                    
                    # 박스 그리기 (두께 3)
                    cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), color, 3)
                    
                    # 레이블 생성 (브라우저와 동일한 정보)
                    name = detection.get("name", "Unknown")
                    confidence = detection.get("confidence", 0)
                    label = f"{name} ({confidence}%)"
                    
                    # 레이블 배경 (가독성 향상)
                    (label_width, label_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    cv2.rectangle(
                        frame_with_boxes,
                        (x1, y1 - label_height - 10),
                        (x1 + label_width, y1),
                        color,
                        -1  # 채워진 사각형
                    )
                    
                    # 레이블 텍스트 (흰색)
                    cv2.putText(
                        frame_with_boxes,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),  # 흰색
                        2
                    )
                    
                    # JSON 어노테이션 데이터 수집
                    face_annotation = {
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "status": status,
                        "name": name,
                        "person_id": detection.get("person_id"),
                        "confidence": confidence,
                        "color": detection.get("color", "yellow"),
                        "angle_type": detection.get("angle_type"),
                        "yaw_angle": detection.get("yaw_angle"),
                        "bank_type": detection.get("bank_type")
                    }
                    frame_annotations["faces"].append(face_annotation)
                    
                    total_faces_detected += 1
                    if detection.get("status") != "unknown":
                        total_matches += 1
                
                # 프레임 저장 (JPEG 형식, 매칭 결과 박스가 그려진 이미지)
                frame_filename = f"frame_{frame_idx:06d}.jpg"
                frame_path = output_dir / frame_filename
                cv2.imwrite(str(frame_path), frame_with_boxes, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                # JSON 어노테이션 저장 (이미지 파일과 쌍으로 저장)
                json_filename = f"frame_{frame_idx:06d}.json"
                json_path = annotations_dir / json_filename
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(frame_annotations, f, indent=2, ensure_ascii=False)
                
                saved_count += 1
                
                # 진행 상황 출력 (100프레임마다)
                if frame_idx % 100 == 0:
                    progress = (frame_idx / total_frames * 100) if total_frames > 0 else 0
                    print(f"  ⏳ 진행 중: {frame_idx}/{total_frames} 프레임 ({progress:.1f}%), 감지된 얼굴: {total_faces_detected}개, 매칭: {total_matches}개")
                
                frame_idx += 1
        finally:
            db.close()
        
        cap.release()
        
        # 임시 파일 삭제
        try:
            os.unlink(input_path)
        except:
            pass
        
        print(f"  ✅ 프레임 추출 완료: {saved_count}개 프레임 저장됨")
        print(f"  👤 총 감지된 얼굴: {total_faces_detected}개")
        print(f"  ✅ 매칭 성공: {total_matches}개")
        print(f"  📁 이미지 저장 위치: {output_dir}")
        print(f"  📄 JSON 저장 위치: {annotations_dir}")
        
        return {
            "success": True,
            "message": f"{saved_count}개의 프레임이 추출되었습니다. (감지된 얼굴: {total_faces_detected}개, 매칭: {total_matches}개)",
            "total_frames": saved_count,
            "total_faces": total_faces_detected,
            "total_matches": total_matches,
            "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
            "annotations_dir": str(annotations_dir.relative_to(PROJECT_ROOT)),
            "video_info": {
                "fps": fps,
                "width": width,
                "height": height,
                "duration": total_frames / fps if fps > 0 else 0
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ [EXTRACT FRAMES] 프레임 추출 실패: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"프레임 추출 중 오류 발생: {str(e)}")

@app.post("/api/extract_clip")
async def extract_clip(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    start_time: float = Form(...),
    end_time: float = Form(...),
    person_name: str = Form("Unknown")
):
    """
    비디오 파일에서 특정 구간을 추출하여 클립 생성
    
    Args:
        video: 비디오 파일
        start_time: 시작 시간 (초)
        end_time: 종료 시간 (초)
        person_name: 범죄자 이름
    
    Returns:
        추출된 클립 파일
    """
    try:
        # 임시 파일 생성
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as input_file:
            input_path = input_file.name
            # 업로드된 비디오 파일 저장
            content = await video.read()
            input_file.write(content)
        
        # 출력 파일 경로
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        
        # ffmpeg를 사용하여 클립 추출
        duration = end_time - start_time
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-ss', str(start_time),
            '-t', str(duration),
            '-c', 'copy',  # 재인코딩 없이 복사 (빠름)
            '-avoid_negative_ts', 'make_zero',
            '-y',  # 덮어쓰기
            output_path
        ]
        
        print(f"🎬 클립 추출 시작: {person_name} ({start_time:.1f}s - {end_time:.1f}s)")
        print(f"📝 명령어: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60  # 60초 타임아웃
        )
        
        if result.returncode != 0:
            print(f"❌ ffmpeg 오류: {result.stderr}")
            raise HTTPException(status_code=500, detail=f"클립 추출 실패: {result.stderr}")
        
        # 임시 입력 파일 삭제
        try:
            os.unlink(input_path)
        except:
            pass
        
        print(f"✅ 클립 추출 완료: {output_path}")
        
        # 응답 후 파일 삭제를 BackgroundTasks로 등록
        background_tasks.add_task(os.unlink, output_path)
        
        # 파일 응답 반환
        return FileResponse(
            output_path,
            media_type='video/mp4',
            filename=f"clip_{person_name}_{start_time:.1f}s-{end_time:.1f}s.mp4"
        )
        
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="클립 추출 시간 초과")
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="ffmpeg가 설치되지 않았습니다. ffmpeg를 설치해주세요.")
    except Exception as e:
        print(f"❌ 클립 추출 오류: {e}")
        raise HTTPException(status_code=500, detail=f"클립 추출 실패: {str(e)}")
    finally:
        # 임시 파일 정리
        try:
            if 'input_path' in locals():
                os.unlink(input_path)
        except:
            pass

# ==========================================
# Static Files 마운트 (프론트엔드 서빙)
# ==========================================
# web 폴더의 정적 파일들을 루트 경로로 서빙
# 이렇게 하면 ngrok으로 외부 접속 시에도 하나의 URL로 통합 가능
web_dir = PROJECT_ROOT / "web"
app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="static")

# 실행 명령: uvicorn backend.main:app --reload --host 0.0.0.0 --port 5000
