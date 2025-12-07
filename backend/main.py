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
# 6.6. Temporal Consistency 필터 함수
# ==========================================

def apply_temporal_filter(websocket: WebSocket, result: Dict) -> Dict:
    """
    개선된 Temporal Filter: Hysteresis 임계값 + 윈도우 기반 투표
    
    기존 문제: 연속 3프레임 매칭 요구 → 임계값 근처에서 깜빡임
    개선 방안:
    1. Hysteresis 임계값: 시작(0%) vs 유지(-3%) → 안정성 향상
    2. 윈도우 기반 투표: 최근 5프레임 중 3프레임 매칭 → 일시적 실패 무시
    
    Args:
        websocket: WebSocket 연결 객체
        result: process_detection의 반환값
    
    Returns:
        temporal filter가 적용된 result
    """
    # 설정
    WINDOW_SIZE = 5              # 최근 N프레임 추적
    MIN_MATCHES = 1              # 필요한 최소 매칭 수 (2 → 1: 최대 안정성)
    MATCH_KEEP_OFFSET = -0.05    # 유지 임계값 완화 (-5%) - 왼쪽 얼굴 인식 개선
    
    if websocket not in connection_states:
        return result
    
    state = connection_states[websocket]
    match_history = state.get("match_history", {})  # {person_id: [(confidence, matched), ...]}
    
    # 현재 프레임의 detection을 person_id별로 매핑
    current_detections = {}
    for det in result.get("detections", []):
        person_id = det.get("person_id")
        if person_id:
            current_detections[person_id] = det
    
    # 사라진 person_id의 히스토리 정리
    for person_id in list(match_history.keys()):
        if person_id not in current_detections:
            del match_history[person_id]
    
    # 각 detection에 temporal filter 적용
    filtered_detections = []
    alert_triggered = False
    detected_metadata = result.get("metadata", {"name": "미상", "confidence": 0, "status": "unknown"})
    
    for det in result.get("detections", []):
        person_id = det.get("person_id")
        status = det.get("status", "unknown")
        confidence = det.get("confidence", 0) / 100.0  # 0-1 범위로 변환
        
        # criminal 또는 normal 상태이고 person_id가 있는 경우만 temporal filter 적용
        if status in ["criminal", "normal"] and person_id:
            # 이력 가져오기
            history = match_history.get(person_id, [])
            
            # 현재 안정 상태 확인 (최근 MIN_MATCHES 프레임이 모두 매칭)
            is_currently_stable = False
            if len(history) >= MIN_MATCHES:
                recent_matches = [matched for _, matched in history[-MIN_MATCHES:]]
                is_currently_stable = all(recent_matches)
            
            # Hysteresis: 안정 상태면 낮은 임계값 사용
            base_threshold = 0.48  # 기본값
            if is_currently_stable:
                effective_threshold = base_threshold + MATCH_KEEP_OFFSET  # 0.45
            else:
                effective_threshold = base_threshold  # 0.48
            
            # 현재 프레임 매칭 여부 판단
            is_matched = confidence >= effective_threshold
            
            # 이력에 추가
            history.append((confidence, is_matched))
            
            # 윈도우 크기 유지
            if len(history) > WINDOW_SIZE:
                history = history[-WINDOW_SIZE:]
            
            match_history[person_id] = history
            
            # 투표: 최근 N프레임 중 M프레임 이상 매칭?
            matched_count = sum(matched for _, matched in history)
            is_stable = matched_count >= MIN_MATCHES
            
            # 디버그 로그
            history_str = "".join(["O" if m else "X" for _, m in history])
            
            if not is_stable:
                # 아직 불안정 → Unknown
                filtered_det = det.copy()
                filtered_det["status"] = "unknown"
                filtered_det["color"] = "yellow"
                filtered_det["name"] = "Unknown"
                filtered_detections.append(filtered_det)
                print(f"   🔄 [TEMPORAL] {person_id[-6:]}: [{history_str}] {matched_count}/{len(history)} → Unknown (th={effective_threshold:.2f})")
            else:
                # 안정 → 원래 상태 유지
                filtered_detections.append(det)
                print(f"   ✅ [TEMPORAL] {person_id[-6:]}: [{history_str}] {matched_count}/{len(history)} → Stable (th={effective_threshold:.2f})")
                
                if status == "criminal":
                    alert_triggered = True
                    detected_metadata = {
                        "name": det.get("name", "Unknown"),
                        "confidence": det.get("confidence", 0),
                        "status": "criminal"
                    }
                elif not alert_triggered:
                    detected_metadata = {
                        "name": det.get("name", "Unknown"),
                        "confidence": det.get("confidence", 0),
                        "status": "normal"
                    }
        else:
            # unknown 상태는 그대로 유지
            filtered_detections.append(det)
    
    # match_history 업데이트
    state["match_history"] = match_history
    
    return {
        "detections": filtered_detections,
        "alert": alert_triggered,
        "metadata": detected_metadata,
        "learning_events": result.get("learning_events", [])
    }

# ==========================================
# 7. 공통 감지 로직 함수
# ==========================================

def process_detection(frame: np.ndarray, suspect_id: Optional[str] = None, suspect_ids: Optional[List[str]] = None, db: Optional[Session] = None, tracking_state: Optional[Dict] = None) -> Dict:
    """
    공통 얼굴 감지 및 인식 로직
    
    Args:
        frame: BGR 이미지 (numpy array)
        suspect_id: 선택적 타겟 ID (단일, 호환성 유지)
        suspect_ids: 선택적 타겟 ID 배열 (여러 명 선택 시)
        db: 데이터베이스 세션 (로그 저장용, None이면 로그 저장 안함)
        tracking_state: bbox tracking 상태 (None이면 자동 생성)
    
    Returns:
        {
            "detections": [...],  # 박스 좌표 및 메타데이터 배열
            "alert": bool,        # 범죄자 감지 여부
            "metadata": {...}      # 주요 감지 정보
        }
    """
    # suspect_ids가 없으면 suspect_id를 배열로 변환
    if suspect_ids is None:
        suspect_ids = [suspect_id] if suspect_id else []
    
    # tracking_state 초기화 (없으면 생성)
    if tracking_state is None:
        tracking_state = {
            "tracks": {}  # {track_id: {"bbox": [...], "person_id": str, "frames": int, "embeddings": [...], "last_frame": int}}
        }
    
    # 1. 저화질 영상 전처리 (업스케일링 및 샤프닝)
    original_height, original_width = frame.shape[:2]
    processed_frame = preprocess_image_for_detection(frame, min_size=640)
    processed_height, processed_width = processed_frame.shape[:2]
    
    # 스케일 비율 계산 (박스 좌표 변환용)
    scale_x = original_width / processed_width
    scale_y = original_height / processed_height

    # 2. InsightFace로 얼굴 탐지 및 특징 추출 (전처리된 이미지 사용)
    faces = model.get(processed_frame)
    
    # 얼굴 감지 개수 로그 출력 (디버깅용)
    print(f"🔍 [얼굴 감지] 감지된 얼굴 개수: {len(faces)}")
    if suspect_ids:
        print(f"   - suspect_ids 모드: {suspect_ids}")
    else:
        print(f"   - 전체 갤러리 모드")
    
    alert_triggered = False
    detected_metadata = {"name": "미상", "confidence": 0, "status": "unknown"}
    detections = []  # 박스 좌표 및 메타데이터 배열
    learning_events = []  # 학습 이벤트 (UI 피드백용)

    # 3. 먼저 모든 얼굴에 대해 매칭 결과 수집 (오인식 방지 필터링을 위해)
    face_results = []
    face_objects = []  # face 객체를 인덱스로 매핑하여 저장 (Dynamic Bank 검증용)
    for face in faces:
        # 바운딩 박스 좌표 (정수형 변환)
        # 전처리된 이미지의 좌표를 원본 이미지 좌표로 변환
        box = face.bbox.astype(float)
        box[0] *= scale_x  # x1
        box[1] *= scale_y  # y1
        box[2] *= scale_x  # x2
        box[3] *= scale_y  # y2
        box = box.astype(int)
        
        embedding = face.embedding.astype("float32")
        embedding_normalized = l2_normalize(embedding)
        
        # 얼굴 각도 추정
        angle_type, yaw_angle = estimate_face_angle(face)
        
        # 화질 추정
        face_quality = estimate_face_quality(box, (original_height, original_width))
        
        # Base Bank, Masked Bank, Dynamic Bank 각각 매칭 (분리 계산)
        base_sim = 0.0
        masked_sim = 0.0
        # Bank 매칭 결과 초기화
        best_base_person_id = "unknown"
        best_mask_person_id = "unknown"
        best_dynamic_person_id = "unknown"
        base_sim = 0.0
        masked_sim = 0.0
        dynamic_sim = 0.0
        second_base_sim = 0.0
        second_mask_sim = 0.0
        second_dynamic_sim = 0.0
        bank_type = "unknown"  # ← 초기화 추가!
        
        # suspect_ids가 지정된 경우: 선택된 용의자들만 검색 (전체 DB 검색 안 함)
        if suspect_ids and len(suspect_ids) > 0:
            # 선택된 용의자들만 포함한 base/masked/dynamic 갤러리 생성
            target_base_gallery = {}
            target_masked_gallery = {}
            target_dynamic_gallery = {}
            for sid in suspect_ids:
                if sid in data_loader.gallery_base_cache:
                    target_base_gallery[sid] = data_loader.gallery_base_cache[sid]
                if sid in data_loader.gallery_masked_cache:
                    target_masked_gallery[sid] = data_loader.gallery_masked_cache[sid]
                if sid in data_loader.gallery_dynamic_cache:
                    target_dynamic_gallery[sid] = data_loader.gallery_dynamic_cache[sid]
            
            # Base Bank 매칭
            if target_base_gallery:
                best_base_person_id, base_sim, second_base_sim = match_with_bank_detailed(embedding, target_base_gallery)
            
            # Masked Bank 매칭
            if target_masked_gallery:
                best_mask_person_id, masked_sim, second_mask_sim = match_with_bank_detailed(embedding, target_masked_gallery)
            
            # Dynamic Bank 매칭 (인식용)
            if target_dynamic_gallery:
                best_dynamic_person_id, dynamic_sim, second_dynamic_sim = match_with_bank_detailed(embedding, target_dynamic_gallery)
            
            # 디버깅: 갤러리 상태 확인
            print(f"   📊 [GALLERY] base={len(target_base_gallery)}, masked={len(target_masked_gallery)}, dynamic={len(target_dynamic_gallery)}")
        
        # suspect_ids가 없거나 비어있는 경우: 매칭 시도하지 않음 (모든 얼굴을 unknown으로 처리)
        else:
            # 인물이 선택되지 않았으므로 매칭을 시도하지 않음
            # best_person_id는 None으로 유지되고, 아래 로직에서 unknown으로 처리됨
            print(f"   - 인물 미선택 모드: 모든 얼굴을 unknown으로 처리")
        
        # =========================================================
        # 2단계 개선: 가중치 기반 매칭 (Weighted Voting) - v2
        # =========================================================
        # 기존 문제: 승자 독식(Winner-Takes-All) 방식
        # - Dynamic Bank가 Base보다 약간만 높아도 무조건 Dynamic 선택
        # - Base Bank(원본 사진)과 유사도가 낮아도 Dynamic/Masked가 높으면 매칭됨
        # 
        # 개선: 가중치 기반 투표 + Base Bank 기준 보정
        # - Base Bank를 Golden Standard로 간주하여 가중치 가장 높게 설정
        # - Base 유사도가 너무 낮으면 다른 Bank 점수도 크게 깎음
        
        # 1. 가중치 상수 정의
        W_BASE = 1.0      # Base Bank: 가장 신뢰 (원본 등록 사진)
        W_DYNAMIC = 0.9   # Dynamic Bank: 높은 신뢰 (0.8 → 0.9: 수집된 옆모습 우선)
        W_MASKED = 0.7    # Masked Bank: 중간 신뢰 (0.6 → 0.7: 마스크/모자 인식 개선)
        
        # [근본 해결] 랜드마크 기반 실제 Occlusion 감지
        # 유사도가 아닌 실제 얼굴 구조로 마스크 착용 여부 확인
        # 
        # 주의: check_face_occlusion 반환값
        #   True = occlusion 없음 (얼굴 전체 보임, 마스크 없음)
        #   False = occlusion 있음 (얼굴 가려짐, 마스크 있음)
        is_face_clear = check_face_occlusion(face, box)
        is_masked = not is_face_clear  # 가려지지 않으면 마스크 없음
        
        # mask_prob는 이제 실제 occlusion 결과에 기반
        if is_masked:
            # 얼굴이 가려져 있음 (마스크 착용)
            mask_prob = 0.9
            print(f"   🎭 [MASKED] 감지됨 (랜드마크 기반, is_face_clear={is_face_clear})")
        else:
            # 얼굴 전체가 보임 (마스크 없음)
            mask_prob = 0.0
        
        # 2. Base Bank 기준 보정 로직 - 단계별 보정 강화
        # Base 유사도가 너무 낮으면 다른 Bank의 높은 점수도 신뢰하지 않음
        # [근본 해결] 단계별 페널티로 Masked/Dynamic Bank 오염 방지
        
        if base_sim < 0.3:
            # [근본 해결] 실제 occlusion 확인 + Masked Bank 유사도 높으면 예외
            # 이전 문제: 유사도만으로 마스크 추정 → 오인식
            # 현재 해결: 실제 랜드마크로 마스크 확인 → 정확
            if is_masked and masked_sim >= 0.50:  # [완화] 0.92 → 0.50: 마스크 인식 개선
                # 실제로 마스크 쓴 것 확인됨 + Masked Bank 중간 이상 → 페널티 면제
                penalty_factor = 1.0
                print(f"   ✅ [MASKED 예외] 실제 마스크 확인, masked_sim={masked_sim:.3f} → penalty 면제")
            elif dynamic_sim >= 0.60:  # [신규] Dynamic Bank 예외: 이미 수집된 옆얼굴 활용
                # Dynamic Bank와 높은 유사도 → 이미 학습된 각도 → 페널티 면제
                penalty_factor = 1.0
                print(f"   ✅ [DYNAMIC 예외] Dynamic Bank 높은 유사도, dynamic_sim={dynamic_sim:.3f} → penalty 면제")
            else:
                # 마스크 안 쓰고 Base 낮음 = 다른 사람 → 보정 (완화)
                penalty_factor = 0.7  # 40% → 70%: 페널티 완화
                if not is_masked:
                    print(f"   ⚠️ [BASE 보정] 마스크 없음, base_sim={base_sim:.3f} < 0.3 → penalty=70%")
                else:
                    print(f"   ⚠️ [BASE 보정] masked_sim={masked_sim:.3f} < 0.50, dynamic_sim={dynamic_sim:.3f} < 0.60 → penalty=70%")
        elif base_sim < 0.5:
            # Base가 낮으면 (50% 미만) 보정 적용 (완화)
            penalty_factor = 0.7  # 50% → 70%: 페널티 완화
            print(f"   ⚠️ [BASE 보정] base_sim={base_sim:.3f} < 0.5 → penalty=70%")
        else:
            # Base가 충분하면 정상 가중치 적용
            penalty_factor = 1.0
        
        # 페널티 적용
        confident_base = base_sim * W_BASE
        confident_dynamic = dynamic_sim * W_DYNAMIC * penalty_factor
        confident_masked = masked_sim * W_MASKED * penalty_factor
        
        # 가장 높은 점수 선택
        scores = [
            (confident_base, best_base_person_id, second_base_sim, "base"),
            (confident_dynamic, best_dynamic_person_id, second_dynamic_sim, "dynamic"),
            (confident_masked, best_mask_person_id, second_mask_sim, "masked")
        ]
        
        # 3. 최고 점수 선택
        scores.sort(key=lambda x: x[0], reverse=True)
        max_similarity, best_person_id, second_similarity, bank_type = scores[0]
        
        # 유사도는 1.0을 넘을 수 없음
        max_similarity = min(max_similarity, 1.0)
        second_similarity = second_similarity if second_similarity > 0 else 0.0
        
        # best_match 찾기
        # suspect_ids가 없거나 비어있으면 매칭을 시도하지 않음 (unknown으로 처리)
        if not suspect_ids or len(suspect_ids) == 0:
            best_match = None
            best_person_id = "unknown"
            max_similarity = 0.0
            second_similarity = 0.0
        elif best_person_id != "unknown" and max_similarity > 0:
            best_match = find_person_info(best_person_id)
        else:
            # suspect_ids가 있는 경우에만 fallback 매칭 시도
            similarities = []
            # 선택된 용의자들만 비교
            for sid in suspect_ids:
                person = find_person_info(sid)
                if person and person.get("embedding") is not None:
                    sim = compute_cosine_similarity(embedding, person["embedding"])
                    similarities.append((sim, person))
            
            # 유사도 순으로 정렬
            similarities.sort(key=lambda x: x[0], reverse=True)
            if similarities:
                max_similarity = similarities[0][0]
                second_similarity = similarities[1][0] if len(similarities) > 1 else 0.0
                best_match = similarities[0][1]
                best_person_id = best_match["id"]
                base_sim = max_similarity  # fallback에서는 base_sim으로 간주
                masked_sim = 0.0
            else:
                best_match = None
        
        # best_match가 None인 경우 처리 (suspect_ids 모드 또는 전체 DB 검색 실패)
        if not best_match:
            # 화질 기반 기본값 설정
            if face_quality == "high":
                main_threshold = 0.42
                gap_margin = 0.12
            elif face_quality == "medium":
                main_threshold = 0.40
                gap_margin = 0.10
            else:
                main_threshold = 0.38
                gap_margin = 0.08
            
            # unknown 상태로 face_results에 추가 (나중에 detections에 포함됨)
            face_results.append({
                "bbox": box.tolist(),
                "embedding": embedding_normalized,
                "angle_type": angle_type,
                "yaw_angle": float(yaw_angle) if yaw_angle is not None else 0.0,
                "face_quality": face_quality,
                "max_similarity": 0.0,
                "second_similarity": 0.0,
                "sim_gap": 0.0,
                "main_threshold": main_threshold,
                "gap_margin": gap_margin,
                "is_match": False,
                "best_match": None,
                "best_person_id": None,
                "mask_prob": 0.0
            })
            continue  # 다음 얼굴로 진행
        
        # 화질 기반 절대 임계값 설정 (마스크와 무관하게)
        # 마스크 기반 threshold 조정 로직 제거: "유사도 낮음 → 마스크겠지 → threshold 내려!" 패턴 폐기
        # 
        # 튜닝 가이드:
        # - False Positive가 많으면 threshold/gap을 높이기 (+0.01 ~ +0.02)
        # - True Positive가 적으면 threshold/gap을 낮추기 (-0.01 ~ -0.02)
        # - 특정 화질에서만 문제가 있으면 해당 화질만 조정
        # - 자세한 튜닝 가이드: python scripts/tune_threshold_gap.py --guide
        if face_quality == "high":
            main_threshold = 0.38  # [최종 조정] 0.42 → 0.38 (-4%): 인식률 극대화
            gap_margin = 0.10      # [흐릿한 영상] 0.14 → 0.10 (-4%): 저화질 영상 대응
        elif face_quality == "medium":
            main_threshold = 0.36  # [최종 조정] 0.40 → 0.36 (-4%): 인식률 극대화
            gap_margin = 0.08      # [흐릿한 영상] 0.12 → 0.08 (-4%): 저화질 영상 대응
        else:  # low
            main_threshold = 0.34  # [최종 조정] 0.38 → 0.34 (-4%): 인식률 극대화
            gap_margin = 0.06      # [흐릿한 영상] 0.10 → 0.06 (-4%): 저화질 영상 대응
        
        # suspect_ids 모드에서 gap만 강화 (threshold는 유지)
        # [최종 조정] threshold 상향 제거: 특정 인물 검색 시에도 인식률 우선
        if suspect_ids:
            main_threshold += 0.00  # threshold 상향 제거 (인식률 우선)
            gap_margin += 0.02      # Gap만 약간 상향 (오인식 방지)
        
        # 두 번째 유사도와의 차이 계산 (오인식 방지)
        sim_gap = max_similarity - second_similarity if second_similarity > 0 else max_similarity
        
        # [mask_prob는 이미 위에서 계산됨 - 중복 제거]
        
        # Masked candidate frame 판단
        # 조건: base_sim < threshold AND base_sim >= 0.25 AND mask_prob >= 0.5
        # 주의: best_person_id가 있어야 tracking 가능 (base_sim이 낮아도 매칭된 인물이 있어야 함)
        is_masked_candidate = False
        if best_person_id != "unknown":  # 매칭된 인물이 있어야 masked candidate로 판단
            # 모든 조건 체크 및 상세 로그
            cond1 = base_sim < main_threshold
            cond2 = base_sim >= MASKED_CANDIDATE_MIN_SIM
            cond3 = mask_prob >= MASKED_BANK_MASK_PROB_THRESHOLD
            
            if cond1 and cond2 and cond3:
                is_masked_candidate = True
                print(f"🎭 [MASKED CAND] ✅ 감지됨! person_id={best_person_id}, base_sim={base_sim:.3f}, mask_prob={mask_prob:.3f}, threshold={main_threshold:.3f}")
            else:
                # 조건 미충족 이유 상세 로그
                reasons = []
                if not cond1:
                    reasons.append(f"base_sim({base_sim:.3f}) >= threshold({main_threshold:.3f})")
                if not cond2:
                    reasons.append(f"base_sim({base_sim:.3f}) < min({MASKED_CANDIDATE_MIN_SIM:.3f})")
                if not cond3:
                    reasons.append(f"mask_prob({mask_prob:.3f}) < min({MASKED_BANK_MASK_PROB_THRESHOLD:.3f})")
                print(f"🎭 [MASKED CAND] ❌ 조건 미충족: person_id={best_person_id}, base_sim={base_sim:.3f}, mask_prob={mask_prob:.3f} | 이유: {', '.join(reasons)}")
        else:
            # best_person_id가 unknown인 경우도 로그 출력 (디버깅용)
            if base_sim > 0:  # base_sim이 0보다 크면 매칭 시도는 했지만 실패한 경우
                print(f"🎭 [MASKED CAND] ⚠️ 매칭 실패: best_person_id=unknown, base_sim={base_sim:.3f}, mask_prob={mask_prob:.3f}")
        
        # 박스 정보 초기화
        box_info = {
            "bbox": box.tolist(),  # [x1, y1, x2, y2]
            "status": "unknown",
            "name": "Unknown",
            "confidence": int(max_similarity * 100),
            "color": "yellow",  # 기본값: 노란색 (미확인)
            "angle_type": angle_type,  # 각도 정보 추가
            "yaw_angle": float(yaw_angle) if yaw_angle is not None else 0.0
        }
        
        # Bank 자동 추가 여부 결정
        AUTO_ADD_TO_BANK = True  # 자동 학습 활성화
        BANK_DUPLICATE_THRESHOLD = 0.8
        bank_added = False
        
        # 강화된 매칭 조건: 두 가지 조건을 모두 만족해야 match 인정
        # 1) 절대 유사도 기준: main_threshold 이상
        # 2) gap 기준: sim_gap >= gap_margin
        # [제거됨] 3) 두 번째 후보 상한 체크 - Gap margin만으로 충분함
        is_match = False
        if max_similarity >= main_threshold:
            # Gap이 충분히 벌어졌을 때만 match 인정
            if sim_gap >= gap_margin:
                is_match = True
        
        # [근본 해결] 최소 Base 유사도 요구사항
        # Base Bank와 10% 미만이면 무조건 차단 (Masked/Dynamic Bank 오염 방지)
        # 단, Masked Bank 사용 + mask_prob 높으면 예외 (정상 마스크 착용자)
        # [완화] 0.30 → 0.15 → 0.10: 마스크/모자 착용 범죄자 인식 극대화
        MIN_BASE_SIMILARITY_REQUIRED = 0.10
        
        if is_match and base_sim < MIN_BASE_SIMILARITY_REQUIRED:
            # [개선] 마스크 착용자 예외 - mask_prob만으로도 판단 (더 관대하게)
            # 실제 마스크 확인 또는 mask_prob이 높으면 예외 처리
            if (bank_type == "masked" and is_masked) or mask_prob >= 0.70:
                print(f"   ✅ [BASE 예외] 마스크 착용자 확인 (mask_prob={mask_prob:.3f}), Base 요구사항 면제")
            else:
                is_match = False
                # 마스크 착용 여부에 따라 다른 메시지 출력
                if not is_masked:
                    print(f"   ⚠️ [BASE 요구사항] 차단: {best_person_id} (base={base_sim:.3f} < {MIN_BASE_SIMILARITY_REQUIRED:.3f}, 마스크 없음)")
                else:
                    print(f"   ⚠️ [BASE 요구사항] 차단: {best_person_id} (base={base_sim:.3f} < {MIN_BASE_SIMILARITY_REQUIRED:.3f}, 마스크 착용했으나 mask_prob 낮음)")
        
        # suspect_ids가 지정된 경우: 추가 강화 규칙 적용
        if suspect_ids:
            # best_match가 이미 선택된 용의자 중 하나임을 보장
            if not best_match:
                is_match = False
            # 절대값 기준은 위에서 설정한 main_threshold를 따름
            # [수정] 하드코딩된 0.48 제거 -> main_threshold 사용
            elif max_similarity < main_threshold:
                is_match = False
        else:
            # 전체 갤러리 모드에서도 best_match가 없으면 match 불가
            if not best_match:
                is_match = False
        
        # Bbox tracking 기반 multi-frame 확인 (masked candidate인 경우)
        track_id = None
        candidate_frames_count = 0
        
        if is_masked_candidate:
            # 기존 track 찾기 (IoU 기반)
            best_iou = 0.0
            for tid, track in tracking_state["tracks"].items():
                if track["person_id"] == best_person_id:
                    # 마지막 bbox와 현재 bbox의 IoU 계산
                    last_bbox = track["bbox"]
                    iou = calculate_bbox_iou(box.tolist(), last_bbox)
                    if iou > best_iou and iou >= MASKED_TRACKING_IOU_THRESHOLD:
                        best_iou = iou
                        track_id = tid
            
            # 기존 track이 있으면 업데이트, 없으면 새로 생성
            if track_id is not None:
                track = tracking_state["tracks"][track_id]
                track["bbox"] = box.tolist()
                track["frames"] += 1
                track["embeddings"].append(embedding_normalized)
                candidate_frames_count = track["frames"]
                
                # 연속 N 프레임 이상 조건 충족 시 masked bank에 추가
                if track["frames"] >= MASKED_CANDIDATE_MIN_FRAMES:
                    # masked bank에 추가 (중복 체크 포함)
                    added = update_gallery_cache_in_memory(best_person_id, embedding_normalized, bank_type="masked")
                    if added:
                        learning_events.append({
                            "person_id": best_person_id,
                            "person_name": best_match["name"] if best_match else "Unknown",
                            "angle_type": angle_type,
                            "yaw_angle": yaw_angle,
                            "embedding": embedding_normalized.tolist(),
                            "bank_type": "masked",
                            "track_frames": track["frames"]
                        })
                        print(f"  ✅ [MASKED BANK] 자동 추가 성공: {best_person_id} (연속 {track['frames']}프레임, base_sim={base_sim:.3f}, mask_prob={mask_prob:.3f})")
                    else:
                        print(f"  ⚠️ [MASKED BANK] 중복으로 스킵: {best_person_id} (연속 {track['frames']}프레임)")
                else:
                    print(f"  📊 [MASKED CAND] 추적 중: {best_person_id} ({track['frames']}/{MASKED_CANDIDATE_MIN_FRAMES}프레임, base_sim={base_sim:.3f})")
            else:
                # 새 track 생성
                track_id = f"track_{len(tracking_state['tracks'])}"
                tracking_state["tracks"][track_id] = {
                    "bbox": box.tolist(),
                    "person_id": best_person_id,
                    "frames": 1,
                    "embeddings": [embedding_normalized],
                    "last_frame": 0  # 프레임 번호는 나중에 업데이트
                }
                candidate_frames_count = 1
                print(f"  🆕 [MASKED CAND] 새 track 생성: {best_person_id} (track_id={track_id}, base_sim={base_sim:.3f})")
        
        # 결과 저장 (나중에 필터링)
        face_index = len(face_results)  # 현재 인덱스
        face_results.append({
            "bbox": box.tolist(),
            "embedding": embedding_normalized,
            "angle_type": angle_type,
            "yaw_angle": float(yaw_angle) if yaw_angle is not None else 0.0,
            "face_quality": face_quality,
            "max_similarity": max_similarity,
            "base_sim": base_sim,  # base bank 유사도
            "masked_sim": masked_sim,  # masked bank 유사도
            "dynamic_sim": dynamic_sim,  # dynamic bank 유사도
            "second_similarity": second_similarity,
            "sim_gap": sim_gap,
            "main_threshold": main_threshold,
            "gap_margin": gap_margin,
            "is_match": is_match,
            "best_match": best_match,
            "best_person_id": best_person_id,
            "mask_prob": mask_prob,
            "bank_type": bank_type,
            "is_masked_candidate": is_masked_candidate,
            "candidate_frames_count": candidate_frames_count,
            "track_id": track_id,
            "face_index": face_index  # face 객체 인덱스 저장
        })
        face_objects.append(face)  # face 객체 저장
    
    # 4. 같은 얼굴 영역에서 여러 인물로 매칭되는 경우 필터링 (오인식 방지)
    print(f"🔍 [필터링 전] face_results 개수: {len(face_results)}")
    filtered_results = []
    used_indices = set()
    
    for i, r1 in enumerate(face_results):
        if i in used_indices:
            continue
        
        # 같은 얼굴 영역 그룹 찾기
        group = [r1]
        used_indices.add(i)
        
        for j, r2 in enumerate(face_results):
            if j <= i or j in used_indices:
                continue
            
            if is_same_face_region(r1["bbox"], r2["bbox"]):
                group.append(r2)
                used_indices.add(j)
        
        # 그룹 처리
        if len(group) == 1:
            # 단일 매칭: 그대로 유지
            filtered_results.append(group[0])
        else:
            # 같은 얼굴 영역에서 여러 인물로 매칭됨 → 오인식 가능성 높음
            # 유사도 순으로 정렬
            group.sort(key=lambda x: x["max_similarity"], reverse=True)
            
            best_match = group[0]
            second_match = group[1] if len(group) > 1 else None
            
            # 더 엄격한 기준 적용 (오인식 방지)
            # 새로운 강화된 매칭 조건 사용
            quality = best_match["face_quality"]
            main_threshold = best_match.get("main_threshold", 0.40)
            gap_margin = best_match.get("gap_margin", 0.10)
            
            # 강화된 조건 재검증
            max_sim = best_match["max_similarity"]
            second_sim = best_match.get("second_similarity", 0.0)
            sim_gap = best_match["sim_gap"]
            
            is_match = False
            if max_sim >= main_threshold:
                if second_sim > 0 and second_sim >= (main_threshold - 0.02):
                    is_match = False
                else:
                    if sim_gap >= gap_margin:
                        is_match = True
            
            if is_match:
                # 확신 있는 매칭
                best_match["is_match"] = True
                filtered_results.append(best_match)
            else:
                # 조건을 만족하지 않으면 매칭 해제 (오인식 방지)
                # 하지만 unknown 상태로라도 detections에 포함되어야 함
                best_match["is_match"] = False
                best_match["best_match"] = None  # 매칭 해제
                filtered_results.append(best_match)  # unknown 상태로 추가
                print(f"  ⚠️ 같은 얼굴 영역에서 여러 인물 매칭됨 → 매칭 해제 (sim={max_sim:.3f} < {main_threshold:.3f} 또는 gap={sim_gap:.3f} < {gap_margin:.3f} 또는 second_sim={second_sim:.3f} >= {main_threshold - 0.02:.3f})")
    
    print(f"🔍 [필터링 후] filtered_results 개수: {len(filtered_results)}")
    
    # 5. 최종 결과 생성
    for result in filtered_results:
        # 최종 결과 생성
        box = result["bbox"]
        max_similarity = result["max_similarity"]
        best_match = result["best_match"]
        is_match = result["is_match"]
        angle_type = result["angle_type"]
        yaw_angle = result["yaw_angle"]
        main_threshold = result.get("main_threshold", 0.40)
        gap_margin = result.get("gap_margin", 0.10)
        sim_gap = result["sim_gap"]
        second_similarity = result.get("second_similarity", 0.0)
        mask_prob = result.get("mask_prob", 0.0)
        bank_type_result = result.get("bank_type", "base")
        
        # 디버깅: 매칭 조건 상세 정보 출력
        bank_type_result = result.get("bank_type", "base")
        base_sim_result = result.get("base_sim", 0.0)
        masked_sim_result = result.get("masked_sim", 0.0)
        dynamic_sim_result = result.get("dynamic_sim", 0.0)
        mask_prob_result = result.get("mask_prob", 0.0)
        is_masked_candidate_result = result.get("is_masked_candidate", False)
        candidate_frames_count_result = result.get("candidate_frames_count", 0)
        
        print(f"🎯 [매칭 디버깅] bank={bank_type_result}, base_sim={base_sim_result:.3f}, masked_sim={masked_sim_result:.3f}, dynamic_sim={dynamic_sim_result:.3f}, best_sim={max_similarity:.3f}")
        print(f"   - main_threshold={main_threshold:.3f}, sim_gap={sim_gap:.3f}, gap_margin={gap_margin:.3f}, 매칭={is_match}")
        print(f"   - mask_prob={mask_prob_result:.3f}, masked_candidate={is_masked_candidate_result}, candidate_frames={candidate_frames_count_result}")
        print(f"   - 유사도 >= main_threshold: {max_similarity:.3f} >= {main_threshold:.3f} = {max_similarity >= main_threshold}")
        print(f"   - sim_gap >= gap_margin: {sim_gap:.3f} >= {gap_margin:.3f} = {sim_gap >= gap_margin}")
        
        if is_match:
            # 매칭 성공
            name = best_match["name"]
            person_id = best_match["id"]
            is_criminal = best_match["is_criminal"]
            
            # [수정 1] 구체적인 person_type 추출 (DB info에서 가져오기)
            # best_match['info'] 딕셔너리에서 'person_type'이나 'category'를 꺼냅니다.
            person_info = best_match.get("info", {})
            person_type = person_info.get("person_type") or person_info.get("category") or ("criminal" if is_criminal else "normal")
            
            # [수정 2] 정확도 소수점 유지 (int -> float)
            confidence_score = round(max_similarity * 100, 2)

            embedding_normalized = result["embedding"]
            
            # 감지 로그 저장 (PostgreSQL) - db가 제공된 경우에만
            if db is not None:
                try:
                    log_detection(
                        db=db,
                        person_id=person_id,
                        person_name=name,
                        similarity=max_similarity,
                        is_criminal=is_criminal,
                        status="criminal" if is_criminal else "normal",
                        metadata={
                            "bbox": box,
                            "threshold": main_threshold
                        }
                    )
                except Exception as e:
                    print(f"⚠️ 로그 저장 실패: {e}")
            
            # 동적 Bank 자동 추가 (매칭 성공 시) - 강화된 필터링 적용
            # 목적: 정면으로 식별된 인물에 대해 CCTV 영상에서 움직일 때 추가 각도 임베딩을 수집
            # 개선: 오인식으로 인한 임베딩 오염 방지를 위한 검증 강화
            AUTO_ADD_TO_DYNAMIC_BANK = True
            BANK_DUPLICATE_THRESHOLD = 0.95
            
            # 1단계 개선: Dynamic Bank 입력 필터 강화 (Hygiene Check) - v2
            # 검증 1: Base Bank와의 최소 유사도 검증 (>= 0.55) ✅ 사용자 승인
            # 검증 2: Occlusion 없는 상태 검증 (랜드마크 기반) ✅ 구현
            # 검증 3: 고화질 검증 (보류 - 향후 필요시 추가)
            
            # =========================================================
            # 🛡️ 오염 방지 로직 (Drift Prevention) - 수정된 코드
            # =========================================================
            
            # 1. 학습용 임계값은 감지용보다 훨씬 높아야 함 (보수적 접근)
            # 예: 감지는 40%면 알람을 울리지만, 학습은 48% 이상일 때만 함
            # [조정] 0.75 → 0.48: 옆모습 각도 임베딩 수집 극대화 (최종 조정 3차)
            LEARNING_THRESHOLD = 0.48  # 75% → 65% → 60% → 55% → 53% → 51% → 48%

            # 2. [핵심] 원본(Base)과의 유사도 검증 (Golden Standard Check)
            # 현재 모습이 'Dynamic Bank(최근 모습)'와 비슷하더라도, 
            # 'Base Bank(원본 등록 사진)'와 너무 다르면 학습하지 않음.
            # -> 이게 없으면 점점 엉뚱한 얼굴로 변해가는 것을 막을 수 없음.
            # [조정] 0.55 → 0.33: 옆모습 각도 수집 극대화 (최종 조정 3차)
            MIN_BASE_SIMILARITY = 0.33  # 55% → 50% → 45% → 40% → 38% → 36% → 33% 

            should_add_to_dynamic_bank = False
            validation_failures = []

            if AUTO_ADD_TO_DYNAMIC_BANK:
                # 조건 1: 전체 유사도가 매우 높아야 함 (확실한 경우만 학습)
                if max_similarity < LEARNING_THRESHOLD:
                    validation_failures.append(f"sim({max_similarity:.2f}) < learn_th({LEARNING_THRESHOLD})")
                
                # 조건 2: 원본 사진과도 어느 정도 닮아야 함 (오염 방지)
                elif base_sim_result < MIN_BASE_SIMILARITY:
                    validation_failures.append(f"base_sim({base_sim_result:.2f}) < min_base({MIN_BASE_SIMILARITY}) - 원본과 너무 다름")
                
                # 조건 3: Occlusion 체크 (v2 신규 - 랜드마크 기반)
                # [제거됨] 다양한 각도 수집을 위해 Occlusion 체크 비활성화
                # - 문제: 정상적인 옆모습/윗모습도 차단 (90% profile, 60% side 차단)
                # - 대안: LEARNING_THRESHOLD (0.48), MIN_BASE_SIMILARITY (0.33), 중복 체크 (0.95)로 오염 방지
                # face 객체가 필요하므로 face_index로 찾아옴
                
                # 조건 3 (신규): 얼굴 품질 체크 (기존 조건 4를 조건 3으로 승격)
                else:
                    face_index = result.get("face_index")
                    if face_index is not None and face_index < len(face_objects):
                        # Occlusion 체크 생략 - 각도 다양성 우선
                        # 얼굴 크기만 체크
                        face_width = box[2] - box[0]
                        face_height = box[3] - box[1]
                        face_size = max(face_width, face_height)
                        
                        if face_size < 100:  # 너무 작은 얼굴은 학습 X
                            validation_failures.append("face too small")
                        else:
                            should_add_to_dynamic_bank = True
                    else:
                        validation_failures.append("face object not found")
                
                if should_add_to_dynamic_bank:
                    # 모든 검증 통과: 동적 bank에 추가 (각도별 다양성 체크 포함)
                    # 모든 각도(front, left, right, top) 수집 가능
                    learning_events.append({
                        "person_id": person_id,
                        "person_name": name,
                        "angle_type": angle_type,
                        "yaw_angle": yaw_angle,
                        "embedding": embedding_normalized.tolist(),  # 파일 저장용
                        "bank_type": "dynamic"  # 동적 bank로 저장
                    })
                    base_sim_result = result.get("base_sim", 0.0)
                    print(f"  ✅ [DYNAMIC BANK] 검증 통과: {person_id} (base_sim={base_sim_result:.3f}, face_size={face_size}px, angle={angle_type})")
                else:
                    # 검증 실패: Dynamic Bank에 추가하지 않음
                    print(f"  ⏭ [DYNAMIC BANK] 검증 실패: {person_id} | 이유: {', '.join(validation_failures)}")
            
            # Bank 자동 추가 (매칭 성공 시) - base bank는 절대 자동 추가하지 않음
            # Dynamic Bank는 위에서 이미 처리됨 (모든 각도 수집 가능)
            # Masked Bank는 마스크 쓴 얼굴만 수집
            # 여기서는 추가적인 각도 학습을 위해 Dynamic Bank에 더 많이 추가하도록 개선
            
            # Dynamic Bank에 추가되지 않은 경우, 추가 시도 (검증 완화)
            # 목적: 다양한 각도의 임베딩을 더 많이 수집하여 인식률 향상
            # [삭제됨] 완화된 조건의 Dynamic Bank 추가 로직 제거 (오염 방지)
            
            # Masked Bank 추가 (마스크 쓴 얼굴만, 측면/프로파일 각도)
            AUTO_ADD_TO_BANK = True
            important_angles = ["left_profile", "right_profile", "left", "right", "front"]  # front도 추가
            
            if AUTO_ADD_TO_BANK and bank_type == "masked":
                # 조건: 고화질 + 고유사도 (main_threshold 이상)
                is_high_confidence = (face_quality == "high" and 
                                     max_similarity >= main_threshold)
                
                # 모든 각도에서 masked bank에 추가 가능 (측면/프로파일 우선, front도 허용)
                is_valid_angle = angle_type in important_angles if angle_type else True
                
                if is_high_confidence and is_valid_angle:
                    # 메모리에서 즉시 업데이트 (실시간 반영)
                    added = update_gallery_cache_in_memory(person_id, embedding_normalized, bank_type="masked")
                    if added:
                        # 학습 이벤트 기록 (masked bank)
                        learning_events.append({
                            "person_id": person_id,
                            "person_name": name,
                            "angle_type": angle_type or "front",
                            "yaw_angle": yaw_angle or 0.0,
                            "embedding": embedding_normalized.tolist(),
                            "bank_type": "masked"
                        })
                        print(f"  ✅ [MASKED BANK] 추가: {person_id} (angle={angle_type or 'front'}, sim={max_similarity:.3f})")
            
            # 박스 정보 설정 (person_id 포함)
            box_info = {
                "bbox": box,
                "status": "criminal" if is_criminal else "normal", # 프론트엔드 색상 결정용 (유지)
                "person_type": person_type,  # <--- [중요] 상세 카테고리 추가 ("missing", "child" 등)
                "name": name,
                "person_id": person_id,  # person_id 필드 추가 (temporal filter용)
                "confidence": confidence_score, # <--- [중요] 소수점 포함된 값 (98.2)
                "color": "red" if is_criminal else "green",
                "angle_type": angle_type,
                "yaw_angle": yaw_angle,
                "bank_type": bank_type  # base 또는 masked
            }
            
            if is_criminal:
                # [범죄자 발견] 빨간색 박스
                alert_triggered = True
                detected_metadata = {
                    "name": name,
                    "confidence": confidence_score, # 수정됨
                    "status": "criminal",
                    "person_type": person_type,     # 추가됨
                    "person_id": person_id          # 추가됨
                }
            else:
                # [일반인] 초록색 박스
                # 현재 화면에 범죄자가 없다면 일반인 정보 표시
                if not alert_triggered:
                    detected_metadata = {
                        "name": name,
                        "confidence": confidence_score, # 수정됨
                        "status": "normal",
                        "person_type": person_type,     # 추가됨
                        "person_id": person_id          # 추가됨
                    }
        else:
            # [미확인] 노란색 박스 (person_id는 None)
            box_info = {
                "bbox": box,
                "status": "unknown",
                "person_type": "unknown", # 미확인
                "name": "Unknown",
                "person_id": None,  # person_id 필드 추가 (temporal filter용)
                "confidence": int(max_similarity * 100), # 미확인은 정수로도 충분
                "color": "yellow",
                "angle_type": angle_type,
                "yaw_angle": yaw_angle
            }
            
            # 미확인 감지도 로그 저장 - db가 제공된 경우에만
            if db is not None:
                try:
                    log_detection(
                        db=db,
                        similarity=max_similarity,
                        status="unknown",
                        metadata={
                            "bbox": box,
                            "threshold": main_threshold
                        }
                    )
                except Exception as e:
                    print(f"⚠️ 로그 저장 실패: {e}")
        
        detections.append(box_info)

    # 최종 결과 로그 출력 (디버깅용)
    print(f"📊 [최종 결과] detections 개수: {len(detections)}, alert: {alert_triggered}")
    if detections:
        for i, det in enumerate(detections):
            print(f"   - [{i+1}] {det.get('name', 'Unknown')} ({det.get('status', 'unknown')}), confidence: {det.get('confidence', 0)}%")

    return {
        "detections": detections,
        "alert": alert_triggered,
        "metadata": detected_metadata,
        "learning_events": learning_events  # 학습 이벤트 (UI 피드백용)
    }

# ==========================================
# 8. API 엔드포인트
# ==========================================

class DetectionRequest(BaseModel):
    image: str       # Base64 이미지
    suspect_id: Optional[str] = None  # (선택적) 특정 타겟 ID (호환성 유지)
    suspect_ids: Optional[List[str]] = None  # (선택적) 여러 타겟 ID

@app.post("/api/detect")
async def detect_faces(request: DetectionRequest, db: Session = Depends(get_db)):
    """
    얼굴 감지 및 인식 (HTTP API - 호환성 유지)
    
    Args:
        request: DetectionRequest (image: Base64, suspect_id: 선택적)
        db: 데이터베이스 세션
    
    Returns:
        {
            "success": bool,
            "detections": [...],  # 박스 좌표 및 메타데이터 배열
            "alert": bool,
            "metadata": {...}
        }
    """
    # 1. 이미지 디코딩
    frame = base64_to_image(request.image)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image data")
    
    # 2. 공통 감지 로직 사용 (suspect_ids 우선, 없으면 suspect_id 사용)
    result = process_detection(
        frame, 
        suspect_id=request.suspect_id, 
        suspect_ids=request.suspect_ids,
        db=db
    )
    
    # 3. 범죄자 감지 시 스냅샷 Base64 인코딩 추가 (HTTP API용)
    snapshot_base64 = None
    video_timestamp = None
    
    if result.get("alert"):  # 범죄자 감지됨
        print(f"🚨 HTTP API: 범죄자 감지됨! 스냅샷 생성 중...")
        try:
            # 프레임을 JPEG로 인코딩하여 Base64 생성
            success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if success and buffer is not None and len(buffer) > 0:
                snapshot_base64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')
                print(f"✅ HTTP API: 스냅샷 생성 완료: 크기={len(snapshot_base64)} bytes")
            else:
                print(f"⚠️ HTTP API: 스냅샷 인코딩 실패 (success={success}, buffer={buffer is not None})")
        except Exception as e:
            print(f"❌ HTTP API: 스냅샷 생성 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
    
    # 4. 결과 반환
    response = {
        "success": True,
        **result
    }
    
    # 범죄자 감지 시 스냅샷 추가
    if snapshot_base64:
        response["snapshot_base64"] = snapshot_base64
        response["video_timestamp"] = video_timestamp  # None이지만 필드 추가
        print(f"📤 HTTP API 응답에 스냅샷 포함: {len(snapshot_base64)} bytes")
    
    return response

@app.websocket("/ws/detect", name="websocket_detect")
async def websocket_detect(websocket: WebSocket):
    """
    WebSocket을 통한 실시간 얼굴 감지 및 인식
    
    메시지 형식:
    - 클라이언트 → 서버:
        {
            "type": "frame",
            "data": {
                "image": "base64_string",
                "suspect_id": "optional_id",
                "frame_id": 123
            }
        }
        또는
        {
            "type": "config",
            "suspect_id": "optional_id"
        }
    
    - 서버 → 클라이언트:
        {
            "type": "detection",
            "data": {
                "frame_id": 123,
                "detections": [...],
                "alert": false,
                "metadata": {...}
            }
        }
        또는
        {
            "type": "error",
            "message": "error message"
        }
    """
    # WebSocket 연결 수락 (CORS 허용)
    try:
        print(f"🔌 [메인] WebSocket 연결 시도: {websocket.client}")
        print(f"   URL: {websocket.url}")
        print(f"   Path: {websocket.url.path}")
        origin = websocket.headers.get("origin")
        print(f"   Origin: {origin}")
        print(f"   Headers: {dict(websocket.headers)}")
        
        # WebSocket 연결 수락 (모든 origin 허용)
        await websocket.accept()
        print(f"✅ [메인] WebSocket 연결 수락됨")
        
        # 연결 등록
        active_connections.add(websocket)
        connection_states[websocket] = {
            "suspect_ids": [],  # 여러 명 선택 가능
            "connected_at": asyncio.get_event_loop().time(),
            "match_counters": {},  # person_id별 연속 매칭 프레임 카운터 (하위 호환용 유지)
            "match_history": {},   # person_id별 최근 프레임 이력: {person_id: [(confidence, matched), ...]}
            "tracking_state": {
                "tracks": {}  # bbox tracking 상태
            }
        }
        print(f"✅ [메인] WebSocket 연결됨 (총 {len(active_connections)}개 연결)")
        
    except Exception as e:
        print(f"❌ [메인] WebSocket 연결 수락 실패: {e}")
        import traceback
        traceback.print_exc()
        try:
            await websocket.close()
        except:
            pass
        return
    
    try:
        while True:
            # 클라이언트로부터 메시지 수신
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                msg_type = message.get("type")
                
                if msg_type == "frame":
                    # 프레임 처리 요청
                    frame_data = message.get("data", {})
                    image_base64 = frame_data.get("image")
                    suspect_ids = frame_data.get("suspect_ids")  # 배열로 받음
                    suspect_id = frame_data.get("suspect_id")  # 호환성 유지 (단일)
                    frame_id = frame_data.get("frame_id", 0)
                    video_time = frame_data.get("video_time")  # 비디오 시간 (초 단위)
                    
                    # 연결 상태에서 suspect_ids 업데이트
                    if suspect_ids is not None:
                        connection_states[websocket]["suspect_ids"] = suspect_ids
                    elif suspect_id is not None:
                        # 단일 suspect_id를 배열로 변환 (호환성)
                        connection_states[websocket]["suspect_ids"] = [suspect_id]
                    else:
                        # 연결 상태에서 suspect_ids 사용
                        suspect_ids = connection_states[websocket].get("suspect_ids", [])
                    
                    if not image_base64:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Missing image data"
                        })
                        continue
                    
                    # 이미지 디코딩
                    frame = base64_to_image(image_base64)
                    if frame is None:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Invalid image data"
                        })
                        continue
                    
                    # 각 요청마다 새로운 DB 세션 생성 (연결 유지 시 세션 문제 방지)
                    db = next(get_db())
                    try:
                        # tracking_state 가져오기
                        tracking_state = connection_states[websocket].get("tracking_state", {"tracks": {}})
                        
                        # 공통 감지 로직 사용 (suspect_ids 우선)
                        result = process_detection(
                            frame, 
                            suspect_id=suspect_id if not suspect_ids else None,
                            suspect_ids=suspect_ids if suspect_ids else None,
                            db=db,
                            tracking_state=tracking_state
                        )
                        
                        # tracking_state 업데이트
                        connection_states[websocket]["tracking_state"] = tracking_state
                    finally:
                        db.close()
                    
                    # Temporal Consistency 필터 적용 (연속 프레임 기반 매칭 확정)
                    result = apply_temporal_filter(websocket, result)
                    
                    # 범죄자 감지 시 스냅샷 Base64 인코딩 추가
                    snapshot_base64 = None
                    
                    # 비디오 타임스탬프 계산 (모든 응답에 포함)
                    if video_time is not None:
                        video_timestamp = float(video_time)
                    else:
                        # 프레임 ID를 사용하여 대략적인 타임스탬프 계산 (10 FPS 가정)
                        video_timestamp = frame_id / 10.0
                    
                    print(f"🔍 WebSocket 감지 결과: alert={result.get('alert')}, detections={len(result.get('detections', []))}, video_time={video_timestamp:.2f}s")
                    
                    if result.get("alert"):  # 범죄자 감지됨
                        print(f"🚨 범죄자 감지됨! 스냅샷 생성 중...")
                        try:
                            # 프레임을 JPEG로 인코딩하여 Base64 생성
                            success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                            if success and buffer is not None and len(buffer) > 0:
                                snapshot_base64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')
                                print(f"✅ 스냅샷 생성 완료: 크기={len(snapshot_base64)} bytes, 타임스탬프={video_timestamp:.1f}s")
                            else:
                                print(f"⚠️ WebSocket: 스냅샷 인코딩 실패 (success={success}, buffer={buffer is not None})")
                        except Exception as e:
                            print(f"❌ WebSocket: 스냅샷 생성 중 오류 발생: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    # 결과 전송 (응답 먼저 - 성능 최우선)
                    response_data = {
                        "type": "detection",
                        "data": {
                            "frame_id": frame_id,
                            "video_timestamp": video_timestamp,  # 항상 포함
                            **result
                        }
                    }
                    
                    # 범죄자 감지 시 스냅샷 추가
                    if snapshot_base64:
                        response_data["data"]["snapshot_base64"] = snapshot_base64
                        print(f"📤 WebSocket 응답에 스냅샷 포함: {len(snapshot_base64)} bytes")
                    
                    await websocket.send_json(response_data)


                    
                    # 학습 이벤트가 있으면 파일 저장 (비동기, 응답 후)
                    learning_events = result.get("learning_events", [])
                    for event in learning_events:
                        # 임베딩을 numpy 배열로 변환
                        embedding_array = np.array(event["embedding"], dtype=np.float32)
                        bank_type = event.get("bank_type", "base")
                        
                        # 동적 bank 저장 (각도별 다양성 체크 및 수집 완료 로직 포함)
                        # ⚠️ Dynamic Bank 자동 수집 활성화
                        if bank_type == "dynamic":
                            # 파일 저장은 백그라운드에서 비동기 처리 (응답 지연 없음)
                            asyncio.create_task(add_embedding_to_dynamic_bank_async(
                                event["person_id"],
                                embedding_array,
                                event.get("angle_type"),
                                event.get("yaw_angle"),
                                similarity_threshold=0.9,
                                verbose=True
                            ))
                        else: # Dynamic이 아니면 Masked/Base 처리
                            # 기존 masked/base bank 저장 (호환성 유지)
                            asyncio.create_task(add_embedding_to_bank_async(
                                event["person_id"],
                                embedding_array,
                                event.get("angle_type"),
                                event.get("yaw_angle"),
                                bank_type=bank_type
                            ))
                
                elif msg_type == "config":
                    # 설정 변경 (suspect_ids 등)
                    suspect_ids = message.get("suspect_ids")  # 배열로 받음
                    suspect_id = message.get("suspect_id")  # 호환성 유지 (단일)
                    
                    if suspect_ids is not None:
                        connection_states[websocket]["suspect_ids"] = suspect_ids
                    elif suspect_id is not None:
                        # 단일 suspect_id를 배열로 변환 (호환성)
                        connection_states[websocket]["suspect_ids"] = [suspect_id]
                    
                    await websocket.send_json({
                        "type": "config_updated",
                        "suspect_ids": connection_states[websocket].get("suspect_ids", [])
                    })
                
                elif msg_type == "ping":
                    # 연결 확인
                    await websocket.send_json({
                        "type": "pong"
                    })
                
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Unknown message type: {msg_type}"
                    })
            
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON format"
                })
            except Exception as e:
                print(f"⚠️ WebSocket 처리 오류: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
    
    except WebSocketDisconnect:
        print("WebSocket 연결이 끊어졌습니다")
    except Exception as e:
        print(f"⚠️ WebSocket 오류: {e}")
    finally:
        unregister_connection(websocket)

@app.get("/api/health")
async def health_check():
    """서버 상태 확인 (WebSocket 연결 테스트용)"""
    return {
        "status": "ok",
        "websocket_endpoint": "/ws/detect",
        "active_connections": len(active_connections),
        "websocket_url": "ws://localhost:5000/ws/detect"
    }

@app.websocket("/ws/test")
async def websocket_test(websocket: WebSocket):
    """WebSocket 연결 테스트용 간단한 엔드포인트"""
    try:
        print(f"🔌 [테스트] WebSocket 연결 시도: {websocket.client}")
        await websocket.accept()
        print(f"✅ [테스트] WebSocket 연결됨")
        
        await websocket.send_json({
            "type": "test",
            "message": "WebSocket 연결 성공!"
        })
        
        # 간단한 에코 테스트
        while True:
            data = await websocket.receive_text()
            await websocket.send_json({
                "type": "echo",
                "message": f"받은 메시지: {data}"
            })
    except WebSocketDisconnect:
        print("⚠️ [테스트] WebSocket 연결 종료")
    except Exception as e:
        print(f"❌ [테스트] WebSocket 오류: {e}")
        import traceback
        traceback.print_exc()

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
