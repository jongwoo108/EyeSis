# backend/services/face_detection.py
"""
얼굴 감지 및 인식 핵심 서비스
"""
from typing import Optional, List, Dict
import numpy as np
from sqlalchemy.orm import Session

# Data loader (module import for accessing updated caches)
from backend.services import data_loader
from backend.services.data_loader import find_person_info

# Image and bbox utilities  
from backend.utils.image_utils import (
    l2_normalize,
    preprocess_image_for_detection,
    compute_cosine_similarity,
    estimate_face_angle,
    estimate_face_quality,
    check_face_occlusion
)
from backend.utils.bbox_utils import (
    is_same_face_region,
    calculate_bbox_iou
)

# Bank manager functions
from backend.services.bank_manager import (
    match_with_bank_detailed,
    update_gallery_cache_in_memory
)

# Database functions
from backend.database import log_detection

# InsightFace model (will be injected from main.py)
model = None

# Constants (will be imported from main.py or defined here)
MASKED_CANDIDATE_MIN_SIM = 0.25
MASKED_BANK_MASK_PROB_THRESHOLD = 0.5
MASKED_TRACKING_IOU_THRESHOLD = 0.5
MASKED_CANDIDATE_MIN_FRAMES = 3


def set_model(face_model):
    """Set the InsightFace model (called from main.py)"""
    global model
    model = face_model



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
