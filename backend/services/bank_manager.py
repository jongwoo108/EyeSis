# backend/services/bank_manager.py
"""
Bank 관리 서비스 (동적 임베딩 추가 및 관리)
"""
from pathlib import Path
from typing import Dict
import numpy as np
import json
from datetime import datetime

from backend.utils.image_utils import l2_normalize
from src.utils.face_angle_detector import is_diverse_angle, is_all_angles_collected

#constants
PROJECT_ROOT = Path(__file__).parent.parent.parent
EMBEDDING_DIR = PROJECT_ROOT / "outputs" / "embeddings"


def save_angle_separated_banks(dynamic_bank: np.ndarray, angles_info: dict, person_dir: Path):
    """
    동적 bank를 각도별로 분리하여 저장 (평가용 - 정답 데이터와 비교하기 위함)
    
    주의: 이 파일들은 인식에는 사용되지 않습니다. 평가 목적으로만 사용됩니다.
    인식에는 bank_dynamic.npy (통합 파일)만 사용됩니다.
    
    정답 데이터 구조(embeddings_manual)와 동일하게 저장:
    - bank_{angle_type}.npy: 해당 각도의 모든 임베딩 배열 (평가용)
    - embedding_{angle_type}.npy: 해당 각도의 centroid(평균) 임베딩 (평가용)
    
    Args:
        dynamic_bank: 동적 bank 배열 (N, 512)
        angles_info: 각도 정보 딕셔너리 {"angle_types": [...], "yaw_angles": [...]}
        person_dir: 사람별 폴더 경로
    """
    if dynamic_bank.shape[0] == 0:
        return
    
    angle_types = angles_info.get("angle_types", [])
    
    # 각도별로 그룹화
    angle_groups = {}
    for i, angle_type in enumerate(angle_types):
        if angle_type not in angle_groups:
            angle_groups[angle_type] = []
        angle_groups[angle_type].append(i)
    
    # 각도별 파일 저장
    for angle_type, indices in angle_groups.items():
        if not indices:
            continue
        
        # 해당 각도의 임베딩 추출
        angle_bank = dynamic_bank[indices]
        
        # 각도별 bank 파일 저장 (정답 데이터와 동일한 구조: bank_{angle_type}.npy)
        angle_bank_path = person_dir / f"bank_{angle_type}.npy"
        np.save(angle_bank_path, angle_bank)
        
        # 각도별 centroid 계산 및 저장 (정답 데이터와 동일한 구조: embedding_{angle_type}.npy)
        angle_centroid = angle_bank.mean(axis=0)
        angle_centroid = l2_normalize(angle_centroid)
        angle_embedding_path = person_dir / f"embedding_{angle_type}.npy"
        np.save(angle_embedding_path, angle_centroid)


async def add_embedding_to_dynamic_bank_async(person_id: str, embedding: np.ndarray,
                                               angle_type: str = None, yaw_angle: float = None,
                                               similarity_threshold: float = 0.95, verbose: bool = False):
    """
    동적 Bank에 임베딩을 비동기로 추가 (각도별 다양성 체크 및 수집 완료 로직 포함)
    
    목적: 정면으로 식별된 인물에 대해 CCTV 영상에서 움직일 때 추가 각도 임베딩을 수집
    - 기존 base 임베딩(bank_base.npy)은 보호
    - 동적 임베딩은 bank_dynamic.npy에 별도 저장
    
    Args:
        person_id: 인물 ID
        embedding: 추가할 임베딩 (512차원, L2 정규화됨)
        angle_type: 얼굴 각도 타입 (front, left, right, top 등)
        yaw_angle: yaw 각도 값 (도 단위)
        similarity_threshold: 중복 체크 임계값
        verbose: 상세 출력 여부
    
    Returns:
        추가 성공 여부 (True: 추가됨, False: 중복/각도 제한/수집 완료로 스킵)
    """
    import json
    from datetime import datetime
    
    person_dir = EMBEDDINGS_DIR / person_id
    bank_base_path = person_dir / "bank_base.npy"
    bank_dynamic_path = person_dir / "bank_dynamic.npy"
    bank_legacy_path = person_dir / "bank.npy"
    angles_path = person_dir / "angles_dynamic.json"
    collection_status_path = person_dir / "collection_status.json"
    
    # 수집 완료 여부 확인 (이미 완료되었으면 추가 수집 중단)
    if collection_status_path.exists():
        try:
            with open(collection_status_path, 'r', encoding='utf-8') as f:
                collection_status = json.load(f)
                if collection_status.get("is_completed", False):
                    if verbose:
                        print(f"     ⏭ Dynamic Bank 스킵 (수집 완료: {person_id}, 모든 필수 각도 수집됨)")
                    return False
        except Exception as e:
            if verbose:
                print(f"     ⚠️ 수집 상태 파일 읽기 실패: {e}")
    
    # Base bank 로드 (참조용, 수정하지 않음)
    base_bank = None
    if bank_base_path.exists():
        try:
            base_bank = np.load(bank_base_path)
            if base_bank.ndim == 1:
                base_bank = base_bank.reshape(1, -1)
        except Exception as e:
            if verbose:
                print(f"     ⚠️ Base Bank 로드 실패 ({person_id}): {e}")
            base_bank = None
    elif bank_legacy_path.exists():
        try:
            base_bank = np.load(bank_legacy_path)
            if base_bank.ndim == 1:
                base_bank = base_bank.reshape(1, -1)
        except Exception as e:
            if verbose:
                print(f"     ⚠️ Legacy Bank 로드 실패 ({person_id}): {e}")
            base_bank = None
    
    # Dynamic bank 로드 (없으면 새로 생성)
    if bank_dynamic_path.exists():
        try:
            dynamic_bank = np.load(bank_dynamic_path)
            if dynamic_bank.ndim == 1:
                dynamic_bank = dynamic_bank.reshape(1, -1)
        except Exception as e:
            if verbose:
                print(f"     ⚠️ Dynamic Bank 로드 실패 ({person_id}): {e}")
            dynamic_bank = np.empty((0, 512), dtype=np.float32)
    else:
        dynamic_bank = np.empty((0, 512), dtype=np.float32)
    
    # 기존 동적 각도 정보 로드
    if angles_path.exists():
        try:
            with open(angles_path, 'r', encoding='utf-8') as f:
                angles_info = json.load(f)
        except Exception as e:
            if verbose:
                print(f"     ⚠️ 각도 정보 로드 실패 ({person_id}): {e}")
            angles_info = {"angle_types": [], "yaw_angles": []}
    else:
        angles_info = {"angle_types": [], "yaw_angles": []}
    
    # 각도 타입이 없으면 기본값 사용 (완화)
    if not angle_type or angle_type == "unknown":
        # 각도 정보가 없어도 "front"로 기본값 설정하여 수집 허용
        angle_type = "front"
        if verbose:
            print(f"     ℹ️ Dynamic Bank: 각도 정보 없음, 기본값 'front'로 설정")
    
    # 각도별 다양성 체크
    collected_angles = angles_info.get("angle_types", [])
    if not is_diverse_angle(collected_angles, angle_type):
        if verbose:
            print(f"     ⏭ Dynamic Bank 스킵 (각도 제한: {angle_type}, 이미 수집된 각도: {collected_angles})")
        return False
    
    # 중복 체크 (Base + Dynamic 모두 확인)
    all_banks = []
    if base_bank is not None and base_bank.shape[0] > 0:
        all_banks.append(base_bank)
    if dynamic_bank.shape[0] > 0:
        all_banks.append(dynamic_bank)
    
    if all_banks:
        combined_bank = np.vstack(all_banks)
        max_sim = float(np.max(combined_bank @ embedding))
        if max_sim >= similarity_threshold:
            if verbose:
                print(f"     ⏭ Dynamic Bank 스킵 (중복: {max_sim:.3f} >= {similarity_threshold})")
            return False
    
    # Dynamic Bank에 추가
    new_emb = embedding.reshape(1, -1)
    updated_dynamic_bank = np.vstack([dynamic_bank, new_emb])
    
    # 각도 정보 추가
    angles_info["angle_types"].append(angle_type)
    angles_info["yaw_angles"].append(float(yaw_angle) if yaw_angle is not None else 0.0)
    
    # 수집 완료 여부 확인
    updated_collected_angles = angles_info.get("angle_types", [])
    is_completed = is_all_angles_collected(updated_collected_angles)
    
    # 수집 완료 상태 저장
    collection_status = {
        "is_completed": is_completed,
        "completed_at": datetime.now().isoformat() if is_completed else None,
        "collected_angles": updated_collected_angles,
        "required_angles": ["front", "left", "right", "top"],
        "completion_criteria": {
            "min_front": 1,
            "min_left": 1,
            "min_right": 1,
            "min_top": 1
        }
    }
    
    # Dynamic Centroid 재계산
    updated_dynamic_centroid = updated_dynamic_bank.mean(axis=0)
    updated_dynamic_centroid = l2_normalize(updated_dynamic_centroid)
    
    # 저장 (Base는 보호, Dynamic만 저장)
    person_dir.mkdir(parents=True, exist_ok=True)
    np.save(bank_dynamic_path, updated_dynamic_bank)
    centroid_dynamic_path = person_dir / "centroid_dynamic.npy"
    np.save(centroid_dynamic_path, updated_dynamic_centroid)
    
    # 각도 정보 저장
    with open(angles_path, 'w', encoding='utf-8') as f:
        json.dump(angles_info, f, indent=2, ensure_ascii=False)
    
    # 수집 완료 상태 저장
    with open(collection_status_path, 'w', encoding='utf-8') as f:
        json.dump(collection_status, f, indent=2, ensure_ascii=False)
    
    # 각도별 파일로 분리하여 저장 (정답 데이터와 동일한 구조 - 평가용)
    save_angle_separated_banks(updated_dynamic_bank, angles_info, person_dir)
    
    # 메모리 캐시 즉시 갱신 (실시간 인식에 반영)
    global gallery_dynamic_cache
    updated_dynamic_bank_normalized = updated_dynamic_bank / (np.linalg.norm(updated_dynamic_bank, axis=1, keepdims=True) + 1e-6)
    gallery_dynamic_cache[person_id] = updated_dynamic_bank_normalized
    
    if verbose:
        completion_msg = " [수집 완료!]" if is_completed else ""
        print(f"     ✅ Dynamic Bank 추가: {person_id} [{angle_type}]{completion_msg} "
              f"(동적: {updated_dynamic_bank.shape[0]}개, "
              f"기준: {base_bank.shape[0] if base_bank is not None else 0}개)")
        print(f"     🔄 메모리 캐시 갱신 완료 (실시간 인식에 즉시 반영)")
        if is_completed:
            print(f"     🎉 모든 필수 각도 수집 완료: {person_id} "
                  f"(front, left, right, top 모두 수집됨)")
    
    return True


async def add_embedding_to_bank_async(person_id: str, embedding: np.ndarray, 
                                      angle_type: str = None, yaw_angle: float = None,
                                      bank_type: str = "base"):
    """
    Bank에 임베딩을 비동기로 추가 (파일 저장)
    
    주의: bank_base.npy는 절대 수정하지 않습니다. bank_masked.npy에만 추가합니다.
    base bank에는 마스크 없는 얼굴만, masked bank에는 마스크 쓴 얼굴만 저장합니다.
    
    Args:
        person_id: 인물 ID
        embedding: 추가할 임베딩 (512차원, L2 정규화됨)
        angle_type: 얼굴 각도 타입
        yaw_angle: yaw 각도 값
        bank_type: "base" 또는 "masked"
    
    Returns:
        추가 성공 여부 (True: 추가됨, False: 중복으로 스킵)
    """
    import json
    
    person_dir = EMBEDDINGS_DIR / person_id
    base_bank_path = person_dir / "bank_base.npy"
    masked_bank_path = person_dir / "bank_masked.npy"
    
    # bank_type에 따라 파일 경로 결정
    if bank_type == "masked":
        target_bank_path = masked_bank_path
        angles_path = person_dir / "angles_masked.json"  # masked용 각도 정보
    else:  # base
        # base bank는 자동 학습으로 추가하지 않음 (read-only)
        # 하지만 호환성을 위해 함수는 동작하도록 함
        target_bank_path = base_bank_path
        angles_path = person_dir / "angles_base.json"
    
    # Base Bank 로드 (중복 체크용, read-only) - 새 구조만 사용
    base_bank = None
    if base_bank_path.exists():
        try:
            base_bank = np.load(base_bank_path)
            if base_bank.ndim == 1:
                base_bank = base_bank.reshape(1, -1)
        except Exception as e:
            print(f"  ⚠️ Base Bank 로드 실패 ({person_id}): {e}")
            base_bank = None
    
    # Masked Bank 로드 (중복 체크용)
    masked_bank = None
    if masked_bank_path.exists():
        try:
            masked_bank = np.load(masked_bank_path)
            if masked_bank.ndim == 1:
                masked_bank = masked_bank.reshape(1, -1)
        except Exception as e:
            print(f"  ⚠️ Masked Bank 로드 실패 ({person_id}): {e}")
            masked_bank = None
    
    # Target Bank 로드 (추가할 bank)
    if target_bank_path.exists():
        try:
            target_bank = np.load(target_bank_path)
            if target_bank.ndim == 1:
                target_bank = target_bank.reshape(1, -1)
        except Exception as e:
            print(f"  ⚠️ Target Bank 로드 실패 ({person_id}): {e}")
            target_bank = np.empty((0, 512), dtype=np.float32)
    else:
        target_bank = np.empty((0, 512), dtype=np.float32)
    
    # 중복 체크: base + masked 전체를 대상으로
    BANK_DUPLICATE_THRESHOLD = 0.85
    all_bank_list = []
    if base_bank is not None:
        all_bank_list.append(base_bank)
    if masked_bank is not None and masked_bank.shape[0] > 0:
        all_bank_list.append(masked_bank)
    
    if all_bank_list:
        all_bank = np.vstack(all_bank_list)
        max_sim = float(np.max(all_bank @ embedding))
        if max_sim >= BANK_DUPLICATE_THRESHOLD:
            return False  # 중복으로 스킵
    
    # Target Bank에 추가
    new_emb = embedding.reshape(1, -1)
    updated_target_bank = np.vstack([target_bank, new_emb])
    
    # 각도 정보 로드 및 추가
    if angles_path.exists():
        try:
            with open(angles_path, 'r', encoding='utf-8') as f:
                angles_info = json.load(f)
        except Exception as e:
            print(f"  ⚠️ 각도 정보 로드 실패 ({person_id}): {e}")
            angles_info = {"angle_types": [], "yaw_angles": [], "bank_types": []}
    else:
        angles_info = {"angle_types": [], "yaw_angles": [], "bank_types": []}
    
    angles_info["angle_types"].append(angle_type if angle_type else "unknown")
    angles_info["yaw_angles"].append(float(yaw_angle) if yaw_angle is not None else 0.0)
    angles_info["bank_types"].append(bank_type)  # bank_type 정보 추가
    
    # 파일 저장 (비동기로 처리)
    person_dir.mkdir(parents=True, exist_ok=True)
    np.save(target_bank_path, updated_target_bank)
    
    with open(angles_path, 'w', encoding='utf-8') as f:
        json.dump(angles_info, f, indent=2, ensure_ascii=False)
    
    bank_name = "Masked" if bank_type == "masked" else "Base"
    file_path_str = str(target_bank_path.relative_to(PROJECT_ROOT)) if target_bank_path.exists() else str(target_bank_path)
    print(f"  ✅ [{bank_name} BANK] 파일 저장: {file_path_str} (총 {updated_target_bank.shape[0]}개 임베딩, angle: {angle_type})")
    
    return True


def update_gallery_cache_in_memory(person_id: str, embedding: np.ndarray, bank_type: str = "base"):
    """
    gallery_cache를 메모리에서 즉시 업데이트 (실시간 반영)
    
    주의: base bank는 절대 수정하지 않습니다. masked bank에만 추가합니다.
    base bank에는 마스크 없는 얼굴만, masked bank에는 마스크 쓴 얼굴만 저장합니다.
    
    Args:
        person_id: 인물 ID
        embedding: 추가할 임베딩 (512차원, L2 정규화됨)
        bank_type: "base" 또는 "masked"
    
    Returns:
        추가 성공 여부 (True: 추가됨, False: 중복으로 스킵)
    """
    global gallery_base_cache, gallery_masked_cache
    
    embedding = l2_normalize(embedding.astype("float32"))
    
    BANK_DUPLICATE_THRESHOLD = 0.95
    
    # Base Bank와 Masked Bank 모두 확인 (중복 체크용)
    base_bank = gallery_base_cache.get(person_id)
    masked_bank = gallery_masked_cache.get(person_id)
    
    # 중복 체크: base + masked 전체를 대상으로
    all_bank_list = []
    if base_bank is not None:
        all_bank_list.append(base_bank)
    if masked_bank is not None:
        all_bank_list.append(masked_bank)
    
    if all_bank_list:
        all_bank = np.vstack(all_bank_list)
        max_sim = float(np.max(all_bank @ embedding))
        if max_sim >= BANK_DUPLICATE_THRESHOLD:
            return False  # 중복으로 스킵
    
    # bank_type에 따라 적절한 캐시에 추가
    if bank_type == "masked":
        # Masked Bank에 추가
        if masked_bank is None:
            masked_bank = np.empty((0, 512), dtype=np.float32)
        
        new_emb = embedding.reshape(1, -1)
        updated_masked_bank = np.vstack([masked_bank, new_emb])
        gallery_masked_cache[person_id] = updated_masked_bank
    else:
        # Base Bank는 자동 학습으로 추가하지 않음 (read-only)
        # 하지만 호환성을 위해 함수는 동작하도록 함
        if base_bank is None:
            print(f"  ⚠️ Base Bank가 없는 상태에서 Base 추가 시도: {person_id}")
            base_bank = np.empty((0, 512), dtype=np.float32)
        
        new_emb = embedding.reshape(1, -1)
        updated_base_bank = np.vstack([base_bank, new_emb])
        gallery_base_cache[person_id] = updated_base_bank
    
    return True