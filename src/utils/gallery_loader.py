# src/utils/gallery_loader.py
"""
갤러리(등록된 얼굴 임베딩) 로드 및 매칭 유틸리티
"""
import numpy as np
from pathlib import Path
from typing import Dict, Tuple


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """벡터를 L2 정규화 (norm=1)"""
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def load_gallery(emb_dir: Path, use_bank: bool = True) -> Dict[str, np.ndarray]:
    """
    임베딩 디렉토리에서 갤러리 로드 (사람별 폴더 구조 지원)
    Base와 Dynamic bank를 모두 로드하여 통합 사용
    
    Args:
        emb_dir: 임베딩 파일들이 있는 디렉토리 (예: outputs/embeddings)
        use_bank: True면 bank.npy 파일을 우선 사용, False면 centroid.npy 사용
    
    Returns:
        {person_id: embedding_array} 딕셔너리
        - bank가 있으면: (N, 512) 2D 배열 (Base + Dynamic 통합)
        - 없으면: (512,) 1D 배열
    """
    emb_dir = Path(emb_dir)
    if not emb_dir.exists():
        raise FileNotFoundError(f"임베딩 디렉토리를 찾을 수 없음: {emb_dir}")
    
    gallery = {}
    
    # 방법 1: 사람별 폴더 구조 확인 (새로운 구조)
    person_dirs = [d for d in emb_dir.iterdir() if d.is_dir()]
    
    for person_dir in person_dirs:
        person_id = person_dir.name
        
        # Base와 Dynamic bank 경로
        bank_base_path = person_dir / "bank_base.npy"
        bank_dynamic_path = person_dir / "bank_dynamic.npy"
        bank_legacy_path = person_dir / "bank.npy"  # Legacy 호환
        centroid_base_path = person_dir / "centroid_base.npy"
        centroid_dynamic_path = person_dir / "centroid_dynamic.npy"
        centroid_path = person_dir / "centroid.npy"  # Legacy 호환
        
        if use_bank:
            # Base와 Dynamic bank 통합 로드
            banks = []
            
            # Base bank 로드
            if bank_base_path.exists():
                base_bank = np.load(bank_base_path)
                if base_bank.ndim == 2:
                    base_bank = base_bank / (np.linalg.norm(base_bank, axis=1, keepdims=True) + 1e-6)
                    banks.append(base_bank)
            elif bank_legacy_path.exists():
                # Legacy bank를 base로 간주
                legacy_bank = np.load(bank_legacy_path)
                if legacy_bank.ndim == 2:
                    legacy_bank = legacy_bank / (np.linalg.norm(legacy_bank, axis=1, keepdims=True) + 1e-6)
                    banks.append(legacy_bank)
            
            # Dynamic bank 로드
            if bank_dynamic_path.exists():
                dynamic_bank = np.load(bank_dynamic_path)
                if dynamic_bank.ndim == 2 and dynamic_bank.shape[0] > 0:
                    dynamic_bank = dynamic_bank / (np.linalg.norm(dynamic_bank, axis=1, keepdims=True) + 1e-6)
                    banks.append(dynamic_bank)
            
            # Base와 Dynamic 통합
            if banks:
                combined_bank = np.vstack(banks)
                gallery[person_id] = combined_bank
                continue
        
        # Centroid 사용 (bank가 없을 때)
        if centroid_base_path.exists() or centroid_dynamic_path.exists() or centroid_path.exists():
            centroids = []
            
            # Base centroid
            if centroid_base_path.exists():
                base_centroid = np.load(centroid_base_path)
                centroids.append(base_centroid)
            elif centroid_path.exists():
                legacy_centroid = np.load(centroid_path)
                centroids.append(legacy_centroid)
            
            # Dynamic centroid (있으면 함께 사용)
            if centroid_dynamic_path.exists():
                dynamic_centroid = np.load(centroid_dynamic_path)
                centroids.append(dynamic_centroid)
            
            # Centroid 평균 계산
            if centroids:
                combined_centroid = np.mean(np.stack(centroids), axis=0)
                emb = l2_normalize(combined_centroid)
                gallery[person_id] = emb
    
    # 방법 2: 루트에 있는 파일들 확인 (기존 구조 호환)
    npy_files = sorted(emb_dir.glob("*.npy"))
    
    # person_id별로 그룹화
    person_ids = set()
    for npy_file in npy_files:
        stem = npy_file.stem
        if stem.endswith("_bank"):
            person_id = stem[:-5]  # "_bank" 제거
            person_ids.add(person_id)
        elif stem.endswith("_centroid"):
            person_id = stem[:-9]  # "_centroid" 제거
            person_ids.add(person_id)
        else:
            # 그냥 .npy 파일 (legacy)
            person_id = stem
            person_ids.add(person_id)
    
    # 각 person_id에 대해 bank 우선 또는 centroid 선택 (이미 폴더에서 로드한 경우 스킵)
    for person_id in person_ids:
        if person_id in gallery:
            continue  # 이미 폴더에서 로드했으면 스킵
        
        bank_path = emb_dir / f"{person_id}_bank.npy"
        centroid_path = emb_dir / f"{person_id}_centroid.npy"
        legacy_path = emb_dir / f"{person_id}.npy"
        
        if use_bank and bank_path.exists():
            # Bank 사용 (2D 배열: N x 512)
            emb = np.load(bank_path)
            if emb.ndim == 2:
                emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-6)
            gallery[person_id] = emb
        elif centroid_path.exists():
            # Centroid 사용 (1D 배열: 512)
            emb = np.load(centroid_path)
            emb = l2_normalize(emb)
            gallery[person_id] = emb
        elif legacy_path.exists():
            # Legacy .npy 파일 사용 (1D 배열: 512)
            emb = np.load(legacy_path)
            emb = l2_normalize(emb)
            gallery[person_id] = emb
    
    return gallery


def match_with_bank(face_emb: np.ndarray, gallery: Dict[str, np.ndarray]) -> Tuple[str, float]:
    """
    얼굴 임베딩과 갤러리를 비교하여 가장 유사한 사람 찾기
    
    Args:
        face_emb: 얼굴 임베딩 벡터 (512차원, 정규화되지 않아도 됨)
        gallery: load_gallery()로 로드한 갤러리 딕셔너리
    
    Returns:
        (best_person_id, best_similarity) 튜플
        - best_person_id: 가장 유사한 사람 ID (갤러리가 비어있으면 "unknown")
        - best_similarity: cosine similarity (0~1)
    """
    if not gallery:
        return "unknown", 0.0
    
    # 얼굴 임베딩 정규화
    face_emb = l2_normalize(face_emb.astype("float32"))
    
    best_person = "unknown"
    best_sim = -1.0
    
    for person_id, emb_data in gallery.items():
        if emb_data.ndim == 2:
            # Bank: (N, 512) - 모든 임베딩과 비교하여 최대값 사용
            similarities = np.dot(emb_data, face_emb)  # (N,)
            max_sim = float(np.max(similarities))
        else:
            # Centroid: (512,) - 단일 벡터와 비교
            max_sim = float(np.dot(emb_data, face_emb))
        
        if max_sim > best_sim:
            best_sim = max_sim
            best_person = person_id
    
    return best_person, best_sim


def match_with_bank_detailed(face_emb: np.ndarray, gallery: Dict) -> Tuple[str, float, float]:
    """
    얼굴 임베딩과 갤러리를 비교하여 가장 유사한 사람 찾기 (상세 정보 포함)
    
    Args:
        face_emb: 얼굴 임베딩 벡터 (512차원)
        gallery: 갤러리 딕셔너리
            - 새 구조: {person_id: {"base": np.ndarray, "dynamic": Optional[np.ndarray]}}
            - 구 구조 (backward compatibility): {person_id: np.ndarray}
    
    Returns:
        (best_person_id, best_similarity, second_similarity) 튜플
        - second_similarity: 두 번째로 높은 유사도
    """
    if not gallery:
        return "unknown", 0.0, 0.0
    
    # 얼굴 임베딩 정규화
    face_emb = l2_normalize(face_emb.astype("float32"))
    
    similarities = []
    for person_id, entry in gallery.items():
        # 새 구조: {"base": ..., "dynamic": ...}
        if isinstance(entry, dict) and "base" in entry:
            base_bank = entry["base"]
            dynamic_bank = entry.get("dynamic")
            
            # Base 유사도 계산
            if base_bank.ndim == 2:
                sim_base = float(np.max(np.dot(base_bank, face_emb)))
            else:
                sim_base = float(np.dot(base_bank, face_emb))
            
            # Dynamic 유사도 계산 (있으면)
            sim_dyn = -1.0
            if dynamic_bank is not None and dynamic_bank.shape[0] > 0:
                if dynamic_bank.ndim == 2:
                    sim_dyn = float(np.max(np.dot(dynamic_bank, face_emb)))
                else:
                    sim_dyn = float(np.dot(dynamic_bank, face_emb))
            
            # Base와 Dynamic 중 최대값 사용
            person_best_sim = max(sim_base, sim_dyn)
            
        else:
            # 구 구조 (backward compatibility): 단일 np.ndarray
            emb_data = entry
            if emb_data.ndim == 2:
                # Bank: 최대값 사용
                sims = np.dot(emb_data, face_emb)
                person_best_sim = float(np.max(sims))
            else:
                # Centroid: 단일 값
                person_best_sim = float(np.dot(emb_data, face_emb))
        
        similarities.append((person_id, person_best_sim))
    
    # 유사도 기준으로 정렬
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    if len(similarities) == 0:
        return "unknown", 0.0, 0.0
    elif len(similarities) == 1:
        return similarities[0][0], similarities[0][1], 0.0
    else:
        return similarities[0][0], similarities[0][1], similarities[1][1]




