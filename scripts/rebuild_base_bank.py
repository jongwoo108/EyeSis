#!/usr/bin/env python3
"""
기준 사진만으로 bank_base.npy를 재생성하는 스크립트

사용법:
    python scripts/rebuild_base_bank.py [--person-id PERSON_ID] [--backup]

기능:
1. 기존 bank.npy를 bank_base.npy로 변환 (backup 옵션 시 백업)
2. bank_dynamic.npy는 그대로 유지 (또는 삭제 옵션)
3. 기준 사진(enroll 폴더)만으로 새로운 bank_base.npy 생성 (선택 사항)
"""
import sys
from pathlib import Path
import numpy as np
import shutil
import argparse
from typing import Optional

# 프로젝트 루트를 Python 경로에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EMBEDDINGS_DIR = PROJECT_ROOT / "outputs" / "embeddings"
ENROLL_DIR = PROJECT_ROOT / "images" / "enroll"  # enroll 폴더는 images/enroll에 있음


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """벡터를 L2 정규화"""
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def rebuild_base_bank_from_enroll(person_id: str, enroll_dir: Path) -> Optional[np.ndarray]:
    """
    enroll 폴더에서 기준 사진만으로 bank_base.npy 재생성
    
    Args:
        person_id: 인물 ID
        enroll_dir: enroll 폴더 경로
    
    Returns:
        bank_base 배열 또는 None
    """
    from src.utils.device_config import _ensure_cuda_in_path
    _ensure_cuda_in_path()
    
    from insightface.app import FaceAnalysis
    from src.utils.device_config import get_device_id, safe_prepare_insightface
    import cv2
    
    person_enroll_dir = enroll_dir / person_id
    if not person_enroll_dir.exists():
        print(f"  ⚠️ Enroll 폴더 없음: {person_enroll_dir}")
        return None
    
    # 이미지 파일 찾기
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = [f for f in person_enroll_dir.iterdir() 
                   if f.suffix.lower() in IMG_EXTS]
    
    if not image_files:
        print(f"  ⚠️ 기준 사진 없음: {person_enroll_dir}")
        return None
    
    print(f"  📸 기준 사진 {len(image_files)}개 발견")
    
    # InsightFace 초기화
    device_id = get_device_id()
    app = FaceAnalysis(name="buffalo_l")
    safe_prepare_insightface(app, device_id, det_size=(640, 640))
    
    embeddings = []
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"    ⚠️ 이미지 읽기 실패: {img_path.name}")
            continue
        
        faces = app.get(img)
        if len(faces) == 0:
            print(f"    ⚠️ 얼굴 미검출: {img_path.name}")
            continue
        
        # 가장 큰 얼굴 선택
        faces_sorted = sorted(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True
        )
        main_face = faces_sorted[0]
        emb = main_face.embedding.astype("float32")
        emb = l2_normalize(emb)
        embeddings.append(emb)
        print(f"    ✅ {img_path.name}")
    
    if not embeddings:
        print(f"  ❌ 임베딩 추출 실패")
        return None
    
    bank_base = np.stack(embeddings, axis=0)
    print(f"  ✅ Bank Base 생성: {bank_base.shape[0]}개 임베딩")
    return bank_base


def convert_legacy_to_base(person_id: str, backup: bool = False) -> bool:
    """
    기존 bank.npy를 bank_base.npy로 변환
    
    Args:
        person_id: 인물 ID
        backup: 백업 생성 여부
    
    Returns:
        성공 여부
    """
    person_dir = EMBEDDINGS_DIR / person_id
    if not person_dir.exists():
        print(f"  ⚠️ 폴더 없음: {person_dir}")
        return False
    
    legacy_bank_path = person_dir / "bank.npy"
    base_bank_path = person_dir / "bank_base.npy"
    legacy_centroid_path = person_dir / "centroid.npy"
    base_centroid_path = person_dir / "centroid_base.npy"
    
    # bank.npy → bank_base.npy 변환
    if legacy_bank_path.exists() and not base_bank_path.exists():
        try:
            bank_data = np.load(legacy_bank_path)
            
            if backup:
                backup_path = person_dir / "bank.npy.backup"
                shutil.copy2(legacy_bank_path, backup_path)
                print(f"  💾 백업 생성: {backup_path}")
            
            np.save(base_bank_path, bank_data)
            print(f"  ✅ bank_base.npy 생성: {bank_data.shape}")
            
            # Centroid도 변환
            if legacy_centroid_path.exists() and not base_centroid_path.exists():
                centroid_data = np.load(legacy_centroid_path)
                np.save(base_centroid_path, centroid_data)
                print(f"  ✅ centroid_base.npy 생성")
            
            return True
        except Exception as e:
            print(f"  ❌ 변환 실패: {e}")
            return False
    elif base_bank_path.exists():
        print(f"  ℹ️ bank_base.npy 이미 존재")
        return True
    else:
        print(f"  ⚠️ bank.npy 없음")
        return False


def main():
    parser = argparse.ArgumentParser(description="기준 사진만으로 bank_base.npy 재생성")
    parser.add_argument("--person-id", type=str, help="특정 인물 ID만 처리 (없으면 전체)")
    parser.add_argument("--backup", action="store_true", help="기존 파일 백업")
    parser.add_argument("--from-enroll", action="store_true", 
                       help="enroll 폴더에서 기준 사진으로 재생성")
    parser.add_argument("--delete-dynamic", action="store_true",
                       help="bank_dynamic.npy 삭제 (선택 사항)")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🔄 Bank Base 재생성 스크립트")
    print("=" * 70)
    
    if not EMBEDDINGS_DIR.exists():
        print(f"❌ Embeddings 디렉토리 없음: {EMBEDDINGS_DIR}")
        return
    
    # 처리할 person_id 목록
    if args.person_id:
        person_ids = [args.person_id]
    else:
        person_dirs = [d for d in EMBEDDINGS_DIR.iterdir() if d.is_dir()]
        person_ids = [d.name for d in person_dirs]
    
    print(f"📂 처리 대상: {len(person_ids)}명\n")
    
    success_count = 0
    for person_id in person_ids:
        print(f"👤 {person_id}:")
        
        person_dir = EMBEDDINGS_DIR / person_id
        dynamic_bank_path = person_dir / "bank_dynamic.npy"
        
        # 1. 기존 bank.npy를 bank_base.npy로 변환
        if not args.from_enroll:
            if convert_legacy_to_base(person_id, backup=args.backup):
                success_count += 1
        else:
            # 2. enroll 폴더에서 재생성
            bank_base = rebuild_base_bank_from_enroll(person_id, ENROLL_DIR)
            if bank_base is not None:
                base_bank_path = person_dir / "bank_base.npy"
                base_centroid_path = person_dir / "centroid_base.npy"
                
                # 백업
                if args.backup and base_bank_path.exists():
                    backup_path = person_dir / "bank_base.npy.backup"
                    shutil.copy2(base_bank_path, backup_path)
                    print(f"  💾 기존 bank_base.npy 백업: {backup_path}")
                
                # 저장
                person_dir.mkdir(parents=True, exist_ok=True)
                np.save(base_bank_path, bank_base)
                
                # Centroid 계산 및 저장
                centroid_base = bank_base.mean(axis=0)
                centroid_base = l2_normalize(centroid_base)
                np.save(base_centroid_path, centroid_base)
                
                print(f"  ✅ bank_base.npy 저장 완료")
                success_count += 1
        
        # 3. bank_dynamic.npy 삭제 (선택 사항)
        if args.delete_dynamic and dynamic_bank_path.exists():
            try:
                dynamic_bank_path.unlink()
                print(f"  🗑️ bank_dynamic.npy 삭제")
            except Exception as e:
                print(f"  ⚠️ 삭제 실패: {e}")
        
        print()
    
    print("=" * 70)
    print(f"✅ 완료: {success_count}/{len(person_ids)}명 처리")
    print("=" * 70)


if __name__ == "__main__":
    main()

