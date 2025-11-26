# src/face_enroll.py
"""
얼굴 임베딩 추출 스크립트
프로젝트 폴더의 정면 사진에서 얼굴 임베딩을 추출하여 bank/centroid 생성

주요 기능:
1. 기본 등록: enroll 폴더에서 이미지 읽어 bank/centroid 생성
2. 수동 추가: 특정 이미지 폴더나 파일들을 기존 bank에 추가

참고: 영상에서 임베딩 수집은 face_match_cctv.py에서 처리합니다.
"""
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# CUDA 경로를 먼저 설정
from src.utils.device_config import _ensure_cuda_in_path
_ensure_cuda_in_path()

from insightface.app import FaceAnalysis
import cv2
import numpy as np
from src.utils.device_config import get_device_id, safe_prepare_insightface

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """벡터를 L2 정규화"""
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def get_main_face_embedding(app: FaceAnalysis, img_path: Path) -> np.ndarray | None:
    """이미지에서 가장 큰 얼굴 한 개의 임베딩을 반환 (측면 얼굴 감지 개선)"""
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  ⚠️ 이미지 읽기 실패: {img_path}")
        return None

    # 먼저 원본 이미지로 시도
    faces = app.get(img)
    
    # 얼굴을 찾지 못한 경우, 이미지 전처리 후 재시도
    if len(faces) == 0:
        # 이미지 밝기/대비 조정
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # 전처리된 이미지로 재시도
        faces = app.get(enhanced)
        
        # 여전히 실패하면 업스케일링 후 재시도
        if len(faces) == 0:
            h, w = img.shape[:2]
            if h < 1280 or w < 1280:
                scale = max(1280 / h, 1280 / w)
                new_h, new_w = int(h * scale), int(w * scale)
                upscaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                faces = app.get(upscaled)
    
    if len(faces) == 0:
        print(f"  ⚠️ 얼굴 미검출: {img_path}")
        return None

    # 가장 큰 얼굴 선택
    faces_sorted = sorted(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        reverse=True
    )
    main_face = faces_sorted[0]
    emb = main_face.embedding.astype("float32")
    emb = l2_normalize(emb)
    return emb


def save_embeddings(person_id: str, emb_list: list[np.ndarray], out_dir: Path, 
                   save_bank: bool = True, save_centroid: bool = True):
    """
    임베딩 리스트를 bank_base와 centroid_base로 저장 (사람별 폴더 구조)
    
    Enrollment 시에는 기준 Bank(base)만 생성하고, dynamic은 생성하지 않습니다.
    
    Args:
        person_id: 사람 ID
        emb_list: 임베딩 리스트
        out_dir: 저장 디렉토리 (예: outputs/embeddings)
        save_bank: bank 저장 여부
        save_centroid: centroid 저장 여부
    """
    if not emb_list:
        return
    
    embs = np.stack(emb_list, axis=0)  # (N, 512)
    centroid = embs.mean(axis=0)       # (512,)
    centroid = l2_normalize(centroid)
    
    # 사람별 폴더 생성
    person_dir = out_dir / person_id
    person_dir.mkdir(parents=True, exist_ok=True)
    
    if save_bank:
        # bank_base.npy 저장 (기준 Bank, read-only)
        bank_base_path = person_dir / "bank_base.npy"
        np.save(bank_base_path, embs)
        print(f"     Base Bank 저장: {bank_base_path} ({embs.shape[0]}개 임베딩)")
        
        # Backward compatibility: 기존 bank.npy가 없으면 생성 (legacy 지원)
        legacy_bank_path = person_dir / "bank.npy"
        if not legacy_bank_path.exists():
            np.save(legacy_bank_path, embs)
            print(f"     Legacy Bank 저장 (backward compatibility): {legacy_bank_path}")
    
    if save_centroid:
        # centroid_base.npy 저장 (기준 Centroid, read-only)
        centroid_base_path = person_dir / "centroid_base.npy"
        np.save(centroid_base_path, centroid)
        print(f"     Base Centroid 저장: {centroid_base_path}")
        
        # Backward compatibility: 기존 centroid.npy가 없으면 생성 (legacy 지원)
        legacy_centroid_path = person_dir / "centroid.npy"
        if not legacy_centroid_path.exists():
            np.save(legacy_centroid_path, centroid)
            print(f"     Legacy Centroid 저장 (backward compatibility): {legacy_centroid_path}")
    
    print(f"     L2 norm: {np.linalg.norm(centroid):.4f}")


# ===== MODE 1: 기본 등록 =====
def mode_basic_enroll(app: FaceAnalysis, enroll_root: Path, out_dir: Path, 
                     save_bank: bool = True, save_centroid: bool = True):
    """
    enroll 폴더에서 모든 사람의 이미지를 읽어 bank/centroid 생성
    
    Args:
        app: FaceAnalysis 인스턴스
        enroll_root: enroll 폴더 경로 (예: images/enroll)
        out_dir: 출력 디렉토리 (예: outputs/embeddings)
        save_bank: bank 저장 여부
        save_centroid: centroid 저장 여부
    """
    print(f"{'='*70}")
    print(f"📝 MODE 1: 기본 등록 (Basic Enrollment)")
    print(f"{'='*70}")
    print(f"   입력 폴더: {enroll_root}")
    print(f"   출력 폴더: {out_dir}")
    print()
    
    if not enroll_root.exists():
        raise FileNotFoundError(f"enroll 폴더를 찾을 수 없음: {enroll_root}")
    
    person_dirs = [p for p in enroll_root.iterdir() if p.is_dir()]
    if not person_dirs:
        print(f"⚠️ {enroll_root} 안에 사람별 폴더가 없습니다.")
        return
    
    print("👥 등록 대상 사람 목록:")
    for d in person_dirs:
        print(f"  - {d.name}")
    print()
    
    for person_dir in person_dirs:
        person_id = person_dir.name
        print(f"\n===== {person_id} 등록 시작 =====")
        
        emb_list = []
        for img_path in sorted(person_dir.glob("*")):
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            
            print(f"  ▶ 이미지 처리: {img_path.name}")
            emb = get_main_face_embedding(app, img_path)
            if emb is None:
                continue
            emb_list.append(emb)
        
        if not emb_list:
            print(f"  ❌ 유효한 얼굴 임베딩 없음 → {person_id} 스킵")
            continue
        
        print(f"  ✅ {person_id} 등록 완료 ({len(emb_list)}장 사용)")
        save_embeddings(person_id, emb_list, out_dir, save_bank, save_centroid)
    
    print(f"\n🎉 기본 등록 완료!")


# ===== MODE 2: 수동 추가 =====
def mode_manual_add(app: FaceAnalysis, person_id: str, image_paths: list[Path],
                   out_dir: Path, similarity_threshold: float = 0.95):
    """
    특정 이미지들을 bank에 수동으로 추가
    
    Args:
        app: FaceAnalysis 인스턴스
        person_id: 사람 ID
        image_paths: 추가할 이미지 경로 리스트
        out_dir: 출력 디렉토리
        similarity_threshold: 중복 체크 임계값
    
    Returns:
        추가된 임베딩 개수
    """
    print(f"{'='*70}")
    print(f"📁 MODE 2: 수동 추가 (Manual Add)")
    print(f"{'='*70}")
    print(f"   대상 인물: {person_id}")
    print(f"   이미지 개수: {len(image_paths)}개")
    print(f"   중복 체크 임계값: {similarity_threshold}")
    print()
    
    # 사람별 폴더 우선, 없으면 루트에서 찾기
    person_dir = out_dir / person_id
    bank_path = person_dir / "bank.npy"
    if not bank_path.exists():
        bank_path = out_dir / f"{person_id}_bank.npy"
    
    # 기존 bank 로드
    if bank_path.exists():
        bank = np.load(bank_path)
        print(f"📚 기존 bank: {bank.shape[0]}개 임베딩 ({bank_path})")
    else:
        bank = np.empty((0, 512), dtype=np.float32)
        print(f"📚 새 bank 생성")
    
    new_embeddings = []
    skipped_count = 0
    
    for img_path in image_paths:
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        
        print(f"  ▶ 처리 중: {img_path.name}")
        emb = get_main_face_embedding(app, img_path)
        
        if emb is None:
            skipped_count += 1
            continue
        
        # 중복 체크
        if bank.shape[0] > 0:
            max_sim = float(np.max(bank @ emb))
            if max_sim >= similarity_threshold:
                print(f"     ⏭ 스킵 (기존 임베딩과 유사도 {max_sim:.3f} >= {similarity_threshold})")
                skipped_count += 1
                continue
        
        new_embeddings.append(emb)
        max_sim = float(np.max(bank @ emb)) if bank.shape[0] > 0 else 0.0
        print(f"     ✅ 추가 (기존 bank와 최대 유사도: {max_sim:.3f})")
    
    if not new_embeddings:
        print(f"\n⚠️ 추가할 새로운 임베딩이 없습니다. (스킵: {skipped_count}개)")
        return 0
    
    # Bank에 추가
    new_embs_array = np.stack(new_embeddings, axis=0)
    updated_bank = np.vstack([bank, new_embs_array])
    
    # Centroid 재계산
    updated_centroid = updated_bank.mean(axis=0)
    updated_centroid = l2_normalize(updated_centroid)
    
    # 저장 (사람별 폴더에 저장)
    person_dir = out_dir / person_id
    person_dir.mkdir(parents=True, exist_ok=True)
    
    bank_path_new = person_dir / "bank.npy"
    np.save(bank_path_new, updated_bank)
    
    centroid_path_new = person_dir / "centroid.npy"
    np.save(centroid_path_new, updated_centroid)
    
    print(f"\n✅ Bank 업데이트 완료!")
    print(f"   추가된 임베딩: {len(new_embeddings)}개")
    print(f"   총 임베딩 수: {updated_bank.shape[0]}개 (기존 {bank.shape[0]}개 + 신규 {len(new_embeddings)}개)")
    print(f"   저장 위치: {person_dir}")
    
    return len(new_embeddings)


def main():
    # ===== 설정 =====
    MODE = 1  # 1: 기본 등록, 2: 수동 추가
    
    enroll_root = Path("images") / "enroll"
    out_dir = Path("outputs") / "embeddings"
    
    # MODE 2 설정
    person_id = "hani"  # 대상 인물 ID
    image_folder = Path("images") / "extracted_frames" / person_id  # 수동 추가할 폴더
    
    print(f"{'='*70}")
    print(f"🎯 얼굴 임베딩 추출 시스템")
    print(f"{'='*70}")
    print(f"   모드: {MODE} (1: 기본 등록, 2: 수동 추가)")
    print(f"   출력 폴더: {out_dir}")
    print()
    
    # InsightFace 초기화
    device_id = get_device_id()
    device_type = "GPU" if device_id >= 0 else "CPU"
    print(f"🔧 디바이스: {device_type} (ctx_id={device_id})")
    
    app = FaceAnalysis(name="buffalo_l")
    actual_device_id = safe_prepare_insightface(app, device_id, det_size=(640, 640))
    if actual_device_id != device_id:
        print(f"   (실제 사용: {'GPU' if actual_device_id >= 0 else 'CPU'})")
    print()
    
    # 모드별 실행
    if MODE == 1:
        # 기본 등록: enroll 폴더에서 모든 사람 등록
        mode_basic_enroll(app, enroll_root, out_dir, save_bank=True, save_centroid=True)
    
    elif MODE == 2:
        # 수동 추가: 특정 폴더의 이미지들을 bank에 추가
        if not image_folder.exists():
            print(f"⚠️ 이미지 폴더를 찾을 수 없음: {image_folder}")
            return
        
        image_paths = [p for p in sorted(image_folder.glob("*")) 
                      if p.suffix.lower() in IMG_EXTS]
        
        if not image_paths:
            print(f"⚠️ {image_folder} 안에 이미지 파일이 없습니다.")
            return
        
        added_count = mode_manual_add(
            app=app,
            person_id=person_id,
            image_paths=image_paths,
            out_dir=out_dir,
            similarity_threshold=0.95
        )
        
        if added_count > 0:
            print(f"\n💡 다음 단계:")
            print(f"   python src/face_match_cctv.py 실행하여 인식 성능 확인")
    
    else:
        print(f"❌ 잘못된 모드: {MODE} (1 또는 2 중 선택)")
    
    print(f"\n{'='*70}")
    print(f"✅ 작업 완료!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

