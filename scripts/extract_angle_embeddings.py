"""
각도별 임베딩 추출 스크립트
수동 촬영한 left, right, top 사진에서 임베딩을 추출하여 비교 테스트용으로 저장
"""
import sys
import json
from pathlib import Path
import numpy as np

# 프로젝트 루트를 Python 경로에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# CUDA 경로를 먼저 설정
from src.utils.device_config import _ensure_cuda_in_path
_ensure_cuda_in_path()

from insightface.app import FaceAnalysis
from src.utils.device_config import get_device_id, safe_prepare_insightface
from src.face_enroll import get_main_face_embedding, l2_normalize

# 설정
ENROLL_DIR = PROJECT_ROOT / "images" / "enroll"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "embeddings_manual"  # 비교 테스트용 별도 폴더

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}

# 각도별 파일명 패턴
ANGLE_PATTERNS = {
    "left": ["left", "_l", "_L"],
    "right": ["right", "_r", "_R"],
    "top": ["top", "_t", "_T", "up", "_u", "_U"]
}


def detect_angle_from_filename(filename: str) -> str:
    """
    파일명에서 각도 추정
    
    Args:
        filename: 이미지 파일명
    
    Returns:
        각도 타입: "left", "right", "top", 또는 "front" (기본값)
    """
    filename_lower = filename.lower()
    
    for angle, patterns in ANGLE_PATTERNS.items():
        for pattern in patterns:
            if pattern in filename_lower:
                return angle
    
    return "front"  # 기본값


def extract_angle_embeddings():
    """각도별 임베딩 추출"""
    print("=" * 70)
    print("📸 각도별 임베딩 추출 스크립트")
    print("=" * 70)
    print(f"입력 폴더: {ENROLL_DIR}")
    print(f"출력 폴더: {OUTPUT_DIR}")
    print()
    
    # 출력 폴더 생성
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # InsightFace 초기화
    print("🔧 InsightFace 초기화 중...")
    device_id = get_device_id()
    device_type = "GPU" if device_id >= 0 else "CPU"
    print(f"   디바이스: {device_type} (ctx_id={device_id})")
    
    app = FaceAnalysis(name="buffalo_l")
    # 측면 얼굴 감지를 위해 detection size를 더 크게 설정
    actual_device_id = safe_prepare_insightface(app, device_id, det_size=(1280, 1280))
    if actual_device_id != device_id:
        print(f"   (실제 사용: {'GPU' if actual_device_id >= 0 else 'CPU'})")
    print()
    
    # 각 인물별 폴더 확인
    if not ENROLL_DIR.exists():
        print(f"❌ enroll 폴더를 찾을 수 없습니다: {ENROLL_DIR}")
        return
    
    person_dirs = [d for d in ENROLL_DIR.iterdir() if d.is_dir()]
    
    if not person_dirs:
        print(f"⚠️ {ENROLL_DIR} 안에 인물 폴더가 없습니다.")
        return
    
    print(f"👥 처리 대상 인물: {len(person_dirs)}명")
    for d in person_dirs:
        print(f"   - {d.name}")
    print()
    
    # 각 인물별 처리
    all_results = {}
    
    for person_dir in person_dirs:
        person_id = person_dir.name
        print(f"\n{'='*70}")
        print(f"👤 {person_id} 처리 중...")
        print(f"{'='*70}")
        
        # 이미지 파일 찾기
        image_files = [
            f for f in sorted(person_dir.iterdir())
            if f.is_file() and f.suffix.lower() in IMG_EXTS
        ]
        
        if not image_files:
            print(f"  ⚠️ 이미지 파일이 없습니다.")
            continue
        
        # 각도별로 그룹화
        angle_groups = {
            "left": [],
            "right": [],
            "top": [],
            "front": []
        }
        
        for img_file in image_files:
            angle = detect_angle_from_filename(img_file.name)
            angle_groups[angle].append(img_file)
        
        # 각 각도별 임베딩 추출
        person_results = {
            "person_id": person_id,
            "embeddings": {}
        }
        
        for angle_type, img_files in angle_groups.items():
            if not img_files:
                continue
            
            print(f"\n  📐 {angle_type.upper()} 각도 ({len(img_files)}개 파일):")
            
            embeddings_list = []
            
            for img_file in img_files:
                print(f"    ▶ {img_file.name}")
                embedding = get_main_face_embedding(app, img_file)
                
                if embedding is None:
                    print(f"      ❌ 얼굴 감지 실패")
                    continue
                
                embeddings_list.append(embedding)
                print(f"      ✅ 임베딩 추출 완료")
            
            if embeddings_list:
                # 여러 임베딩의 평균 (centroid)
                embeddings_array = np.stack(embeddings_list, axis=0)
                centroid = embeddings_array.mean(axis=0)
                centroid = l2_normalize(centroid)
                
                # 저장
                person_output_dir = OUTPUT_DIR / person_id
                person_output_dir.mkdir(parents=True, exist_ok=True)
                
                # 각도별 임베딩 저장
                embedding_file = person_output_dir / f"embedding_{angle_type}.npy"
                np.save(embedding_file, centroid)
                
                # 모든 임베딩도 저장 (선택적)
                bank_file = person_output_dir / f"bank_{angle_type}.npy"
                np.save(bank_file, embeddings_array)
                
                person_results["embeddings"][angle_type] = {
                    "file": str(embedding_file.relative_to(PROJECT_ROOT)),
                    "count": len(embeddings_list),
                    "centroid_norm": float(np.linalg.norm(centroid))
                }
                
                print(f"    💾 저장 완료: {embedding_file.name} ({len(embeddings_list)}개 임베딩)")
        
        all_results[person_id] = person_results
    
    # 전체 결과 요약 저장
    summary_file = OUTPUT_DIR / "extraction_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # 결과 출력
    print("\n" + "=" * 70)
    print("✅ 추출 완료!")
    print("=" * 70)
    print(f"\n📊 추출 결과:")
    
    for person_id, result in all_results.items():
        print(f"\n  👤 {person_id}:")
        for angle_type, info in result["embeddings"].items():
            print(f"    - {angle_type}: {info['count']}개 임베딩")
    
    print(f"\n📁 저장 위치:")
    print(f"   {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")
    print(f"\n📄 요약 파일:")
    print(f"   {summary_file.relative_to(PROJECT_ROOT)}")
    print("=" * 70)


if __name__ == "__main__":
    extract_angle_embeddings()


