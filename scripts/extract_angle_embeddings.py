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
from src.utils.face_angle_detector import estimate_face_angle
import cv2

# 설정
ENROLL_DIR = PROJECT_ROOT / "images" / "enroll"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "embeddings_manual"  # 비교 테스트용 별도 폴더

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}

# 각도별 파일명 패턴
ANGLE_PATTERNS = {
    "left": ["left", "_l", "_L", "fleft"],  # fleft = front-left (정면이면서 왼쪽)
    "right": ["right", "_r", "_R", "fright"],  # fright = front-right (정면이면서 오른쪽)
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
    
    # fleft, fright는 우선적으로 처리 (front-left, front-right)
    # frignt는 fright의 오타로 보이지만 인식하도록 처리
    if "fleft" in filename_lower:
        return "left"  # front-left는 left 카테고리로 분류
    if "fright" in filename_lower or "frignt" in filename_lower:
        return "right"  # front-right는 right 카테고리로 분류
    
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
        
        # 각도 정보 저장용 (평가 시 사용)
        angles_info = {
            "angle_types": [],
            "yaw_angles": [],
            "pitch_angles": [],
            "file_mapping": []  # 각 임베딩이 어떤 파일에서 왔는지
        }
        
        for angle_type, img_files in angle_groups.items():
            if not img_files:
                continue
            
            print(f"\n  📐 {angle_type.upper()} 각도 ({len(img_files)}개 파일):")
            
            embeddings_list = []
            angle_data_list = []  # 각 이미지의 각도 정보
            
            for img_file in img_files:
                print(f"    ▶ {img_file.name}")
                
                # 이미지 로드 및 얼굴 감지
                img = cv2.imread(str(img_file))
                if img is None:
                    print(f"      ❌ 이미지 읽기 실패")
                    continue
                
                faces = app.get(img)
                if len(faces) == 0:
                    # 전처리 후 재시도
                    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    l = clahe.apply(l)
                    enhanced = cv2.merge([l, a, b])
                    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
                    faces = app.get(enhanced)
                    
                    if len(faces) == 0:
                        h, w = img.shape[:2]
                        if h < 1280 or w < 1280:
                            scale = max(1280 / h, 1280 / w)
                            new_h, new_w = int(h * scale), int(w * scale)
                            upscaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                            faces = app.get(upscaled)
                            
                            # 업스케일링 후에도 실패하면 전처리 + 업스케일링 조합 시도
                            if len(faces) == 0:
                                upscaled_enhanced = cv2.resize(enhanced, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                                faces = app.get(upscaled_enhanced)
                
                if len(faces) == 0:
                    print(f"      ❌ 얼굴 감지 실패")
                    continue
                
                # 가장 큰 얼굴 선택
                faces_sorted = sorted(
                    faces,
                    key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                    reverse=True
                )
                main_face = faces_sorted[0]
                
                # 임베딩 추출
                emb = main_face.embedding.astype("float32")
                emb = l2_normalize(emb)
                embeddings_list.append(emb)
                
                # 각도 측정
                detected_angle_type, yaw_angle = estimate_face_angle(main_face)
                
                # Pitch 각도도 계산 (estimate_face_angle 내부 로직 사용)
                kps = main_face.kps
                left_eye = kps[0]
                right_eye = kps[1]
                nose = kps[2]
                left_mouth = kps[3]
                right_mouth = kps[4]
                
                eye_center_y = (left_eye[1] + right_eye[1]) / 2
                mouth_center_y = (left_mouth[1] + right_mouth[1]) / 2
                eye_to_mouth_distance = abs(mouth_center_y - eye_center_y)
                eye_to_nose_distance = abs(nose[1] - eye_center_y)
                
                if eye_to_mouth_distance > 1e-6:
                    nose_ratio = eye_to_nose_distance / eye_to_mouth_distance
                    pitch_angle = (0.5 - nose_ratio) * 90.0
                else:
                    pitch_angle = 0.0
                
                angle_data_list.append({
                    "file": img_file.name,
                    "detected_angle_type": detected_angle_type,
                    "yaw_angle": float(yaw_angle),
                    "pitch_angle": float(pitch_angle),
                    "file_angle_type": angle_type  # 파일명 기반 각도
                })
                
                angles_info["angle_types"].append(angle_type)
                angles_info["yaw_angles"].append(float(yaw_angle))
                angles_info["pitch_angles"].append(float(pitch_angle))
                angles_info["file_mapping"].append({
                    "file": img_file.name,
                    "angle_type": angle_type,
                    "detected_angle_type": detected_angle_type,
                    "yaw": float(yaw_angle),
                    "pitch": float(pitch_angle)
                })
                
                print(f"      ✅ 임베딩 추출 완료 (각도: {detected_angle_type}, yaw: {yaw_angle:.1f}°, pitch: {pitch_angle:.1f}°)")
            
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
                
                # 각도 정보 저장 (평가 시 사용)
                angles_file = person_output_dir / "angles_manual.json"
                with open(angles_file, 'w', encoding='utf-8') as f:
                    json.dump(angles_info, f, indent=2, ensure_ascii=False)
                
                person_results["embeddings"][angle_type] = {
                    "file": str(embedding_file.relative_to(PROJECT_ROOT)),
                    "count": len(embeddings_list),
                    "centroid_norm": float(np.linalg.norm(centroid)),
                    "angles": angle_data_list
                }
                
                print(f"    💾 저장 완료: {embedding_file.name} ({len(embeddings_list)}개 임베딩)")
                print(f"    💾 각도 정보 저장: {angles_file.name}")
        
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


