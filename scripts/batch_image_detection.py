"""
이미지 배치 인식 스크립트
images/test/ 폴더의 모든 이미지에 대해 얼굴 인식 수행 및 결과 저장
"""
import sys
import json
from pathlib import Path
import cv2
import numpy as np
from typing import List, Dict

# 프로젝트 루트를 Python 경로에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# CUDA 경로를 먼저 설정
from src.utils.device_config import _ensure_cuda_in_path
_ensure_cuda_in_path()

from insightface.app import FaceAnalysis
from src.utils.device_config import get_device_id, safe_prepare_insightface
from backend.database import SessionLocal, get_db
from backend.main import process_detection, load_persons_from_db

# 설정
TEST_IMAGE_DIR = PROJECT_ROOT / "images" / "test_easy"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "test_results_base_only_easy"  # Base Bank만 사용 모드
OUTPUT_IMAGES_DIR = OUTPUT_DIR / "images"
OUTPUT_JSON_DIR = OUTPUT_DIR / "annotations"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}


def process_single_image(image_path: Path, db) -> Dict:
    """
    단일 이미지 처리
    
    Args:
        image_path: 이미지 파일 경로
        db: 데이터베이스 세션
    
    Returns:
        처리 결과 딕셔너리
    """
    print(f"\n📷 처리 중: {image_path.name}")
    
    # 이미지 로드
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  ❌ 이미지 로드 실패")
        return None
    
    # BGR to RGB 변환 (process_detection은 RGB를 기대할 수 있음)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 얼굴 인식 수행 (전체 DB 검색)
    tracking_state = {"tracks": {}}
    detection_result = process_detection(
        frame=img_rgb,
        suspect_ids=None,  # 전체 갤러리 검색
        db=db,
        tracking_state=tracking_state
    )
    
    # 결과 추출
    detections = detection_result.get("detections", [])
    
    # 박스가 그려진 이미지 생성
    img_with_boxes = img.copy()
    
    # 어노테이션 데이터
    annotation = {
        "image_path": str(image_path.relative_to(PROJECT_ROOT)),
        "image_name": image_path.name,
        "faces": []
    }
    
    # 각 감지 결과에 대해 박스 그리기 및 어노테이션 수집
    for detection in detections:
        bbox = detection["bbox"]
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        
        # 색상 결정
        status = detection.get("status", "unknown")
        if status == "criminal":
            color = (0, 0, 255)  # 빨간색 (BGR)
        elif status == "normal":
            color = (0, 255, 0)  # 초록색 (BGR)
        else:  # unknown
            color = (0, 255, 255)  # 노란색 (BGR)
        
        # 박스 그리기 (두께 3)
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 3)
        
        # 레이블 생성
        name = detection.get("name", "Unknown")
        confidence = detection.get("confidence", 0)
        label = f"{name} ({confidence}%)"
        
        # 레이블 배경 (가독성 향상)
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(
            img_with_boxes,
            (x1, y1 - label_height - 10),
            (x1 + label_width, y1),
            color,
            -1  # 채워진 사각형
        )
        
        # 레이블 텍스트 (흰색)
        cv2.putText(
            img_with_boxes,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),  # 흰색
            2
        )
        
        # JSON 어노테이션 데이터 수집
        face_annotation = {
            "bbox": [x1, y1, x2, y2],
            "status": status,
            "name": name,
            "person_id": detection.get("person_id"),
            "confidence": confidence,
            "color": detection.get("color", "yellow"),
            "angle_type": detection.get("angle_type"),
            "yaw_angle": detection.get("yaw_angle"),
            "bank_type": detection.get("bank_type")
        }
        annotation["faces"].append(face_annotation)
    
    return {
        "image": img_with_boxes,
        "annotation": annotation,
        "detection_count": len(detections)
    }


def main():
    """메인 함수"""
    print("=" * 70)
    print("🖼️  이미지 배치 인식 스크립트")
    print("=" * 70)
    
    # 1. 입력 폴더 확인
    if not TEST_IMAGE_DIR.exists():
        print(f"❌ 테스트 이미지 폴더를 찾을 수 없습니다: {TEST_IMAGE_DIR}")
        print(f"   폴더를 생성하거나 이미지를 추가해주세요.")
        return
    
    # 2. 출력 폴더 생성
    OUTPUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON_DIR.mkdir(parents=True, exist_ok=True)
    
    # 3. 이미지 파일 목록 수집
    image_files = [
        f for f in sorted(TEST_IMAGE_DIR.iterdir())
        if f.is_file() and f.suffix.lower() in IMG_EXTS
    ]
    
    if not image_files:
        print(f"⚠️ {TEST_IMAGE_DIR} 안에 이미지 파일이 없습니다.")
        return
    
    print(f"\n📂 입력 폴더: {TEST_IMAGE_DIR}")
    print(f"📂 출력 폴더: {OUTPUT_DIR}")
    print(f"📊 처리할 이미지: {len(image_files)}개")
    print()
    
    # 4. InsightFace 초기화
    print("🔧 InsightFace 초기화 중...")
    device_id = get_device_id()
    device_type = "GPU" if device_id >= 0 else "CPU"
    print(f"   디바이스: {device_type} (ctx_id={device_id})")
    
    model = FaceAnalysis(name="buffalo_l")
    actual_device_id = safe_prepare_insightface(model, device_id, det_size=(640, 640))
    if actual_device_id != device_id:
        print(f"   (실제 사용: {'GPU' if actual_device_id >= 0 else 'CPU'})")
    print()
    
    # 5. 데이터베이스 연결 및 데이터 로드 (base bank만 사용)
    print("🗄️  데이터베이스 연결 중...")
    print("   ⚠️ Base Bank만 사용 모드 (Masked Bank 제외)")
    db = SessionLocal()
    try:
        load_persons_from_db(db)
        # Masked Bank 비우기 (Base Bank만 사용)
        from backend.main import gallery_masked_cache
        gallery_masked_cache.clear()
        print("   ✅ 데이터베이스 로드 완료 (Base Bank만 사용)")
    except Exception as e:
        print(f"   ⚠️ 데이터베이스 로드 실패: {e}")
        print("   파일 시스템에서 로드 시도...")
        from backend.main import load_persons_from_embeddings, gallery_masked_cache
        load_persons_from_embeddings()
        # Masked Bank 비우기 (Base Bank만 사용)
        gallery_masked_cache.clear()
        print("   ✅ 파일 시스템 로드 완료 (Base Bank만 사용)")
    
    # 6. 각 이미지 처리
    print("\n" + "=" * 70)
    print("🔄 이미지 처리 시작")
    print("=" * 70)
    
    total_faces = 0
    processed_count = 0
    
    for image_path in image_files:
        try:
            result = process_single_image(image_path, db)
            
            if result is None:
                continue
            
            # 결과 이미지 저장
            output_image_path = OUTPUT_IMAGES_DIR / image_path.name
            cv2.imwrite(str(output_image_path), result["image"], [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # JSON 어노테이션 저장
            json_filename = image_path.stem + ".json"
            json_path = OUTPUT_JSON_DIR / json_filename
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result["annotation"], f, indent=2, ensure_ascii=False)
            
            face_count = result["detection_count"]
            total_faces += face_count
            processed_count += 1
            
            print(f"  ✅ 완료: {image_path.name} (감지된 얼굴: {face_count}개)")
            
        except Exception as e:
            print(f"  ❌ 처리 실패: {image_path.name} - {e}")
            import traceback
            traceback.print_exc()
    
    db.close()
    
    # 7. 결과 요약
    print("\n" + "=" * 70)
    print("✅ 처리 완료!")
    print("=" * 70)
    print(f"📊 처리 통계:")
    print(f"   - 처리된 이미지: {processed_count}/{len(image_files)}개")
    print(f"   - 총 감지된 얼굴: {total_faces}개")
    print(f"\n📁 결과 저장 위치:")
    print(f"   - 이미지: {OUTPUT_IMAGES_DIR.relative_to(PROJECT_ROOT)}")
    print(f"   - JSON: {OUTPUT_JSON_DIR.relative_to(PROJECT_ROOT)}")
    print("=" * 70)


if __name__ == "__main__":
    main()

