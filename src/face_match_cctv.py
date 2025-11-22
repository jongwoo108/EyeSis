# src/face_match_cctv_final.py
"""
CCTV 용의자 식별 최종 통합 스크립트
모든 고급 기능을 통합한 프로덕션 레벨 코드

주요 기능:
- 얼굴 각도 감지 (정면/측면/프로필)
- 마스크 감지 및 적응형 임계값
- sim_gap 체크로 오탐 방지
- 중복 얼굴 필터링
- CSV 로그 저장
- 스냅샷 저장
- 상세한 통계 출력
"""
# CUDA 경로를 먼저 설정 (가장 먼저 import)
from utils.device_config import _ensure_cuda_in_path
_ensure_cuda_in_path()

from insightface.app import FaceAnalysis
import cv2
import numpy as np
from pathlib import Path
import csv
import time
from datetime import datetime
from collections import defaultdict
from utils.gallery_loader import load_gallery, match_with_bank, match_with_bank_detailed
from utils.device_config import get_device_id, safe_prepare_insightface
from utils.mask_detector import estimate_mask_from_similarity, get_adjusted_threshold, estimate_face_quality
from utils.face_angle_detector import estimate_face_angle


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """벡터를 L2 정규화"""
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def add_embedding_to_bank(person_id: str, embedding: np.ndarray, emb_dir: Path, 
                          similarity_threshold: float = 0.95, verbose: bool = False,
                          angle_type: str = None, yaw_angle: float = None):
    """
    매칭된 얼굴의 임베딩을 Bank에 추가
    
    Args:
        person_id: 인물 ID
        embedding: 추가할 임베딩 (512차원, L2 정규화됨)
        emb_dir: 임베딩 저장 디렉토리
        similarity_threshold: 중복 체크 임계값 (이상이면 중복으로 간주)
        verbose: 상세 출력 여부
        angle_type: 얼굴 각도 타입 (front, left, right, left_profile, right_profile)
        yaw_angle: yaw 각도 값 (도 단위)
    
    Returns:
        추가 성공 여부 (True: 추가됨, False: 중복으로 스킵)
    """
    import json
    
    # 사람별 폴더 경로
    person_dir = emb_dir / person_id
    bank_path = person_dir / "bank.npy"
    angles_path = person_dir / "angles.json"  # 각도 정보 저장 파일
    
    # 기존 bank 로드
    if bank_path.exists():
        bank = np.load(bank_path)
    else:
        bank = np.empty((0, 512), dtype=np.float32)
    
    # 기존 각도 정보 로드
    if angles_path.exists():
        with open(angles_path, 'r', encoding='utf-8') as f:
            angles_info = json.load(f)
    else:
        angles_info = {"angle_types": [], "yaw_angles": []}
    
    # 중복 체크
    if bank.shape[0] > 0:
        max_sim = float(np.max(bank @ embedding))
        if max_sim >= similarity_threshold:
            if verbose:
                print(f"     ⏭ Bank 스킵 (중복: {max_sim:.3f} >= {similarity_threshold})")
            return False  # 중복으로 스킵
    
    # Bank에 추가
    new_emb = embedding.reshape(1, -1)  # (1, 512)
    updated_bank = np.vstack([bank, new_emb])
    
    # 각도 정보 추가
    angles_info["angle_types"].append(angle_type if angle_type else "unknown")
    angles_info["yaw_angles"].append(float(yaw_angle) if yaw_angle is not None else 0.0)
    
    # Centroid 재계산
    updated_centroid = updated_bank.mean(axis=0)
    updated_centroid = l2_normalize(updated_centroid)
    
    # 저장
    person_dir.mkdir(parents=True, exist_ok=True)
    np.save(bank_path, updated_bank)
    centroid_path = person_dir / "centroid.npy"
    np.save(centroid_path, updated_centroid)
    
    # 각도 정보 저장
    with open(angles_path, 'w', encoding='utf-8') as f:
        json.dump(angles_info, f, indent=2, ensure_ascii=False)
    
    if verbose:
        angle_info = f" [{angle_type}]" if angle_type else ""
        print(f"     ✅ Bank 추가: {person_id} (총 {updated_bank.shape[0]}개 임베딩{angle_info})")
    
    return True  # 추가 성공


def calculate_bbox_iou(bbox1, bbox2):
    """
    두 bbox 간의 IoU(Intersection over Union) 계산
    
    Args:
        bbox1, bbox2: [x1, y1, x2, y2] 형식의 바운딩 박스
    
    Returns:
        IoU 값 (0.0 ~ 1.0)
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # 교집합 영역 계산
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # 각 bbox의 면적
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def calculate_bbox_center_distance(bbox1, bbox2):
    """
    두 bbox의 중심점 간 거리 계산
    
    Args:
        bbox1, bbox2: [x1, y1, x2, y2] 형식의 바운딩 박스
    
    Returns:
        중심점 간 유클리드 거리
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    center1_x = (x1_1 + x2_1) / 2
    center1_y = (y1_1 + y2_1) / 2
    center2_x = (x1_2 + x2_2) / 2
    center2_y = (y1_2 + y2_2) / 2
    
    distance = np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
    return distance


def is_same_face_region(bbox1, bbox2, iou_threshold=0.3, distance_threshold=None):
    """
    두 bbox가 같은 얼굴 영역을 가리키는지 판단
    
    Args:
        bbox1, bbox2: [x1, y1, x2, y2] 형식의 바운딩 박스
        iou_threshold: IoU 임계값 (기본 0.3)
        distance_threshold: 중심점 거리 임계값 (None이면 bbox 크기 기반 자동 계산)
    
    Returns:
        같은 얼굴 영역이면 True, 아니면 False
    """
    # IoU 기반 판단
    iou = calculate_bbox_iou(bbox1, bbox2)
    if iou >= iou_threshold:
        return True
    
    # 중심점 거리 기반 판단 (보조)
    if distance_threshold is None:
        # bbox 크기의 평균을 기준으로 임계값 설정
        w1 = bbox1[2] - bbox1[0]
        h1 = bbox1[3] - bbox1[1]
        w2 = bbox2[2] - bbox2[0]
        h2 = bbox2[3] - bbox2[1]
        avg_size = (w1 + h1 + w2 + h2) / 4
        distance_threshold = avg_size * 0.5  # bbox 크기의 50% 이내면 같은 얼굴로 간주
    
    distance = calculate_bbox_center_distance(bbox1, bbox2)
    if distance <= distance_threshold:
        return True
    
    return False


def process_frame(img, app, gallery, BASE_THRESH, frame_idx=None):
    """
    단일 프레임 처리 - 모든 얼굴 감지 및 매칭
    
    Args:
        img: BGR 이미지 (numpy array)
        app: FaceAnalysis 객체
        gallery: 갤러리 딕셔너리
        BASE_THRESH: 기본 임계값
        frame_idx: 프레임 인덱스 (None이면 이미지)
    
    Returns:
        results: 얼굴별 분석 결과 리스트
    """
    # 얼굴 검출
    faces = app.get(img)
    
    if len(faces) == 0:
        return []
    
    results = []
    
    for i, face in enumerate(faces):
        face_emb = face.embedding.astype("float32")
        face_emb_normalized = l2_normalize(face_emb)
        
        # 얼굴 각도 추정
        angle_type, yaw_angle = estimate_face_angle(face)
        
        # Bank 기반 매칭 (상세 정보 포함)
        best_id, best_sim, second_sim = match_with_bank_detailed(face_emb, gallery)
        
        # 2차 확인: 최고 유사도와 두 번째 유사도의 차이 확인 (오탐 방지)
        sim_gap = best_sim - second_sim if second_sim > -1 else best_sim
        min_gap = 0.05  # 최소 차이 (5% 이상 차이 필요)
        
        # 화질 추정 (얼굴 크기 기반)
        img_height, img_width = img.shape[:2]
        face_quality = estimate_face_quality(face.bbox, (img_height, img_width))
        
        # 마스크 가능성 추정 및 적응형 임계값 (화질 고려)
        mask_prob = estimate_mask_from_similarity(best_sim)
        use_thresh = get_adjusted_threshold(BASE_THRESH, mask_prob, best_sim, face_quality)
        
        # 매칭 여부: 임계값 통과 + 유사도 차이가 충분해야 함
        is_match = (best_sim >= use_thresh) and (sim_gap >= min_gap)
        
        # 결과 저장
        results.append({
            "face_idx": i,
            "frame_idx": frame_idx,
            "angle_type": angle_type,
            "yaw_angle": yaw_angle,
            "best_id": best_id,
            "similarity": best_sim,
            "second_similarity": second_sim,
            "sim_gap": sim_gap,
            "threshold": use_thresh,
            "is_match": is_match,
            "bbox": face.bbox,
            "mask_prob": mask_prob,
            "face_quality": face_quality,  # 화질 정보
            "embedding": face_emb_normalized  # 중복 체크용
        })
    
    # 같은 프레임 내에서 매칭 필터링 (bbox 기반 다중 매칭 처리)
    if len(results) > 1:
        matched_results = []
        unmatched_results = []
        
        for r in results:
            if r["is_match"]:
                matched_results.append(r)
            else:
                unmatched_results.append(r)
        
        if len(matched_results) > 1:
            # bbox 기반으로 같은 얼굴 영역 그룹화
            face_groups = []
            used_indices = set()
            
            for i, r1 in enumerate(matched_results):
                if i in used_indices:
                    continue
                
                # 새로운 그룹 시작
                group = [r1]
                used_indices.add(i)
                
                # 같은 얼굴 영역인 다른 매칭 찾기
                for j, r2 in enumerate(matched_results):
                    if j <= i or j in used_indices:
                        continue
                    
                    if is_same_face_region(r1["bbox"], r2["bbox"]):
                        group.append(r2)
                        used_indices.add(j)
                
                face_groups.append(group)
            
            # 각 그룹 처리
            filtered_matched = []
            review_candidates = []  # 검토 대상
            
            for group in face_groups:
                if len(group) == 1:
                    # 단일 매칭: 그대로 유지
                    filtered_matched.append(group[0])
                else:
                    # 같은 얼굴 영역에서 여러 인물로 매칭됨 → 오탐 가능성
                    # 유사도 순으로 정렬
                    group.sort(key=lambda x: x["similarity"], reverse=True)
                    
                    best_match = group[0]
                    second_match = group[1] if len(group) > 1 else None
                    
                    # sim_gap이 충분히 크면 가장 높은 유사도만 인정
                    min_gap_for_confidence = 0.10  # 10% 이상 차이 필요
                    if second_match and (best_match["sim_gap"] >= min_gap_for_confidence):
                        # 확신 있는 매칭
                        filtered_matched.append(best_match)
                        # 나머지는 검토 대상
                        for other in group[1:]:
                            other["is_match"] = False
                            other["review_reason"] = "same_face_multiple_persons"
                            review_candidates.append(other)
                            unmatched_results.append(other)
                    else:
                        # sim_gap이 작아서 애매한 경우 → 모두 검토 대상
                        for match in group:
                            match["is_match"] = False
                            match["review_reason"] = "ambiguous_match"
                            review_candidates.append(match)
                            unmatched_results.append(match)
            
            # 다른 얼굴 영역의 매칭들도 검토
            # 낮은 유사도나 작은 sim_gap인 경우 검토 대상으로 분리
            # 화질에 따라 임계값 조정
            for match in filtered_matched:
                quality = match.get("face_quality", "medium")
                # 고화질일 때는 더 엄격하게, 저화질일 때는 관대하게
                sim_threshold = 0.38 if quality == "high" else (0.35 if quality == "medium" else 0.32)
                gap_threshold = 0.10 if quality == "high" else (0.08 if quality == "medium" else 0.06)
                
                if match["similarity"] < sim_threshold or match["sim_gap"] < gap_threshold:
                    match["review_reason"] = "low_confidence"
                    review_candidates.append(match)
            
            results = filtered_matched + unmatched_results
            
            # review_reason이 있는 결과에 플래그 추가
            for r in results:
                if "review_reason" not in r:
                    r["review_reason"] = None
        elif len(matched_results) == 1:
            # 단일 매칭도 낮은 신뢰도면 검토 대상
            match = matched_results[0]
            quality = match.get("face_quality", "medium")
            # 화질에 따라 임계값 조정
            sim_threshold = 0.38 if quality == "high" else (0.35 if quality == "medium" else 0.32)
            gap_threshold = 0.10 if quality == "high" else (0.08 if quality == "medium" else 0.06)
            
            if match["similarity"] < sim_threshold or match["sim_gap"] < gap_threshold:
                match["review_reason"] = "low_confidence"
            else:
                match["review_reason"] = None
            results = matched_results + unmatched_results
        else:
            # 매칭이 없는 경우에도 review_reason 초기화
            for r in unmatched_results:
                if "review_reason" not in r:
                    r["review_reason"] = None
            results = unmatched_results
    
    # 모든 결과에 review_reason이 있는지 확인
    for r in results:
        if "review_reason" not in r:
            r["review_reason"] = None
    
    return results


def main():
    # ===== 설정 =====
    # 입력 파일 경로 설정 (추출용 소스 파일)
    # 우선순위: images/source/ 또는 videos/source/ → 루트 폴더 (호환성)
    input_filename = "catch_criminal.MOV"  # 파일명만 지정 (확장자로 자동 감지)
    
    # 파일 타입에 따라 폴더 선택
    file_ext = Path(input_filename).suffix.lower()
    IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}
    VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.gif', '.webm'}
    
    # 추출용 소스 파일 경로 찾기 (우선순위: source 폴더 → 루트 폴더)
    if file_ext in IMAGE_EXTS:
        # 이미지: images/source/ 우선, 없으면 images/ 루트
        input_path = Path("images") / "source" / input_filename
        if not input_path.exists():
            input_path = Path("images") / input_filename
    elif file_ext in VIDEO_EXTS:
        # 영상: videos/source/ 우선, 없으면 videos/ 루트, 마지막으로 images/ (호환성)
        input_path = Path("videos") / "source" / input_filename
        if not input_path.exists():
            input_path = Path("videos") / input_filename
        if not input_path.exists():
            input_path = Path("images") / input_filename
    else:
        # 확장자가 없거나 알 수 없는 경우, 모든 가능한 위치 확인
        input_path = Path("videos") / "source" / input_filename
        if not input_path.exists():
            input_path = Path("videos") / input_filename
        if not input_path.exists():
            input_path = Path("images") / "source" / input_filename
        if not input_path.exists():
            input_path = Path("images") / input_filename
    
    emb_dir = Path("outputs") / "embeddings"  # 등록 임베딩 폴더
    BASE_THRESH = 0.32                        # 기본 임계값 (화질 기반 조정 전)
    
    # Bank 자동 추가 설정
    AUTO_ADD_TO_BANK = True  # 매칭 성공 시 Bank에 자동 추가 여부
    BANK_DUPLICATE_THRESHOLD = 0.95  # 중복 체크 임계값 (0.95 이상이면 중복으로 간주)
    
    # 파일명 기반 출력 폴더 구조 (타임스탬프 포함)
    stem = input_path.stem  # 파일명 (확장자 제외)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 예: 20240101_120000
    output_base_dir = Path("outputs") / "results" / f"{stem}_{timestamp}"  # outputs/results/ive_iam_20240101_120000/
    
    # 하위 폴더들
    matches_dir = output_base_dir / "matches"      # outputs/results/yh/matches/ (매칭된 스냅샷)
    review_dir = output_base_dir / "matches" / "review"  # 검토 대상 스냅샷
    logs_dir = output_base_dir / "logs"            # outputs/results/yh/logs/ (CSV 로그)
    frames_dir = output_base_dir / "frames"        # outputs/results/yh/frames/ (추출된 프레임)
    
    # 폴더 생성
    matches_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    log_path = logs_dir / "detection_log.csv"
    
    # 파일 존재 확인
    if not input_path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없음: {input_path}")
    
    # 파일 타입 자동 감지 (이미 위에서 설정됨)
    file_ext = input_path.suffix.lower()
    is_image = file_ext in IMAGE_EXTS
    is_video = file_ext in VIDEO_EXTS
    
    if not (is_image or is_video):
        raise ValueError(f"지원하지 않는 파일 형식: {file_ext}\n"
                        f"지원 형식: 이미지 {IMAGE_EXTS}, 영상 {VIDEO_EXTS}")
    
    print(f"{'='*70}")
    print(f"🕵️ CCTV 용의자 식별 시스템 (최종 통합 버전)")
    print(f"{'='*70}")
    print(f"   입력 파일: {input_path}")
    print(f"   파일 타입: {'이미지' if is_image else '영상'}")
    print(f"   임베딩 폴더: {emb_dir}")
    print(f"   기본 임계값: {BASE_THRESH}")
    print(f"   Bank 자동 추가: {'활성화' if AUTO_ADD_TO_BANK else '비활성화'}")
    if AUTO_ADD_TO_BANK:
        print(f"     - 중복 체크 임계값: {BANK_DUPLICATE_THRESHOLD}")
    print(f"   출력 폴더: {output_base_dir}")
    print(f"     - 매칭 스냅샷: {matches_dir}")
    print(f"     - 검토 대상: {review_dir}")
    print(f"     - 로그 파일: {logs_dir}")
    print(f"     - 프레임 이미지: {frames_dir}")
    print()
    
    # 1. 갤러리 로드 (bank 우선)
    gallery = load_gallery(emb_dir, use_bank=True)
    if not gallery:
        raise RuntimeError(f"갤러리 비어 있음: {emb_dir}")
    
    print("👥 갤러리 로드 완료:", list(gallery.keys()))
    for pid, data in gallery.items():
        if data.ndim == 2:
            print(f"  - {pid}: bank ({data.shape[0]}개 임베딩)")
        else:
            print(f"  - {pid}: centroid")
    print()
    
    # 2. InsightFace 준비 (GPU 우선, 없으면 CPU)
    device_id = get_device_id()
    device_type = "GPU" if device_id >= 0 else "CPU"
    print(f"🔧 디바이스: {device_type} (ctx_id={device_id})")
    
    app = FaceAnalysis(name="buffalo_l")
    actual_device_id = safe_prepare_insightface(app, device_id, det_size=(640, 640))
    if actual_device_id != device_id:
        print(f"   (실제 사용: {'GPU' if actual_device_id >= 0 else 'CPU'})")
    print("   Detection size: (640, 640)")
    print()
    
    # 3. CSV 로그 파일 열기
    log_f = open(log_path, "w", newline="", encoding="utf-8")
    log_writer = csv.writer(log_f)
    log_writer.writerow([
        "frame", "person_id", "similarity", "threshold", "is_match",
        "angle_type", "yaw_angle", "mask_prob", "sim_gap", "face_quality",
        "x1", "y1", "x2", "y2", "review_reason"
    ])
    
    # 4. 통계 변수 초기화
    frame_idx = 0
    hit_count = 0
    total_faces_detected = 0
    max_sim_ever = -1.0
    bank_added_count = 0  # Bank에 추가된 임베딩 개수
    person_stats = defaultdict(lambda: {"count": 0, "max_sim": 0.0, "angles": defaultdict(int)})
    angle_stats = defaultdict(lambda: {"total": 0, "matched": 0})
    
    # 프레임 저장 옵션 (영상일 때만 사용)
    SAVE_FRAMES = False  # 기본값
    FRAME_INTERVAL = 30  # N프레임마다 저장
    
    start_time = time.time()
    
    # 5. 이미지 또는 영상 처리
    if is_image:
        # ===== 이미지 처리 =====
        print(f"🖼 이미지 분석 시작...")
        print()
        
        img = cv2.imread(str(input_path))
        if img is None:
            raise FileNotFoundError(f"이미지를 읽을 수 없음: {input_path}")
        
        print(f"   이미지 크기: {img.shape[1]}x{img.shape[0]}")
        print()
        
        # 이미지 처리
        frame_results = process_frame(img, app, gallery, BASE_THRESH, None)
        
        if frame_results:
            total_faces_detected = len(frame_results)
            
            for r in frame_results:
                x1, y1, x2, y2 = map(int, r["bbox"])
                
                # CSV 로그 기록
                review_reason = r.get("review_reason", None) or ""
                face_quality = r.get("face_quality", "unknown")
                log_writer.writerow([
                    None, r["best_id"], r["similarity"], r["threshold"],
                    int(r["is_match"]), r["angle_type"], r["yaw_angle"],
                    r["mask_prob"], r["sim_gap"], face_quality,
                    x1, y1, x2, y2, review_reason
                ])
                
                # 통계 업데이트
                if r["similarity"] > max_sim_ever:
                    max_sim_ever = r["similarity"]
                
                # Bank에 자동 추가 (매칭 성공 시)
                bank_added = False
                if r["is_match"] and AUTO_ADD_TO_BANK:
                    bank_added = add_embedding_to_bank(
                        person_id=r["best_id"],
                        embedding=r["embedding"],
                        emb_dir=emb_dir,
                        similarity_threshold=BANK_DUPLICATE_THRESHOLD,
                        verbose=False,
                        angle_type=r.get("angle_type"),
                        yaw_angle=r.get("yaw_angle")
                    )
                    if bank_added:
                        bank_added_count += 1
                
                # 통계 업데이트
                if r["similarity"] > max_sim_ever:
                    max_sim_ever = r["similarity"]
                
                angle_stats[r["angle_type"]]["total"] += 1
                if r["is_match"]:
                    angle_stats[r["angle_type"]]["matched"] += 1
                    person_stats[r["best_id"]]["count"] += 1
                    if r["similarity"] > person_stats[r["best_id"]]["max_sim"]:
                        person_stats[r["best_id"]]["max_sim"] = r["similarity"]
                    person_stats[r["best_id"]]["angles"][r["angle_type"]] += 1
                
                # 결과 표시
                label = f"{r['best_id']} {r['similarity']:.2f}"
                if r.get("face_quality"):
                    quality_emoji = {"high": "🔍", "medium": "📷", "low": "📱"}.get(r["face_quality"], "")
                    label += f" [{r['face_quality']}{quality_emoji}]"
                if r["mask_prob"] > 0.3:
                    label += f" [M:{r['mask_prob']:.1f}]"
                if r["angle_type"] != "front":
                    label += f" [{r['angle_type']}]"
                if r.get("review_reason"):
                    label += f" [REVIEW:{r['review_reason']}]"
                if bank_added:
                    label += " [BANK+]"
                
                if r["is_match"]:
                    color = (0, 255, 0)  # 초록
                    hit_count += 1
                elif r.get("review_reason"):
                    color = (0, 255, 255)  # 노란색 (검토 대상)
                else:
                    color = (0, 0, 255)  # 빨강
                
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label, (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # 결과 출력
                match_status = "✅ 매칭" if r["is_match"] else "❌ 미매칭"
                mask_info = f" [마스크:{r['mask_prob']:.1f}]" if r["mask_prob"] > 0.3 else ""
                quality_info = f" [화질:{r.get('face_quality', 'unknown')}]" if r.get("face_quality") else ""
                print(f"[얼굴 {r['face_idx']}] {match_status}")
                print(f"  인물: {r['best_id']}, 유사도: {r['similarity']:.3f}, "
                      f"임계값: {r['threshold']:.3f}{quality_info}")
                print(f"  각도: {r['angle_type']} (yaw={r['yaw_angle']:.1f}°){mask_info}")
                if r["sim_gap"] > 0:
                    print(f"  유사도 차이: {r['sim_gap']:.3f}")
                print()
            
            # 결과 이미지 저장
            out_name = "result.jpg"
            cv2.imwrite(str(matches_dir / out_name), img)
            print(f"✅ 결과 이미지 저장: {matches_dir / out_name}")
            print()
        
        else:
            print("⚠ 얼굴을 하나도 찾지 못했습니다.")
    
    else:
        # ===== 영상 파일 처리 =====
        cap = cv2.VideoCapture(str(input_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if total_frames <= 0:
            total_frames = None
        
        print(f"🎥 영상 정보:")
        print(f"   총 프레임 수: {total_frames if total_frames else '알 수 없음'}")
        print(f"   FPS: {fps:.2f}")
        print(f"   분석 시작...")
        print()
        
        # 프레임 저장 옵션
        SAVE_FRAMES = True  # 프레임 이미지 저장 여부 (False로 변경하면 저장 안함)
        FRAME_INTERVAL = 30  # N프레임마다 저장 (성능 고려, 1이면 모든 프레임 저장)
        PROCESS_EVERY_N_FRAMES = 1  # 영상 파일은 모든 프레임 처리
        
        print(f"   프레임 저장: {'활성화' if SAVE_FRAMES else '비활성화'} (간격: {FRAME_INTERVAL}프레임)")
        print()
        
        # 프레임 간 연속성 체크를 위한 히스토리 저장
        # 각 인물별로 최근 N프레임 동안의 매칭 기록 저장
        frame_history = defaultdict(list)  # {person_id: [frame_idx1, frame_idx2, ...]}
        continuity_window = 5  # 연속성 체크를 위한 프레임 범위
        
        # 프레임별 처리
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 프레임 이미지 저장 (선택적)
            if SAVE_FRAMES and frame_idx % FRAME_INTERVAL == 0:
                frame_filename = f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(frames_dir / frame_filename), frame)
            
            # 프레임 처리
            frame_results = process_frame(frame, app, gallery, BASE_THRESH, frame_idx)
            
            if frame_results:
                total_faces_detected += len(frame_results)
                
                # 프레임 간 연속성 체크 (매칭된 결과에 대해)
                matched_in_frame = [r for r in frame_results if r["is_match"]]
                
                for r in matched_in_frame:
                    person_id = r["best_id"]
                    # 이전 프레임들에서 같은 인물이 매칭되었는지 확인
                    recent_frames = frame_history[person_id]
                    
                    # 연속성 체크: 최근 continuity_window 프레임 내에 같은 인물이 있었는지
                    has_continuity = False
                    if recent_frames:
                        # 최근 프레임과의 거리 확인
                        last_frame = recent_frames[-1]
                        frame_gap = frame_idx - last_frame
                        if frame_gap <= continuity_window:
                            has_continuity = True
                    
                    # 연속성이 없고 유사도가 낮으면 검토 대상
                    # 화질에 따라 임계값 조정
                    quality = r.get("face_quality", "medium")
                    continuity_threshold = 0.42 if quality == "high" else (0.40 if quality == "medium" else 0.38)
                    if not has_continuity and r["similarity"] < continuity_threshold:
                        # review_reason이 이미 있으면 유지, 없으면 설정
                        if "review_reason" not in r or r["review_reason"] is None:
                            r["review_reason"] = "no_continuity"
                        r["is_match"] = False  # 일단 매칭 해제
                
                for r in frame_results:
                    x1, y1, x2, y2 = map(int, r["bbox"])
                    
                    # CSV 로그 기록 (review_reason, face_quality 추가)
                    review_reason = r.get("review_reason", None) or ""
                    face_quality = r.get("face_quality", "unknown")
                    log_writer.writerow([
                        frame_idx, r["best_id"], r["similarity"], r["threshold"],
                        int(r["is_match"]), r["angle_type"], r["yaw_angle"],
                        r["mask_prob"], r["sim_gap"], face_quality,
                        x1, y1, x2, y2, review_reason
                    ])
                    
                    # Bank에 자동 추가 (매칭 성공 시)
                    bank_added = False
                    if r["is_match"] and AUTO_ADD_TO_BANK:
                        bank_added = add_embedding_to_bank(
                            person_id=r["best_id"],
                            embedding=r["embedding"],
                            emb_dir=emb_dir,
                            similarity_threshold=BANK_DUPLICATE_THRESHOLD,
                            verbose=False,
                            angle_type=r.get("angle_type"),
                            yaw_angle=r.get("yaw_angle")
                        )
                        if bank_added:
                            bank_added_count += 1
                    
                    # 통계 업데이트
                    if r["similarity"] > max_sim_ever:
                        max_sim_ever = r["similarity"]
                    
                    angle_stats[r["angle_type"]]["total"] += 1
                    if r["is_match"]:
                        angle_stats[r["angle_type"]]["matched"] += 1
                        person_stats[r["best_id"]]["count"] += 1
                        if r["similarity"] > person_stats[r["best_id"]]["max_sim"]:
                            person_stats[r["best_id"]]["max_sim"] = r["similarity"]
                        person_stats[r["best_id"]]["angles"][r["angle_type"]] += 1
                        
                        # 히스토리 업데이트
                        frame_history[r["best_id"]].append(frame_idx)
                        # 오래된 기록 제거 (메모리 관리)
                        if len(frame_history[r["best_id"]]) > continuity_window * 2:
                            frame_history[r["best_id"]] = frame_history[r["best_id"]][-continuity_window:]
                    
                    # 매칭된 경우 또는 검토 대상인 경우 화면에 표시 및 저장
                    if r["is_match"]:
                        hit_count += 1
                        
                        # 이미지에 표시
                        label = f"{r['best_id']} {r['similarity']:.2f}"
                        if r.get("face_quality"):
                            quality_emoji = {"high": "🔍", "medium": "📷", "low": "📱"}.get(r["face_quality"], "")
                            label += f" [{r['face_quality']}{quality_emoji}]"
                        if r["mask_prob"] > 0.3:
                            label += f" [M:{r['mask_prob']:.1f}]"
                        if r["angle_type"] != "front":
                            label += f" [{r['angle_type']}]"
                        if bank_added:
                            label += " [BANK+]"
                        
                        color = (0, 255, 0)  # 초록
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, max(0, y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        # 스냅샷 저장
                        out_name = f"match_f{frame_idx:06d}_{r['best_id']}_{r['similarity']:.2f}.jpg"
                        cv2.imwrite(str(matches_dir / out_name), frame)
                    
                    # 검토 대상인 경우 별도 폴더에 저장
                    elif r.get("review_reason"):
                        # 이미지에 표시 (노란색)
                        label = f"{r['best_id']} {r['similarity']:.2f} [REVIEW]"
                        if r.get("face_quality"):
                            quality_emoji = {"high": "🔍", "medium": "📷", "low": "📱"}.get(r["face_quality"], "")
                            label += f" [{r['face_quality']}{quality_emoji}]"
                        if r["mask_prob"] > 0.3:
                            label += f" [M:{r['mask_prob']:.1f}]"
                        if r["angle_type"] != "front":
                            label += f" [{r['angle_type']}]"
                        
                        color = (0, 255, 255)  # 노란색
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, max(0, y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        # 검토 대상 스냅샷 저장
                        reason = r["review_reason"]
                        out_name = f"review_f{frame_idx:06d}_{r['best_id']}_{r['similarity']:.2f}_{reason}.jpg"
                        cv2.imwrite(str(review_dir / out_name), frame)
                
                # 프레임별 요약 출력 (매칭된 얼굴만)
                matched_in_frame = [r for r in frame_results if r["is_match"]]
                if matched_in_frame:
                    print(f"[프레임 {frame_idx:5d}] 감지: {len(frame_results)}개 얼굴, "
                          f"매칭: {len(matched_in_frame)}개")
                    for r in matched_in_frame:
                        mask_info = f" [마스크:{r['mask_prob']:.1f}]" if r["mask_prob"] > 0.3 else ""
                        quality_info = f" [화질:{r.get('face_quality', 'unknown')}]" if r.get("face_quality") else ""
                        print(f"  → {r['best_id']}: {r['similarity']:.3f} "
                              f"({r['angle_type']}{quality_info}{mask_info})")
            
            frame_idx += 1
            
            # 진행 상황 출력 (100프레임마다)
            if frame_idx % 100 == 0:
                elapsed = time.time() - start_time
                fps_actual = frame_idx / elapsed if elapsed > 0 else 0
                print(f"[진행] {frame_idx}프레임 처리 완료 "
                      f"({fps_actual:.1f} FPS, 매칭: {hit_count}건)")
        
        cap.release()
    
    log_f.close()
    
    elapsed = time.time() - start_time
    
    # 7. 최종 통계 출력
    print(f"\n{'='*70}")
    print(f"✅ 분석 완료")
    print(f"{'='*70}")
    print(f"   처리 시간: {elapsed:.2f}초")
    if is_video:
        print(f"   총 프레임 수: {frame_idx}")
        print(f"   처리 속도: {frame_idx/elapsed:.2f} FPS")
    print(f"   감지된 얼굴 수: {total_faces_detected}개")
    print(f"   매칭된 얼굴 수: {hit_count}개")
    print(f"   관측된 최대 유사도: {max_sim_ever:.3f}")
    if AUTO_ADD_TO_BANK:
        print(f"   Bank에 추가된 임베딩: {bank_added_count}개")
    print()
    
    # 인물별 통계
    if person_stats:
        print(f"📊 인물별 매칭 통계:")
        for person_id, stats in sorted(person_stats.items(), 
                                       key=lambda x: x[1]["count"], reverse=True):
            print(f"   {person_id:10s}: {stats['count']:4d}회 매칭, "
                  f"최고 유사도: {stats['max_sim']:.3f}")
            if stats["angles"]:
                angle_str = ", ".join([f"{k}:{v}" for k, v in sorted(stats["angles"].items())])
                print(f"              각도 분포: {angle_str}")
        print()
    
    # 각도별 통계
    if angle_stats:
        print(f"📈 각도별 인식 성공률:")
        for angle_type in sorted(angle_stats.keys()):
            stats = angle_stats[angle_type]
            success_rate = (stats["matched"] / stats["total"] * 100) if stats["total"] > 0 else 0
            print(f"   {angle_type:15s}: {stats['matched']:4d}/{stats['total']:4d} "
                  f"({success_rate:5.1f}%)")
        print()
    
    # 출력 파일 정보
    print(f"📁 출력 파일:")
    print(f"   출력 폴더: {output_base_dir}")
    print(f"   CSV 로그: {log_path}")
    print(f"   매칭 스냅샷: {matches_dir} ({hit_count}장)")
    if is_video:
        review_count = len(list(review_dir.glob("review_*.jpg"))) if review_dir.exists() else 0
        if review_count > 0:
            print(f"   검토 대상: {review_dir} ({review_count}장)")
    if is_video and SAVE_FRAMES:
        saved_frames = len(list(frames_dir.glob("frame_*.jpg")))
        print(f"   프레임 이미지: {frames_dir} ({saved_frames}장)")
    print()
    
    print(f"💡 해석:")
    print(f"   - CSV 로그에는 모든 얼굴 감지 기록이 저장됩니다")
    print(f"   - 스냅샷은 매칭된 얼굴만 저장됩니다")
    print(f"   - 검토 대상은 matches/review/ 폴더에 별도 저장됩니다")
    print(f"   - 각도 정보와 마스크 가능성이 라벨에 표시됩니다")
    print(f"   - 오탐 방지: bbox 기반 다중 매칭 필터링 및 프레임 간 연속성 체크 적용")


if __name__ == "__main__":
    main()

