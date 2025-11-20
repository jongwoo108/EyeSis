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
from collections import defaultdict
from utils.gallery_loader import load_gallery, match_with_bank, match_with_bank_detailed
from utils.device_config import get_device_id, safe_prepare_insightface
from utils.mask_detector import estimate_mask_from_similarity, get_adjusted_threshold
from utils.face_angle_detector import estimate_face_angle


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """벡터를 L2 정규화"""
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


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
        
        # 마스크 가능성 추정 및 적응형 임계값
        mask_prob = estimate_mask_from_similarity(best_sim)
        use_thresh = get_adjusted_threshold(BASE_THRESH, mask_prob, best_sim)
        
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
            "embedding": face_emb_normalized  # 중복 체크용
        })
    
    # 같은 프레임 내에서 같은 사람이 여러 번 감지된 경우 필터링
    if len(results) > 1:
        matched_results = []
        unmatched_results = []
        
        for r in results:
            if r["is_match"]:
                matched_results.append(r)
            else:
                unmatched_results.append(r)
        
        if len(matched_results) > 1:
            # 같은 사람으로 인식된 얼굴들 그룹화
            person_groups = {}
            for r in matched_results:
                person_id = r["best_id"]
                if person_id not in person_groups:
                    person_groups[person_id] = []
                person_groups[person_id].append(r)
            
            # 각 그룹에서 실제로 같은 사람인지 임베딩 비교
            filtered_matched = []
            for person_id, group in person_groups.items():
                if len(group) == 1:
                    filtered_matched.append(group[0])
                else:
                    # 여러 명이면 임베딩 간 유사도 비교
                    same_person_threshold = 0.85
                    
                    # 가장 유사도가 높은 얼굴을 기준으로 선택
                    group.sort(key=lambda x: x["similarity"], reverse=True)
                    best_face = group[0]
                    filtered_matched.append(best_face)
                    
                    # 나머지 얼굴들과 임베딩 비교
                    for other_face in group[1:]:
                        emb_sim = float(np.dot(best_face["embedding"], other_face["embedding"]))
                        if emb_sim < same_person_threshold:
                            # 다른 사람으로 판단 → 매칭 해제 (오탐 가능성)
                            other_face["is_match"] = False
                            unmatched_results.append(other_face)
            
            results = filtered_matched + unmatched_results
        elif len(matched_results) == 1:
            results = matched_results + unmatched_results
        else:
            results = unmatched_results
    
    return results


def main():
    # ===== 설정 =====
    # 입력 파일 경로 설정 (추출용 소스 파일)
    # 우선순위: images/source/ 또는 videos/source/ → 루트 폴더 (호환성)
    input_filename = "yh.MOV"  # 파일명만 지정 (확장자로 자동 감지)
    
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
    BASE_THRESH = 0.30                        # 기본 임계값
    
    # 파일명 기반 출력 폴더 구조
    stem = input_path.stem  # 파일명 (확장자 제외)
    output_base_dir = Path("outputs") / "results" / stem  # outputs/results/yh/
    
    # 하위 폴더들
    matches_dir = output_base_dir / "matches"      # outputs/results/yh/matches/ (매칭된 스냅샷)
    logs_dir = output_base_dir / "logs"            # outputs/results/yh/logs/ (CSV 로그)
    frames_dir = output_base_dir / "frames"        # outputs/results/yh/frames/ (추출된 프레임)
    
    # 폴더 생성
    matches_dir.mkdir(parents=True, exist_ok=True)
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
    print(f"   출력 폴더: {output_base_dir}")
    print(f"     - 매칭 스냅샷: {matches_dir}")
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
        "angle_type", "yaw_angle", "mask_prob", "sim_gap",
        "x1", "y1", "x2", "y2"
    ])
    
    # 4. 통계 변수 초기화
    frame_idx = 0
    hit_count = 0
    total_faces_detected = 0
    max_sim_ever = -1.0
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
                log_writer.writerow([
                    None, r["best_id"], r["similarity"], r["threshold"],
                    int(r["is_match"]), r["angle_type"], r["yaw_angle"],
                    r["mask_prob"], r["sim_gap"], x1, y1, x2, y2
                ])
                
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
                if r["mask_prob"] > 0.3:
                    label += f" [M:{r['mask_prob']:.1f}]"
                if r["angle_type"] != "front":
                    label += f" [{r['angle_type']}]"
                
                if r["is_match"]:
                    color = (0, 255, 0)  # 초록
                    hit_count += 1
                else:
                    color = (0, 0, 255)  # 빨강
                
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label, (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # 결과 출력
                match_status = "✅ 매칭" if r["is_match"] else "❌ 미매칭"
                mask_info = f" [마스크:{r['mask_prob']:.1f}]" if r["mask_prob"] > 0.3 else ""
                print(f"[얼굴 {r['face_idx']}] {match_status}")
                print(f"  인물: {r['best_id']}, 유사도: {r['similarity']:.3f}, "
                      f"임계값: {r['threshold']:.3f}")
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
        # ===== 영상 처리 =====
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
        
        print(f"   프레임 저장: {'활성화' if SAVE_FRAMES else '비활성화'} (간격: {FRAME_INTERVAL}프레임)")
        print()
        
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
                
                for r in frame_results:
                    x1, y1, x2, y2 = map(int, r["bbox"])
                    
                    # CSV 로그 기록
                    log_writer.writerow([
                        frame_idx, r["best_id"], r["similarity"], r["threshold"],
                        int(r["is_match"]), r["angle_type"], r["yaw_angle"],
                        r["mask_prob"], r["sim_gap"], x1, y1, x2, y2
                    ])
                    
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
                    
                    # 매칭된 경우 스냅샷 저장
                    if r["is_match"]:
                        hit_count += 1
                        
                        # 이미지에 표시
                        label = f"{r['best_id']} {r['similarity']:.2f}"
                        if r["mask_prob"] > 0.3:
                            label += f" [M:{r['mask_prob']:.1f}]"
                        if r["angle_type"] != "front":
                            label += f" [{r['angle_type']}]"
                        
                        color = (0, 255, 0)  # 초록
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, max(0, y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        # 스냅샷 저장
                        out_name = f"match_f{frame_idx:06d}_{r['best_id']}_{r['similarity']:.2f}.jpg"
                        cv2.imwrite(str(matches_dir / out_name), frame)
                
                # 프레임별 요약 출력 (매칭된 얼굴만)
                matched_in_frame = [r for r in frame_results if r["is_match"]]
                if matched_in_frame:
                    print(f"[프레임 {frame_idx:5d}] 감지: {len(frame_results)}개 얼굴, "
                          f"매칭: {len(matched_in_frame)}개")
                    for r in matched_in_frame:
                        mask_info = f" [마스크:{r['mask_prob']:.1f}]" if r["mask_prob"] > 0.3 else ""
                        print(f"  → {r['best_id']}: {r['similarity']:.3f} "
                              f"({r['angle_type']}{mask_info})")
            
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
    if is_video and SAVE_FRAMES:
        saved_frames = len(list(frames_dir.glob("frame_*.jpg")))
        print(f"   프레임 이미지: {frames_dir} ({saved_frames}장)")
    print()
    
    print(f"💡 해석:")
    print(f"   - CSV 로그에는 모든 얼굴 감지 기록이 저장됩니다")
    print(f"   - 스냅샷은 매칭된 얼굴만 저장됩니다")
    print(f"   - 각도 정보와 마스크 가능성이 라벨에 표시됩니다")


if __name__ == "__main__":
    main()

