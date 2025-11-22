# src/face_match_webcam.py
"""
웹캠 실시간 얼굴 식별 스크립트
웹캠에서 실시간으로 얼굴을 감지하고 등록된 인물을 식별합니다.

주요 기능:
- 웹캠 실시간 스트리밍
- 얼굴 감지 및 매칭
- 실시간 화면 표시
- 마스크 감지 및 적응형 임계값
- 화질 기반 적응형 임계값
- 오탐 방지 (bbox 기반 다중 매칭 필터링, 프레임 간 연속성 체크)
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
from utils.gallery_loader import load_gallery, match_with_bank_detailed
from utils.device_config import get_device_id, safe_prepare_insightface
from utils.mask_detector import (
    estimate_mask_from_similarity,
    get_adjusted_threshold,
    estimate_face_quality,
)
from utils.face_angle_detector import estimate_face_angle


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """벡터를 L2 정규화"""
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


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

    distance = np.sqrt((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2)
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
        use_thresh = get_adjusted_threshold(
            BASE_THRESH, mask_prob, best_sim, face_quality
        )

        # 매칭 여부: 임계값 통과 + 유사도 차이가 충분해야 함
        is_match = (best_sim >= use_thresh) and (sim_gap >= min_gap)

        # 결과 저장
        results.append(
            {
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
                "face_quality": face_quality,
                "embedding": face_emb_normalized,
            }
        )

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

            for group in face_groups:
                if len(group) == 1:
                    # 단일 매칭: 그대로 유지
                    filtered_matched.append(group[0])
                else:
                    # 같은 얼굴 영역에서 여러 인물로 매칭됨 → 오탐 가능성
                    group.sort(key=lambda x: x["similarity"], reverse=True)

                    best_match = group[0]
                    second_match = group[1] if len(group) > 1 else None

                    # sim_gap이 충분히 크면 가장 높은 유사도만 인정
                    min_gap_for_confidence = 0.10
                    if second_match and (best_match["sim_gap"] >= min_gap_for_confidence):
                        filtered_matched.append(best_match)
                        # 나머지는 매칭 해제
                        for other in group[1:]:
                            other["is_match"] = False
                            unmatched_results.append(other)
                    else:
                        # 애매한 경우 모두 매칭 해제
                        for match in group:
                            match["is_match"] = False
                            unmatched_results.append(match)

            # 낮은 신뢰도 체크
            final_filtered = []
            for match in filtered_matched:
                quality = match.get("face_quality", "medium")
                sim_threshold = (
                    0.38 if quality == "high" else (0.35 if quality == "medium" else 0.32)
                )
                gap_threshold = (
                    0.10 if quality == "high" else (0.08 if quality == "medium" else 0.06)
                )

                if match["similarity"] >= sim_threshold and match["sim_gap"] >= gap_threshold:
                    final_filtered.append(match)
                else:
                    match["is_match"] = False
                    unmatched_results.append(match)

            results = final_filtered + unmatched_results

        elif len(matched_results) == 1:
            # 단일 매칭도 낮은 신뢰도면 매칭 해제
            match = matched_results[0]
            quality = match.get("face_quality", "medium")
            sim_threshold = (
                0.38 if quality == "high" else (0.35 if quality == "medium" else 0.32)
            )
            gap_threshold = (
                0.10 if quality == "high" else (0.08 if quality == "medium" else 0.06)
            )

            if match["similarity"] < sim_threshold or match["sim_gap"] < gap_threshold:
                match["is_match"] = False
                unmatched_results.append(match)
                results = unmatched_results
            else:
                results = matched_results + unmatched_results
        else:
            results = unmatched_results

    return results


def main():
    # ===== 설정 =====
    emb_dir = Path("outputs") / "embeddings"  # 등록 임베딩 폴더
    BASE_THRESH = 0.32  # 기본 임계값 (화질 기반 조정 전)

    # 출력 폴더 구조 (타임스탬프 포함)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base_dir = Path("outputs") / "results" / f"webcam_{timestamp}"

    matches_dir = output_base_dir / "matches"
    logs_dir = output_base_dir / "logs"

    # 폴더 생성
    matches_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_path = logs_dir / "detection_log.csv"

    print("=" * 70)
    print("📹 웹캠 실시간 얼굴 식별 시스템")
    print("=" * 70)
    print(f"   임베딩 폴더: {emb_dir}")
    print(f"   기본 임계값: {BASE_THRESH}")
    print(f"   출력 폴더: {output_base_dir}")
    print(f"     - 매칭 스냅샷: {matches_dir}")
    print(f"     - 로그 파일: {logs_dir}")
    print()

    # 1. 갤러리 로드
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

    # 2. InsightFace 준비
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
    log_writer.writerow(
        [
            "frame",
            "person_id",
            "similarity",
            "threshold",
            "is_match",
            "angle_type",
            "yaw_angle",
            "mask_prob",
            "sim_gap",
            "face_quality",
            "x1",
            "y1",
            "x2",
            "y2",
        ]
    )

    # 4. 통계 변수 초기화
    frame_idx = 0
    hit_count = 0
    total_faces_detected = 0
    max_sim_ever = -1.0
    person_stats = defaultdict(lambda: {"count": 0, "max_sim": 0.0})
    frame_history = defaultdict(list)  # 프레임 간 연속성 체크용
    continuity_window = 5

    start_time = time.time()

    # 5. 웹캠 초기화
    print("📹 웹캠 초기화 중...")
    print()

    # 👉 cam_test.py 스타일: Windows에서 DirectShow 백엔드 사용
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError(
            "웹캠을 열 수 없습니다.\n"
            "가능한 원인:\n"
            "  1. 웹캠이 연결되어 있지 않음\n"
            "  2. 다른 프로그램에서 웹캠을 사용 중\n"
            "  3. 웹캠 드라이버가 설치되지 않음\n"
            "  4. 권한 문제 (Windows: 카메라 권한 확인)"
        )

    # 해상도는 우선 기본값 사용 (문제 없으면 이후에 조정 가능)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"   해상도: {width}x{height}")
    print(f"   FPS: {fps:.2f}")
    print("   실시간 식별 시작... (종료: 'q' 키 누르기)")
    print()

    # 성능 최적화를 위한 프레임 스킵 설정
    PROCESS_EVERY_N_FRAMES = 2  # 2프레임마다 처리 (성능 고려)

    # 6. 실시간 프레임 처리 루프
    print("💡 실시간 화면이 표시됩니다. 'q' 키를 누르면 종료됩니다.")
    print()

    # 이전 프레임의 결과를 저장 (스킵된 프레임에도 표시하기 위해)
    last_frame_with_boxes = None

    while True:
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            print("⚠ 웹캠에서 프레임을 읽을 수 없습니다. 종료합니다.")
            break

        # 프레임 스킵으로 성능 최적화
        if frame_idx % PROCESS_EVERY_N_FRAMES != 0:
            frame_idx += 1
            # 스킵된 프레임에는 이전 결과 표시
            if last_frame_with_boxes is not None:
                cv2.imshow("FaceWatch - 실시간 얼굴 식별", last_frame_with_boxes)
            else:
                cv2.imshow("FaceWatch - 실시간 얼굴 식별", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n⚠ 사용자가 종료를 요청했습니다.")
                break
            continue

        # 프레임 처리
        frame_results = process_frame(frame, app, gallery, BASE_THRESH, frame_idx)
        
        # 결과를 표시할 프레임 복사
        display_frame = frame.copy()

        if frame_results:
            total_faces_detected += len(frame_results)

            # 프레임 간 연속성 체크 (매칭된 결과에 대해)
            # 웹캠은 실시간이므로 연속성 체크를 완화
            matched_in_frame = [r for r in frame_results if r["is_match"]]

            for r in matched_in_frame:
                person_id = r["best_id"]
                recent_frames = frame_history[person_id]

                # 연속성 체크 (웹캠은 실시간이므로 완화된 기준 사용)
                has_continuity = False
                if recent_frames:
                    last_frame = recent_frames[-1]
                    frame_gap = frame_idx - last_frame
                    # 웹캠은 연속성 윈도우를 더 크게 설정 (10프레임)
                    if frame_gap <= continuity_window * 2:
                        has_continuity = True

                # 연속성이 없고 유사도가 매우 낮은 경우만 매칭 해제 (웹캠은 더 관대하게)
                quality = r.get("face_quality", "medium")
                continuity_threshold = (
                    0.35 if quality == "high" else (0.33 if quality == "medium" else 0.30)
                )
                if not has_continuity and r["similarity"] < continuity_threshold:
                    r["is_match"] = False

            for r in frame_results:
                x1, y1, x2, y2 = map(int, r["bbox"])

                # CSV 로그 기록
                face_quality = r.get("face_quality", "unknown")
                log_writer.writerow(
                    [
                        frame_idx,
                        r["best_id"],
                        r["similarity"],
                        r["threshold"],
                        int(r["is_match"]),
                        r["angle_type"],
                        r["yaw_angle"],
                        r["mask_prob"],
                        r["sim_gap"],
                        face_quality,
                        x1,
                        y1,
                        x2,
                        y2,
                    ]
                )

                # 통계 업데이트
                if r["similarity"] > max_sim_ever:
                    max_sim_ever = r["similarity"]

                if r["is_match"]:
                    hit_count += 1
                    person_stats[r["best_id"]]["count"] += 1
                    if r["similarity"] > person_stats[r["best_id"]]["max_sim"]:
                        person_stats[r["best_id"]]["max_sim"] = r["similarity"]

                    # 히스토리 업데이트
                    frame_history[r["best_id"]].append(frame_idx)
                    if len(frame_history[r["best_id"]]) > continuity_window * 2:
                        frame_history[r["best_id"]] = frame_history[r["best_id"]][
                            -continuity_window:
                        ]

                    # 스냅샷 저장 (30프레임마다, 매칭된 경우만)
                    if frame_idx % 30 == 0:
                        out_name = (
                            f"match_f{frame_idx:06d}_{r['best_id']}_{r['similarity']:.2f}.jpg"
                        )
                        cv2.imwrite(str(matches_dir / out_name), display_frame)

                # 화면에 표시
                label = f"{r['best_id']} {r['similarity']:.2f}"
                if r.get("face_quality"):
                    quality_emoji = {
                        "high": "🔍",
                        "medium": "📷",
                        "low": "📱",
                    }.get(r["face_quality"], "")
                    label += f" [{r['face_quality']}{quality_emoji}]"
                if r["mask_prob"] > 0.3:
                    label += f" [M:{r['mask_prob']:.1f}]"
                if r["angle_type"] != "front":
                    label += f" [{r['angle_type']}]"

                color = (0, 255, 0) if r["is_match"] else (0, 0, 255)

                # 박스와 텍스트를 display_frame에 그리기
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    display_frame,
                    label,
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

        # 화면 상단에 정보 표시
        info_text = f"Frame: {frame_idx} | Matches: {hit_count} | Faces: {total_faces_detected}"
        cv2.putText(
            display_frame,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            display_frame,
            "Press 'q' to quit",
            (10, display_frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # 이전 프레임 결과 저장 (스킵된 프레임에 표시하기 위해)
        last_frame_with_boxes = display_frame.copy()

        # GUI 표시
        cv2.imshow("FaceWatch - 실시간 얼굴 식별", display_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n⚠ 사용자가 종료를 요청했습니다.")
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    log_f.close()

    elapsed = time.time() - start_time

    # 7. 최종 통계 출력
    print("\n" + "=" * 70)
    print("✅ 분석 완료")
    print("=" * 70)
    print(f"   처리 시간: {elapsed:.2f}초")
    print(f"   총 프레임 수: {frame_idx}")
    print(f"   처리 속도: {frame_idx / elapsed:.2f} FPS")
    print(f"   감지된 얼굴 수: {total_faces_detected}개")
    print(f"   매칭된 얼굴 수: {hit_count}개")
    print(f"   관측된 최대 유사도: {max_sim_ever:.3f}")
    print()

    # 인물별 통계
    if person_stats:
        print("📊 인물별 매칭 통계:")
        for person_id, stats in sorted(
            person_stats.items(), key=lambda x: x[1]["count"], reverse=True
        ):
            print(
                f"   {person_id:10s}: {stats['count']:4d}회 매칭, "
                f"최고 유사도: {stats['max_sim']:.3f}"
            )
        print()

    # 출력 파일 정보
    print("📁 출력 파일:")
    print(f"   출력 폴더: {output_base_dir}")
    print(f"   CSV 로그: {log_path}")
    print(f"   매칭 스냅샷: {matches_dir} ({hit_count}장)")
    print()


if __name__ == "__main__":
    main()
