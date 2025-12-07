# src/vis_embedding_heatmap.py
from insightface.app import FaceAnalysis
import cv2
import numpy as np
from pathlib import Path
from utils.device_config import get_device_id

def cosine_sim(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.dot(a, b))

def get_main_face(app, img):
    faces = app.get(img)
    if len(faces) == 0:
        raise RuntimeError("얼굴을 찾지 못했습니다.")
    faces_sorted = sorted(
        faces,
        key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]),
        reverse=True,
    )
    return faces_sorted[0]

def main():
    # ---------- 설정 ----------
    enroll_img_path = Path("images/enroll/newjeans_hani.jpg")
    heatmap_out_path = Path("outputs/vis/hani_heatmap.jpg")

    GRID_H = 16  # 세로 방향 그리드 개수
    GRID_W = 16  # 가로 방향 그리드 개수
    OCCLUSION_COLOR = (0, 0, 0)  # 가릴 때 쓸 색 (검정)

    # ---------- 모델 로드 ----------
    device_id = get_device_id()
    device_type = "GPU" if device_id >= 0 else "CPU"
    print(f"🔧 디바이스: {device_type} (ctx_id={device_id})")
    
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=device_id, det_size=(640, 640))

    # ---------- 원본 이미지 & 기준 임베딩 ----------
    img = cv2.imread(str(enroll_img_path))
    if img is None:
        raise FileNotFoundError(f"이미지를 찾지 못했어: {enroll_img_path}")

    base_face = get_main_face(app, img)
    base_emb = base_face.embedding

    h, w, _ = img.shape
    print(f"이미지 크기: {w}x{h}")

    # ---------- 그리드 설정 ----------
    cell_h = h // GRID_H
    cell_w = w // GRID_W

    importance = np.zeros((GRID_H, GRID_W), dtype=np.float32)

    # ---------- 각 영역을 하나씩 가리면서 similarity 측정 ----------
    for gy in range(GRID_H):
        for gx in range(GRID_W):
            y1 = gy * cell_h
            x1 = gx * cell_w
            y2 = h if gy == GRID_H - 1 else (gy + 1) * cell_h
            x2 = w if gx == GRID_W - 1 else (gx + 1) * cell_w

            occluded = img.copy()
            cv2.rectangle(occluded, (x1, y1), (x2, y2), OCCLUSION_COLOR, thickness=-1)

            try:
                face_occ = get_main_face(app, occluded)
            except RuntimeError:
                # 얼굴을 못 찾으면 영향이 없다고 가정
                importance[gy, gx] = 0.0
                continue

            sim = cosine_sim(base_emb, face_occ.embedding)
            # 원본과 자신 사이 유사도는 1에 가까움.
            # 가렸을 때 얼마나 떨어졌는지 = 중요도
            importance[gy, gx] = 1.0 - sim

    # ---------- 중요도 정규화 ----------
    imp_min, imp_max = float(importance.min()), float(importance.max())
    if imp_max > imp_min:
        norm_imp = (importance - imp_min) / (imp_max - imp_min)
    else:
        norm_imp = importance.copy()

    # ---------- heatmap을 이미지 크기로 키우기 ----------
    heatmap_small = (norm_imp * 255).astype(np.uint8)
    heatmap = cv2.resize(heatmap_small, (w, h), interpolation=cv2.INTER_CUBIC)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # 원본과 합성 (0.5, 0.5 비율)
    overlay = cv2.addWeighted(img, 0.5, heatmap_color, 0.5, 0)

    # 출력 폴더 생성 & 저장
    heatmap_out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(heatmap_out_path), overlay)

    print(f"✅ heatmap 저장 완료: {heatmap_out_path}")

if __name__ == "__main__":
    main()
