# -*- coding: utf-8 -*-
"""
부분 얼굴 임베딩 실험 스크립트

마스크 영역을 제외한 부분(눈 영역)만으로 임베딩을 추출하여
전체 얼굴 임베딩과 비교하는 실험

실험 목표:
1. 부분 얼굴 임베딩의 품질 확인
2. 전체 얼굴 vs 부분 얼굴 호환성 확인
3. Masked Bank 대체 가능성 판단
"""

import cv2
import numpy as np
from insightface.app import FaceAnalysis
import os
from pathlib import Path

def l2_normalize(embedding):
    """L2 정규화"""
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm

def extract_upper_region(image, bbox, ratio=0.5):
    """
    얼굴 이미지에서 상반부(눈 영역)만 추출
    
    Args:
        image: 원본 이미지
        bbox: (x1, y1, x2, y2) 얼굴 영역
        ratio: 상반부 비율 (0.5 = 상위 50%)
    
    Returns:
        상반부만 추출한 이미지
    """
    x1, y1, x2, y2 = bbox.astype(int)
    face_height = y2 - y1
    
    # 상반부만 자르기
    upper_y2 = int(y1 + face_height * ratio)
    
    upper_region = image[y1:upper_y2, x1:x2]
    return upper_region, (x1, y1, x2, upper_y2)

def test_partial_embedding(image_path, app):
    """
    단일 이미지로 전체 얼굴 vs 부분 얼굴 임베딩 테스트
    
    Returns:
        (full_embedding, partial_embedding, bbox)
    """
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 이미지 로드 실패: {image_path}")
        return None, None, None
    
    # 얼굴 감지
    faces = app.get(image)
    if len(faces) == 0:
        print(f"❌ 얼굴 감지 실패: {image_path}")
        return None, None, None
    
    face = faces[0]  # 첫 번째 얼굴
    bbox = face.bbox
    
    # 1. 전체 얼굴 임베딩
    full_embedding = l2_normalize(face.embedding)
    
    # 2. 상반부만 추출 (마스크 영역 제외)
    # 50%는 너무 작아서 얼굴 감지 실패 → 65%, 70% 시도
    upper_image_65, upper_bbox_65 = extract_upper_region(image, bbox, ratio=0.65)
    upper_image_70, upper_bbox_70 = extract_upper_region(image, bbox, ratio=0.7)
    
    # 65% 시도
    debug_path = image_path.replace('.jpg', '_upper65.jpg').replace('.png', '_upper65.png')
    cv2.imwrite(debug_path, upper_image_65)
    print(f"  📸 상반부 65% 이미지 저장: {debug_path}")
    
    # 70% 시도
    debug_path_70 = image_path.replace('.jpg', '_upper70.jpg').replace('.png', '_upper70.png')
    cv2.imwrite(debug_path_70, upper_image_70)  
    print(f"  📸 상반부 70% 이미지 저장: {debug_path_70}")
    
    # 3. 65%로 임베딩 추출 시도
    partial_embedding = None
    used_ratio = None
    
    try:
        upper_faces = app.get(upper_image_65)
        if len(upper_faces) > 0:
            upper_face = upper_faces[0]
            partial_embedding = l2_normalize(upper_face.embedding)
            used_ratio = 0.65
            print(f"  ✅ 부분 임베딩 추출 성공 (65% 영역)")
    except Exception as e:
        print(f"  ⚠️ 65% 영역으로 실패: {e}")
    
    # 65% 실패 시 70% 시도
    if partial_embedding is None:
        try:
            upper_faces = app.get(upper_image_70)
            if len(upper_faces) > 0:
                upper_face = upper_faces[0]
                partial_embedding = l2_normalize(upper_face.embedding)
                used_ratio = 0.7
                print(f"  ✅ 부분 임베딩 추출 성공 (70% 영역)")
        except Exception as e:
            print(f"  ❌ 70% 영역으로도 실패: {e}")
    
    if partial_embedding is None:
        print(f"  ⚠️ 모든 비율에서 얼굴 감지 실패")
    
    return full_embedding, partial_embedding, bbox

def compare_embeddings(emb1, emb2, label1="Embedding 1", label2="Embedding 2"):
    """두 임베딩의 유사도 계산"""
    if emb1 is None or emb2 is None:
        return None
    
    similarity = float(np.dot(emb1, emb2))
    print(f"  📊 {label1} vs {label2}: {similarity:.4f} ({similarity*100:.2f}%)")
    return similarity

def main():
    print("=" * 70)
    print("🧪 부분 얼굴 임베딩 실험")
    print("=" * 70)
    
    # InsightFace 모델 초기화
    print("\n1️⃣ InsightFace 모델 로딩...")
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("✅ 모델 로딩 완료\n")
    
    # 테스트할 이미지 경로
    test_dir = Path("c:/FaceWatch/outputs/snapshots")
    
    # 등록된 사람들의 이미지 찾기
    person_dirs = list(Path("c:/FaceWatch/outputs/embeddings").glob("person_*"))
    
    if not person_dirs:
        print("❌ 등록된 인물이 없습니다.")
        print("   outputs/embeddings/person_* 폴더에 인물을 먼저 등록해주세요.")
        return
    
    print(f"2️⃣ 등록된 인물: {len(person_dirs)}명\n")
    
    # 각 인물의 Base Bank 이미지로 테스트
    for person_dir in person_dirs[:3]:  # 최대 3명까지만
        person_id = person_dir.name
        print(f"\n{'='*70}")
        print(f"👤 테스트 인물: {person_id}")
        print(f"{'='*70}")
        
        # Base Bank 이미지 찾기
        base_images = list(person_dir.glob("base_*.jpg")) + list(person_dir.glob("base_*.png"))
        
        if not base_images:
            print(f"  ⚠️ Base Bank 이미지 없음")
            continue
        
        # 첫 번째 Base 이미지로 테스트
        base_image = str(base_images[0])
        print(f"\n  📸 Base 이미지: {Path(base_image).name}")
        
        # 실험 1: 동일 이미지에서 전체 vs 부분 임베딩
        print(f"\n  🔬 실험 1: 동일 이미지 - 전체 얼굴 vs 부분 얼굴")
        full_emb, partial_emb, bbox = test_partial_embedding(base_image, app)
        
        if full_emb is None:
            continue
        
        if partial_emb is not None:
            # 전체 vs 부분 유사도
            sim = compare_embeddings(full_emb, partial_emb, 
                                    "전체 얼굴", "부분 얼굴(눈)")
            
            print(f"\n  💡 분석:")
            if sim and sim > 0.7:
                print(f"     ✅ 부분 임베딩 품질 우수 (유사도 {sim*100:.1f}%)")
            elif sim and sim > 0.5:
                print(f"     ⚠️ 부분 임베딩 품질 보통 (유사도 {sim*100:.1f}%)")
            else:
                print(f"     ❌ 부분 임베딩 품질 불량 (유사도 {sim*100:.1f}%)")
        
        # 실험 2: 다른 이미지가 있다면 교차 테스트
        if len(base_images) > 1:
            base_image2 = str(base_images[1])
            print(f"\n  🔬 실험 2: 다른 이미지 - 교차 비교")
            print(f"  📸 Base 이미지 2: {Path(base_image2).name}")
            
            full_emb2, partial_emb2, _ = test_partial_embedding(base_image2, app)
            
            if full_emb2 is not None and partial_emb is not None and partial_emb2 is not None:
                # 부분 vs 부분 (같은 사람)
                sim_partial = compare_embeddings(partial_emb, partial_emb2,
                                                "이미지1 부분", "이미지2 부분")
                
                # 전체 vs 전체 (같은 사람)
                sim_full = compare_embeddings(full_emb, full_emb2,
                                             "이미지1 전체", "이미지2 전체")
                
                # 전체 vs 부분 (교차)
                sim_cross = compare_embeddings(full_emb, partial_emb2,
                                              "이미지1 전체", "이미지2 부분")
                
                print(f"\n  💡 분석:")
                print(f"     - 전체 vs 전체: {sim_full*100:.1f}%")
                print(f"     - 부분 vs 부분: {sim_partial*100:.1f}% {'(✅ 우수)' if sim_partial and sim_partial > 0.7 else ''}")
                print(f"     - 전체 vs 부분: {sim_cross*100:.1f}% {'(✅ 호환 가능)' if sim_cross and sim_cross > 0.6 else '(❌ 호환 불가)'}")
    
    print(f"\n{'='*70}")
    print("🎯 실험 완료")
    print("='*70")
    print("\n📋 다음 단계:")
    print("   1. 위 결과를 확인하여 부분 임베딩의 품질을 판단하세요")
    print("   2. 유사도가 70% 이상이면 실용적으로 사용 가능합니다")
    print("   3. 호환성(전체 vs 부분)이 60% 이상이면 Base Bank 활용 가능합니다")
    print("\n⚠️ 주의:")
    print("   - 부분 임베딩 품질이 낮으면 Masked Bank 방식을 유지해야 합니다")
    print("   - 품질이 우수하면 Masked Bank를 완전히 대체할 수 있습니다")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️ 사용자가 중단했습니다.")
    except Exception as e:
        print(f"\n\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
