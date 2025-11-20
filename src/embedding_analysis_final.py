# src/embedding_analysis_final.py
"""
임베딩 분석 및 시각화 통합 스크립트 (최종 버전)
모든 분석/시각화 기능을 하나로 통합

주요 기능:
1. 갤러리 통계 표시: Bank/Centroid 정보, 임베딩 개수 등
2. 유사도 히트맵: 사람 간 유사도 매트릭스 시각화
3. 임베딩 분포 비교: 히스토그램으로 분포 비교
4. 3D 시각화: PCA를 이용한 3D scatter plot
"""
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from utils.gallery_loader import load_gallery

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """벡터를 L2 정규화"""
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


# ===== MODE 1: 갤러리 통계 표시 =====
def mode_gallery_stats(emb_dir: Path):
    """갤러리 전체 통계 표시"""
    print(f"{'='*70}")
    print(f"📊 MODE 1: 갤러리 통계 (Gallery Statistics)")
    print(f"{'='*70}")
    print()
    
    gallery = load_gallery(emb_dir, use_bank=True)
    if not gallery:
        print("⚠️ 갤러리가 비어있습니다.")
        return
    
    print("👥 등록된 인물 목록:")
    print()
    
    for person_id, data in sorted(gallery.items()):
        if data.ndim == 2:
            # Bank
            bank_size = data.shape[0]
            centroid = data.mean(axis=0)
            centroid = l2_normalize(centroid)
            
            # Bank 내부 유사도 통계
            if bank_size > 1:
                # 모든 임베딩 쌍의 유사도 계산
                similarities = []
                for i in range(bank_size):
                    for j in range(i + 1, bank_size):
                        sim = float(np.dot(data[i], data[j]))
                        similarities.append(sim)
                
                avg_sim = np.mean(similarities)
                min_sim = np.min(similarities)
                max_sim = np.max(similarities)
                
                print(f"  {person_id:10s}: Bank ({bank_size:3d}개 임베딩)")
                print(f"              평균 유사도: {avg_sim:.3f} (범위: {min_sim:.3f} ~ {max_sim:.3f})")
            else:
                print(f"  {person_id:10s}: Bank ({bank_size:3d}개 임베딩)")
        else:
            # Centroid
            print(f"  {person_id:10s}: Centroid")
        
        print(f"              벡터 차원: {data.shape[-1]}")
        print()
    
    # 전체 통계
    total_people = len(gallery)
    total_embeddings = sum(data.shape[0] if data.ndim == 2 else 1 for data in gallery.values())
    
    print(f"📈 전체 통계:")
    print(f"   등록된 인물 수: {total_people}명")
    print(f"   총 임베딩 수: {total_embeddings}개")
    print()


# ===== MODE 2: 유사도 히트맵 =====
def mode_similarity_heatmap(emb_dir: Path, output_path: Path = None):
    """사람 간 유사도 히트맵 생성"""
    print(f"{'='*70}")
    print(f"🔥 MODE 2: 유사도 히트맵 (Similarity Heatmap)")
    print(f"{'='*70}")
    print()
    
    gallery = load_gallery(emb_dir, use_bank=True)
    if not gallery:
        print("⚠️ 갤러리가 비어있습니다.")
        return
    
    # 각 사람의 대표 임베딩 추출 (bank가 있으면 centroid, 없으면 centroid)
    person_ids = sorted(gallery.keys())
    embeddings = []
    
    for pid in person_ids:
        data = gallery[pid]
        if data.ndim == 2:
            # Bank의 centroid 사용
            centroid = data.mean(axis=0)
            centroid = l2_normalize(centroid)
        else:
            centroid = data
        embeddings.append(centroid)
    
    embeddings = np.stack(embeddings, axis=0)  # (N, 512)
    
    # 유사도 매트릭스 계산
    similarity_matrix = embeddings @ embeddings.T  # (N, N)
    
    # 히트맵 생성
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        similarity_matrix,
        annot=True,
        fmt='.3f',
        cmap='RdYlBu_r',
        xticklabels=person_ids,
        yticklabels=person_ids,
        vmin=0.0,
        vmax=1.0,
        square=True,
        linewidths=0.5
    )
    plt.title('사람 간 얼굴 유사도 매트릭스', fontsize=14, pad=20)
    plt.xlabel('인물', fontsize=12)
    plt.ylabel('인물', fontsize=12)
    plt.tight_layout()
    
    if output_path is None:
        output_path = Path("outputs") / "analysis" / "similarity_heatmap.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 히트맵 저장: {output_path}")
    plt.close()
    
    # 통계 출력
    print()
    print("📊 유사도 통계:")
    for i, pid1 in enumerate(person_ids):
        for j, pid2 in enumerate(person_ids):
            if i < j:  # 상삼각 행렬만 출력
                sim = similarity_matrix[i, j]
                print(f"   {pid1:10s} ↔ {pid2:10s}: {sim:.3f}")
    print()


# ===== MODE 3: 임베딩 분포 비교 =====
def mode_distribution_compare(emb_dir: Path, output_path: Path = None):
    """임베딩 분포 히스토그램 비교"""
    print(f"{'='*70}")
    print(f"📊 MODE 3: 임베딩 분포 비교 (Distribution Comparison)")
    print(f"{'='*70}")
    print()
    
    gallery = load_gallery(emb_dir, use_bank=True)
    if not gallery:
        print("⚠️ 갤러리가 비어있습니다.")
        return
    
    # Bank가 있는 사람들의 임베딩 수집
    person_data = {}
    for person_id, data in gallery.items():
        if data.ndim == 2 and data.shape[0] > 1:
            # Bank 내부 유사도 분포 계산
            similarities = []
            for i in range(data.shape[0]):
                for j in range(i + 1, data.shape[0]):
                    sim = float(np.dot(data[i], data[j]))
                    similarities.append(sim)
            
            if similarities:
                person_data[person_id] = similarities
    
    if not person_data:
        print("⚠️ Bank가 2개 이상인 인물이 없습니다.")
        return
    
    # 히스토그램 생성
    fig, axes = plt.subplots(len(person_data), 1, figsize=(10, 4 * len(person_data)))
    if len(person_data) == 1:
        axes = [axes]
    
    for idx, (person_id, similarities) in enumerate(sorted(person_data.items())):
        axes[idx].hist(similarities, bins=20, alpha=0.7, edgecolor='black')
        axes[idx].set_title(f'{person_id} - Bank 내부 유사도 분포 ({len(similarities)}개 쌍)')
        axes[idx].set_xlabel('유사도')
        axes[idx].set_ylabel('빈도')
        axes[idx].grid(True, alpha=0.3)
        
        # 통계 표시
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        axes[idx].axvline(mean_sim, color='red', linestyle='--', label=f'평균: {mean_sim:.3f}')
        axes[idx].legend()
    
    plt.tight_layout()
    
    if output_path is None:
        output_path = Path("outputs") / "analysis" / "distribution_compare.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 분포 비교 그래프 저장: {output_path}")
    plt.close()
    
    # 통계 출력
    print()
    print("📊 Bank 내부 유사도 통계:")
    for person_id, similarities in sorted(person_data.items()):
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        min_sim = np.min(similarities)
        max_sim = np.max(similarities)
        print(f"   {person_id:10s}: 평균={mean_sim:.3f}, 표준편차={std_sim:.3f}, "
              f"범위=[{min_sim:.3f}, {max_sim:.3f}]")
    print()


# ===== MODE 4: 3D 시각화 =====
def mode_3d_visualization(emb_dir: Path, output_path: Path = None):
    """PCA를 이용한 3D 시각화"""
    print(f"{'='*70}")
    print(f"🎨 MODE 4: 3D 시각화 (PCA Visualization)")
    print(f"{'='*70}")
    print()
    
    gallery = load_gallery(emb_dir, use_bank=True)
    if not gallery:
        print("⚠️ 갤러리가 비어있습니다.")
        return
    
    # 모든 임베딩 수집
    all_embeddings = []
    labels = []
    
    for person_id, data in sorted(gallery.items()):
        if data.ndim == 2:
            # Bank의 모든 임베딩
            for emb in data:
                all_embeddings.append(emb)
                labels.append(person_id)
        else:
            # Centroid
            all_embeddings.append(data)
            labels.append(person_id)
    
    if len(all_embeddings) < 3:
        print("⚠️ 임베딩이 3개 미만입니다. 3D 시각화를 위해 최소 3개 필요합니다.")
        return
    
    embeddings_array = np.stack(all_embeddings, axis=0)  # (N, 512)
    
    # PCA로 3차원으로 축소
    print(f"   원본 차원: {embeddings_array.shape[1]}차원")
    print(f"   임베딩 개수: {embeddings_array.shape[0]}개")
    print(f"   PCA 차원 축소 중...")
    
    pca = PCA(n_components=3)
    embeddings_3d = pca.fit_transform(embeddings_array)
    
    explained_variance = pca.explained_variance_ratio_
    print(f"   설명된 분산: PC1={explained_variance[0]:.1%}, "
          f"PC2={explained_variance[1]:.1%}, PC3={explained_variance[2]:.1%}")
    print(f"   총 설명 분산: {sum(explained_variance):.1%}")
    print()
    
    # 3D scatter plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 사람별로 색상 지정
    unique_labels = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
    
    for label in unique_labels:
        mask = np.array(labels) == label
        points = embeddings_3d[mask]
        ax.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            c=[color_map[label]], label=label, alpha=0.6, s=50
        )
    
    ax.set_xlabel(f'PC1 ({explained_variance[0]:.1%})')
    ax.set_ylabel(f'PC2 ({explained_variance[1]:.1%})')
    ax.set_zlabel(f'PC3 ({explained_variance[2]:.1%})')
    ax.set_title('임베딩 3D 시각화 (PCA)', fontsize=14, pad=20)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    if output_path is None:
        output_path = Path("outputs") / "analysis" / "3d_visualization.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 3D 시각화 저장: {output_path}")
    plt.close()
    
    # 클러스터링 분석
    print()
    print("📊 클러스터링 분석:")
    for label in unique_labels:
        mask = np.array(labels) == label
        points = embeddings_3d[mask]
        if len(points) > 1:
            centroid_3d = points.mean(axis=0)
            distances = np.linalg.norm(points - centroid_3d, axis=1)
            avg_distance = np.mean(distances)
            print(f"   {label:10s}: {len(points)}개 점, 평균 중심 거리={avg_distance:.3f}")
        else:
            print(f"   {label:10s}: {len(points)}개 점")
    print()


def main():
    # ===== 설정 =====
    MODE = 1  # 1: 갤러리 통계, 2: 유사도 히트맵, 3: 분포 비교, 4: 3D 시각화
    
    emb_dir = Path("outputs") / "embeddings"
    output_dir = Path("outputs") / "analysis"
    
    print(f"{'='*70}")
    print(f"📊 임베딩 분석 및 시각화 통합 시스템")
    print(f"{'='*70}")
    print(f"   모드: {MODE}")
    print(f"   임베딩 폴더: {emb_dir}")
    print(f"   출력 폴더: {output_dir}")
    print()
    
    if not emb_dir.exists():
        print(f"❌ 임베딩 폴더를 찾을 수 없음: {emb_dir}")
        return
    
    # 모드별 실행
    if MODE == 1:
        mode_gallery_stats(emb_dir)
    
    elif MODE == 2:
        output_path = output_dir / "similarity_heatmap.png"
        mode_similarity_heatmap(emb_dir, output_path)
    
    elif MODE == 3:
        output_path = output_dir / "distribution_compare.png"
        mode_distribution_compare(emb_dir, output_path)
    
    elif MODE == 4:
        output_path = output_dir / "3d_visualization.png"
        mode_3d_visualization(emb_dir, output_path)
    
    else:
        print(f"❌ 잘못된 모드: {MODE} (1, 2, 3, 4 중 선택)")
    
    print(f"{'='*70}")
    print(f"✅ 분석 완료!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

