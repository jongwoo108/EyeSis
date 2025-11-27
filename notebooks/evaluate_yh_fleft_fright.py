"""
yh의 fleft/fright 임베딩 평가 스크립트

목적: 새로 추출한 fleft/fright 임베딩을 정답 데이터로 삼아
      CCTV에서 수집된 동적 임베딩과의 유사도를 평가

정답 데이터: outputs/embeddings_yh_frontal/ (fleft, fright)
CCTV 데이터: outputs/embeddings/yh/bank_dynamic.npy + angles_dynamic.json
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.device_config import _ensure_cuda_in_path
_ensure_cuda_in_path()

import numpy as np
import json
from collections import defaultdict
from datetime import datetime


def l2_normalize(vec):
    """L2 normalization"""
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def load_ground_truth_embeddings():
    """
    Load ground truth embeddings (fleft, fright)
    
    Returns:
        dict: {angle_type: embedding}
    """
    gt_dir = PROJECT_ROOT / "outputs" / "embeddings_yh_frontal"
    
    embeddings = {}
    angle_info = {}
    
    for angle_type in ['fleft', 'fright']:
        emb_file = gt_dir / f"embedding_{angle_type}.npy"
        angle_file = gt_dir / f"angle_{angle_type}.json"
        
        if emb_file.exists():
            emb = np.load(emb_file)
            emb = l2_normalize(emb)
            embeddings[angle_type] = emb
            
            if angle_file.exists():
                with open(angle_file, 'r') as f:
                    angle_info[angle_type] = json.load(f)
            
            print(f"[LOADED] {angle_type}: embedding shape {emb.shape}")
        else:
            print(f"[WARNING] {angle_type} embedding not found: {emb_file}")
    
    return embeddings, angle_info


def load_cctv_embeddings():
    """
    Load CCTV dynamic embeddings and angle information
    
    Returns:
        tuple: (bank_array, angle_info_list)
    """
    cctv_dir = PROJECT_ROOT / "outputs" / "embeddings" / "yh"
    
    # Load dynamic bank
    bank_file = cctv_dir / "bank_dynamic.npy"
    if not bank_file.exists():
        print(f"[ERROR] CCTV bank not found: {bank_file}")
        return None, None
    
    bank = np.load(bank_file)
    print(f"[LOADED] CCTV bank: {bank.shape} embeddings")
    
    # Load angle information
    angle_file = cctv_dir / "angles_dynamic.json"
    angle_info_list = None
    if angle_file.exists():
        with open(angle_file, 'r') as f:
            angle_data = json.load(f)
        
        # Convert to list format
        # Structure: {'angle_types': [...], 'yaw_angles': [...], ...}
        if isinstance(angle_data, dict) and 'angle_types' in angle_data:
            angle_types = angle_data.get('angle_types', [])
            yaw_angles = angle_data.get('yaw_angles', [])
            pitch_angles = angle_data.get('pitch_angles', [])
            
            # Create list of dicts
            angle_info_list = []
            for i in range(len(angle_types)):
                angle_info_list.append({
                    'angle_type': angle_types[i] if i < len(angle_types) else 'unknown',
                    'yaw': yaw_angles[i] if i < len(yaw_angles) else 0,
                    'pitch': pitch_angles[i] if i < len(pitch_angles) else 0
                })
            
            print(f"[LOADED] Angle info: {len(angle_info_list)} entries")
        else:
            print(f"[WARNING] Unexpected angle data structure: {list(angle_data.keys())}")
    else:
        print(f"[WARNING] Angle info not found: {angle_file}")
    
    return bank, angle_info_list


def evaluate_similarities(gt_embeddings, gt_angle_info, cctv_bank, cctv_angle_info):
    """
    Evaluate similarity between ground truth and CCTV embeddings
    
    Args:
        gt_embeddings: Ground truth embeddings {angle_type: embedding}
        gt_angle_info: Ground truth angle info {angle_type: info}
        cctv_bank: CCTV embeddings (N, 512)
        cctv_angle_info: List of angle info for each CCTV embedding
    
    Returns:
        dict: Evaluation results
    """
    results = {
        'timestamp': datetime.now().isoformat(),
        'ground_truth': {},
        'cctv_total': cctv_bank.shape[0],
        'comparisons': {}
    }
    
    # Store ground truth info
    for angle_type, info in gt_angle_info.items():
        results['ground_truth'][angle_type] = {
            'yaw': info['yaw'],
            'pitch': info['pitch'],
            'filename': info['filename']
        }
    
    # Calculate similarities for each ground truth embedding
    for gt_type, gt_emb in gt_embeddings.items():
        print(f"\n{'='*70}")
        print(f"[EVALUATE] Ground Truth: {gt_type}")
        print(f"  Yaw: {gt_angle_info[gt_type]['yaw']:.1f}, Pitch: {gt_angle_info[gt_type]['pitch']:.1f}")
        print(f"{'='*70}")
        
        # Calculate similarities with all CCTV embeddings
        similarities = np.dot(cctv_bank, gt_emb)
        
        # Find angle matches
        angle_matches = defaultdict(list)
        angle_similarities = defaultdict(list)
        
        if cctv_angle_info:
            for idx, (sim, angle_data) in enumerate(zip(similarities, cctv_angle_info)):
                angle_type = angle_data.get('angle_type', 'unknown')
                yaw = angle_data.get('yaw', 0)
                pitch = angle_data.get('pitch', 0)
                
                angle_matches[angle_type].append({
                    'index': idx,
                    'similarity': float(sim),
                    'yaw': yaw,
                    'pitch': pitch
                })
                angle_similarities[angle_type].append(sim)
        
        # Statistics
        print(f"\n[OVERALL STATISTICS]")
        print(f"  Total CCTV embeddings: {len(similarities)}")
        print(f"  Mean similarity: {np.mean(similarities):.4f}")
        print(f"  Std similarity: {np.std(similarities):.4f}")
        print(f"  Max similarity: {np.max(similarities):.4f}")
        print(f"  Min similarity: {np.min(similarities):.4f}")
        
        # Top matches
        top_indices = np.argsort(similarities)[::-1][:10]
        print(f"\n[TOP 10 MATCHES]")
        for rank, idx in enumerate(top_indices, 1):
            sim = similarities[idx]
            angle_data = cctv_angle_info[idx] if cctv_angle_info else {}
            angle_type = angle_data.get('angle_type', 'unknown')
            yaw = angle_data.get('yaw', 0)
            pitch = angle_data.get('pitch', 0)
            
            print(f"  {rank:2d}. Similarity: {sim:.4f}, Angle: {angle_type:15s}, Yaw: {yaw:6.1f}, Pitch: {pitch:6.1f}")
        
        # Angle-based statistics
        print(f"\n[ANGLE-BASED STATISTICS]")
        for angle_type in sorted(angle_similarities.keys()):
            sims = angle_similarities[angle_type]
            if len(sims) > 0:
                print(f"  {angle_type:15s}: Count={len(sims):4d}, Mean={np.mean(sims):.4f}, "
                      f"Std={np.std(sims):.4f}, Max={np.max(sims):.4f}")
        
        # Threshold analysis
        print(f"\n[THRESHOLD ANALYSIS]")
        for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
            count = np.sum(similarities >= threshold)
            percentage = count / len(similarities) * 100
            print(f"  Similarity >= {threshold:.1f}: {count:4d} ({percentage:5.1f}%)")
        
        # Store results
        results['comparisons'][gt_type] = {
            'overall': {
                'mean': float(np.mean(similarities)),
                'std': float(np.std(similarities)),
                'max': float(np.max(similarities)),
                'min': float(np.min(similarities)),
                'total_count': int(len(similarities))
            },
            'top_10': [
                {
                    'rank': rank,
                    'index': int(idx),
                    'similarity': float(similarities[idx]),
                    'angle_type': cctv_angle_info[idx].get('angle_type', 'unknown') if cctv_angle_info else 'unknown',
                    'yaw': cctv_angle_info[idx].get('yaw', 0) if cctv_angle_info else 0,
                    'pitch': cctv_angle_info[idx].get('pitch', 0) if cctv_angle_info else 0
                }
                for rank, idx in enumerate(np.argsort(similarities)[::-1][:10], 1)
            ],
            'angle_statistics': {
                angle_type: {
                    'count': len(sims),
                    'mean': float(np.mean(sims)),
                    'std': float(np.std(sims)),
                    'max': float(np.max(sims)),
                    'min': float(np.min(sims))
                }
                for angle_type, sims in angle_similarities.items()
            },
            'threshold_analysis': {
                f'{threshold:.1f}': {
                    'count': int(np.sum(similarities >= threshold)),
                    'percentage': float(np.sum(similarities >= threshold) / len(similarities) * 100)
                }
                for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]
            }
        }
    
    return results


def compare_fleft_fright(gt_embeddings):
    """
    Compare fleft and fright embeddings
    """
    if 'fleft' in gt_embeddings and 'fright' in gt_embeddings:
        sim = np.dot(gt_embeddings['fleft'], gt_embeddings['fright'])
        print(f"\n{'='*70}")
        print(f"[GROUND TRUTH COMPARISON]")
        print(f"  fleft <-> fright similarity: {sim:.4f}")
        if sim > 0.6:
            print(f"  [HIGH] Same person confirmed")
        elif sim > 0.4:
            print(f"  [MEDIUM] Verification needed")
        else:
            print(f"  [LOW] Different person suspected")
        print(f"{'='*70}")
        return sim
    return None


def save_results(results, output_file):
    """
    Save evaluation results to JSON
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVED] Results saved to: {output_file}")


def main():
    print("="*70)
    print("[YH EVALUATION] fleft/fright vs CCTV Embeddings")
    print("="*70)
    
    # Load ground truth
    print("\n[STEP 1] Loading ground truth embeddings...")
    gt_embeddings, gt_angle_info = load_ground_truth_embeddings()
    
    if len(gt_embeddings) == 0:
        print("[ERROR] No ground truth embeddings found!")
        return
    
    # Compare ground truth embeddings
    compare_fleft_fright(gt_embeddings)
    
    # Load CCTV data
    print("\n[STEP 2] Loading CCTV embeddings...")
    cctv_bank, cctv_angle_info = load_cctv_embeddings()
    
    if cctv_bank is None:
        print("[ERROR] Failed to load CCTV data!")
        return
    
    # Evaluate similarities
    print("\n[STEP 3] Evaluating similarities...")
    results = evaluate_similarities(gt_embeddings, gt_angle_info, cctv_bank, cctv_angle_info)
    
    # Save results
    output_dir = PROJECT_ROOT / "outputs" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"yh_fleft_fright_evaluation_{timestamp}.json"
    save_results(results, output_file)
    
    print("\n[COMPLETE] Evaluation finished!")
    print(f"  Results: {output_file}")
    print("="*70)


if __name__ == "__main__":
    main()
