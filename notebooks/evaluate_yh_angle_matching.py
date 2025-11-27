"""
yh 각도 매칭 기반 평가 스크립트

목적: 정답 데이터의 각도와 일치하는 CCTV 임베딩만 선별해서 비교
- fleft (Yaw: -5.8°, 정면 왼쪽) → CCTV left 각도만
- fright (Yaw: -4.6°, 정면 오른쪽) → CCTV right/front 각도만
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.device_config import _ensure_cuda_in_path
_ensure_cuda_in_path()

import numpy as np
import json
from datetime import datetime


def l2_normalize(vec):
    """L2 normalization"""
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def load_ground_truth():
    """Load ground truth embeddings"""
    gt_dir = PROJECT_ROOT / "outputs" / "embeddings_yh_frontal"
    
    results = {}
    
    for angle_type in ['fleft', 'fright']:
        emb_file = gt_dir / f"embedding_{angle_type}.npy"
        angle_file = gt_dir / f"angle_{angle_type}.json"
        
        if emb_file.exists() and angle_file.exists():
            emb = np.load(emb_file)
            emb = l2_normalize(emb)
            
            with open(angle_file, 'r') as f:
                angle_info = json.load(f)
            
            results[angle_type] = {
                'embedding': emb,
                'yaw': angle_info['yaw'],
                'pitch': angle_info['pitch'],
                'filename': angle_info['filename']
            }
            
            print(f"[LOADED] {angle_type}: Yaw={angle_info['yaw']:.1f}, Pitch={angle_info['pitch']:.1f}")
    
    return results


def load_cctv_data():
    """Load CCTV embeddings with angle info"""
    cctv_dir = PROJECT_ROOT / "outputs" / "embeddings" / "yh"
    
    # Load bank
    bank_file = cctv_dir / "bank_dynamic.npy"
    bank = np.load(bank_file)
    
    # Load angle info
    angle_file = cctv_dir / "angles_dynamic.json"
    with open(angle_file, 'r') as f:
        angle_data = json.load(f)
    
    angle_types = angle_data['angle_types']
    yaw_angles = angle_data['yaw_angles']
    pitch_angles = angle_data.get('pitch_angles', [0] * len(angle_types))
    
    print(f"\n[LOADED] CCTV bank: {bank.shape[0]} embeddings")
    
    return bank, angle_types, yaw_angles, pitch_angles


def match_angles_for_gt(gt_angle_type, gt_yaw, cctv_angle_types):
    """
    Determine which CCTV angle types match the ground truth angle
    
    Args:
        gt_angle_type: 'fleft' or 'fright'
        gt_yaw: Ground truth yaw angle
        cctv_angle_types: List of CCTV angle types
    
    Returns:
        list: Matched angle types for CCTV
    """
    if gt_angle_type == 'fleft':
        # fleft (Yaw: -5.8°) → left 각도
        return ['left']
    elif gt_angle_type == 'fright':
        # fright (Yaw: -4.6°) → front, right 각도
        return ['front', 'right']
    else:
        return []


def evaluate_angle_matching(gt_data, cctv_bank, cctv_angle_types, cctv_yaw, cctv_pitch):
    """
    Evaluate similarity with angle matching
    """
    results = {
        'timestamp': datetime.now().isoformat(),
        'ground_truth': {},
        'comparisons': {}
    }
    
    # Store ground truth info
    for gt_type, gt_info in gt_data.items():
        results['ground_truth'][gt_type] = {
            'yaw': gt_info['yaw'],
            'pitch': gt_info['pitch'],
            'filename': gt_info['filename']
        }
    
    # Evaluate each ground truth
    for gt_type, gt_info in gt_data.items():
        print(f"\n{'='*70}")
        print(f"[EVALUATE] Ground Truth: {gt_type}")
        print(f"  Yaw: {gt_info['yaw']:.1f}, Pitch: {gt_info['pitch']:.1f}")
        print(f"{'='*70}")
        
        gt_emb = gt_info['embedding']
        
        # Determine matching angle types
        matched_angle_types = match_angles_for_gt(gt_type, gt_info['yaw'], cctv_angle_types)
        print(f"\n[ANGLE MATCHING] Looking for CCTV angles: {matched_angle_types}")
        
        # Filter CCTV embeddings by angle
        matched_indices = []
        for idx, angle_type in enumerate(cctv_angle_types):
            if angle_type in matched_angle_types:
                matched_indices.append(idx)
        
        print(f"[MATCHED] Found {len(matched_indices)} CCTV embeddings with matching angles")
        
        if len(matched_indices) == 0:
            print(f"[WARNING] No matching CCTV embeddings found!")
            results['comparisons'][gt_type] = {
                'matched_count': 0,
                'message': 'No matching CCTV embeddings found'
            }
            continue
        
        # Get matched embeddings
        matched_embeddings = cctv_bank[matched_indices]
        matched_angles = [cctv_angle_types[i] for i in matched_indices]
        matched_yaws = [cctv_yaw[i] for i in matched_indices]
        matched_pitches = [cctv_pitch[i] for i in matched_indices]
        
        # Calculate similarities
        similarities = np.dot(matched_embeddings, gt_emb)
        
        # Statistics
        print(f"\n[STATISTICS]")
        print(f"  Matched CCTV count: {len(similarities)}")
        print(f"  Mean similarity: {np.mean(similarities):.4f}")
        print(f"  Std similarity: {np.std(similarities):.4f}")
        print(f"  Max similarity: {np.max(similarities):.4f}")
        print(f"  Min similarity: {np.min(similarities):.4f}")
        
        # Top matches
        print(f"\n[TOP MATCHES]")
        sorted_indices = np.argsort(similarities)[::-1]
        for rank, idx in enumerate(sorted_indices[:min(10, len(sorted_indices))], 1):
            sim = similarities[idx]
            angle = matched_angles[idx]
            yaw = matched_yaws[idx]
            pitch = matched_pitches[idx]
            print(f"  {rank:2d}. Similarity: {sim:.4f}, Angle: {angle:10s}, Yaw: {yaw:6.1f}, Pitch: {pitch:6.1f}")
        
        # Angle breakdown
        print(f"\n[ANGLE BREAKDOWN]")
        from collections import defaultdict
        angle_sims = defaultdict(list)
        for idx, (sim, angle) in enumerate(zip(similarities, matched_angles)):
            angle_sims[angle].append(sim)
        
        for angle in sorted(angle_sims.keys()):
            sims = angle_sims[angle]
            print(f"  {angle:10s}: Count={len(sims):3d}, Mean={np.mean(sims):.4f}, "
                  f"Std={np.std(sims):.4f}, Max={np.max(sims):.4f}")
        
        # Threshold analysis
        print(f"\n[THRESHOLD ANALYSIS]")
        for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
            count = np.sum(similarities >= threshold)
            percentage = count / len(similarities) * 100
            print(f"  Similarity >= {threshold:.1f}: {count:3d} ({percentage:5.1f}%)")
        
        # Store results
        results['comparisons'][gt_type] = {
            'matched_count': len(similarities),
            'matched_angles': matched_angle_types,
            'statistics': {
                'mean': float(np.mean(similarities)),
                'std': float(np.std(similarities)),
                'max': float(np.max(similarities)),
                'min': float(np.min(similarities))
            },
            'top_matches': [
                {
                    'rank': rank,
                    'similarity': float(similarities[idx]),
                    'angle': matched_angles[idx],
                    'yaw': float(matched_yaws[idx]),
                    'pitch': float(matched_pitches[idx]),
                    'cctv_index': int(matched_indices[idx])
                }
                for rank, idx in enumerate(np.argsort(similarities)[::-1][:10], 1)
            ],
            'angle_breakdown': {
                angle: {
                    'count': len(sims),
                    'mean': float(np.mean(sims)),
                    'std': float(np.std(sims)),
                    'max': float(np.max(sims)),
                    'min': float(np.min(sims))
                }
                for angle, sims in angle_sims.items()
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


def main():
    print("="*70)
    print("[YH ANGLE MATCHING EVALUATION]")
    print("="*70)
    
    # Load ground truth
    print("\n[STEP 1] Loading ground truth...")
    gt_data = load_ground_truth()
    
    # Load CCTV
    print("\n[STEP 2] Loading CCTV data...")
    cctv_bank, cctv_angles, cctv_yaw, cctv_pitch = load_cctv_data()
    
    # Evaluate
    print("\n[STEP 3] Evaluating with angle matching...")
    results = evaluate_angle_matching(gt_data, cctv_bank, cctv_angles, cctv_yaw, cctv_pitch)
    
    # Save results
    output_dir = PROJECT_ROOT / "outputs" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"yh_angle_matching_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n[SAVED] Results saved to: {output_file}")
    print("="*70)


if __name__ == "__main__":
    main()
