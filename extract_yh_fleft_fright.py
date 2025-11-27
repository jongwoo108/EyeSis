"""
yh_fleft.jpeg와 yh_frignt.jpeg 임베딩 추출 전용 스크립트
감지 임계값을 낮추고 다양한 설정으로 시도
"""

import cv2
import numpy as np
from pathlib import Path
import insightface
from insightface.app import FaceAnalysis
import json

def extract_embedding_for_file(app, image_path, output_dir):
    """
    Extract embedding from a single image file
    """
    print(f"\n{'='*70}")
    print(f"[Processing] {image_path.name}")
    print(f"{'='*70}")
    
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  [ERROR] Failed to read image: {image_path}")
        return None
    
    print(f"  [INFO] Image size: {img.shape[1]}x{img.shape[0]}")
    
    # Try multiple detection sizes
    det_sizes = [(640, 640), (1280, 1280), (320, 320), (960, 960)]
    
    for det_size in det_sizes:
        print(f"\n  [DETECT] Detection size: {det_size}")
        
        # Re-prepare with different settings
        app.prepare(ctx_id=0, det_size=det_size, det_thresh=0.3)
        
        # Face detection (lower threshold)
        faces = app.get(img)
        
        if len(faces) > 0:
            print(f"  [SUCCESS] {len(faces)} face(s) detected!")
            
            # Select largest face
            face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            
            # Extract embedding
            embedding = face.embedding
            
            # L2 normalization
            embedding = embedding / np.linalg.norm(embedding)
            
            # Angle information
            if hasattr(face, 'pose'):
                yaw = face.pose[0] if face.pose is not None else 0.0
                pitch = face.pose[1] if face.pose is not None else 0.0
            else:
                yaw = 0.0
                pitch = 0.0
            
            print(f"  [INFO] Embedding shape: {embedding.shape}")
            print(f"  [INFO] Angle - Yaw: {yaw:.1f}, Pitch: {pitch:.1f}")
            print(f"  [INFO] Bounding box: {face.bbox}")
            
            # Extract angle from filename
            filename = image_path.stem
            if 'fleft' in filename:
                angle_type = 'fleft'
            elif 'fright' in filename or 'frignt' in filename:
                angle_type = 'fright'
            else:
                angle_type = 'unknown'
            
            # Save embedding
            output_file = output_dir / f"embedding_{angle_type}.npy"
            np.save(output_file, embedding)
            print(f"  [SAVED] Embedding saved: {output_file}")
            
            # Save angle information
            angle_info = {
                "filename": image_path.name,
                "angle_type": angle_type,
                "yaw": float(yaw),
                "pitch": float(pitch),
                "det_size": det_size,
                "bbox": face.bbox.tolist()
            }
            
            angle_file = output_dir / f"angle_{angle_type}.json"
            with open(angle_file, 'w', encoding='utf-8') as f:
                json.dump(angle_info, f, indent=2, ensure_ascii=False)
            print(f"  [SAVED] Angle info saved: {angle_file}")
            
            return {
                "embedding": embedding,
                "angle_info": angle_info
            }
    
    print(f"  [ERROR] Face detection failed for all detection sizes")
    print(f"  [INFO] Tried detection sizes: {det_sizes}")
    
    # Image analysis info
    print(f"\n  [ANALYSIS] Image analysis:")
    print(f"     - Brightness mean: {np.mean(img):.1f}")
    print(f"     - Brightness std: {np.std(img):.1f}")
    print(f"     - Channels: {img.shape[2] if len(img.shape) > 2 else 1}")
    
    return None


def main():
    print("="*70)
    print("[YH EXTRACTION] yh_fleft & yh_fright Embedding Extraction")
    print("="*70)
    
    # Set paths
    image_dir = Path("images/enroll/yh")
    output_dir = Path("outputs/embeddings_yh_frontal")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[INPUT]  Directory: {image_dir.absolute()}")
    print(f"[OUTPUT] Directory: {output_dir.absolute()}")
    
    # Target files
    target_files = [
        image_dir / "yh_fleft.jpeg",
        image_dir / "yh_frignt.jpeg"
    ]
    
    # Check file existence
    print(f"\n[TARGET] Files:")
    for file in target_files:
        status = "[OK]" if file.exists() else "[MISSING]"
        print(f"  {status} {file.name}")
    
    # Initialize InsightFace
    print(f"\n[INIT] Initializing InsightFace...")
    app = FaceAnalysis(
        name='buffalo_l',
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    print(f"[INIT] InsightFace initialized successfully")
    
    # Process each file
    results = {}
    for image_path in target_files:
        if not image_path.exists():
            print(f"\n[WARNING] File does not exist: {image_path}")
            continue
        
        result = extract_embedding_for_file(app, image_path, output_dir)
        results[image_path.name] = result
    
    # Summary
    print(f"\n{'='*70}")
    print("[SUMMARY] Extraction Results")
    print("="*70)
    
    success_count = sum(1 for r in results.values() if r is not None)
    total_count = len(results)
    
    for filename, result in results.items():
        status = "[SUCCESS]" if result else "[FAILED]"
        print(f"  {status} {filename}")
        if result:
            print(f"      Angle: {result['angle_info']['angle_type']}")
            print(f"      Yaw: {result['angle_info']['yaw']:.1f}")
    
    print(f"\nSuccess: {success_count}/{total_count}")
    print(f"\n[OUTPUT] Saved to: {output_dir.absolute()}")
    print("="*70)
    
    # Similarity comparison (if both succeeded)
    if success_count == 2:
        print(f"\n[SIMILARITY] Embedding Similarity Analysis")
        print("="*70)
        
        # Load embeddings
        emb_fleft = np.load(output_dir / "embedding_fleft.npy")
        emb_fright = np.load(output_dir / "embedding_fright.npy")
        
        # Cosine similarity
        similarity = np.dot(emb_fleft, emb_fright)
        print(f"  fleft <-> fright similarity: {similarity:.4f}")
        
        if similarity > 0.6:
            print(f"  [HIGH] Same person (high similarity)")
        elif similarity > 0.4:
            print(f"  [MEDIUM] Needs verification (medium similarity)")
        else:
            print(f"  [LOW] Different person suspected (low similarity)")


if __name__ == "__main__":
    main()
