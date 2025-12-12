# backend/api/video.py
"""
비디오 처리 및 로그 API 엔드포인트
"""
import os
import json
import time
import tempfile
import subprocess
import cv2

from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.services.face_detection import process_detection

# 프로젝트 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent.parent

router = APIRouter()

@router.get("/api/logs")
async def get_logs(limit: int = 100, db: Session = Depends(get_db)):
    """감지 로그 조회"""
    from backend.database import DetectionLog
    try:
        logs = db.query(DetectionLog).order_by(DetectionLog.detected_at.desc()).limit(limit).all()
        return {
            "success": True,
            "count": len(logs),
            "logs": [
                {
                    "id": log.id,
                    "person_id": log.person_id,
                    "person_name": log.person_name,
                    "similarity": log.similarity,
                    "is_criminal": log.is_criminal,
                    "status": log.status,
                    "detected_at": log.detected_at.isoformat(),
                    "metadata": log.detection_metadata
                }
                for log in logs
            ]
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "count": 0,
            "logs": []
        }

@router.post("/api/extract_frames")
async def extract_frames(
    video: UploadFile = File(...)
):
    """
    비디오 파일에서 모든 프레임을 추출하여 저장 (라벨링용)
    
    Args:
        video: 비디오 파일
    
    Returns:
        {
            "success": bool,
            "message": str,
            "total_frames": int,
            "output_dir": str
        }
    """
    try:
        print(f"📹 [EXTRACT FRAMES] 프레임 추출 요청: {video.filename}")
        
        # 임시 파일로 비디오 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as input_file:
            input_path = input_file.name
            content = await video.read()
            input_file.write(content)
        
        # 출력 디렉토리 생성 (비디오 파일명 기반)
        video_name = Path(video.filename).stem if video.filename else f"video_{int(time.time())}"
        output_dir = PROJECT_ROOT / "outputs" / "extracted_frames" / video_name
        annotations_dir = output_dir / "annotations"  # JSON 파일 저장 폴더
        output_dir.mkdir(parents=True, exist_ok=True)
        annotations_dir.mkdir(parents=True, exist_ok=True)
        
        # OpenCV로 비디오 열기
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="비디오 파일을 열 수 없습니다.")
        
        # 비디오 정보
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"  📊 비디오 정보:")
        print(f"     - 총 프레임: {total_frames}")
        print(f"     - FPS: {fps:.2f}")
        print(f"     - 해상도: {width}x{height}")
        print(f"     - 출력 디렉토리: {output_dir}")
        print(f"  🔍 얼굴 감지 및 매칭 결과 박스 그리기 활성화")
        
        # DB 세션 생성 (매칭을 위해 필요)
        from backend.database import SessionLocal
        db = SessionLocal()
        
        try:
            # 모든 프레임 추출 (매칭 결과 포함 박스 그리기)
            frame_idx = 0
            saved_count = 0
            total_faces_detected = 0
            total_matches = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 매칭 로직 실행 (브라우저에서 보는 것과 동일한 로직)
                # tracking_state 초기화 (tracks 키 필요)
                tracking_state = {"tracks": {}}
                
                detection_result = process_detection(
                    frame=frame,
                    suspect_ids=None,  # 전체 갤러리 검색
                    db=db,
                    tracking_state=tracking_state  # 프레임별로 독립적으로 처리
                )
                
                # 박스가 그려진 프레임 복사
                frame_with_boxes = frame.copy()
                
                # 매칭 결과에 따라 박스 그리기 및 JSON 데이터 수집
                detections = detection_result.get("detections", [])
                frame_annotations = {
                    "frame_idx": frame_idx,
                    "timestamp": frame_idx / fps if fps > 0 else 0.0,
                    "faces": []
                }
                
                for detection in detections:
                    bbox = detection["bbox"]
                    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                    
                    # 색상 결정 (브라우저와 동일한 로직)
                    status = detection.get("status", "unknown")
                    if status == "criminal":
                        color = (0, 0, 255)  # 빨간색 (BGR)
                        label_color = (0, 0, 255)
                    elif status == "normal":
                        color = (0, 255, 0)  # 초록색 (BGR)
                        label_color = (0, 255, 0)
                    else:  # unknown
                        color = (0, 255, 255)  # 노란색 (BGR)
                        label_color = (0, 255, 255)
                    
                    # 박스 그리기 (두께 3)
                    cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), color, 3)
                    
                    # 레이블 생성 (브라우저와 동일한 정보)
                    name = detection.get("name", "Unknown")
                    confidence = detection.get("confidence", 0)
                    label = f"{name} ({confidence}%)"
                    
                    # 레이블 배경 (가독성 향상)
                    (label_width, label_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    cv2.rectangle(
                        frame_with_boxes,
                        (x1, y1 - label_height - 10),
                        (x1 + label_width, y1),
                        color,
                        -1  # 채워진 사각형
                    )
                    
                    # 레이블 텍스트 (흰색)
                    cv2.putText(
                        frame_with_boxes,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),  # 흰색
                        2
                    )
                    
                    # JSON 어노테이션 데이터 수집
                    face_annotation = {
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "status": status,
                        "name": name,
                        "person_id": detection.get("person_id"),
                        "confidence": confidence,
                        "color": detection.get("color", "yellow"),
                        "angle_type": detection.get("angle_type"),
                        "yaw_angle": detection.get("yaw_angle"),
                        "bank_type": detection.get("bank_type")
                    }
                    frame_annotations["faces"].append(face_annotation)
                    
                    total_faces_detected += 1
                    if detection.get("status") != "unknown":
                        total_matches += 1
                
                # 프레임 저장 (JPEG 형식, 매칭 결과 박스가 그려진 이미지)
                frame_filename = f"frame_{frame_idx:06d}.jpg"
                frame_path = output_dir / frame_filename
                cv2.imwrite(str(frame_path), frame_with_boxes, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                # JSON 어노테이션 저장 (이미지 파일과 쌍으로 저장)
                json_filename = f"frame_{frame_idx:06d}.json"
                json_path = annotations_dir / json_filename
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(frame_annotations, f, indent=2, ensure_ascii=False)
                
                saved_count += 1
                
                # 진행 상황 출력 (100프레임마다)
                if frame_idx % 100 == 0:
                    progress = (frame_idx / total_frames * 100) if total_frames > 0 else 0
                    print(f"  ⏳ 진행 중: {frame_idx}/{total_frames} 프레임 ({progress:.1f}%), 감지된 얼굴: {total_faces_detected}개, 매칭: {total_matches}개")
                
                frame_idx += 1
        finally:
            db.close()
        
        cap.release()
        
        # 임시 파일 삭제
        try:
            os.unlink(input_path)
        except:
            pass
        
        print(f"  ✅ 프레임 추출 완료: {saved_count}개 프레임 저장됨")
        print(f"  👤 총 감지된 얼굴: {total_faces_detected}개")
        print(f"  ✅ 매칭 성공: {total_matches}개")
        print(f"  📁 이미지 저장 위치: {output_dir}")
        print(f"  📄 JSON 저장 위치: {annotations_dir}")
        
        return {
            "success": True,
            "message": f"{saved_count}개의 프레임이 추출되었습니다. (감지된 얼굴: {total_faces_detected}개, 매칭: {total_matches}개)",
            "total_frames": saved_count,
            "total_faces": total_faces_detected,
            "total_matches": total_matches,
            "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
            "annotations_dir": str(annotations_dir.relative_to(PROJECT_ROOT)),
            "video_info": {
                "fps": fps,
                "width": width,
                "height": height,
                "duration": total_frames / fps if fps > 0 else 0
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ [EXTRACT FRAMES] 프레임 추출 실패: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"프레임 추출 중 오류 발생: {str(e)}")

@router.post("/api/extract_clip")
async def extract_clip(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    start_time: float = Form(...),
    end_time: float = Form(...),
    person_name: str = Form("Unknown")
):
    """
    비디오 파일에서 특정 구간을 추출하여 클립 생성
    
    Args:
        video: 비디오 파일
        start_time: 시작 시간 (초)
        end_time: 종료 시간 (초)
        person_name: 범죄자 이름
    
    Returns:
        추출된 클립 파일
    """
    try:
        # 임시 파일 생성
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as input_file:
            input_path = input_file.name
            # 업로드된 비디오 파일 저장
            content = await video.read()
            input_file.write(content)
        
        # 출력 파일 경로
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        
        # ffmpeg를 사용하여 클립 추출
        duration = end_time - start_time
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-ss', str(start_time),
            '-t', str(duration),
            '-c', 'copy',  # 재인코딩 없이 복사 (빠름)
            '-avoid_negative_ts', 'make_zero',
            '-y',  # 덮어쓰기
            output_path
        ]
        
        print(f"🎬 클립 추출 시작: {person_name} ({start_time:.1f}s - {end_time:.1f}s)")
        print(f"📝 명령어: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60  # 60초 타임아웃
        )
        
        if result.returncode != 0:
            print(f"❌ ffmpeg 오류: {result.stderr}")
            raise HTTPException(status_code=500, detail=f"클립 추출 실패: {result.stderr}")
        
        # 임시 입력 파일 삭제
        try:
            os.unlink(input_path)
        except:
            pass
        
        print(f"✅ 클립 추출 완료: {output_path}")
        
        # 응답 후 파일 삭제를 BackgroundTasks로 등록
        background_tasks.add_task(os.unlink, output_path)
        
        # 파일 응답 반환
        return FileResponse(
            output_path,
            media_type='video/mp4',
            filename=f"clip_{person_name}_{start_time:.1f}s-{end_time:.1f}s.mp4"
        )
        
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="클립 추출 시간 초과")
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="ffmpeg가 설치되지 않았습니다. ffmpeg를 설치해주세요.")
    except Exception as e:
        print(f"❌ 클립 추출 오류: {e}")
        raise HTTPException(status_code=500, detail=f"클립 추출 실패: {str(e)}")
    finally:
        # 임시 파일 정리
        try:
            if 'input_path' in locals():
                os.unlink(input_path)
        except:
            pass