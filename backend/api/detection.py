# backend/api/detection.py
"""
얼굴 감지 API 엔드포인트
"""
import base64
import json
import asyncio
import cv2
import numpy as np

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models.schemas import DetectionRequest
from backend.services.face_detection import process_detection
from backend.services.temporal_filter import apply_temporal_filter
from backend.services.bank_manager import (
    add_embedding_to_bank_async,
    add_embedding_to_dynamic_bank_async
)
from backend.utils.image_utils import base64_to_image
from backend.utils.websocket_manager import (
    active_connections,
    register_connection,
    unregister_connection,
    connection_states
)

router = APIRouter()


@router.post("/api/detect")
async def detect_faces(request: DetectionRequest, db: Session = Depends(get_db)):
    """
    얼굴 감지 및 인식 (HTTP API - 호환성 유지)
    
    Args:
        request: DetectionRequest (image: Base64, suspect_id: 선택적)
        db: 데이터베이스 세션
    
    Returns:
        {
            "success": bool,
            "detections": [...],  # 박스 좌표 및 메타데이터 배열
            "alert": bool,
            "metadata": {...}
        }
    """
    # 1. 이미지 디코딩
    frame = base64_to_image(request.image)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image data")
    
    # 2. 공통 감지 로직 사용 (suspect_ids 우선, 없으면 suspect_id 사용)
    result = process_detection(
        frame, 
        suspect_id=request.suspect_id, 
        suspect_ids=request.suspect_ids,
        db=db
    )
    
    # 3. 범죄자 감지 시 스냅샷 Base64 인코딩 추가 (HTTP API용)
    snapshot_base64 = None
    video_timestamp = None
    
    if result.get("alert"):  # 범죄자 감지됨
        print(f"🚨 HTTP API: 범죄자 감지됨! 스냅샷 생성 중...")
        try:
            # 프레임을 JPEG로 인코딩하여 Base64 생성
            success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if success and buffer is not None and len(buffer) > 0:
                snapshot_base64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')
                print(f"✅ HTTP API: 스냅샷 생성 완료: 크기={len(snapshot_base64)} bytes")
            else:
                print(f"⚠️ HTTP API: 스냅샷 인코딩 실패 (success={success}, buffer={buffer is not None})")
        except Exception as e:
            print(f"❌ HTTP API: 스냅샷 생성 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
    
    # 4. 결과 반환
    response = {
        "success": True,
        **result
    }
    
    # 범죄자 감지 시 스냅샷 추가
    if snapshot_base64:
        response["snapshot_base64"] = snapshot_base64
        response["video_timestamp"] = video_timestamp  # None이지만 필드 추가
        print(f"📤 HTTP API 응답에 스냅샷 포함: {len(snapshot_base64)} bytes")
    
    return response


@router.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    """
    WebSocket을 통한 실시간 얼굴 감지 및 인식
    
    메시지 형식:
    - 클라이언트 → 서버:
        {
            "type": "frame",
            "data": {
                "image": "base64_string",
                "suspect_id": "optional_id",
                "frame_id": 123
            }
        }
        또는
        {
            "type": "config",
            "suspect_id": "optional_id"
        }
    
    - 서버 → 클라이언트:
        {
            "type": "detection",
            "data": {
                "frame_id": 123,
                "detections": [...],
                "alert": false,
                "metadata": {...}
            }
        }
        또는
        {
            "type": "error",
            "message": "error message"
        }
    """
    # WebSocket 연결 수락 (CORS 허용)
    try:
        print(f"🔌 [메인] WebSocket 연결 시도: {websocket.client}")
        print(f"   URL: {websocket.url}")
        print(f"   Path: {websocket.url.path}")
        origin = websocket.headers.get("origin")
        print(f"   Origin: {origin}")
        print(f"   Headers: {dict(websocket.headers)}")
        
        # WebSocket 연결 수락 (모든 origin 허용)
        await websocket.accept()
        print(f"✅ [메인] WebSocket 연결 수락됨")
        
        # 연결 등록
        active_connections.add(websocket)
        connection_states[websocket] = {
            "suspect_ids": [],  # 여러 명 선택 가능
            "connected_at": asyncio.get_event_loop().time(),
            "match_counters": {},  # person_id별 연속 매칭 프레임 카운터 (하위 호환용 유지)
            "match_history": {},   # person_id별 최근 프레임 이력: {person_id: [(confidence, matched), ...]}
            "tracking_state": {
                "tracks": {}  # bbox tracking 상태
            }
        }
        print(f"✅ [메인] WebSocket 연결됨 (총 {len(active_connections)}개 연결)")
        
    except Exception as e:
        print(f"❌ [메인] WebSocket 연결 수락 실패: {e}")
        import traceback
        traceback.print_exc()
        try:
            await websocket.close()
        except:
            pass
        return
    
    try:
        while True:
            # 클라이언트로부터 메시지 수신
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                msg_type = message.get("type")
                
                if msg_type == "frame":
                    # 프레임 처리 요청
                    frame_data = message.get("data", {})
                    image_base64 = frame_data.get("image")
                    suspect_ids = frame_data.get("suspect_ids")  # 배열로 받음
                    suspect_id = frame_data.get("suspect_id")  # 호환성 유지 (단일)
                    frame_id = frame_data.get("frame_id", 0)
                    video_time = frame_data.get("video_time")  # 비디오 시간 (초 단위)
                    
                    # 연결 상태에서 suspect_ids 업데이트
                    if suspect_ids is not None:
                        connection_states[websocket]["suspect_ids"] = suspect_ids
                    elif suspect_id is not None:
                        # 단일 suspect_id를 배열로 변환 (호환성)
                        connection_states[websocket]["suspect_ids"] = [suspect_id]
                    else:
                        # 연결 상태에서 suspect_ids 사용
                        suspect_ids = connection_states[websocket].get("suspect_ids", [])
                    
                    if not image_base64:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Missing image data"
                        })
                        continue
                    
                    # 이미지 디코딩
                    frame = base64_to_image(image_base64)
                    if frame is None:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Invalid image data"
                        })
                        continue
                    
                    # 각 요청마다 새로운 DB 세션 생성 (연결 유지 시 세션 문제 방지)
                    db = next(get_db())
                    try:
                        # tracking_state 가져오기
                        tracking_state = connection_states[websocket].get("tracking_state", {"tracks": {}})
                        
                        # 공통 감지 로직 사용 (suspect_ids 우선)
                        result = process_detection(
                            frame, 
                            suspect_id=suspect_id if not suspect_ids else None,
                            suspect_ids=suspect_ids if suspect_ids else None,
                            db=db,
                            tracking_state=tracking_state
                        )
                        
                        # tracking_state 업데이트
                        connection_states[websocket]["tracking_state"] = tracking_state
                    finally:
                        db.close()
                    
                    # Temporal Consistency 필터 적용 (연속 프레임 기반 매칭 확정)
                    result = apply_temporal_filter(websocket, result)
                    
                    # 범죄자 감지 시 스냅샷 Base64 인코딩 추가
                    snapshot_base64 = None
                    
                    # 비디오 타임스탬프 계산 (모든 응답에 포함)
                    if video_time is not None:
                        video_timestamp = float(video_time)
                    else:
                        # 프레임 ID를 사용하여 대략적인 타임스탬프 계산 (10 FPS 가정)
                        video_timestamp = frame_id / 10.0
                    
                    print(f"🔍 WebSocket 감지 결과: alert={result.get('alert')}, detections={len(result.get('detections', []))}, video_time={video_timestamp:.2f}s")
                    
                    if result.get("alert"):  # 범죄자 감지됨
                        print(f"🚨 범죄자 감지됨! 스냅샷 생성 중...")
                        try:
                            # 프레임을 JPEG로 인코딩하여 Base64 생성
                            success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                            if success and buffer is not None and len(buffer) > 0:
                                snapshot_base64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')
                                print(f"✅ 스냅샷 생성 완료: 크기={len(snapshot_base64)} bytes, 타임스탬프={video_timestamp:.1f}s")
                            else:
                                print(f"⚠️ WebSocket: 스냅샷 인코딩 실패 (success={success}, buffer={buffer is not None})")
                        except Exception as e:
                            print(f"❌ WebSocket: 스냅샷 생성 중 오류 발생: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    # 결과 전송 (응답 먼저 - 성능 최우선)
                    response_data = {
                        "type": "detection",
                        "data": {
                            "frame_id": frame_id,
                            "video_timestamp": video_timestamp,  # 항상 포함
                            **result
                        }
                    }
                    
                    # 범죄자 감지 시 스냅샷 추가
                    if snapshot_base64:
                        response_data["data"]["snapshot_base64"] = snapshot_base64
                        print(f"📤 WebSocket 응답에 스냅샷 포함: {len(snapshot_base64)} bytes")
                    
                    await websocket.send_json(response_data)

                    
                    # 학습 이벤트가 있으면 파일 저장 (비동기, 응답 후)
                    learning_events = result.get("learning_events", [])
                    for event in learning_events:
                        # 임베딩을 numpy 배열로 변환
                        embedding_array = np.array(event["embedding"], dtype=np.float32)
                        bank_type = event.get("bank_type", "base")
                        
                        # 동적 bank 저장 (각도별 다양성 체크 및 수집 완료 로직 포함)
                        # ⚠️ Dynamic Bank 자동 수집 활성화
                        if bank_type == "dynamic":
                            # 파일 저장은 백그라운드에서 비동기 처리 (응답 지연 없음)
                            asyncio.create_task(add_embedding_to_dynamic_bank_async(
                                event["person_id"],
                                embedding_array,
                                event.get("angle_type"),
                                event.get("yaw_angle"),
                                similarity_threshold=0.9,
                                verbose=True
                            ))
                        else: # Dynamic이 아니면 Masked/Base 처리
                            # 기존 masked/base bank 저장 (호환성 유지)
                            asyncio.create_task(add_embedding_to_bank_async(
                                event["person_id"],
                                embedding_array,
                                event.get("angle_type"),
                                event.get("yaw_angle"),
                                bank_type=bank_type
                            ))
                
                elif msg_type == "config":
                    # 설정 변경 (suspect_ids 등)
                    suspect_ids = message.get("suspect_ids")  # 배열로 받음
                    suspect_id = message.get("suspect_id")  # 호환성 유지 (단일)
                    
                    if suspect_ids is not None:
                        connection_states[websocket]["suspect_ids"] = suspect_ids
                    elif suspect_id is not None:
                        # 단일 suspect_id를 배열로 변환 (호환성)
                        connection_states[websocket]["suspect_ids"] = [suspect_id]
                    
                    await websocket.send_json({
                        "type": "config_updated",
                        "suspect_ids": connection_states[websocket].get("suspect_ids", [])
                    })
                
                elif msg_type == "ping":
                    # 연결 확인
                    await websocket.send_json({
                        "type": "pong"
                    })
                
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Unknown message type: {msg_type}"
                    })
            
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON format"
                })
            except Exception as e:
                print(f"⚠️ WebSocket 처리 오류: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
    
    except WebSocketDisconnect:
        print("WebSocket 연결이 끊어졌습니다")
    except Exception as e:
        print(f"⚠️ WebSocket 오류: {e}")
    finally:
        unregister_connection(websocket)


@router.get("/api/health")
async def health_check():
    """서버 상태 확인 (WebSocket 연결 테스트용)"""
    return {
        "status": "ok",
        "websocket_endpoint": "/ws/detect",
        "active_connections": len(active_connections),
        "websocket_url": "ws://localhost:5000/ws/detect"
    }


@router.websocket("/ws/test")
async def websocket_test(websocket: WebSocket):
    """WebSocket 연결 테스트용 간단한 엔드포인트"""
    try:
        print(f"🔌 [테스트] WebSocket 연결 시도: {websocket.client}")
        await websocket.accept()
        print(f"✅ [테스트] WebSocket 연결됨")
        
        await websocket.send_json({
            "type": "test",
            "message": "WebSocket 연결 성공!"
        })
        
        # 간단한 에코 테스트
        while True:
            data = await websocket.receive_text()
            await websocket.send_json({
                "type": "echo",
                "message": f"받은 메시지: {data}"
            })
    except WebSocketDisconnect:
        print("⚠️ [테스트] WebSocket 연결 종료")
    except Exception as e:
        print(f"❌ [테스트] WebSocket 오류: {e}")
        import traceback
        traceback.print_exc()